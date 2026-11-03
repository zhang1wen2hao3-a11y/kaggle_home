#!/usr/bin/env python3
"""A1：BC 监督微调。

从 `tools/bc_materialize.py --materialize` 产出的 npz 上做交叉熵训练。

关键实现约束（来自 A0 实测）：
  * `forward_eval` 已把非法位设成 -1e9 ⇒ **不需要额外 mask**，`CE(logits, y)` 直接可用
  * 但物化时已剔除"目标位被模型判非法"的帧；这里仍要断言，防止数据换版本后静默失效
  * 每帧的 `n_opt` 只用于统计，不参与 loss（非法位已是 -1e9）

用法：
    # 先看基线（不训练，只报端口在 train/val 上的 loss 与一致率）
    python3 tools/bc_train.py --npz data/bc_rows_v1.npz --baseline
    # 训练
    python3 tools/bc_train.py --npz data/bc_rows_v1.npz --epochs 6 --lr 2e-4 \
        --out models/bc_v1.pt
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
# `ptcg` 这个 python 包住在端口 bundle 里（build/port_plain/ptcg），不在仓库根
# ⇒ 必须把 bundle 一起塞进 sys.path；否则在只有 kit 的机器上（或 cwd 不对时）
#   直接 ModuleNotFoundError: No module named 'ptcg'
BUNDLE = pathlib.Path(os.environ.get("PTCG_BUNDLE") or (ROOT / "build" / "port_plain"))
for p in (ROOT, ROOT / "port_repo", ROOT / "tools", BUNDLE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np            # noqa: E402
import torch                  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ptcg.rl.search_agent import SearchAgent  # noqa: E402


def build_policy(device: str):
    AG = BUNDLE
    ag = SearchAgent(model_path=str(AG / "model.pt"), deck_path=str(AG / "deck.csv"),
                     cfg=None, seed=0, device=device, data_dir=str(AG))
    return ag.policy


@torch.no_grad()
def _shuffled_order(idx, bs, block, rng):
    """生成一个"块内连续、块间打乱"的遍历顺序。

    ⚠️ 为什么不能直接 `rng.permutation(idx)`：memmap 上一批 256 行会被打散到
       平均间隔 34 KB 的位置，200k 次寻道 ⇒ 单个 epoch 55 分钟（实测）。
       按 block 行一块打乱后，读变成顺序 I/O，同一 epoch 只要几分钟。
    """
    n = len(idx)
    block = int(block or 0)
    if block <= max(1, bs):
        return rng.permutation(idx)
    nb = n // block
    if nb == 0:
        return rng.permutation(idx)
    head = idx[:nb * block].reshape(nb, block)[rng.permutation(nb)].reshape(-1)
    tail = idx[nb * block:]
    if len(tail):
        tail = tail[rng.permutation(len(tail))]
    return np.concatenate([head, tail])


def _subsample_blocks(idx, target, block=512):
    """**按连续块**抽样，不是按固定步长跳。

    为什么：memmap 上一次读 512 行 = 22 MB 连续 I/O；如果改成"每隔 32 帧取一帧"，
    每批 256 帧就散在 256 个相隔 1.4 MB 的位置上 —— 实测一个 epoch 从十几秒变成 **133 秒**，
    纯粹被寻道拖死。分块抽样既保住时间轴覆盖（块均匀铺开），又保住顺序 I/O。
    """
    n = len(idx)
    if target >= n:
        return idx
    nb = max(1, target // block)
    block = max(1, min(block, n // nb))
    starts = np.linspace(0, n - block, nb).astype(np.int64)
    return np.sort(np.concatenate([idx[s:s + block] for s in starts]))


def _drop_cache(path):
    """主动丢弃 X 的页缓存。

    memmap 读过的页会计入**进程 RSS**，别人（评测）看起来就是"BC 吃了 3 GB"。
    每轮结束 fadvise(DONTNEED) 一次，把页还给内核 —— 下一轮重新读，但我们本来就是
    I/O 瓶颈，页缓存留着也没让谁变快（同一 epoch 内仍然命中）。
    """
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        pass


def _rss_mb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def _avail_mb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def evaluate(policy, X, y, device, bs=512, idx=None):
    """在 `idx` 选的帧上算 loss/一致率。**必须按批索引进 memmap**。

    ⚠️ 早先这里收的是 `X[tr]`（fancy index on memmap）——那会**把整个子集读进 RAM**。
       254,526 帧 × 10944 × 4 B = 11.1 GB，在 7 GB 机器上直接被内核 OOM kill，
       而且**没有任何报错**（`Killed`，日志只剩 Killed）。所以改成只传下标。
    """
    policy.eval()
    tot = correct = 0
    loss_sum = 0.0
    n = len(idx) if idx is not None else len(y)
    for i in range(0, n, bs):
        sel = slice(i, min(i + bs, n)) if idx is None else np.sort(idx[i:i + bs])
        xb = torch.from_numpy(np.asarray(X[sel])).to(device)
        yb = torch.from_numpy(np.asarray(y[sel]).astype(np.int64)).to(device)
        lg = policy.forward_eval(xb)[0]
        loss_sum += F.cross_entropy(lg, yb, reduction="sum").item()
        correct += int((lg.argmax(1) == yb).sum())
        tot += len(yb)
    policy.train()
    return loss_sum / max(tot, 1), correct / max(tot, 1), tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/bc_rows_v1.npz")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--out", default="models/bc_v1.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-deck", default="", help="只训 deck 前缀匹配的帧（如 0d86）")
    ap.add_argument("--only-agent", default="", help="只训 agent 正则匹配的帧")
    ap.add_argument("--winners-only", action="store_true", help="只训赢家帧（需 meta.won）")
    ap.add_argument("--x-dat", default=None, help="X 的流式 float32 文件（默认与 --npz 同名 .f32）")
    ap.add_argument("--shuffle-blocks", type=int, default=256,
                    help="打乱顺序时以多少帧为一块（0=逐帧全随机）。"
                         "memmap 上逐帧乱序会让每批 256 行散落在 8.8 GB 里，"
                         "实测一个 epoch 从 ~5 分钟拖到 55 分钟（寻道，不是带宽）。"
                         "块内连续 + 块间随机基本无损随机性。")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="只用最多这么多帧（均匀跨整个时间轴抽样）。0=全用。"
                         "本机磁盘只有 ~36 MB/s，202k 帧一个 epoch 就要 55 分钟；"
                         "限到 60k 约 7 分钟，且内存抖动小得多")
    ap.add_argument("--train-eval-n", type=int, default=8000,
                    help="每轮在 train 上抽多少帧算指标（-1=全量）")
    a = ap.parse_args()

    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    z = np.load(ROOT / a.npz)
    y, nopt, arch, split = z["y"], z["nopt"], z["arch"], z["split"]
    # X 走 memmap：本机只有 ~4 GB 可用内存，np.stack 会 OOM（2026-10-28 实测 7.3 万帧即崩）
    xdat = pathlib.Path(a.x_dat) if a.x_dat else (ROOT / a.npz).with_suffix(".f32")
    row_w = 10944
    if not xdat.exists():
        raise SystemExit(f"✗ 缺 {xdat}（物化时流式写出的 X）")
    X = np.memmap(xdat, dtype=np.float32, mode="r", shape=(len(y), row_w))
    import re as _re
    ok = np.ones(len(y), dtype=bool)
    if a.only_deck:
        ok &= np.char.startswith(z["deck"].astype(str), a.only_deck)
    if a.only_agent:
        pat = _re.compile(a.only_agent)
        ok &= np.array([bool(pat.search(str(x))) for x in z["agent"]])
    if a.winners_only:
        if "won" not in z.files:
            raise SystemExit("✗ meta 里没有 won 列")
        ok &= z["won"].astype(bool)
    if a.only_deck or a.only_agent or a.winners_only:
        print(f"元数据过滤：命中 {int(ok.sum()):,}/{len(y):,} 帧"
              f"（deck={a.only_deck or '-'} agent={a.only_agent or '-'} winners={a.winners_only}）")
    tr = np.where((split == "train") & ok)[0]
    va = np.where((split == "val") & ok)[0]

    if a.max_frames and len(tr) > a.max_frames:
        tr = _subsample_blocks(tr, a.max_frames)
        va = _subsample_blocks(va, max(1, int(a.max_frames * 0.25)))
        print(f"--max-frames {a.max_frames}：train {len(tr):,} 帧 / val {len(va):,} 帧（分块抽样）")
    print(f"数据 {X.shape}  行宽 {X.shape[1]}")
    print(f"train {len(tr):,} 帧   val {len(va):,} 帧   原型 {sorted(set(arch.tolist()))}")
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit("✗ train/val 有一侧为空，检查切分")

    print(f"内存预算：可用 {_avail_mb()} MB（本脚本会增加 memmap 页缓存，"
          f"页缓存可回收，但别和评测同时跑）")

    policy = build_policy(dev)

    # ── 一次性校验：目标动作合法性（**只用端口权重**，与物化时的过滤规则一致）
    #    绝不能用训练中不断变化的权重来判断，否则同一帧会时而保留时而丢弃。
    with torch.no_grad():
        checked = 0
        bad = 0
        # 只查**过滤后**的帧：全量查在 529k 帧上要读 23 GB memmap，
        # 实测白白多花 20 分钟，而合法性与 filter 无关。
        # 只查**这次真正要用**的帧（tr+va）：再退一步，只查过滤后的全量也没必要
        keep = np.sort(np.concatenate([tr, va]))
        LEGAL_BS = int(os.environ.get("PTCG_LEGAL_BS", "512"))   # 4096 行在 4 GB 卡上会 OOM
        for i in range(0, len(keep), LEGAL_BS):
            sel = np.sort(keep[i:i + LEGAL_BS])
            xb = torch.from_numpy(np.asarray(X[sel])).to(dev)
            yb = torch.from_numpy(y[sel].astype(np.int64)).to(dev)
            tv = policy.forward_eval(xb)[0].gather(1, yb[:, None]).squeeze(1)
            bad += int((tv < -1e8).sum())
            checked += len(yb)
        if dev == "cuda":
            torch.cuda.empty_cache()
    ratio = 1 - bad / max(checked, 1)
    print(f"\n=== 数据合法性（端口权重，过滤后 {checked:,} 帧）===")
    print(f"  目标动作合法比例 {ratio:.4%}（非法 {bad}）")
    if ratio <= 0.98:
        raise SystemExit(f"✗ 非法目标帧 {1-ratio:.2%} > 2%：停止，检查动作索引/候选顺序/编码约定；"
                         f"**不要放宽阈值**")
    print(f"\n=== 基线（端口权重，未训练）===")
    l_tr, a_tr, _ = evaluate(policy, X, y, dev, idx=tr)
    l_va, a_va, _ = evaluate(policy, X, y, dev, idx=va)
    print(f"  train loss {l_tr:.4f}  一致率 {a_tr:.1%}")
    print(f"  val   loss {l_va:.4f}  一致率 {a_va:.1%}   ← A1 要超过这个")
    base = dict(train_loss=l_tr, train_acc=a_tr, val_loss=l_va, val_acc=a_va)
    if a.baseline:
        json.dump(dict(filter=dict(deck=a.only_deck, agent=a.only_agent,
                                  winners=bool(a.winners_only)),
                       n_train=int(len(tr)), n_val=int(len(va)),
                       baseline=base), open(str(ROOT / a.out) + ".json", "w"),
                  ensure_ascii=False, indent=1)
        return 0

    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_ep = max(1, len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=max(1, steps_per_ep * a.epochs), pct_start=0.15)
    policy.train()
    best = -1.0
    best_ep = {}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== 训练 {a.epochs} epoch, lr {a.lr}, bs {a.bs}, {steps_per_ep} step/epoch ===")
    for ep in range(1, a.epochs + 1):
        perm = _shuffled_order(tr, a.bs, a.shuffle_blocks, np.random.default_rng(ep))
        t0 = time.time()
        run = 0.0
        nb = 0
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            # memmap 上**升序**取行 => 近似顺序 I/O。实测随机 fancy indexing 会让
            # 训练慢 20 倍（6 epoch 从 2 分钟变 44 分钟）。
            idx = np.sort(perm[i:i + a.bs])
            xb = torch.from_numpy(np.asarray(X[idx])).to(dev)
            yb = torch.from_numpy(y[idx].astype(np.int64)).to(dev)
            lg = policy.forward_eval(xb)[0]
            loss = F.cross_entropy(lg, yb)
            if not torch.isfinite(loss):
                raise SystemExit(f"✗ loss 非有限值（{loss.item()}），检查数据/编码")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            nb += 1
        l_va, a_va, _ = evaluate(policy, X, y, dev, idx=va)
        tr_idx = tr if a.train_eval_n < 0 else tr[: min(a.train_eval_n, len(tr))]
        l_tr_e, a_tr_e, _ = evaluate(policy, X, y, dev, idx=tr_idx)
        tag = ""
        if a_va > best:
            best = a_va
            best_ep = dict(epoch=ep, val_loss=l_va, val_acc=a_va,
                           train_loss=l_tr_e, train_acc=a_tr_e)
            sd = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            torch.save(sd, out)
            tag = f"  ✅ 存 {out.name}"
        _drop_cache(xdat)
        print(f"  ep{ep}  tr_loss {l_tr_e:.4f}  tr_acc {a_tr_e:.1%}  "
              f"va_loss {l_va:.4f}  va_acc {a_va:.1%}  ({time.time()-t0:.0f}s)"
              f"  rss {_rss_mb()}M{tag}")
    print(f"\n最佳 val 一致率 {best:.1%}（基线 {base['val_acc']:.1%}）→ {out}")
    json.dump(dict(filter=dict(deck=a.only_deck, agent=a.only_agent,
                              winners=bool(a.winners_only)),
                   n_train=int(len(tr)), n_val=int(len(va)), epochs=a.epochs,
                   lr=a.lr, bs=a.bs, archs=sorted(set(arch.tolist())),
                   baseline=base, best=best_ep, out=str(out)),
              open(str(out) + ".json", "w"), ensure_ascii=False, indent=1)
    print("下一步：")
    print(f"  python3 tools/stage_ckpt.py {out} --out /tmp/ag_bc --bundle build/port_plain \\")
    print("      --deck dunsparce_abra__db04 --force")
    print("  # 同卡组双向 A/B 80 场（EXP-0135），>0.55 才进 gate real")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
