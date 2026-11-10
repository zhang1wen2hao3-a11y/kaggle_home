#!/usr/bin/env python3
"""Phase A：冻结 Encoder + 新 Head 的 BC 训练（"共享 Encoder + N 个 specialist Head" 的最小验证）。

问题：G6 的 Encoder（3.75M 参数，74.9%）是 db04 的 PPO 训出来的。
      换一个 specialist（如 26c6），只靠一个 495k 参数的 Head 能不能适应？

做法：加载 `--agent` 的权重（默认 g6_db04）→ 冻结除 head 外所有参数 →
      在 `--npz` 的帧上做 BC 交叉熵 → 存成完整 state_dict（可直接 stage 成 agent）。

head = q_mlp + score_mlp + opt_type_emb + opt_card + opt_attack（495k）
      （这些正好是 model.py §5 option scorer 的可替换部分；readout 属于 encoder）

--reset-head：把头重新随机初始化（用同一个类的 __init__，保证 init 与从零训练一致）。
              **这是最纯粹的检验**：encoder 若真的 deck-agnostic，随机头也能训起来；
              若 encoder 已把 db04 的打法刻进 readout/ln_f，随机头会卡住。

用法：
    python3 tools/head_train.py --npz data/bc_26c6.npz --agent build/agents/g6_db04 \
        --reset-head --epochs 12 --lr 2e-4 --bs 256 --device cuda \
        --out /tmp/head_26c6.pt
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUNDLE = pathlib.Path(os.environ.get("PTCG_BUNDLE") or (ROOT / "build" / "port_plain"))
for p in (ROOT, ROOT / "port_repo", ROOT / "tools", BUNDLE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np            # noqa: E402
import torch                  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ptcg.rl.search_agent import SearchAgent                # noqa: E402
from ptcg.rl.model import PTCGTransformer, arch_from_state_dict  # noqa: E402

# head = 可替换部分（§5 option scorer）。readout 属于 encoder，**不在**这里。
HEAD_PREFIXES = ("q_mlp.", "score_mlp.", "opt_type_emb.", "opt_card.", "opt_attack.")


def is_head(name: str) -> bool:
    return name.startswith(HEAD_PREFIXES)


def build_agent(agent_dir: pathlib.Path, device: str):
    return SearchAgent(model_path=str(agent_dir / "model.pt"),
                       deck_path=str(agent_dir / "deck.csv"),
                       cfg=None, seed=0, device=device, data_dir=str(agent_dir))


def reset_head(policy: PTCGTransformer, device: str):
    """用**同一个类**重新构造并只取 head 权重。

    ⚠️ 不要手写 init：score_mlp 最后一层是 `orthogonal_(gain=0.01)`，
       手抄容易漏。构造一个同构新模型，其 __init__ 已保证 init 一致。
    """
    sd = {k.replace("module.", ""): v for k, v in policy.state_dict().items()}
    fresh = PTCGTransformer(**arch_from_state_dict(sd, heads=8)).to(device)
    fsd = fresh.state_dict()
    with torch.no_grad():
        for n, p in policy.named_parameters():
            if is_head(n) and n in fsd:
                p.copy_(fsd[n])
    del fresh
    return policy


def freeze_encoder(policy: PTCGTransformer, extra_prefixes=()):
    """默认只放开 head；`extra_prefixes` 里的模块一并放开（部分解冻实验）。

    "冻结 encoder 打不出强度"之后要回答的是：encoder 到底要改多少才够。
    按 Transformer 块从后往前解冻，是最省的改法（后层承载卡组相关的抽象）。
    """
    n_head = n_enc = 0
    for n, p in policy.named_parameters():
        trainable = is_head(n) or (extra_prefixes and n.startswith(tuple(extra_prefixes)))
        p.requires_grad_(trainable)
        if trainable:
            n_head += p.numel()
        else:
            n_enc += p.numel()
    return n_head, n_enc


@torch.no_grad()
def evaluate(policy, X, y, device, bs=512, idx=None):
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


def _subsample_blocks(idx, target, block=512):
    n = len(idx)
    if target >= n:
        return idx
    nb = max(1, target // block)
    block = max(1, min(block, n // nb))
    starts = np.linspace(0, n - block, nb).astype(np.int64)
    return np.sort(np.concatenate([idx[s:s + block] for s in starts]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--x-dat", default=None)
    ap.add_argument("--agent", default="build/agents/g6_db04",
                    help="提供 encoder 权重的 agent 目录")
    ap.add_argument("--unfreeze", default="",
                    help="逗号分隔的参数前缀，连同 head 一起训练（部分解冻）。"
                         "例：\"blocks.2.,blocks.3.,ln_f.\"（只改后两层 + 归一化）")
    ap.add_argument("--reset-head", action="store_true",
                    help="把 head 重新随机初始化（最纯粹的'encoder 是否通用'检验）")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--shuffle-blocks", type=int, default=256)
    ap.add_argument("--train-eval-n", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    z = np.load(ROOT / a.npz)
    y, arch, split = z["y"], z["arch"], z["split"]
    xdat = pathlib.Path(a.x_dat) if a.x_dat else (ROOT / a.npz).with_suffix(".f32")
    if not xdat.exists():
        raise SystemExit(f"✗ 缺 {xdat}")
    X = np.memmap(xdat, dtype=np.float32, mode="r", shape=(len(y), 10944))
    tr = np.where(split == "train")[0]
    va = np.where(split == "val")[0]
    if a.max_frames and len(tr) > a.max_frames:
        tr = _subsample_blocks(tr, a.max_frames)
        va = _subsample_blocks(va, max(1, int(a.max_frames * 0.25)))
    print(f"数据 {X.shape}  train {len(tr):,}  val {len(va):,}  原型 {sorted(set(arch.tolist()))}")

    agent_dir = ROOT / a.agent
    ag = build_agent(agent_dir, dev)
    policy = ag.policy
    print(f"encoder 来源 {a.agent}")

    if a.reset_head:
        reset_head(policy, dev)
        print("head 已重置为随机初始化")

    extra = tuple(x.strip() for x in a.unfreeze.split(",") if x.strip())
    n_head, n_enc = freeze_encoder(policy, extra)
    print(f"冻结 {n_enc:,} 参数；可训 {n_head:,} 参数"
          + (f"（head + 解冻 {extra}）" if extra else "（仅 head）"))
    head_params = [p for n, p in policy.named_parameters() if p.requires_grad]

    l_tr, a_tr, _ = evaluate(policy, X, y, dev, idx=tr)
    l_va, a_va, _ = evaluate(policy, X, y, dev, idx=va)
    print(f"\n=== 起点（未训练）===")
    print(f"  train loss {l_tr:.4f}  一致率 {a_tr:.1%}")
    print(f"  val   loss {l_va:.4f}  一致率 {a_va:.1%}")
    base = dict(train_loss=l_tr, train_acc=a_tr, val_loss=l_va, val_acc=a_va)

    opt = torch.optim.AdamW(head_params, lr=a.lr, weight_decay=a.wd)
    steps_per_ep = max(1, len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=max(1, steps_per_ep * a.epochs), pct_start=0.15)
    policy.train()
    best, best_ep = -1.0, {}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== 训练 {a.epochs} epoch, lr {a.lr}, bs {a.bs}, {steps_per_ep} step/epoch ===")
    for ep in range(1, a.epochs + 1):
        rng = np.random.default_rng(ep)
        perm = rng.permutation(tr)
        t0 = time.time()
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            idx = np.sort(perm[i:i + a.bs])
            xb = torch.from_numpy(np.asarray(X[idx])).to(dev)
            yb = torch.from_numpy(y[idx].astype(np.int64)).to(dev)
            lg = policy.forward_eval(xb)[0]
            loss = F.cross_entropy(lg, yb)
            if not torch.isfinite(loss):
                raise SystemExit(f"✗ loss 非有限值 {loss.item()}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            opt.step()
            sched.step()
        l_va, a_va, _ = evaluate(policy, X, y, dev, idx=va)
        tr_idx = tr if a.train_eval_n < 0 else tr[: min(a.train_eval_n, len(tr))]
        l_tr_e, a_tr_e, _ = evaluate(policy, X, y, dev, idx=tr_idx)
        tag = ""
        if a_va > best:
            best = a_va
            best_ep = dict(epoch=ep, val_loss=l_va, val_acc=a_va,
                           train_loss=l_tr_e, train_acc=a_tr_e)
            torch.save({k: v.detach().clone() for k, v in policy.state_dict().items()}, out)
            tag = f"  ✅ 存 {out.name}"
        print(f"  ep{ep}  tr_loss {l_tr_e:.4f}  tr_acc {a_tr_e:.1%}  "
              f"va_loss {l_va:.4f}  va_acc {a_va:.1%}  ({time.time()-t0:.0f}s){tag}")

    json.dump(dict(agent=a.agent, reset_head=bool(a.reset_head), npz=a.npz,
                   n_train=int(len(tr)), n_val=int(len(va)), epochs=a.epochs,
                   lr=a.lr, bs=a.bs, archs=sorted(set(arch.tolist())),
                   n_head_params=int(n_head), n_enc_params=int(n_enc),
                   baseline=base, best=best_ep, out=str(out)),
              open(str(out) + ".json", "w"), ensure_ascii=False, indent=1)
    print(f"\n最佳 val 一致率 {best:.1%} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
