#!/usr/bin/env python3
"""从**强对手的回放**里做行为克隆（BC）——可行性探针。

## 为什么这条路和之前所有学习尝试不同

| 尝试 | 老师是谁 | 强弱 |
|---|---|---|
| Q（失败） | 我们自己的 rollout 打的标签 | 只比规则强 +70 LB |
| RL（失败） | 环境胜负（1 bit / 217 决策） | 无信号 |
| **BC** | **榜首 YumeNeko 1369 / sue124 1277** | **是我们的 3.6 倍** |

这是今天第一次有**真正比我们强的老师**。

## 关键事实（已实测确认）

回放里每个 agent 自己的 `observation` **包含它自己的手牌**，对手的手牌才是 None。
所以每一步都带着"它当时看到了什么 + 它选了什么" —— 这正是 BC 需要的。

## 探针判据（跑之前定）

    留出局的 **top-1 准确率 ≥ 60%**（随机基线 = 每步 1/k，平均约 20%）
      → 通过：策略可学，值得投集成（当 rollout 的续弈策略）
      → 卡在 25~35%：说明这套特征表达不了强策略，不必再谈集成

用法：
    ./run.sh train_bc.py --dir epsoide_field --epochs 8
    ./run.sh train_bc.py --dir epsoide_field --main-only      # 只学 MAIN 决策
"""
import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from framework.features import (  # noqa: E402
    OPTION_FEATURE_KEYS, STATE_FEATURE_KEYS, option_features, state_features,
)
from framework.rollout import is_main, rule_action  # noqa: E402

N_S, N_O = len(STATE_FEATURE_KEYS), len(OPTION_FEATURE_KEYS)
CARD_VOCAB_N = 320          # 最高频的 320 张牌做 one-hot，其余进 "other"


def resolve_cid(obs, opt):
    """把 option 的 (area,index) 解析成 cardId。

    ⚠️ 引擎的一个坑：MAIN 里的 `PLAY`(7) / `EVOLVE`(9) 选项**只有 index、没有 area**，
    隐藏的区是**手牌**；`ATTACH`(8) 才带 area=2 + inPlayArea/inPlayIndex（目标是哪只）。
    不兜这一层的话，MAIN 决策的卡牌身份会全部丢失（实测词表只剩 54 张）。
    """
    area, idx = opt.get("area"), opt.get("index")
    if idx is None:
        return None
    if area is None:
        if int(opt.get("type", -1)) in (7, 8, 9):     # PLAY / ATTACH / EVOLVE
            area = 2                                   # 隐含：手牌
        else:
            return None
    cur = obs.get("current") or {}
    pls = cur.get("players") or []
    j = int(cur.get("yourIndex", 0))
    if area == 1:
        arr = (obs.get("select") or {}).get("deck") or []
        return (arr[idx].get("id") if 0 <= idx < len(arr) and arr[idx] else None)
    if len(pls) < 2:
        return None
    arr = {2: pls[j].get("hand"), 3: pls[j].get("discard"), 4: pls[j].get("active"),
           5: pls[j].get("bench"), 6: pls[j].get("prize"),
           12: cur.get("looking")}.get(area) or []
    if 0 <= idx < len(arr) and arr[idx]:
        return arr[idx].get("id")
    return None

OUR_NAME = "zhang1wen2hao3"
CACHE = os.path.join(ROOT, "data", "bc_dataset.npz")


def extract(directory, max_games=None, main_only=False):
    """directory 支持逗号分隔的多个目录。

    ⚠️ `epsoide/`（我们自己的 316 局）也要抽 —— **我们的对手也是强 agent**，
    每局给出一条完整的对手决策轨迹。不抽它等于白丢 4 倍局数。
    """
    files = []
    for d in str(directory).split(","):
        files += sorted(glob.glob(os.path.join(ROOT, d.strip(), "*.json")))
    files = sorted(set(files))
    if max_games:
        files = files[:max_games]
    print(f"  扫描 {len(files)} 个回放 …", flush=True)
    S, O, Y, G, K, AGREE, CID = [], [], [], [], [], [], []
    stats = collections.Counter()
    for gi, f in enumerate(files):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        names = [a.get("Name") or "" for a in ((d.get("info") or {}).get("Agents") or [])]
        if len(names) < 2:
            continue
        for st in d.get("steps") or []:
            for j, ent in enumerate(st):
                obs = ent.get("observation") or {}
                act = ent.get("action")
                sel = obs.get("select")
                if sel is None or not act or not obs.get("current"):
                    continue
                if OUR_NAME in (names[j] if j < len(names) else ""):
                    continue                      # 跳过我们自己
                opts = sel.get("option") or []
                if len(opts) < 2 or not (0 <= act[0] < len(opts)):
                    continue
                is_m = is_main(sel)
                if main_only and not is_m:
                    continue
                try:
                    sf = np.array([state_features(obs).get(k, 0.0)
                                   for k in STATE_FEATURE_KEYS], np.float32)
                    ofs = np.stack([
                        np.array([option_features(obs, i).get(k, 0.0)
                                  for k in OPTION_FEATURE_KEYS], np.float32)
                        for i in range(len(opts))
                    ])
                except Exception:
                    stats["特征提取失败"] += 1
                    continue
                if not np.isfinite(sf).all() or not np.isfinite(ofs).all():
                    stats["含 NaN"] += 1
                    continue
                ids = [resolve_cid(obs, o) for o in opts]
                S.append(sf); O.append(ofs); Y.append(int(act[0])); CID.append(ids)
                G.append(gi); K.append(len(opts))
                # 我们的规则会不会做同样选择（衡量"强 agent 与我们的差异"）
                try:
                    rb = rule_action(sel)
                    AGREE.append(int(bool(rb) and rb[0] == act[0]))
                except Exception:
                    AGREE.append(0)
                stats["MAIN" if is_m else "非MAIN"] += 1
    print(f"  抽到 {len(S)} 个决策：{dict(stats)}")
    return S, O, Y, G, K, AGREE, CID


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="epsoide_field")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--main-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(max(os.cpu_count() // 2, 2))
    from framework.policy_nn import OptionScorer
    torch.manual_seed(0); np.random.seed(0)

    if args.reuse_cache and os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        S = list(z["S"]); O = list(z["O"]); Y = list(z["Y"])
        G = list(z["G"]); K = list(z["K"]); A = list(z["AGREE"])
        CID = [list(x) for x in z["CID"]] if "CID" in z.files else [[] for _ in S]
        print(f"  复用缓存：{len(S)} 个决策")
    else:
        print("=" * 74)
        print(f"从强对手回放抽 BC 数据：{args.dir}  main_only={args.main_only}")
        print("=" * 74)
        S, O, Y, G, K, A, CID = extract(args.dir, args.max_games, args.main_only)
        if not S:
            raise SystemExit("没抽到数据")
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, S=np.array(S), O=np.array(O, dtype=object),
                            Y=np.array(Y), G=np.array(G), K=np.array(K),
                            AGREE=np.array(A), CID=np.array(CID, dtype=object))
        CID = [list(x) for x in np.array(CID, dtype=object)]
        print(f"  已缓存到 {os.path.relpath(CACHE, ROOT)}")

    G = np.asarray(G)
    games = np.unique(G)
    rng = np.random.RandomState(0); rng.shuffle(games)
    n_ho = max(1, int(len(games) * args.holdout))
    ho = set(games[:n_ho].tolist())
    tr_idx = [i for i in range(len(S)) if G[i] not in ho]
    ho_idx = [i for i in range(len(S)) if G[i] in ho]
    print(f"  按局切分：train {len(tr_idx)} 决策 / {len(games) - n_ho} 局；"
          f"holdout {len(ho_idx)} 决策 / {n_ho} 局")
    print(f"  **强 agent 与我们的规则一致率** = {np.mean(A):.1%}"
          f"（越低说明他们的策略越不同）")
    print(f"  随机基线 top-1 ≈ 1/E[k] = {np.mean([1.0 / k for k in K]):.1%}")

    # ---- 卡牌词表（按出现频次取前 CARD_VOCAB_N 张）
    freq = collections.Counter()
    for ids in CID:
        for c in ids:
            if c is not None:
                freq[c] += 1
    vocab = {c: i for i, (c, _) in enumerate(freq.most_common(CARD_VOCAB_N))}
    other = len(vocab)
    n_eff = N_O + other + 1
    print(f"  卡牌词表：{len(vocab)} 张 + other ⇒ 选项特征 {N_O} → {n_eff}")

    def with_cards(i):
        """把第 i 个决策的选项特征拼上卡牌 one-hot。"""
        ofs = O[i]; ids = CID[i]
        extra = np.zeros((ofs.shape[0], other + 1), np.float32)
        for k, c in enumerate(ids):
            extra[k, vocab.get(c, other) if c is not None else other] = 1.0
        return np.concatenate([ofs, extra], axis=1)

    # ---- 特征标准化（用训练集的统计量）
    # 不加这一步会出事：特征里有 HP≈350 这种大数、也有 0/1 布尔，MLP 初始 logit 会很大，
    # 实测 loss 反而 **高于均匀预测**（2.07 > ln(k)≈1.96），也就是"自信地预测错"。
    Xs = np.stack([S[i] for i in tr_idx])
    mu_s, sd_s = Xs.mean(0), Xs.std(0) + 1e-6
    ao = np.concatenate([with_cards(i) for i in tr_idx], axis=0)
    mu_o, sd_o = ao.mean(0), ao.std(0) + 1e-6
    print(f"  特征标准化完成（option 维度 {ao.shape[1]}）")
    _wc_raw = with_cards

    def with_cards(i):                                  # noqa: F811
        ofs = _wc_raw(i)
        return (ofs - mu_o) / sd_o

    def state_of(i):
        return (S[i] - mu_s) / sd_s

    mean_k = float(np.mean([K[i] for i in tr_idx]))
    import math
    print(f"  均匀预测的 loss 参考值 = ln(E[k]) ≈ {math.log(mean_k):.3f}")

    model = OptionScorer(N_S, n_eff)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    def batches(idx, bs=64):
        order = np.random.permutation(len(idx))
        for i in range(0, len(order), bs):
            yield [idx[j] for j in order[i:i + bs]]

    for ep in range(args.epochs):
        model.train()
        tot = corr = 0; loss_sum = 0.0
        for chunk in batches(tr_idx):
            opt.zero_grad()
            losses = []
            for i in chunk:
                st = torch.from_numpy(state_of(i)); of = torch.from_numpy(with_cards(i))
                sc = model.scores(st, of)                     # 不含规则锚
                losses.append(torch.nn.functional.cross_entropy(
                    sc.unsqueeze(0), torch.tensor([Y[i]])))
                corr += int(torch.argmax(sc).item() == Y[i]); tot += 1
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss) * len(chunk)
        # holdout
        model.eval()
        hc = ht = 0
        with torch.no_grad():
            for i in ho_idx:
                st = torch.from_numpy(state_of(i)); of = torch.from_numpy(with_cards(i))
                sc = model.scores(st, of)
                hc += int(torch.argmax(sc).item() == Y[i]); ht += 1
        print(f"  epoch {ep + 1}/{args.epochs}  loss={loss_sum / max(tot, 1):.4f}  "
              f"train top-1={corr / max(tot, 1):.1%}  **holdout top-1={hc / max(ht, 1):.1%}**",
              flush=True)

    torch.save(model.state_dict(), os.path.join(ROOT, "models", "bc_policy.pt"))
    print(f"\n最终 holdout top-1 = {hc / max(ht, 1):.1%}"
          f"（随机 {np.mean([1.0 / k for k in K]):.1%}；判据 ≥60%）")
    print("模型已存 models/bc_policy.pt")


if __name__ == "__main__":
    main()
