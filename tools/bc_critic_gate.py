#!/usr/bin/env python3
"""偏离门控（critic）离线评估：规则 + 学习到的"该不该偏离"判据。

## 背景：为什么不是"BC 直接当策略"

`logs/bc_rulefeat.log` / `logs/bc_deviate.log` 量出的三段事实：

| 目标 | 模型最好成绩 | 说明 |
|---|---|---|
| 直接学强 agent（11 选 1，top-1） | index 33.0% / 语义 42.0% | **不如我们的规则**（38.9% / 51.3%） |
| 学我们自己的规则（确定性函数） | index 98.2% | 特征足够表达规则 |
| **二选一：规则的选项 vs 强 agent 的选项** | **index 76.8% / 语义 83.6%** | **特征里真的有"强 agent 更偏好谁"的信息** |

第三条是关键：绝对排序学不出来，**相对规则偏好谁学得很好**。所以正确用法不是"模型替代规则"，
而是"模型当门控"—— 只有模型强烈偏好另一个选项时才偏离规则：

    action = argmax_o score(o)   当 score(argmax) - score(rule) > τ
             rule                否则

本脚本就是标定 τ：在 holdout 上扫一遍，看一致率能不能显著超过规则的 51.3%（语义）。

⚠️ 判据（跑之前定死）：
    **语义一致率 ≥ 56%（= 规则 51.3% + 5pp）且偏离率 ≤ 15%**
    → 通过，值得做本地对局 A/B（n≥2000，对 ro_attack_w10）
    → 不过：门控无价值，整数条路线关闭

用法：
    ./run.sh tools/bc_critic_gate.py --epochs 12 --hidden 256
"""
import argparse
import collections
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BC_NO_ANCHOR", "1")      # 必须无锚：锚会让模型直接抄规则

import numpy as np  # noqa: E402

CACHE = os.path.join(ROOT, "data", "bc_dataset_noanchor.npz")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--load", action="store_true", help="跳过训练，直接加载 models/bc_critic.pt 做标定")
    ap.add_argument("--same-type", action="store_true",
                    help="只在**与规则同类**的选项里挑替代（绕开跨类型分数尺度不可比的问题）")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(max(os.cpu_count() // 2, 2))
    torch.manual_seed(0); np.random.seed(0)
    from framework.policy_nn import OptionScorer
    import train_bc

    z = np.load(CACHE, allow_pickle=True)
    S = list(z["S"]); O = list(z["O"]); Y = list(z["Y"])
    G = np.asarray(z["G"]); K = list(z["K"])
    CID = [list(x) for x in z["CID"]]
    TY = [list(x) for x in z["TY"]]
    R = [int(x) for x in z["R"]]
    print(f"缓存 {CACHE.split('/')[-1]}：{len(S)} 个决策 / {len(set(G.tolist()))} 局")

    games = np.unique(G); np.random.RandomState(0).shuffle(games)
    n_ho = max(1, int(len(games) * args.holdout))
    ho = set(games[:n_ho].tolist())
    tr_idx = [i for i in range(len(S)) if G[i] not in ho]
    ho_idx = [i for i in range(len(S)) if G[i] in ho]

    # ---------------- 卡牌词表 + 标准化（只用训练集）
    freq = collections.Counter()
    for i in tr_idx:
        for c in CID[i]:
            if c is not None:
                freq[c] += 1
    vocab = {c: n for n, (c, _) in enumerate(freq.most_common(train_bc.CARD_VOCAB_N))}
    other = len(vocab)
    n_eff = train_bc.N_O + other + 1

    def cards(ids):
        ex = np.zeros((len(ids), other + 1), np.float32)
        for k, c in enumerate(ids):
            ex[k, vocab.get(c, other) if c is not None else other] = 1.0
        return ex

    def full(i):
        return np.concatenate([O[i], cards(CID[i])], axis=1)

    mu_s = np.stack([S[i] for i in tr_idx]).mean(0)
    sd_s = np.stack([S[i] for i in tr_idx]).std(0) + 1e-6
    acc = np.zeros(n_eff, np.float64); sq = np.zeros(n_eff, np.float64); cnt = 0
    for i in tr_idx:                                     # 流式统计，避免 3000 局时爆内存
        a = full(i)
        acc += a.sum(0); sq += (a.astype(np.float64) ** 2).sum(0); cnt += a.shape[0]
    mu_o = acc / cnt
    sd_o = np.sqrt(np.maximum(sq / cnt - mu_o ** 2, 0.0)) + 1e-6

    def state_of(i):
        return ((S[i] - mu_s) / sd_s).astype(np.float32)

    def opt_of(i):
        return ((full(i) - mu_o) / sd_o).astype(np.float32)

    # ---------------- 二选一（规则 vs 强 agent）训练集
    pairs = [i for i in tr_idx if 0 <= R[i] < K[i] and R[i] != Y[i]]
    print(f"二选一训练样本：{len(pairs)}（规则与强 agent 不同的决策）")
    model = OptionScorer(train_bc.N_S, n_eff, hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def batch_of(idx):
        st = torch.from_numpy(np.stack([state_of(i) for i in idx]))
        pair_rows = np.stack([opt_of(i)[[R[i], Y[i]]] for i in idx])   # 0=规则 1=强 agent
        of = torch.from_numpy(pair_rows)
        y = torch.ones(len(idx), dtype=torch.long)
        return st, of, y

    _ckpt = os.path.join(ROOT, "models", "bc_critic.pt")
    if args.load and os.path.exists(_ckpt):
        model.load_state_dict(torch.load(_ckpt)); args.epochs = 0
        print(f"  加载已有 critic：{_ckpt}")
    for ep in range(args.epochs):
        model.train()
        order = np.random.permutation(len(pairs)); tot = corr = 0; loss_sum = 0.0
        for b in range(0, len(order), args.bs):
            idx = [pairs[j] for j in order[b:b + args.bs]]
            st, of, y = batch_of(idx)
            opt.zero_grad()
            sc = torch.stack([model.scores(st[k], of[k]) for k in range(len(idx))])
            loss = torch.nn.functional.cross_entropy(sc, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            corr += int((sc.argmax(1) == y).sum()); tot += len(idx)
            loss_sum += float(loss.detach()) * len(idx)
        print(f"  epoch {ep + 1}/{args.epochs}  loss={loss_sum / max(tot, 1):.4f}  "
              f"train 二选一 top-1={corr / max(tot, 1):.1%}", flush=True)

    # ---------------- holdout：扫描门控阈值
    model.eval()
    rows = []
    with torch.no_grad():
        for i in ho_idx:
            if not (0 <= R[i] < K[i]):
                continue
            st = torch.from_numpy(state_of(i)); of = torch.from_numpy(opt_of(i))
            sc = model.scores(st, of).numpy()
            # 用 log-softmax 归一化，让 τ 与选项数无关
            lg = sc - sc.max()
            p = np.exp(lg); p /= p.sum()
            mask = np.zeros(K[i], bool)
            if args.same_type:
                # 模型的跨类型分数尺度不可比（11 选 1 只有 18.7%），但**同类内部**比较可能仍然有效
                for k in range(K[i]):
                    mask[k] = (TY[i][k] == TY[i][R[i]])
            else:
                mask[:] = True
            lp = np.where(mask, np.log(np.maximum(p, 1e-12)), -1e9)
            rows.append((i, lp, R[i]))
    print(f"holdout 可评估决策 {len(rows)}")

    def sig(i, k):
        return (TY[i][k], CID[i][k])

    base_sem = np.mean([sig(i, R[i]) == sig(i, Y[i]) for i, _, r in rows])
    base_idx = np.mean([R[i] == Y[i] for i, _, r in rows])
    print(f"\n参照：规则 index {base_idx:.1%} / 语义 {base_sem:.1%}")

    print(f"\n{'τ':>6} {'偏离率':>7} {'index':>7} {'语义':>7} {'语义Δ':>7}")
    best = None
    for tau in [99.0, 60.0, 40.0, 30.0, 25.0, 20.0, 17.0, 15.0, 13.0, 11.0, 9.9, 5.0, 2.0, 0.0]:
        ok_i = ok_s = dev = 0
        for i, lp, r in rows:
            k = int(np.argmax(lp))
            if lp[k] - lp[r] > tau:
                pick = k; dev += 1
            else:
                pick = r
            ok_i += int(pick == Y[i])
            ok_s += int(sig(i, pick) == sig(i, Y[i]))
        n = len(rows)
        de = ok_s / n - base_sem
        print(f"{tau:>6} {dev / n:>6.1%} {ok_i / n:>7.1%} {ok_s / n:>7.1%} {de:>+7.1%}")
        if best is None or ok_s > best[1]:
            best = (tau, ok_s / n, dev / n)
    tau, sem, dev = best
    print(f"\n最佳 τ={tau}：语义 {sem:.1%}（规则 {base_sem:.1%}，Δ{sem - base_sem:+.1%}）"
          f"  偏离率 {dev:.1%}")
    ok = (sem >= base_sem + 0.05) and (dev <= 0.15)
    print(f"判据（语义 ≥{base_sem + 0.05:.1%} 且偏离率 ≤15%）：{'**通过**' if ok else '不通过'}")
    torch.save(model.state_dict(), os.path.join(ROOT, "models", "bc_critic.pt"))
    print("模型已存 models/bc_critic.pt")


if __name__ == "__main__":
    main()
