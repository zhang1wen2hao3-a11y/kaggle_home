#!/usr/bin/env python3
"""把"卡组强度"从"驾驶员（队伍）强度"里分离出来 —— ridge 惩罚 logistic Bradley-Terry。

## 为什么必须分离

如果只看"某副卡的胜率"，强队用过它就会被算成它的功劳。公开界（kellydannimao）的做法是：

    logit P(i beats j) = (team_i + deck_i) − (team_j + deck_j)

用"同一支队伍换过多副卡组"的对局来识别 deck 效应，ridge 惩罚防分离，标准误取惩罚 Hessian 逆的对角线。
我们的语料里 **298 支队伍中有 62 支用过 ≥2 副不同卡组**，这是能识别 deck 效应的前提。

## 已知局限（必须写在结论里）

语料不是天梯随机抽样：`epsoide/` 是我们自己的 317 局，`epsoide_teams2`/`epsoide_field` 是
前几名队伍的对局。所以这是**条件于该对手池**的相对强度，不能读成"全池排名"。
另外每局只用单局结果（不是 Bo3 得分），方差更大 —— 所以只看区间，不看排序细粒度。

用法：
    ./run.sh tools/deck_bt.py --corpus data/deck_corpus.csv --lam 1.0 --top 20
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_pairs(path):
    """把 corpus 读成 (game, teamA, deckA, wonA, teamB, deckB, wonB) 的对局列表。"""
    by_game = collections.defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            by_game[r["file"]].append(r)
    pairs = []
    for g, rows in by_game.items():
        if len(rows) != 2:
            continue
        a, b = rows
        if a["won"] == "" or b["won"] == "":
            continue
        pairs.append((a["team"], a["deck_hash"], int(a["won"]),
                      b["team"], b["deck_hash"], int(b["won"])))
    return pairs


def fit(pairs, lam=1.0, iters=30):
    """牛顿法拟合 theta = [team..., deck...]，带 ridge 惩罚。"""
    teams = sorted({p[0] for p in pairs} | {p[3] for p in pairs})
    decks = sorted({p[1] for p in pairs} | {p[4] for p in pairs})
    ti = {t: i for i, t in enumerate(teams)}
    di = {d: i for i, d in enumerate(decks)}
    T, D = len(teams), len(decks)
    n = T + D
    # 设计矩阵：每局一行，y = 1 表示"前者赢"
    X = np.zeros((len(pairs), n))
    y = np.zeros(len(pairs))
    for k, (ta, da, wa, tb, db, wb) in enumerate(pairs):
        X[k, ti[ta]] += 1.0
        X[k, T + di[da]] += 1.0
        X[k, ti[tb]] -= 1.0
        X[k, T + di[db]] -= 1.0
        y[k] = 1.0 if wa > wb else 0.0
    # 只用非平局
    keep = np.array([wa != wb for (_, _, wa, _, _, wb) in pairs], bool)
    X, y = X[keep], y[keep]
    theta = np.zeros(n)
    P = np.eye(n) * lam
    for _ in range(iters):
        z = X @ theta
        p = 1.0 / (1.0 + np.exp(-z))
        W = p * (1 - p)
        g = X.T @ (y - p) - lam * theta
        H = (X * W[:, None]).T @ X + P
        step = np.linalg.solve(H, g)
        theta = theta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    z = X @ theta
    p = 1.0 / (1.0 + np.exp(-z))
    W = p * (1 - p)
    H = (X * W[:, None]).T @ X + P
    cov = np.linalg.inv(H)
    se = np.sqrt(np.diag(cov))
    # 可识别性：加性常数不可辨 → 中心化（与公开界一致：以全体为锚）
    dock = theta[T:] - theta[T:].mean()
    dock_se = se[T:]                      # 中心化只平移均值，SE 近似不变
    return teams, decks, theta[:T], dict(
        decks=[(d, float(v), float(s)) for d, v, s in zip(decks, dock, dock_se)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/deck_corpus.csv")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    path = args.corpus if os.path.isabs(args.corpus) else os.path.join(ROOT, args.corpus)
    pairs = load_pairs(path)
    share = collections.Counter()
    for ta, da, wa, tb, db, wb in pairs:
        share[da] += 1
        share[db] += 1
    teams, decks, team_eff, res = fit(pairs, lam=args.lam)
    print(f"对局 {len(pairs)} 局；队伍 {len(teams)} 支；卡组 {len(decks)} 副；λ={args.lam}")
    multi = sum(1 for t in teams if sum(1 for p in pairs if p[0] == t or p[3] == t) > 0)
    print(f"可识别性：使用 ≥2 副卡组的队伍是 deck 效应的识别来源（见 corpus 统计）")

    rows = sorted(res["decks"], key=lambda x: -x[1])
    print(f"\n{'deck':<10}{'deck 效应':>10}{'SE':>7}{'局数':>7}{'份额':>7}  95% 区间")
    for d, v, s in rows[:args.top]:
        n = share[d]
        print(f"{d:<10}{v:>+10.2f}{s:>7.2f}{n:>7}{n / max(2 * len(pairs), 1):>7.1%}"
              f"  [{v - 1.96 * s:+.2f}, {v + 1.96 * s:+.2f}]")
    print("\n最强 5 副：", ", ".join(d for d, _, _ in rows[:5]))
    print("最弱 5 副：", ", ".join(d for d, _, _ in rows[-5:]))
    print("\n⚠️ 语料非天梯随机抽样（我们自己 317 局 + 前几名队伍），"
          "只能当『条件于该对手池』的相对强度；且是单局结果，只看区间不看细粒度排序。")


if __name__ == "__main__":
    main()
