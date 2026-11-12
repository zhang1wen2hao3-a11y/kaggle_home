#!/usr/bin/env python3
"""**用 a→b 矩阵 + 天梯牌占比算"该用哪副舰牌、该补哪一格"**

    ./tools/py tools/deck_field_matrix.py

数据：docs/the-pockmon-battle-choices-win-rate.csv（132 格实跑/老 eval）
       decks/weights_field_b05.txt（天梯 187 副的覆盖率权重，归一化到 12 副短名）

口径：
  · 行 = 我方舰牌 a，列 = 对手牌 b，格 = 我方 a 打对手 b 的胜率（表里已按最强驾驶员口径）
  · 场地加权期望胜率 = Σ_b w_b · P(a 赢 b) / Σ_b w_b
  · 对角（a→a 自己打自己）按 **0.5** 计入（座位均衡的期望值，表里写作 `/`）

标定（对照 12 格面板实测，同协议 n=200 seed 950000）：
    db04 行 0.7260 vs 面板 g6_db04@db04 0.7261 → Δ −0.0pp
    9d6e 行 0.7184 vs 面板 port@9d6e 0.6842 → Δ +3.4pp（4 格老数据与面板差 4~6pp，属抽样噪声）
"""
from __future__ import annotations

import collections
import csv
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "docs", "the-pockmon-battle-choices-win-rate.csv")
WF = os.path.join(ROOT, "decks", "weights_field_b05.txt")


def field_weights():
    w = {}
    for line in open(WF):
        p = line.split()
        if len(p) == 2 and "__" in p[0]:
            w[p[0].split("__")[-1][:4]] = float(p[1])
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


def matrix():
    txt = open(CSV, "rb").read().decode("gb18030", errors="replace")
    rows = list(csv.reader(io.StringIO(txt)))
    hdr = rows[0][1:]
    return hdr, {r[0]: {hdr[i]: r[i + 1] for i in range(len(hdr))} for r in rows[1:]}


def main():
    W = field_weights()
    hdr, M = matrix()
    print("=== ① 我方舰牌排行（场地加权期望胜率，对角记 0.5）===")
    print(f"{'舰牌':<8}{'期望胜率':>10}{'最大权重短板（非镜像）':>38}")
    rank = []
    for a, row in M.items():
        num = den = 0.0
        leaks = []
        for b, v in row.items():
            if v == "":
                continue
            w = W.get(b, 0.0)
            p = 0.5 if v == "/" else float(v)
            num += w * p
            den += w
            if a != b and w > 0.01:
                leaks.append((w * (1 - p), b, p, w))
        leaks.sort(reverse=True)
        loss, b, p, w = leaks[0]
        rank.append((num / den, a, b, p, w, loss))
    for sc, a, b, p, w, loss in sorted(rank, reverse=True):
        print(f"{a:<8}{sc:>10.4f}   {b} 胜率{p:.2f} 占比{w:.1%} → 丢 {loss:.3f} 权重·点")

    print("\n=== ② 各对手牌（列）把谁打崩（权重 >2%）===")
    per_b = collections.defaultdict(list)
    for a, row in M.items():
        for b, v in row.items():
            if v in ("/", "") or W.get(b, 0) < 0.02 or a == b:
                continue
            per_b[b].append((float(v), a))
    for b in sorted(per_b, key=lambda x: -W[x]):
        vals = sorted(per_b[b])
        s = "  ".join(f"{a}={v:.2f}" for v, a in vals[:3])
        print(f"  {b}（占比 {W[b]:5.1%}）最低三: {s}")

    print("\n=== ③ 镜像格（a→a，表里是 `/`，此处不计）")
    print("  提交时镜像对位按 0.5 起算，所以舰牌选择只看非镜像格。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
