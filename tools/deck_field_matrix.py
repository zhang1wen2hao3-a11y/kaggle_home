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

import argparse
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

    print("\n=== ③ 对手权重建议：场地占比 × 改进空间 ===")
    ap = argparse.ArgumentParser(description="把'场地占比'与'改进空间(1-p)'混合成训练对手权重")
    ap.add_argument("--ship", default="9d6e", help="我方舰牌（决定 p 从哪一行取）")
    ap.add_argument("--alpha", type=float, default=0.5, help="0=纯场地占比，1=纯改进空间（默认 0.5 混合）")
    ap.add_argument("--emit", default="", help="写出一份 DWF 权重文件（训练器 --deck-weights-file 可直接用）")
    a = ap.parse_args()
    row = M[a.ship]
    rows = []
    for b, v in row.items():
        if v == "":
            continue
        p = 0.5 if v == "/" else float(v)
        w = W.get(b, 0.0)
        rows.append([b, w, p, w * (1 - p)])
    base = sum(r[1] for r in rows); imp = sum(r[3] for r in rows)
    print(f"  舰牌 {a.ship}：α={a.alpha}（0=纯场地占比，1=纯改进空间）")
    print(f"  {'对手牌':<10}{'场地占比':>10}{'胜率':>8}{'纯场地':>10}{'改进空间':>10}{'混合后':>10}")
    tw = 0.0
    for b, w, p, wi in rows:
        v = (1 - a.alpha) * w / base + a.alpha * wi / imp
        tw += v
        rows[rows.index([b, w, p, wi])] = [b, w, p, wi, v]
    for b, w, p, wi, v in sorted(rows, key=lambda r: -r[4]):
        print(f"  {b:<10}{w:>9.1%}{p:>8.2f}{w / base:>10.1%}{wi / imp:>10.1%}{v / tw:>10.1%}")
    if a.emit:
        # 写成训练器认的 DWF 格式：**保留原文件全部 36 条**，只把 12 副已知牌按混合权重重新分配，
        # 其余 24 副按原相对比例缩放（保持"这 12 副占总质量"不变，避免把长尾权重放大 10 倍）。
        import glob
        full = {}
        for f in glob.glob(os.path.join(ROOT, "decks", "pool_field", "*.csv")):
            base_name = os.path.basename(f)[:-4]
            if "__" in base_name:
                full.setdefault(base_name.split("__")[-1][:4], base_name)
        orig = []          # [(full_name, weight)]
        for line in open(WF):
            sp = line.split()
            if len(sp) == 2 and "__" in sp[0]:
                orig.append((sp[0], float(sp[1])))
        short_of = {name.split("__")[-1][:4]: name for name, _ in orig}
        known = {b: v / tw for b, w, p, wi, v in rows}
        mass12 = sum(w for name, w in orig if name.split("__")[-1][:4] in known)
        out = []
        for name, w in orig:
            k = name.split("__")[-1][:4]
            if k in known:
                out.append((name, known[k] * mass12))
            else:
                out.append((name, w))
        tot = sum(w for _, w in out)
        with open(a.emit, "w") as fh:
            fh.write("version 3\n")
            fh.write(f"default 0.0001\n")
            for name, w in sorted(out, key=lambda r: -r[1]):
                fh.write(f"{name} {w / tot:.6f}\n")
        print(f"  → 已写出 {a.emit}（{len(out)} 副，12 副重分配后总和归一；这 12 副占总质量 "
              f"{mass12:.1%} 不变）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
