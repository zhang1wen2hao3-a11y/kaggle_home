#!/usr/bin/env python3
"""**12 格面板横向对比**：把一个基线 tag 与若干候选 tag 的 summary.json 摆成一张表。

    ./tools/py tools/panel_compare.py --base meta_port_9d6e --tag r1A_9d6e --tag r1B_9d6e

口径来自 tools/meta_eval.sh（12 格 × n=200、seed 950000、与基线同机同种子同协议）：
    meta_w  按我们 582 局实测场地占比加权（champ 格权重为 0）
    champ   4 副冠军牌的平均   local  8 副本地牌的平均   worst  12 格里最低
    Δ 一律相对 --base，CI 由单格 n=200 推（单格 ±6.5pp，加权后 ≈ ±2.5pp）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(tag):
    p = os.path.join(ROOT, "logs", "meta", tag, "summary.json")
    if not os.path.exists(p):
        raise SystemExit(f"✗ 缺 {p}")
    s = json.load(open(p))
    s.setdefault("cells", {})
    s.setdefault("worst", ("?", 0.0))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tag", action="append", default=[])
    a = ap.parse_args()
    base = load(a.base)
    cands = [(t, load(t)) for t in a.tag]
    if not cands:
        raise SystemExit("✗ 至少给一个 --tag")

    hdr = f"{'指标':<22}{a.base:>16}" + "".join(f"{t:>18}" for t, _ in cands)
    print(hdr)
    print("-" * len(hdr))
    for key, fmt in (("meta_w", "{:.4f}"), ("champ", "{:.3f}"), ("local", "{:.3f}")):
        row = f"{key:<22}{fmt.format(base[key]):>16}"
        for _, s in cands:
            d = (s[key] - base[key]) * 100
            row += f"{fmt.format(s[key]):>10} ({d:+5.1f}pp)"
        print(row)
    row = f"{'worst':<22}{base['worst'][0][:12] + ' ' + format(base['worst'][1], '.3f'):>16}"
    for _, s in cands:
        row += f"{s['worst'][0][:10] + ' ' + format(s['worst'][1], '.3f'):>18}"
    print(row)

    cells = sorted(set(base["cells"]) | set().union(*[set(s["cells"]) for _, s in cands]))
    print("\n逐格（Δ 相对基线）：")
    hdr2 = f"{'格子':<18}{a.base:>8}" + "".join(f"{t:>16}" for t, _ in cands)
    print(hdr2)
    for c in cells:
        b = base["cells"].get(c)
        row = f"{c:<18}{(format(b, '.4f') if b is not None else '—'):>8}"
        for _, s in cands:
            v = s["cells"].get(c)
            row += (f"{format(v, '.4f'):>9}{(v - b) * 100:+6.1f}pp" if (v is not None and b is not None) else f"{'—':>16}")
        print(row)
    print("\n提示：单格 n=200 的 CI ≈ ±6.5pp；12 格加权后 ≈ ±2.5pp。Δ 小于 2.5pp 不能当真实提升。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
