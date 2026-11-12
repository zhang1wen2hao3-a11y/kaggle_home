#!/usr/bin/env python3
"""**给 a→b 矩阵的某一行指定驾驶员**（写 sidecar，供 build_winrate_table.py 重建时沿用）。

    ./tools/py tools/set_winrate_pilot.py --ship 9d6e --pilot ab9d6e_r1A \\
        --tag ab9d6e_r1A --tag ab9d6e_r1A_s950001 --tag ab9d6e_r1A_s950002

为什么需要 sidecar：`the-pockmon-battle-choices-win-rate.csv` 是纯数字矩阵，看不出"这一行用的是谁"。
sidecar 记 `我方卡组 → 驾驶员 → 用了哪些面板 tag → 每格 n`，这样：
  1. 表里每行的驾驶员可追溯（不会有人把它当成 init 的结果）；
  2. 重建表时（build_winrate_table.py）按 sidecar 把这一行换成该驾驶员的面板实测值，
     而不是回落到老的 init 数据；
  3. 驾驶员名带命名空间前缀（如 `ab9d6e_r1A`），与上一轮 slowking 的 `r1A` 不撞车。

多个 --tag 会被视为同协议的多个种子（胜场与场次相加）。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDECAR = os.path.join(ROOT, "docs", "the-pockmon-battle-choices-win-rate-pilot.csv")


def load_tag(tag):
    d = os.path.join(ROOT, "logs", "meta", tag)
    files = glob.glob(os.path.join(d, "vs_*.json"))
    if not files:
        raise SystemExit(f"✗ {tag} 没有面板结果（{d}/vs_*.json）")
    out = {}
    for f in files:
        o = json.load(open(f))["overall"]
        out[os.path.basename(f)[3:-5]] = (float(o["score"]), int(o["n"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ship", required=True, help="我方卡组短名，如 9d6e")
    ap.add_argument("--pilot", required=True, help="驾驶员名（agent 名），如 ab9d6e_r1A")
    ap.add_argument("--tag", action="append", required=True, help="面板 tag（可多个=多种子合并）")
    ap.add_argument("--note", default="", help="备注，如'3 种子合并 n=600'")
    a = ap.parse_args()

    pooled = {}
    for t in a.tag:
        for cell, (sc, n) in load_tag(t).items():
            k, n0 = pooled.get(cell, (0.0, 0))
            pooled[cell] = (k + sc * n, n0 + n)
    pooled = {c: (k / n0, n0) for c, (k, n0) in pooled.items()}
    print(f"{a.pilot} @ {a.ship}：{len(pooled)} 格，合并自 {len(a.tag)} 个 tag（{', '.join(a.tag)}）")
    for c in sorted(pooled):
        print(f"  {c:<18}{pooled[c][0]:.4f}  n={pooled[c][1]}")

    rows = []
    if os.path.exists(SIDECAR):
        with open(SIDECAR, newline="", encoding="gb18030") as f:
            rows = [r for r in csv.DictReader(f) if r["我方卡组"] != a.ship]
    rows.append({"我方卡组": a.ship, "驾驶员": a.pilot, "面板tag": ";".join(a.tag),
                 "每格样本": str(min(n for _, n in pooled.values())),
                 "备注": a.note or "同协议 n=200 面板"})
    rows.sort(key=lambda r: r["我方卡组"])
    with open(SIDECAR, "w", newline="", encoding="gb18030") as f:
        w = csv.DictWriter(f, fieldnames=["我方卡组", "驾驶员", "面板tag", "每格样本", "备注"],
                           lineterminator="\r\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"→ {SIDECAR}")
    print("下一步： ./tools/py tools/build_winrate_table.py   （会按 sidecar 覆盖该行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
