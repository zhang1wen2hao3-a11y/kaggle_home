#!/usr/bin/env python3
"""**LB 读数台账**：每次读 publicScore 都追加一行，避免"读一次就下结论"。

## 为什么（2026-11-22 踩到的坑）

同一个提交的 publicScore 会**随时间漂**：`c0904R1` 提交后 20 分钟读 701.9、1 小时读 919.9、
再往后 1025.3 → 971.9。如果只看第一个数，会得出"训练把牌训坏了"的错误结论。

规则：
1. 每次比较只用**同一批读数**（同一时间窗），且**至少读 3 次取中位**；
2. 任何"某提交更好"的判断，必须写在台账里、附上可回溯的读数序列。

## 用法

    ./tools/py tools/lb_log.py                      # 读一次 kaggle 榜单并追加到台账
    ./tools/py tools/lb_log.py --show               # 只看台账（按提交名汇总中位/极值）
    ./tools/py tools/lb_log.py --show --since 2026-11-22
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP = "the-pokemon-company-ptcg-ai-battle-challenge-playground"
TSV = os.path.join(ROOT, "data", "lb_readings.tsv")


def read_lb() -> list:
    out = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP],
                         capture_output=True, text=True, timeout=180)
    rows = []
    for ln in out.stdout.splitlines():
        f = ln.split()
        if len(f) < 4 or not f[0].isdigit():
            continue
        ref, fname = f[0], f[1]
        score = None
        for tok in reversed(f):
            try:
                score = float(tok)
                break
            except ValueError:
                continue
        if score is None:
            continue
        rows.append((ref, fname, score))
    return rows


def append(rows: list) -> int:
    new = not os.path.exists(TSV)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(TSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        if new:
            w.writerow(["时间", "ref", "提交文件", "publicScore"])
        for ref, fname, score in rows:
            w.writerow([ts, ref, fname, f"{score:.1f}"])
    return len(rows)


def show(since: str = "") -> int:
    if not os.path.exists(TSV):
        print("（台账还没有数据，先跑一次 ./tools/py tools/lb_log.py）")
        return 1
    per = collections.defaultdict(list)
    for r in csv.DictReader(open(TSV), delimiter="\t"):
        if since and r["时间"] < since:
            continue
        per[(r["ref"], r["提交文件"])].append((r["时间"], float(r["publicScore"])))
    print(f"{'ref':10s}{'提交文件':30s}{'读数次数':>8s}{'最新':>9s}{'中位':>9s}{'区间':>18s}")
    for (ref, fname), xs in sorted(per.items(), key=lambda kv: kv[1][-1][0]):
        vals = [v for _, v in xs]
        med = sorted(vals)[len(vals) // 2]
        print(f"{ref:10s}{fname:30s}{len(vals):8d}{vals[-1]:9.1f}{med:9.1f}"
              f"{f'{min(vals):.1f}~{max(vals):.1f}':>18s}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--since", default="")
    a = ap.parse_args()
    if a.show:
        return show(a.since)
    rows = read_lb()
    if not rows:
        print("✗ 读不到榜单（kaggle CLI 或网络）", file=sys.stderr)
        return 1
    n = append(rows)
    print(f"✓ 追加 {n} 条读数 → data/lb_readings.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
