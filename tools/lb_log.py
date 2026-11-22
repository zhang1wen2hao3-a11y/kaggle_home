#!/usr/bin/env python3
"""**LB 读数台账**：每次读 publicScore 都追加一行，避免"读一次就下结论"。

## 为什么（2026-11-22 踩到的坑）

同一个提交的 publicScore 会**随时间漂**：`c0904R1` 提交后 20 分钟读 701.9、1 小时读 919.9、
再往后 1025.3 → 971.9。如果只看第一个数，会得出"训练把牌训坏了"的错误结论。

规则：
0. **提交后至少 1 小时再记**（2026-11-22 用户明确要求："现在的出分先别记下来，目前还没稳定，再至少等 1 个小时"）。
   publicScore 在头 1 小时内会剧烈变化（实测 600.0 → 700.2 → 901.4…），早记只会污染台账；
   本工具默认 `--min-age-min 60`，不到 1 小时的提交直接跳过。
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


def _age_min(date_s: str) -> float:
    """提交时间（UTC）到现在多少分钟。解析不出来就当作"很旧"（放行）。"""
    try:
        d = datetime.datetime.strptime(date_s.strip()[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return 1e9
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    return (now - d).total_seconds() / 60.0


def read_lb(min_age_min: float = 60.0) -> list:
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
        dt = " ".join(f[2:4])
        if _age_min(dt) < min_age_min:
            continue                      # 太新，读数不可信
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
    ap.add_argument("--min-age-min", type=float, default=60.0,
                    help="提交多久之后的读数才记（默认 60 分钟；用户 2026-11-22 要求）")
    a = ap.parse_args()
    if a.show:
        return show(a.since)
    rows = read_lb(a.min_age_min)
    if not rows:
        print("✗ 读不到榜单（kaggle CLI 或网络）", file=sys.stderr)
        return 1
    n = append(rows)
    print(f"✓ 追加 {n} 条读数 → data/lb_readings.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
