#!/usr/bin/env python3
"""**生成方案 A 的测试集**：每副对手牌都由"当前最佳驾驶员"来开。

    ./tools/py tools/make_testset_A.py            # → decks/testset_meta_A.tsv
    ./tools/py tools/make_testset_A.py --dry

方案 A（用户选定）= 标准对手 = 当前最佳驾驶员（最贴近天梯），代价是对手升级后要补测。
本工具读 `docs/the-pockmon-battle-choice-self.csv`（每副牌最佳驾驶员），
输出与 `decks/testset_meta.tsv` 同格式的测试集，`meta_eval.sh` 用 `TS=` 指向它即可。
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "decks", "testset_meta.tsv")
SELF = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-self.csv")
OUT = os.path.join(ROOT, "decks", "testset_meta_A.tsv")

# 驾驶员 → 评估机上的 agent 目录名（未列出的回退该牌自己的 port agent）
DRIVER_AGENT = {
    "init": None,                       # None = 用 base 测试集里该格原有的 port agent
    "shuppet_specialist": "g8_bc_parent",
    "g8_bc_parent": "g8_bc_parent",
    "g8_shuppet": "g8_shuppet",
    "r1A": "r1A_26c6",
    "bc_26c6": "bc_26c6",
    "cand7e39S": "cand7e39S",
    "candE1D7S": "candE1D7S",
    "g6_db04": "g6_db04",
    "ph3_2617_v1": "ph3_2617_v1", "ph3_2617_v2": "ph3_2617_v2",
    "ph3_9d6e_v2": "ph3_9d6e_v2",
    "ab9d6e_r1A": "ab9d6e_r1A", "ab9d6e_r2A": "ab9d6e_r2A",
    "ab9d6e_r1B": "ab9d6e_r1B", "ab9d6e_r2B": "ab9d6e_r2B",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    best = {}
    if os.path.exists(SELF):
        for r in csv.DictReader(io.StringIO(open(SELF, "rb").read().decode("gb18030", errors="replace"))):
            d = r.get("最佳驾驶员") or "init"
            best[r["卡组"]] = "init" if d in ("", "待测", None) else d
    rows = []
    for line in open(BASE):
        if line.startswith("#") or not line.strip():
            continue
        cell, dir_, deck, grp, wt = [x.strip() for x in line.rstrip("\n").split("\t")]
        short = deck.split("__")[-1][:4]
        drv = best.get(short, "init")
        agent = DRIVER_AGENT.get(drv, None)
        new_dir = f"build/agents/{agent}" if agent else dir_
        rows.append((cell, new_dir, deck, grp, wt, drv))
        print(f"  {cell:<16}{(drv or 'init'):<20}{new_dir}")
    n_strong = sum(1 for r in rows if r[5] != "init")
    print(f"\n{len(rows)} 格中 {n_strong} 格改用强驾驶员")
    if a.dry:
        return 0
    with open(OUT, "w") as f:
        f.write("# 方案 A 测试集：每副对手牌由「当前最佳驾驶员」开（tools/make_testset_A.py 生成）\n")
        f.write("# 列：cell\tagent_dir\tdeck\tgroup\tfield_weight\n")
        for cell, dir_, deck, grp, wt, _drv in rows:
            f.write(f"{cell}\t{dir_}\t{deck}\t{grp}\t{wt}\n")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
