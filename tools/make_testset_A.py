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

# ★ 名字 → agent 目录的唯一来源是 tools/driver_registry.py（别在这里再抄一份！
#   抄一份的下场：表里出现脚本不认识的名字时静默回退 port，评测记成"强驾驶员"实际是弱端口。）
sys.path.insert(0, os.path.join(ROOT, "tools"))
import driver_registry as REG

# 名字 → 评估机上的 agent 目录名；None = 用 base 测试集里该格原有的 port agent
DRIVER_AGENT = {n: (d.get("agent") or None) for n, d in REG.DRIVERS.items()}
DRIVER_AGENT["init"] = None


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
        try:
            agent = REG.agent_dir(drv)
        except REG.UnknownDriver as e:
            sys.exit(f"✗ {cell} 的驾驶员 {drv!r} 没在 tools/driver_registry.py 登记：{e}")
        new_dir = agent if agent else dir_
        rows.append((cell, new_dir, deck, grp, wt, drv))
        print(f"  {cell:<16}{(drv or 'init'):<20}{new_dir}")
    n_strong = sum(1 for r in rows if r[5] != "init")
    print(f"\n{len(rows)} 格中 {n_strong} 格改用强驾驶员")
    if a.dry:
        return 0
    with open(OUT, "w") as f:
        f.write("# 方案 A 测试集：每副对手牌由「当前最佳驾驶员」开（tools/make_testset_A.py 生成）\n")
        f.write("# 列：cell\tagent_dir\tdeck\tgroup\tfield_weight\tdriver\n")
        f.write("# driver = 该格对手的驾驶员名字（来自 driver_registry + choice-self），供 meta_eval 打印审计\n")
        for cell, dir_, deck, grp, wt, drv in rows:
            f.write(f"{cell}\t{dir_}\t{deck}\t{grp}\t{wt}\t{drv}\n")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
