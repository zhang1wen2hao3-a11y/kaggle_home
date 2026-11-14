#!/usr/bin/env python3
"""**检查哪些格子需要补测**：当"对手驾驶员"升级后，旧结果就过期了。

    ./tools/py tools/check_stale.py            # 列出需要补测的 (我方牌, 对手牌, 模型)

## 为什么会过期

母表里每条记录都带 **对手驾驶员**：
- 现在所有面板评测的对手都是 **端口权重（init）**；
- 一旦 `the-pockmon-battle-choice-self.csv` 把某副对手牌升级成 `g8_bc_parent` / `r1A` / `cand7e39S` 等，
  那么"我方模型 vs 该副牌"的旧胜率就是**对更弱的对手**测出来的，不能再用。

## 输出

按对手牌分组，列出：(对手牌, 当前最佳对手驾驶员, 受影响的我方模型数, 需要补测的格数)。
补测方式：用新对手驾驶员铺一个 agent，跑 `meta_eval --opp` 或 `h2h`，结果进母表时把
`对手驾驶员` 填成新模型即可（母表会同时保留旧记录，便于对比"换强对手后掉多少"）。
"""
from __future__ import annotations

import collections
import csv
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-in-table.csv")
SELF = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-self.csv")


def load(path, enc="gb18030"):
    return list(csv.DictReader(io.StringIO(open(path, "rb").read().decode(enc, errors="replace"))))


def main():
    if not os.path.exists(SELF):
        print(f"✗ 缺 {SELF}（先跑 tools/build_pilot_table.py）")
        return 1
    best_opp = {r["卡组"]: r["最佳驾驶员"] for r in load(SELF)
                if r.get("最佳驾驶员") not in ("", "待测", None)}   # 待测=还不知道，按 init 算
    rec = load(MASTER)
    stale = collections.defaultdict(lambda: collections.defaultdict(set))   # opp -> drv -> {our decks}
    for r in rec:
        opp = r["被挑战者"]
        cur = best_opp.get(opp, "init")
        if cur != (r.get("对手驾驶员") or "init"):
            stale[opp][cur].add(r["挑战者"])
    if not stale:
        print("✓ 没有过期格子：所有记录用的对手驾驶员都还是当前最佳")
        return 0
    total = 0
    print(f"{'对手牌':<6}{'当前最佳对手驾驶员':<22}{'受影响我方牌数':>14}{'需补测格数':>10}")
    for opp, byd in sorted(stale.items()):
        for drv, ours in byd.items():
            total += len(ours)
            print(f"{opp:<6}{drv:<22}{len(ours):>14}{len(ours):>10}   ← {'/'.join(sorted(ours))}")
    print(f"\n合计需要补测 {total} 格（每格 3 种子 × 200 场 ≈ 25 分钟/格的面板折算）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
