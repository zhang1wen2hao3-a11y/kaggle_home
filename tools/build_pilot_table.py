#!/usr/bin/env python3
"""**生成 docs/the-pockmon-battle-choice-self.csv：每副牌"自己的最强驾驶员"**

    ./tools/py tools/build_pilot_table.py            # 生成/刷新
    ./tools/py tools/build_pilot_table.py --dry      # 只看结论

## 为什么单独一张表

`the-pockmon-battle-choice.csv` 的格子是「**我方**打这格该用谁」；它回答不了
"这**副牌**交给谁驾驶最强"，而训练时的对手池需要后者（`choice_to_banks.py --strong`）。

本表把这层推导**落成文件**，成为唯一真源：训练对手驾驶员一律从这张表读，
不再在脚本里硬编码字典。

## 推导规则

对每副牌 b，收集"b 作为挑战者"那一行里所有**非 init** 的驾驶员记录（`choice-detail.csv`），
按 `胜率` 取最高者；若该行全是 init（例：`7e39` 行目前就是全部 init），则 b 的最强驾驶员 = `init`。

同时给出 `行内 init 胜率` 与 `提升`，便于判断"这条到底有多可信"。
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETAIL = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-detail.csv")
OUT = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-self.csv")


def load_detail():
    txt = open(DETAIL, "rb").read().decode("gb18030", errors="replace")
    return list(csv.DictReader(io.StringIO(txt)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    rows = load_detail()

    # 同格 init 基线（从 choice-all 里取，便于算"同一格的提升"）
    init_cell = {}
    p_all = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-all.csv")
    if os.path.exists(p_all):
        t = open(p_all, "rb").read().decode("gb18030", errors="replace")
        for r in csv.DictReader(io.StringIO(t)):
            if r.get("驾驶员") == "init" and r.get("胜率"):
                try: init_cell[(r["挑战者"], r["被挑战者"])] = float(r["胜率"])
                except ValueError: pass

    by_row = collections.defaultdict(list)
    for r in rows:
        try:
            wr = float(r["胜率"]) if r.get("胜率") else None
        except ValueError:
            wr = None
        by_row[r["挑战者"]].append((r["驾驶员"], wr, r.get("样本场次", ""), r["被挑战者"], r["状态"]))

    out = []
    for deck in sorted(by_row):
        cells = by_row[deck]
        trained = [(p, wr, n, b) for (p, wr, n, b, st) in cells if p != "init" and wr is not None]
        if trained:
            p, wr, n, b = max(trained, key=lambda t: t[1])
            base = init_cell.get((deck, b))
            gain = (wr - base) * 100 if base is not None else None
            why = (f"{len(trained)} 格非 init；最优在 →{b}（同格 init {base:.4f}）" if base is not None
                   else f"{len(trained)} 格非 init；最优在 →{b}（该格无 init 数据）")
            out.append([deck, p, f"{wr:.4f}", n, f"{gain:+.1f}pp" if gain is not None else "NA", why])
        else:
            why = "该行无非 init 驾驶员（全部格子 init 就最合适）"
            out.append([deck, "init", "NA", "NA", "NA", why])

    print(f"{'卡组':<8}{'最强驾驶员':<20}{'胜率':>8}{'样本':>7}{'提升':>9}  依据")
    for r in out:
        print(f"{r[0]:<8}{r[1]:<20}{r[2]:>8}{str(r[3]):>7}{r[4]:>9}  {r[5][:52]}")
    n_init = sum(1 for r in out if r[1] == "init")
    print(f"\n共 {len(out)} 副牌：{len(out)-n_init} 副有非 init 驾驶员，{n_init} 副只能用 init")
    if a.dry:
        return 0
    with open(OUT, "w", newline="", encoding="gb18030") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(["卡组", "最强驾驶员", "胜率", "样本场次", "相对 init 提升", "依据"])
        w.writerows(out)
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
