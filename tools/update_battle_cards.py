#!/usr/bin/env python3
"""**重建 `docs/the-pockmon-battle-cards.csv`（卡组宇宙表）** —— 数据全部取自投资分表。

为什么重写（2026-11-22 用户指出）：老版由 `tools/build_battle_tables.py` 从
`data/ladder_games.tsv`（1756 局**旧**档案）生成，于是出现三个错：

1. `e1d7` 在旧档案里没出现 ⇒ `Meta 占比=0、样本局数=0`（真实场份额其实是 **12.9%**）；
2. 这种"档案里没有、但我们提交过"的牌会被补一行 `…（<4hex>·我们）` —— 那个 **·我们** 只是
   "这行是因为我们打过才补的"标记，**不是**"这副牌是我们设计的"（e1d7 其实是 ミワハルキ 本人的 60 张）；
3. `最高分玩家` 留空、`最高LB` 落成我们自己的分（外部天花板 1600.4 / knshnb 在另一份语料里算的，
   老脚本没接上）。

新口径（全部来自 `docs/卡组天花板与投资分.csv`，它已由 refresh_tables 保证是最新）：
    DeckID        = 4hex
    卡组          = 「原型（4hex·外部最高分玩家 / 我们）」，例：`ogerpon（e1d7·knshnb）`
    Meta 占比     = 真实场份额（对手覆盖率%）
    样本局数      = 该牌在天梯语料里的局数（没有就留 0）
    最高LB        = 外部天花板（没有就 NA）
    自己最高分    = 我们自己的最高 LB（没有就空）
    最高分玩家    = 外部天花板对应的玩家

    ./tools/py tools/update_battle_cards.py
"""
from __future__ import annotations

import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "卡组天花板与投资分.csv")
OUT = os.path.join(ROOT, "docs", "the-pockmon-battle-cards.csv")
HDR = ["DeckID", "卡组", "Meta 占比", "样本局数", "最高LB", "自己最高分", "最高分玩家"]


def fnum(x, default=0.0):
    """'18.3%' / '1404.0' / '' 都要能吃。"""
    t = str(x if x is not None else "").strip().rstrip("%").replace(",", "")
    try:
        return float(t)
    except (TypeError, ValueError):
        return default


def main() -> int:
    rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
    out = []
    for r in rows:
        h = (r.get("卡组") or "").strip()
        if not h:
            continue
        proto = (r.get("原型") or "").strip() or h
        share = fnum(r.get("对手覆盖率"), 0.0)          # 已经是百分数
        games = fnum(r.get("牌天梯局数"), 0.0)
        ceil = fnum(r.get("外部天花板"), 0.0)
        who = (r.get("外部最高分玩家") or "").strip()
        ours = fnum(r.get("我们最高分"), 0.0)
        deck_name = f"{proto}（{h}·{who}）" if who else f"{proto}（{h}）"
        out.append([h, deck_name, f"{share:.2f}", f"{int(games)}",
                    f"{ceil:.1f}" if ceil else "NA",
                    f"{ours:.1f}" if ours else "", who])
    # ★ 2026-11-22 用户指定排序：**最高LB 为第一排序键（降序）**；没有值(NA)排最后；
    #   最高LB 相同则按 DeckID（ASCII 升序，数字在字母前）。
    def sort_key(r):
        lb = fnum(r[4], None) if str(r[4]).strip() not in ("NA", "") else None
        return (0, -lb) if lb is not None else (1, 0.0), r[0]
    out.sort(key=sort_key)
    with open(OUT, "w", newline="", encoding="gbk", errors="replace") as fh:
        w = csv.writer(fh)
        w.writerow(HDR)
        w.writerows(out)
    print(f"✓ {os.path.relpath(OUT, ROOT)}：{len(out)} 行（来源 {os.path.relpath(SRC, ROOT)}）")
    for r in out[:6]:
        print(f"   {r[0]:6s} {r[1][:34]:36s} 份额 {r[2]:>5s}%  天花板 {r[4]:>7s}  我们 {r[5] or '-':>7s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
