#!/usr/bin/env python3
"""**面板 vs 真实天梯：逐格校准**（2026-11-22）

问题：我们一直用"强对手 17 格面板"当训练判据，但它到底能不能预测真实天梯？
做法：拿**我们自己的天梯对局档案** `data/our_games.tsv`（每局有我方牌/对手牌/胜负/提交名），
按"我方牌"聚合出对每副对手牌的真实胜率，再和**同一副牌的面板候选**逐格比。

对齐口径：
* 天梯侧：`our_deck` → 我们的牌；`opp_deck` → 对手牌；`result` → 1 胜。
* 面板侧：`logs/meta/<候选>_real_s95*/vs_<格>.json`，格 → 对手牌 用 `decks/testset_meta_real.tsv` 第 3 列。
* 只比较**两侧都有的对手牌**，且天梯侧 n≥10。

⚠️ 两侧对手强度不同（面板=我们最强的 AI；天梯=真实对手，含我们自己的提交），所以看的是**偏差方向和大小**，
以及"面板排序是否可信"，不是绝对值相等。

    ./tools/py tools/panel_vs_ladder.py --map "dunsparce_abra__9d6e=cand9D6ER3,dunsparce_abra__db04=cand9D6ER3"
    ./tools/py tools/panel_vs_ladder.py --auto        # 用投资分表里的 最佳驾驶员 自动配对
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cell_to_deck() -> dict:
    out = {}
    for ln in open(os.path.join(ROOT, "decks", "testset_meta_real.tsv")):
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.rstrip("\n").split("\t")
        if len(f) >= 3 and f[0] != "tail_mix":
            out[f[0]] = f[2]
    return out


def auto_map() -> dict:
    """投资分表：卡组(4hex) → 最佳驾驶员；再把牌名(带后缀)映射过去。"""
    path = os.path.join(ROOT, "docs", "卡组天花板与投资分.csv")
    out = {}
    try:
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            h4, drv = (r.get("卡组") or "").strip(), (r.get("最佳驾驶员") or "").strip()
            if h4 and drv and drv != "init":
                out[h4] = drv
    except OSError:
        pass
    return out


def ladder_stats() -> dict:
    """{我方牌: {对手牌: (wins, n)}}"""
    agg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in csv.DictReader(open(os.path.join(ROOT, "data", "our_games.tsv")), delimiter="\t"):
        try:
            win = int(str(r.get("result")).strip() in ("1", "True", "win"))
        except (TypeError, ValueError):
            continue
        agg[r["our_deck"]][r["opp_deck"]][0] += win
        agg[r["our_deck"]][r["opp_deck"]][1] += 1
    return agg


def panel_stats(cand: str) -> dict:
    c2d = cell_to_deck()
    agg = collections.defaultdict(lambda: [0.0, 0])
    for p in glob.glob(os.path.join(ROOT, "logs", "meta", f"{cand}_real_s95*", "vs_*.json")):
        cell = os.path.basename(p)[3:-5]
        try:
            o = json.load(open(p)).get("overall") or {}
        except (OSError, ValueError):
            continue
        n = int(o.get("n") or 0)
        d = c2d.get(cell)
        if not n or not d:
            continue
        agg[d][0] += float(o.get("score") or 0) * n
        agg[d][1] += n
    return {k: (v[0] / v[1], v[1]) for k, v in agg.items() if v[1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="", help="我方牌名=面板候选,...")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--min-n", type=int, default=10)
    a = ap.parse_args()

    pair = {}
    for kv in a.map.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            pair[k.strip()] = v.strip()
    if a.auto or not pair:
        am = auto_map()
        for deck in ladder_stats():
            h4 = deck.split("__")[-1][:4]
            if h4 in am:
                pair.setdefault(deck, am[h4])

    lg = ladder_stats()
    print(f"{'我方牌':34s}{'对手牌':34s}{'天梯':>8s}{'n':>5s}{'面板':>8s}{'n':>5s}{'偏差':>8s}")
    diffs = []
    for deck, cand in sorted(pair.items()):
        if deck not in lg:
            continue
        ps = panel_stats(cand)
        for opp, (w, n) in sorted(lg[deck].items(), key=lambda kv: -kv[1][1]):
            if n < a.min_n or opp not in ps:
                continue
            lwr = w / n
            pwr, pn = ps[opp]
            diffs.append(lwr - pwr)
            print(f"{deck:34s}{opp:34s}{lwr*100:7.1f}%{n:5d}{pwr*100:7.1f}%{pn:5d}{(lwr-pwr)*100:+7.1f}")
    if diffs:
        m = sum(diffs) / len(diffs)
        print(f"\n共 {len(diffs)} 格；平均偏差 {(m*100):+.1f}pp（正=天梯比面板好；负=面板偏乐观）")
        print("提示：若整体明显为负 ⇒ 面板偏乐观（本地对手比真实对手弱）；明显为正 ⇒ 面板偏悲观。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
