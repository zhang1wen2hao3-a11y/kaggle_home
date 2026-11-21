#!/usr/bin/env python3
"""**逐格缺口：我们打不过、但高分真人打得过的对手**（2026-11-22 用户要求）

用户原话：「顺便看下主要的缺陷，比如哪些对手我们打不过但是高分选手可以」。

## 两侧数据

* **我们**：强对手 17 格面板 `logs/meta/<候选>_real_s95*/vs_<格>.json`
  （每格 n≈150，对手是"该牌实测最硬的驾驶员"，口径见 `decks/testset_meta_real.tsv`）。
* **高分真人**：`data/replay_matchups.tsv`（3807 局真实天梯对局，带双方队名/牌/胜方）
  × `data/lb_scores.json`（选手天梯分）。只统计**真人**（排除我们自己的提交 `zhang1wen2hao3`、
  解析失败的 `PARSE_ERR` 等），并按分数线分档。

## 为什么这样比

面板给的是"对最强 AI 驾驶员的胜率"，真人给的是"对真人的胜率"，两者的**对手驾驶员强度不同**，
所以绝对数不能直接相减 —— 但**排序**可以用：若某格高分真人 70%、我们 10%，那这一格是
"结构上我们不会打"，值得优先修；若真人 40%、我们 45%，说明我们已经打到人类水平，不用再投。

## 用法

    ./tools/py tools/cell_gap_vs_humans.py --deck z_dragapult_dreepy_petit__0904 --cand cand0904R1
    ./tools/py tools/cell_gap_vs_humans.py --deck dunsparce_abra__9d6e --cand cand9D6ER3
    ./tools/py tools/cell_gap_vs_humans.py --deck X --cand Y --min-human-n 5 --min-score 1300 --csv out.csv
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
OURS = {"zhang1wen2hao3"}          # 我们自己的提交，不算"高分真人"
JUNK = {"PARSE_ERR", "", "PTCG_Agent"}


def lb_scores() -> dict:
    try:
        return {k: float(v) for k, v in json.load(open(os.path.join(ROOT, "data", "lb_scores.json"))).items()}
    except OSError:
        return {}


def cell_to_deck() -> dict:
    """格 → 对手牌名（取自真实口径测试集表第 3 列）。"""
    out = {}
    p = os.path.join(ROOT, "decks", "testset_meta_real.tsv")
    for ln in open(p):
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.rstrip("\n").split("\t")
        if len(f) >= 3 and f[0] != "tail_mix":
            out[f[0]] = f[2]
    return out


def human_stats(deck: str, lb: dict, min_score: float):
    """我们的这副牌在天梯上，各对手牌的真人战绩。返回 {opp_deck: dict}。"""
    per = collections.defaultdict(lambda: dict(n=0, w=0, hn=0, hw=0, by=collections.Counter(), byg=collections.Counter()))
    path = os.path.join(ROOT, "data", "replay_matchups.tsv")
    for r in csv.DictReader(open(path), delimiter="\t"):
        d0, d1 = (r.get("deck0") or ""), (r.get("deck1") or "")
        if deck not in (d0, d1):
            continue
        try:
            win = int(r["winner"])
        except (TypeError, ValueError):
            continue
        me, opp = (0, d1) if d0 == deck else (1, d0)
        team = (r.get(f"team{me}") or "").strip()
        if not team or team in OURS or team in JUNK:
            continue
        sc = lb.get(team)
        rec = per[opp]
        rec["n"] += 1
        rec["w"] += int(win == me)
        if sc is not None and sc >= min_score:
            rec["hn"] += 1
            rec["hw"] += int(win == me)
        if sc is not None:
            rec["by"][team] += int(win == me)
            rec["byg"][team] += 1
    for rec in per.values():
        best = [(sc, t) for t, sc in lb.items() if rec["byg"].get(t, 0) >= 3]
        best.sort(reverse=True)
        rec["top"] = ""
        if best:
            sc, t = best[0]
            rec["top"] = f"{t} {sc:.0f}（{rec['by'][t]}/{rec['byg'][t]}）"
    return per


def panel_stats(cand: str) -> dict:
    """{对手牌: (胜率, n, 格)} —— 同一副对手牌若有多格（如 champ/out 前缀）就合并。"""
    c2d = cell_to_deck()
    agg = collections.defaultdict(lambda: [0.0, 0])
    for p in glob.glob(os.path.join(ROOT, "logs", "meta", f"{cand}_real_s95*", "vs_*.json")):
        cell = os.path.basename(p)[3:-5]
        try:
            o = json.load(open(p)).get("overall") or {}
        except (OSError, ValueError):
            continue
        n = int(o.get("n") or 0)
        if not n:
            continue
        d = c2d.get(cell)
        if not d:
            continue
        a = agg[d]
        a[0] += float(o.get("score") or 0) * n
        a[1] += n
    return {d: (v[0] / v[1], v[1]) for d, v in agg.items() if v[1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True, help="我们的牌（replay_matchups 里的牌名，如 z_dragapult_dreepy_petit__0904）")
    ap.add_argument("--cand", default="", help="面板候选名（logs/meta/<cand>_real_s95*）；不给就只出真人侧")
    ap.add_argument("--min-human-n", type=int, default=10, help="真人侧该格最少局数（默认 10）")
    ap.add_argument("--min-score", type=float, default=1300.0, help="算作'高分选手'的分数线（默认 1300）")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    lb = lb_scores()
    hs = human_stats(a.deck, lb, a.min_score)
    ps = panel_stats(a.cand) if a.cand else {}

    shares = {}
    try:
        for ln in open(os.path.join(ROOT, "decks", "testset_meta_real.tsv")):
            if ln.startswith("#") or not ln.strip():
                continue
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 5 and f[0] != "tail_mix":
                shares[f[2]] = float(f[4])
    except OSError:
        pass

    rows = []
    for d, rec in hs.items():
        if rec["hn"] < a.min_human_n:
            continue
        hwr = rec["hw"] / rec["hn"]
        if d in ps:
            pwr, pn = ps[d]
            gap = hwr - pwr
        else:
            pwr, pn, gap = None, 0, None
        rows.append(dict(deck=d, share=shares.get(d), hwr=hwr, hn=rec["hn"],
                         allwr=(rec["w"] / rec["n"]) if rec["n"] else 0.0, alln=rec["n"],
                         pwr=pwr, pn=pn, gap=gap, top=rec["top"]))
    rows.sort(key=lambda r: (-(r["gap"] if r["gap"] is not None else -9), -(r["share"] or 0)))

    print(f"我方牌：{a.deck}   面板候选：{a.cand or '（未给）'}   高分选手线：{a.min_score:.0f} 分   真人格门槛 n≥{a.min_human_n}")
    hdr = f"{'对手牌':38s}{'份额':>6s}{'高分真人':>9s}{'n':>5s}{'我们':>8s}{'n':>5s}{'缺口':>8s}  最高分玩家(该格战绩)"
    print(hdr)
    for r in rows:
        p = f"{r['pwr']*100:6.1f}%" if r["pwr"] is not None else "     —"
        g = f"{r['gap']*100:+6.1f}" if r["gap"] is not None else "     —"
        sh = f"{r['share']:5.1f}%" if r["share"] is not None else "    —"
        print(f"{r['deck']:38s}{sh}{r['hwr']*100:8.1f}%{r['hn']:5d}{p}{r['pn']:5d}{g}  {r['top']}")

    hard = [r for r in rows if r["gap"] is not None and r["gap"] >= 0.20]
    if hard:
        print("\n**打不过但高分真人打得过（缺口 ≥20pp）**：")
        for r in hard[:8]:
            sh = f"{r['share']:.1f}%" if r["share"] is not None else "非17格"
            print(f"  · {r['deck']:36s} 真人 {r['hwr']*100:.0f}% (n={r['hn']}) vs 我们 {r['pwr']*100:.0f}% "
                  f"⇒ {r['gap']*100:+.0f}pp   份额 {sh}")

    if a.csv:
        out = os.path.join(ROOT, a.csv) if not os.path.isabs(a.csv) else a.csv
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["我方牌", "对手牌", "真实场份额%", "高分真人胜率", "真人n", "全体真人胜率", "全体n",
                        "我方面板胜率", "面板n", "缺口pp", "最高分玩家(该格战绩)"])
            for r in rows:
                w.writerow([a.deck, r["deck"], "" if r["share"] is None else f"{r['share']:.2f}", f"{r['hwr']*100:.1f}", r["hn"],
                            f"{r['allwr']*100:.1f}", r["alln"],
                            "" if r["pwr"] is None else f"{r['pwr']*100:.1f}", r["pn"] or "",
                            "" if r["gap"] is None else f"{r['gap']*100:+.1f}", r["top"]])
        print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
