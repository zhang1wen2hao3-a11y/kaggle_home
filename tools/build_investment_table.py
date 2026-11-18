#!/usr/bin/env python3
"""**投资分表**：把「面板补后水平」× 「天花板系数」合成一个可排序的补强优先级。

为什么需要单独一张表（2026-11-17 用户要求接进 refresh_tables）：
  · 面板分只回答"这牌现在多强"；天花板（外部高分玩家打到的分数）回答"这牌能多强"；
  · 两者相乘才是"还值不值得投"，而且**必须区分**外部记录与我们的记录（后者是循环论证）。

口径：
  面板补后水平 = 覆盖加权胜率 + 可补缺口
      · 覆盖加权胜率：真实份额权重（decks/testset_meta_real.tsv）+ 母表里该牌最佳驾驶员的逐格胜率
      · 可补缺口：Σ 真实份额 × max(0, 端口基线 − 该牌最佳驾驶员)，端口基线取母表 init-vs-init
  天花板 = max(外部最高分, 我们最高分)，外部分来自 data/deck_corpus.csv × data/lb_scores.json
  K = clip(天花板 / 全表中位天花板, 0.85, 1.25)
  投资分 = 面板补后水平 × K；上限差距 = 天花板 − 我们最高分

输出：docs/卡组天花板与投资分.csv（同时打印排序，供"下一轮训什么"直接用）
    ./tools/py tools/build_investment_table.py
"""
from __future__ import annotations
import collections, csv, glob, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-in-table.csv")
TS = os.path.join(ROOT, "decks", "testset_meta_real.tsv")
OUT = os.path.join(ROOT, "docs", "卡组天花板与投资分.csv")


def real_weights():
    W = {}
    for ln in open(TS):
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.rstrip("\n").split("\t")
        if f[0] == "tail_mix":
            continue
        s = f[0].split("_")[-1]
        W[s] = W.get(s, 0.0) + float(f[4])
    t = sum(W.values()) or 1.0
    return {k: v / t for k, v in W.items()}


def load_master():
    rec = collections.defaultdict(lambda: collections.defaultdict(dict))
    txt = open(MASTER, "rb").read().decode("gb18030", errors="replace")
    for r in csv.DictReader(txt.splitlines()):
        if (r.get("对手驾驶员") or "init") != "init":
            continue
        try:
            rec[r["挑战者"]][r["被挑战者"]][r["驾驶员"]] = (float(r["胜率"]), int(float(r["样本场次"])))
        except (TypeError, ValueError):
            pass
    return rec


def _load_ours():
    p = os.path.join(ROOT, "data", "lb_ours_by_deck.json")
    if os.path.exists(p):
        return {k: float(v["score"]) for k, v in json.load(open(p)).items()}
    return {}


def lb_maps():
    """外部天花板：**按牌算，不按人算**。

    坑（2026-11-18 实测）：`lb_scores.json` 是**玩家级**分数。若把他用过的每副牌都记上他的分，
    会严重高估 —— Ondrej Kubicek 名下挂了 `0bf5`，但他 0bf5 只打过 **4 局、0 胜**；
    `f5f8` 也只打 4 局。他的 1428 其实来自 shuppet（0d86，51 条记录、23/41 胜）。
    ⇒ 只有该玩家在这副牌上 **≥MIN_GAMES 局**，才把他的分数算作这副牌的天花板。
    """
    LB = json.load(open(os.path.join(ROOT, "data", "lb_scores.json")))
    MIN_GAMES = 10
    games = collections.Counter()          # (h4, team) -> 局数（从真人档案统计）
    try:
        for r in csv.DictReader(open(os.path.join(ROOT, "data", "ladder_games.tsv")), delimiter="\t"):
            for i in (0, 1):
                t = (r.get(f"seat{i}_agent") or "").strip()
                h4 = (r.get(f"seat{i}_deck") or "")[:4]
                if t and h4:
                    games[(h4, t)] += 1
    except OSError:
        pass
    users = collections.defaultdict(set)
    for r in csv.DictReader(open(os.path.join(ROOT, "data", "deck_corpus.csv"))):
        users[(r.get("deck_hash") or "")[:4]].add(r.get("team") or "")
    ext, extwho = {}, {}
    for h4, teams in users.items():
        best, who = 0.0, ""
        for t in teams:
            if games.get((h4, t), 0) < MIN_GAMES:
                continue
            s_ = LB.get(t, 0)
            if s_ > best:
                best, who = s_, t
        if best:
            ext[h4], extwho[h4] = best, who
    return ext, extwho, _load_ours()


def real_panel_levels():
    """从 logs/meta/<cand>_real_s95*/vs_*.json 算真实口径 17 格加权分（若有）。"""
    import glob, json
    out = {}
    for tag in glob.glob(os.path.join(ROOT, "logs", "meta", "*_real_s95*")):
        cand = os.path.basename(tag).split("_real_s95")[0]
        num = den = 0.0
        for p in glob.glob(os.path.join(tag, "vs_*.json")):
            try:
                o = json.load(open(p))["overall"]
            except Exception:
                continue
            cell = os.path.basename(p)[3:-5].split("_")[-1]
            w = REALW.get(cell, 0.0)
            if w:
                num += w * o["score"] * o["n"]; den += w * o["n"]
        if den:
            out[cand] = (num / den * 100.0, den)
    return out


def cand_deck_map():
    """面板候选名 → 它开的卡组 4 位后缀（走注册表；查不到就用名字里的卡组号猜）。"""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    try:
        import driver_registry as REG
    except Exception:
        REG = None
    out = {}
    for cand in real_panel_levels():
        dk = None
        if REG:
            try:
                dk = str(REG.info(cand).get("deck") or "").lower()
            except Exception:
                dk = None
        if not dk:
            for tok in ("f5f8", "0bf5", "9d6e", "db04", "26c6", "7e39", "0d86", "2617", "3eaa", "e1d7", "0904", "e8ee"):
                if tok in cand.lower():
                    dk = tok; break
        if dk:
            out[cand] = dk
    return out


def main():
    W = real_weights()
    global REALW
    REALW = W
    D = load_master()
    PANEL = real_panel_levels()
    ext, extwho, ours = lb_maps()
    rows = []
    for deck, cells in D.items():
        def sc(drv):
            num = wn = 0.0
            for c, v in cells.items():
                if drv in v and W.get(c):
                    num += W[c] * v[drv][0]
                    wn += W[c]
            return (num / wn if wn else None)
        base = sc("init")
        cands = [(d, sc(d)) for d in {d for v in cells.values() for d in v} if sc(d) is not None]
        if not cands or base is None:
            continue
        best = max(cands, key=lambda t: t[1])
        gap = 0.0
        for c, v in cells.items():
            if W.get(c) and "init" in v and best[0] in v:
                gap += W[c] * max(0.0, v["init"][0] - v[best[0]][0]) * 100
        lvl = best[1] * 100
        src = "母表12格"
        # ★ 有真实 17 格实测就用实测（更接近真实场）；此时"可补缺口"也失去意义 → 置 0
        CD = cand_deck_map()
        same = [(cand, PANEL[cand]) for cand, dk in CD.items() if dk == deck.lower() and cand in PANEL]
        if same:
            panel_key, (lvl, pn) = max(same, key=lambda t: t[1][0])
            src, gap = f"真实17格({panel_key},n={int(pn)})", 0.0
            best = (panel_key, lvl / 100.0)
        rows.append(dict(deck=deck, best=best[0], lvl=lvl, gap=gap, after=lvl + gap, src=src,
                         ext=ext.get(deck), extwho=extwho.get(deck, ""), ours=ours.get(deck)))
    vals = sorted({r["c"] for r in [dict(c=max([x for x in (r["ext"], r["ours"]) if x] or [0])) for r in rows] if r["c"]})
    med = vals[len(vals) // 2] if vals else 1.0
    for r in rows:
        ceil_ = max([x for x in (r["ext"], r["ours"]) if x] or [0]) or None
        r["ceil"] = ceil_
        r["K"] = max(0.85, min(1.25, ceil_ / med)) if ceil_ else 1.0
        r["inv"] = r["after"] * r["K"]
        r["gapceil"] = (ceil_ - r["ours"]) if (ceil_ and r["ours"]) else None
    rows.sort(key=lambda r: -r["inv"])
    with open(OUT, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["卡组", "最佳驾驶员", "覆盖加权胜率%", "可补缺口pp", "面板补后水平%", "口径",
                    "外部天花板", "外部最高分玩家", "我们最高分", "天花板", "K", "上限差距", "投资分"])
        for r in rows:
            w.writerow([r["deck"], r["best"], round(r["lvl"], 1), round(r["gap"], 2), round(r["after"], 1), r["src"],
                        r["ext"] or "", r["extwho"], r["ours"] or "", r["ceil"] or "",
                        round(r["K"], 3), "" if r["gapceil"] is None else round(r["gapceil"], 1),
                        round(r["inv"], 1)])
    print(f"→ {os.path.relpath(OUT, ROOT)}（{len(rows)} 副，天花板中位数 {med:.1f}）")
    print(f"{'排名':<4}{'卡组':7s}{'驾驶员':13s}{'补后水平':>9s}{'天花板':>8s}{'上限差距':>9s}{'K':>6s}{'投资分':>8s}")
    for i, r in enumerate(rows[:10], 1):
        print(f"{i:<4}{r['deck']:7s}{r['best']:13s}{r['after']:8.1f}%{(r['ceil'] or 0):8.0f}"
              f"{('—' if r['gapceil'] is None else f'{r[chr(103)+chr(97)+chr(112)+chr(99)+chr(101)+chr(105)+chr(108)]:.0f}'):>9s}"
              f"{r['K']:6.2f}{r['inv']:8.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
