#!/usr/bin/env python3
"""**高分玩家对位表**：把已归档的真人天梯回放折成「玩家×我方卡组×对手卡组」的胜率表，
并与我们自己的实测胜率并排比较（按对手卡组维度）。

数据源：
  · `data/ladder_games.tsv`  —— 1756 局真人天梯档案（`tools/ladder_archive.py` 生成，双座位都有）
  · `logs/sub_feedback/scan.jsonl` —— 我们提交的真实对局（对手卡组 + 胜负）

输出：
  · `docs/高分玩家对位表.csv`        玩家 / 玩家卡组 / 对手卡组 / 胜 / 负 / 胜率 / 95%CI
  · `docs/我们vs高分玩家_按对手卡组.csv` 对手卡组 / 我们胜率 / 高分玩家胜率 / 差 / 各自样本

    ./tools/py tools/highrank_matchups.py --min-games 5
"""
from __future__ import annotations
import argparse, collections, csv, glob, hashlib, io, json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = os.path.join(ROOT, "decks", "pool_field")


def pool_short():
    """8/4 位 deck hash → (池内全名, 4 位后缀)。"""
    m = {}
    for p in glob.glob(os.path.join(POOL, "*.csv")):
        ids = sorted(int(x) for x in open(p) if x.strip())
        h8 = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:8]
        name = os.path.basename(p)[:-4]
        m[h8] = name
        m.setdefault(h8[:4], name)
    return m


def wilson(k, n, z=1.96):
    if not n: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0, (c - h) / d), min(1, (c + h) / d)


def short_of(m, h):
    n = m.get(h) or m.get((h or "")[:4]) or h
    return n.split("__")[-1] if "__" in n else n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-games", type=int, default=5)
    ap.add_argument("--players", default="", help="只统计这些玩家（逗号分隔，留空=全部）")
    a = ap.parse_args()
    m = pool_short()
    only = {p.strip() for p in a.players.split(",") if p.strip()}

    # ① 高分玩家档案 → (玩家, 我方卡组, 对手卡组) 胜负
    acc = collections.Counter(); gw = collections.Counter()
    for r in csv.DictReader(open(os.path.join(ROOT, "data", "ladder_games.tsv")), delimiter="\t"):
        try: win = int(r["winner"])
        except (TypeError, ValueError): continue
        for i in (0, 1):
            me = (r.get(f"seat{i}_agent") or "").strip()
            mine = r.get(f"seat{i}_deck") or ""
            theirs = r.get(f"seat{1-i}_deck") or ""
            if not me or not mine or not theirs: continue
            if only and me not in only: continue
            k = (me, short_of(m, mine), short_of(m, theirs))
            acc[k] += 1
            if win == i: gw[k] += 1

    rows = []
    for (pl, own, opp), n in acc.items():
        if n < a.min_games: continue
        lo, hi = wilson(gw[(pl, own, opp)], n)
        rows.append(dict(玩家=pl, 玩家卡组=own, 对手卡组=opp, 胜=gw[(pl, own, opp)], 负=n - gw[(pl, own, opp)],
                         场次=n, 胜率=round(gw[(pl, own, opp)] / n, 4), CI_low=round(lo, 3), CI_high=round(hi, 3)))
    rows.sort(key=lambda r: (r["玩家"], -r["场次"]))
    out1 = os.path.join(ROOT, "docs", "高分玩家对位表.csv")
    with open(out1, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["玩家", "玩家卡组", "对手卡组", "胜", "负", "场次", "胜率", "CI_low", "CI_high"])
        w.writeheader(); w.writerows(rows)
    print(f"→ docs/高分玩家对位表.csv（{len(rows)} 行，阈值 ≥{a.min_games} 局）")

    # ② 按「对手卡组」比较：高分玩家 vs 我们
    by_opp_h = collections.Counter(); by_opp_hw = collections.Counter()
    for (pl, own, opp), n in acc.items():
        by_opp_h[opp] += n; by_opp_hw[opp] += gw[(pl, own, opp)]
    ours = collections.Counter(); oursw = collections.Counter()
    sp = os.path.join(ROOT, "logs", "sub_feedback", "scan.jsonl")
    if os.path.exists(sp):
        for ln in open(sp):
            try: r = json.loads(ln)
            except Exception: continue
            if not r.get("opp_hash") or r.get("win") is None or r.get("selfmirror"): continue
            o = short_of(m, r["opp_hash"])
            ours[o] += 1; oursw[o] += 1 if r["win"] else 0
    cmp_rows = []
    for opp in sorted(set(by_opp_h) | set(ours), key=lambda x: -(by_opp_h[x] + ours[x])):
        hn, hk = by_opp_h[opp], by_opp_hw[opp]
        on, ok = ours[opp], oursw[opp]
        if hn < a.min_games and on < 3: continue
        hwr = hk / hn if hn else None
        owr = ok / on if on else None
        cmp_rows.append(dict(对手卡组=opp,
                             我们胜率=f"{owr*100:.1f}%" if owr is not None else "",
                             我们样本=on,
                             高分玩家胜率=f"{hwr*100:.1f}%" if hwr is not None else "",
                             高分玩家样本=hn,
                             差pp=f"{(owr-hwr)*100:+.1f}" if (owr is not None and hwr is not None) else ""))
    out2 = os.path.join(ROOT, "docs", "我们vs高分玩家_按对手卡组.csv")
    with open(out2, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["对手卡组", "我们胜率", "我们样本", "高分玩家胜率", "高分玩家样本", "差pp"])
        w.writeheader(); w.writerows(cmp_rows)
    print(f"→ docs/我们vs高分玩家_按对手卡组.csv（{len(cmp_rows)} 行）\n")
    print(f"{'对手卡组':22s}{'我们':>7s}{'n':>4s}{'高分玩家':>9s}{'n':>5s}{'差':>8s}")
    for r in cmp_rows[:16]:
        print(f"{r['对手卡组']:22s}{r['我们胜率']:>7s}{r['我们样本']:>4d}{r['高分玩家胜率']:>9s}"
              f"{r['高分玩家样本']:>5d}{r['差pp']:>8s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
