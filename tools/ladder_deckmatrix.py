#!/usr/bin/env python3
"""**从天梯回放直接统计"卡组 × 对手卡组"的胜率矩阵**（不需要玩家评分，也不需要知道是谁）。

为什么这个能回答"某副牌在 ladder 上强不强"：回放里**双方 60 张卡表都可见**
（`steps[1][j].action`），胜者由 `rewards` 给 ⇒ 每场都是一条 (deck_a, deck_b, winner)。
把语料里所有场次折成对局表，就能得到"这副牌在真实天梯里对各类对手的实际胜率"。

    ./tools/py tools/ladder_deckmatrix.py --dirs epsoide_teams2,epsoide_miwa,epsoide_petit
    ./tools/py tools/ladder_deckmatrix.py --dirs ... --min-games 20      # 只列样本足够多的牌

输出：`data/ladder_deckmatrix.tsv`（file/seat0deck/seat1deck/winner）+ 卡组级胜率表 + 指定牌的对位表。
口径：卡组 tag = sha256(排序后卡 id)[:8]，与 decks/pool_field 的 4 位后缀同源。
分类会把"同一族的 4 位后缀"合并（如 `dreepy_drakloak__3eaa` 归入 `3eaa`）。
"""
from __future__ import annotations
import argparse, collections, csv, glob, hashlib, json, math, multiprocessing as mp, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from framework.replay import decks_of  # noqa: E402


def tag(ids):
    return hashlib.sha256(",".join(map(str, sorted(ids))).encode()).hexdigest()[:8]


def one(path):
    try:
        e = json.load(open(path))
        steps = e.get("steps") or []
        if len(steps) < 3:
            return None
        dk = decks_of(steps)
        if len(dk) < 2:
            return None
        rw = e.get("rewards") or []
        if len(rw) < 2 or rw[0] is None or rw[0] == rw[1]:
            return None
        s0, s1 = sorted(dk)[0], sorted(dk)[1]
        return (os.path.basename(path), tag(dk[s0]), tag(dk[s1]), 0 if rw[s0] > rw[s1] else 1)
    except Exception:
        return None


def wilson(k, n, z=1.96):
    if not n: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0, (c - h) / d), min(1, (c + h) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--min-games", type=int, default=15)
    ap.add_argument("--out", default="data/ladder_deckmatrix.tsv")
    ap.add_argument("--focus", default="", help="逗号分隔的 4 位后缀，打印这些牌的对位表")
    a = ap.parse_args()
    files = []
    for d in a.dirs.split(","):
        files += sorted(glob.glob(os.path.join(ROOT, d.strip(), "*.json")))
    jobs = a.jobs or min(32, (os.cpu_count() or 8) * 2)
    rows = []
    with mp.Pool(jobs) as pool:
        for r in pool.imap_unordered(one, files, chunksize=8):
            if r: rows.append(r)
    with open(os.path.join(ROOT, a.out), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t"); w.writerow(["file", "seat0deck", "seat1deck", "winner"])
        w.writerows(rows)
    # 卡组级胜率（两个座位都算）
    agg = collections.Counter(); win = collections.Counter()
    pair = collections.Counter(); pairw = collections.Counter()
    for f, d0, d1, wnr in rows:
        for i, d in ((0, d0), (1, d1)):
            agg[d] += 1
            if wnr == i: win[d] += 1
        pair[(d0, d1)] += 1
        pairw[(d0, d1)] += (wnr == 0)
    # 4 位后缀 → 池内名字
    SHORT = {}
    for p in glob.glob(os.path.join(ROOT, "decks", "pool_field", "*.csv")):
        ids = sorted(int(x) for x in open(p) if x.strip())
        SHORT[tag(ids)[:4]] = os.path.basename(p)[:-4]
    print(f"对局 {len(rows)} 场（{len(agg)} 副卡表）；输出 data/ladder_deckmatrix.tsv\n")
    print(f"{'卡组':28s}{'场次':>6s}{'胜率':>7s}  95%CI")
    for d, n in agg.most_common(15):
        if n < a.min_games: continue
        lo, hi = wilson(win[d], n)
        name = SHORT.get(d[:4], d)
        print(f"{name:28s}{n:6d}{win[d]/n*100:6.1f}%  [{lo*100:.0f},{hi*100:.0f}]")
    for short in [x for x in a.focus.split(",") if x]:
        rows2 = [(d, n, win[d]) for d, n in agg.items() if d.startswith(short) and n >= a.min_games]
        if not rows2: continue
        print(f"\n== {short} 的对位（我方为该牌，样本≥{a.min_games}）")
        for d, n, k in sorted(rows2, key=lambda t: -t[2] / t[1]):
            lo, hi = wilson(k, n)
            print(f"  {SHORT.get(d[:4], d):28s}{k:3d}/{n:<4d} {k/n*100:5.1f}%  CI[{lo*100:.0f},{hi*100:.0f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
