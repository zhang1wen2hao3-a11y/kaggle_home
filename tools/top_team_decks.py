#!/usr/bin/env python3
"""**高分选手现在在打什么牌**：从抓下来的顶级队伍回放里统计 (队伍, 卡组) 与胜率，
并与我们的覆盖（17 格测试集 / 母表 / 池）逐一比对，标出"他们在用但我们没覆盖"的牌。

    ./tools/py tools/top_team_decks.py --dirs epsoide_top18
"""
from __future__ import annotations
import argparse, collections, csv, glob, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from framework.replay import decks_of  # noqa: E402


def pool_map():
    m = {}
    for p in glob.glob(os.path.join(ROOT, "decks", "pool_field", "*.csv")):
        ids = sorted(int(x) for x in open(p) if x.strip())
        h = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:8]
        m[h] = os.path.basename(p)[:-4]
        m.setdefault(h[:4], os.path.basename(p)[:-4])
    return m


def covered_sets():
    ts = set()
    for ln in open(os.path.join(ROOT, "decks", "testset_meta_real.tsv")):
        if ln.startswith("#") or not ln.strip():
            continue
        ts.add(ln.split("\t")[0].split("_")[-1])
    master = set()
    txt = open(os.path.join(ROOT, "docs", "the-pockmon-battle-choice-in-table.csv"), "rb").read().decode("gb18030", "replace")
    for r in csv.DictReader(txt.splitlines()):
        master.add(r["被挑战者"])
    return ts, master


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="epsoide_top18")
    a = ap.parse_args()
    m = pool_map()
    ts, master = covered_sets()
    # 队伍分（从榜单 CSV）
    score = {}
    lb = glob.glob(os.path.join(ROOT, "data", "leaderboard", "*.csv"))
    if lb:
        for r in csv.DictReader(open(sorted(lb)[-1], encoding="utf-8-sig", errors="replace")):
            try:
                score[(r.get("TeamName") or "").strip()] = float(r.get("Score") or 0)
            except (TypeError, ValueError):
                pass
    games = collections.Counter(); wins = collections.Counter(); decks_of_team = collections.defaultdict(set)
    files = []
    for d in a.dirs.split(","):
        files += sorted(glob.glob(os.path.join(ROOT, d.strip(), "*.json")))
    for p in files:
        try:
            e = json.load(open(p))
        except Exception:
            continue
        dk = decks_of(e.get("steps") or [])
        names = [(x.get("Name") or "").strip() for x in ((e.get("info") or {}).get("Agents") or [])]
        try:
            win = int(e.get("rewards", [None, None]).index(1))
        except ValueError:
            win = -1
        for i, nm in enumerate(names):
            if i not in dk or not nm:
                continue
            h = hashlib.sha256(",".join(map(str, sorted(dk[i]))).encode()).hexdigest()[:8]
            name = m.get(h) or m.get(h[:4]) or f"未入库:{h}"
            games[(nm, name)] += 1
            if i == win:
                wins[(nm, name)] += 1
            decks_of_team[nm].add(name.split("__")[-1])
    rows = sorted(games.items(), key=lambda t: (-score.get(t[0][0], 0), -t[1]))
    print(f"回放 {len(files)} 局；队伍 {len(decks_of_team)} 个\n")
    print(f"{'队伍':24s}{'榜分':>7s}{'卡组':32s}{'局数':>5s}{'胜率':>7s}  覆盖")
    for (team, deck), n in rows:
        short = deck.split("__")[-1]
        cov = []
        cov.append("17格" if short in ts else "—")
        cov.append("母表" if short in master else "—")
        print(f"{team[:22]:24s}{score.get(team, 0):7.0f}{deck:32s}{n:5d}{wins[(team, deck)]/n*100:6.1f}%  {'/'.join(cov)}")
    print("\n=== 他们在用、但我们 17 格测试集里没有的牌 ===")
    miss = collections.Counter()
    for (team, deck), n in rows:
        if deck.split("__")[-1] not in ts:
            miss[deck] += n
    for deck, n in miss.most_common():
        users = [t for (t, d) in games if d == deck]
        print(f"  {deck:34s}{n:4d} 局  使用者: {', '.join(sorted(set(users)))[:60]}")
    if not miss:
        print("  （无：他们在用的牌都在我们测试集里）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
