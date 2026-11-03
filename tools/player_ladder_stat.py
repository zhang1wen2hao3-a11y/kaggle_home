#!/usr/bin/env python3
"""某个真人玩家的天梯战绩统计（从回放目录直接算，不依赖已蒸馏数据）。

## 为什么需要
抓下来的回放目录（`epsoide_petit` 等）只存原始 JSON，没存"谁赢了/用的哪副卡组"。
要判断**某位高手那副卡组值不值得抄**（卡组轴）、以及**抄到手能打赢谁**（对位），
就必须先把这些回放折成一张对局表：一行一场，带双方卡组 8 位 tag、双方 agent 名、胜方。

## 口径（与 `framework/replay.py` 一致，已在本仓库独立验证）
* 卡表在 `steps[1][j].action`，长度恰为 60
* 卡组 tag = `sha256(",".join(sorted(ids)))[:8]`（和 `decks/pool_field` 的 4 位后缀同源）
* 胜者 = `rewards` 里为 +1 的座位；`rewards=[1,-1]` 表示 seat0 胜
* agent 名在 `info.Agents[i].Name`；**同一名字可能两个座位都有**（镜像/同队）

## 用法
    ./tools/py tools/player_ladder_stat.py --dir epsoide_petit --player "Petit Canard"
    ./tools/py tools/player_ladder_stat.py --dir epsoide_miwa --player "やる気元気ミワハルキ" --cache data/miwa_games.tsv
    ./tools/py tools/player_ladder_stat.py --table data/petit_games.tsv   # 只重算统计
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from framework.replay import decks_of  # noqa: E402

FIELDS = ["file", "seat0", "seat1", "agent0", "agent1", "win", "n_steps"]


def tag(ids):
    return hashlib.sha256(",".join(str(x) for x in sorted(ids)).encode()).hexdigest()[:8]


def one(path):
    try:
        d = json.load(open(path))
        steps = d.get("steps") or []
        if len(steps) < 2:
            return None
        dk = decks_of(steps)
        if len(dk) != 2:
            return None
        names = [a.get("Name", "?") for a in (d.get("info") or {}).get("Agents", [])]
        while len(names) < 2:
            names.append("?")
        rw = d.get("rewards") or [0, 0]
        win = 0 if (rw[0] or 0) > (rw[1] or 0) else (1 if (rw[1] or 0) > (rw[0] or 0) else -1)
        return [os.path.basename(path), tag(dk[0]), tag(dk[1]), names[0], names[1],
                str(win), str(len(steps))]
    except Exception:
        return None


def build(directory, cache, workers):
    files = sorted(glob.glob(os.path.join(directory, "*.json")))
    print(f"扫描 {len(files)} 个回放 → {cache}", file=sys.stderr)
    rows = []
    if workers <= 1:
        for i, f in enumerate(files, 1):
            r = one(f)
            if r:
                rows.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(files)}", file=sys.stderr)
    else:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(one, files, chunksize=2), 1):
                if r:
                    rows.append(r)
                if i % 100 == 0:
                    print(f"  {i}/{len(files)}", file=sys.stderr)
    rows.sort()
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    with open(cache, "w") as fh:
        fh.write("\t".join(FIELDS) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    print(f"写出 {len(rows)} 场 → {cache}", file=sys.stderr)
    return [dict(zip(FIELDS, r)) for r in rows]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    dd = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, (c - h) / dd), min(1.0, (c + h) / dd)


def load(cache):
    rows = []
    with open(cache) as fh:
        head = fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(FIELDS):
                continue
            rows.append(dict(zip(FIELDS, p)))
    return rows


def report(rows, player):
    """player 坐任意座位的所有对局都算；镜像对局(两边同名)单独标出。"""
    mine = collections.defaultdict(lambda: [0, 0])       # 我的卡组 → [w, n]
    opp = collections.defaultdict(lambda: [0, 0])        # 对手卡组 → [w, n]
    n_all = n_dec = 0
    for r in rows:
        win = int(r["win"])
        if win < 0:
            n_dec += 1
            continue
        s0, s1 = r["seat0"], r["seat1"]
        a0, a1 = r["agent0"], r["agent1"]
        i_am = [i for i, a in enumerate((a0, a1)) if player in a]
        if not i_am:
            continue
        n_all += 1
        my_deck = (s0, s1)[i_am[0]]
        op_deck = (s0, s1)[1 - i_am[0]]
        w = 1 if win == i_am[0] else 0
        mine[my_deck][0] += w
        mine[my_deck][1] += 1
        if my_deck != op_deck:
            opp[op_deck][0] += w
            opp[op_deck][1] += 1

    print(f"\n== {player} · {n_all} 场（无胜负记录 {n_dec}）==")
    tot = [sum(v[0] for v in mine.values()), sum(v[1] for v in mine.values())]
    p, lo, hi = wilson(*tot)
    print(f"{'自己的卡组':14s}{'胜率':>7}{'95% CI':>18}{'场次':>6}")
    for d, (w, n) in sorted(mine.items(), key=lambda kv: -kv[1][1]):
        p, lo, hi = wilson(w, n)
        print(f"{d:14s}{p:>7.3f}{'[%.3f,%.3f]' % (lo, hi):>18}{n:>6}")
    print(f"{'合计':14s}{tot[0] / max(1, tot[1]):>7.3f}"
          f"{'[%.3f,%.3f]' % wilson(*tot)[1:]:>18}{tot[1]:>6}")
    print(f"\n{'对手卡组(非镜像)':14s}{'胜率':>7}{'95% CI':>18}{'场次':>6}")
    for d, (w, n) in sorted(opp.items(), key=lambda kv: -kv[1][1]):
        if n < 5:
            continue
        p, lo, hi = wilson(w, n)
        print(f"{d:14s}{p:>7.3f}{'[%.3f,%.3f]' % (lo, hi):>18}{n:>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="回放目录")
    ap.add_argument("--player", default="", help="玩家名（子串匹配 info.Agents[i].Name）")
    ap.add_argument("--cache", default="", help="对局表缓存路径（默认 data/<dir>_games.tsv）")
    ap.add_argument("--table", default="", help="跳过解析，直接读已有对局表")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    if a.table:
        rows = load(a.table)
    else:
        if not a.dir:
            ap.error("要么给 --dir，要么给 --table")
        cache = a.cache or os.path.join(
            ROOT, "data", os.path.basename(a.dir.rstrip("/")) + "_games.tsv")
        rows = load(cache) if os.path.exists(cache) else build(a.dir, cache, a.workers)
    print(f"对局表 {len(rows)} 行")
    if a.player:
        report(rows, a.player)


if __name__ == "__main__":
    main()
