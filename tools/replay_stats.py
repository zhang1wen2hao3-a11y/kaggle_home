#!/usr/bin/env python3
"""从回放里算出**真实对位胜率**：谁打谁、谁赢。

为什么要这个：我们要找的不是"我们哪格最差"，而是"**高分队伍**在多高的胜率，
而我们差多少"。LB 只回总分，分格数据只能自己从回放算。

数据来源：`epsoide_*/episode-<id>-replay.json`。结构（实测）：

    info.TeamNames = [队名0, 队名1]
    rewards        = [1, -1]                     ← 玩家 0 赢
    steps[t][p]    = {action, observation, reward, status, visualize, ...}
    steps[1][p].action[0] = 60 张牌的卡组（deck-request ABI 的作答）

    ./tools/py tools/replay_stats.py --jobs 8 --out data/replay_matchups.tsv
    ./tools/py tools/replay_stats.py --summary data/replay_matchups.tsv

输出 TSV：dir  episode  team0  team1  deck0  deck1  winner(0/1/-1)

卡组 → 名字：按 60 张牌的**多重集合**哈希匹配 `decks/pool_field/*.csv`（不是去重集合 ——
11 张基础能量的多重性必须算进去，否则所有卡组会撞成同一个）。
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def deck_key(ids):
    return hashlib.md5(",".join(str(x) for x in sorted(ids)).encode()).hexdigest()[:12]


def pool_index():
    idx = {}
    for f in glob.glob(os.path.join(ROOT, "decks", "pool_field", "*.csv")):
        ids = [int(x) for x in open(f) if x.strip()]
        if len(ids) == 60:
            idx[deck_key(ids)] = os.path.basename(f)[:-4]
    return idx


_IDX = None


def init_worker():
    global _IDX
    _IDX = pool_index()


def parse_one(path):
    try:
        r = json.load(open(path))
    except Exception as e:                                    # 坏文件就跳过，别中断整批
        return [os.path.basename(os.path.dirname(path)), os.path.basename(path), "PARSE_ERR", str(e)[:60], "", "", ""]
    teams = (r.get("info") or {}).get("TeamNames") or ["", ""]
    rw = r.get("rewards") or [0, 0]
    # 回放里 rewards 可能是 None（未结算/异常局）→ 当 0 处理，最后判成 -1 丢弃
    r0 = rw[0] if isinstance(rw[0], (int, float)) else 0
    r1 = rw[1] if isinstance(rw[1], (int, float)) else 0
    win = 0 if r0 > r1 else (1 if r1 > r0 else -1)
    decks = ["", ""]
    steps = r.get("steps") or []
    for p in (0, 1):
        for t in steps[:4]:                                   # deck-request 在第 1~2 步
            try:
                a = t[p].get("action")
            except Exception:
                continue
            # deck-request 的作答就是**扁平**的 60 个 card id（steps[1][p].action）
            if isinstance(a, list) and len(a) == 60 and all(isinstance(x, int) for x in a):
                decks[p] = _IDX.get(deck_key(a), "unknown:" + deck_key(a))
                break
    return [os.path.basename(os.path.dirname(path)), os.path.basename(path),
            teams[0], teams[1], decks[0], decks[1], str(win)]


def cmd_build(a):
    files = sorted(glob.glob(os.path.join(ROOT, "epsoide_*", "episode-*-replay.json")))
    if a.dirs:
        want = set(a.dirs.split(","))
        files = [f for f in files if os.path.basename(os.path.dirname(f)) in want]
    print(f"待解析 {len(files)} 个回放（{len(files) * 9.6 / 1024:.1f} GB 量级）")
    ctx = mp.get_context("fork")
    rows = []
    with ctx.Pool(a.jobs, initializer=init_worker) as pool:
        for i, r in enumerate(pool.imap_unordered(parse_one, files, chunksize=8), 1):
            rows.append(r)
            if i % 400 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)) or ".", exist_ok=True)
    with open(os.path.join(ROOT, a.out), "w") as f:
        f.write("dir\tepisode\tteam0\tteam1\tdeck0\tdeck1\twinner\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    bad = sum(1 for r in rows if r[2] == "PARSE_ERR")
    unk = sum(1 for r in rows if r[4].startswith("unknown") or r[5].startswith("unknown"))
    print(f"✅ {len(rows)} 条 → {a.out}（解析失败 {bad}，卡组未识别 {unk}）")
    return 0


def cmd_summary(a):
    path = os.path.join(ROOT, a.summary)
    per_deck = collections.defaultdict(lambda: [0, 0])        # deck -> [win, n]
    matchup = collections.defaultdict(lambda: [0, 0])         # (mine, opp) -> [win, n]
    per_dir = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    rows = 0
    for line in open(path):
        if line.startswith("dir\t"):
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) != 7 or p[6] in ("", "PARSE_ERR"):
            continue
        d, _, _, _, d0, d1, w = p
        w = int(w)
        if w < 0 or not d0 or not d1:
            continue
        rows += 1
        # ★ TSV 里 winner=0 表示 **deck0 赢**（build 时按回放 rewards 判：r0>r1 ⇒ 0）。
        #   早先这里写成 `per_deck[d0][0] += w`，等于把每一行都判给了 deck1 ——
        #   全部胜率恒为补数，docs/strength_table.md §B 因此整张表反向
        #   （把"我们赢 85%"读成"我们输 85%"，于是把 0bf5/7e39 当成最大失血格）。
        #   2026-11-08 用回放 125821646 / 125895425 两条原始记录对死。
        win0 = 1 - w
        per_deck[d0][0] += win0; per_deck[d0][1] += 1
        per_deck[d1][0] += w; per_deck[d1][1] += 1
        for mine, opp, win in ((d0, d1, win0), (d1, d0, w)):
            matchup[(mine, opp)][0] += win; matchup[(mine, opp)][1] += 1
            per_dir[d][mine][0] += win; per_dir[d][mine][1] += 1

    print(f"有效对局 {rows} 条，涉及卡组 {len(per_deck)} 副\n")
    print("== 各回放目录里各卡组的胜率（n≥30）==")
    for d in sorted(per_dir):
        items = [(k, v) for k, v in per_dir[d].items() if v[1] >= 30]
        if not items:
            continue
        items.sort(key=lambda kv: -kv[1][1])
        print(f"-- {d}（{sum(v[1] for k, v in per_dir[d].items())} 局）")
        for k, (w, n) in items[:6]:
            print(f"   {k:40s} {w/n:.3f}  n={n}")
    print("\n== 全局：出现最多的卡组（n≥50）==")
    top = sorted(per_deck.items(), key=lambda kv: -kv[1][1])[:15]
    for k, (w, n) in top:
        if n >= 50:
            print(f"   {k:40s} {w/n:.3f}  n={n}")
    if a.deck:
        print(f"\n== 「{a.deck}」对位明细（n≥10）==")
        items = [(k2, v) for (k1, k2), v in matchup.items() if k1 == a.deck and v[1] >= 10]
        items.sort(key=lambda kv: kv[1][0] / kv[1][1])
        for opp, (w, n) in items:
            print(f"   vs {opp:38s} {w/n:.3f}  n={n}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="解析回放 → TSV")
    b.add_argument("--jobs", type=int, default=8)
    b.add_argument("--dirs", default="", help="只解析这些目录（逗号分隔），默认全部")
    b.add_argument("--out", default="data/replay_matchups.tsv")
    b.set_defaults(fn=cmd_build)
    s = sub.add_parser("summary", help="从 TSV 出对位统计")
    s.add_argument("summary", nargs="?", default="data/replay_matchups.tsv")
    s.add_argument("--deck", default="", help="看这副牌的对位明细")
    s.set_defaults(fn=cmd_summary)
    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
