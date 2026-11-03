#!/usr/bin/env python3
"""Route C · 真实天梯档案：把回放变成**对局级表格**，用于回答"Agent 到底输给谁"。

设计口径（2026-11-01 与用户对齐）：
  * 一行 = 一场对局（不是一次决策）
  * 卡组身份用索引里的 8 位 hex（= pool 名的 4 位后缀 + 4 位区分码）⇒ **精确卡组**，
    绝不把 db04 和 9d6e 混成一个"原型"
  * 座位、对手身份、步数、胜负全部落表；胜负需要读回放的 `rewards`
  * 镜像胜率只当**外部观测指标**，不当"策略能力分数"；必须给 Wilson CI 与样本量

两个数据源：
  1. `--archive`：`data/distill_index.jsonl` 索引的 1756 个回放（17 GB，别人的对局）
     —— 用来建"真实天梯对手档案"与"镜像胜率参考分布"
  2. `--ours`：`data/episodes*.csv`（我们自己的提交扫出来的对局，带 submission 版本）
     —— 用来算"我们 per-deck 的镜像胜率"

用法：
    ./tools/py tools/ladder_archive.py --stage index        # 只从索引建表（秒级，无 rewards）
    ./tools/py tools/ladder_archive.py --stage results      # 读回放补 rewards（17 GB，较慢）
    ./tools/py tools/ladder_archive.py --stage ours         # 我们自己的对局
    ./tools/py tools/ladder_archive.py --report             # 汇总 + 镜像基准
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "data", "distill_index.jsonl")
OUT_ARCHIVE = os.path.join(ROOT, "data", "ladder_games.tsv")
OUT_OURS = os.path.join(ROOT, "data", "our_games.tsv")

ARCHIVE_FIELDS = ["replay_id", "file", "seat0_deck", "seat0_agent",
                  "seat1_deck", "seat1_agent", "mirror", "n_steps",
                  "winner", "result_known"]
OURS_FIELDS = ["replay_id", "submission", "our_deck", "opp_deck", "opponent",
               "our_seat", "result", "n_steps", "seed", "mirror"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, (c - h) / d), min(1.0, (c + h) / d))


# ── stage 1：只读索引 ──────────────────────────────────────────────────
def stage_index(limit=0):
    per_game = {}
    maxstep = collections.Counter()
    t0 = time.time()
    n = 0
    for line in open(INDEX, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        n += 1
        f = r.get("file")
        if not f:
            continue
        g = per_game.setdefault(f, {})
        seat = r.get("seat")
        if seat in (0, 1):
            g[seat] = (r.get("deck") or "?", r.get("agent") or "?")
        st = r.get("step")
        if isinstance(st, int):
            maxstep[f] = max(maxstep[f], st)
        if limit and len(per_game) >= limit:
            break
    rows = []
    for f, g in per_game.items():
        if 0 not in g or 1 not in g:
            continue
        rows.append(dict(replay_id=os.path.basename(f).replace("episode-", "").replace("-replay.json", ""),
                         file=f, seat0_deck=g[0][0], seat0_agent=g[0][1],
                         seat1_deck=g[1][0], seat1_agent=g[1][1],
                         mirror=int(g[0][0] == g[1][0]), n_steps=maxstep.get(f, 0),
                         winner="", result_known=0))
    with open(OUT_ARCHIVE, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ARCHIVE_FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"索引 {n} 条 → {len(rows)} 局（镜像 {sum(r['mirror'] for r in rows)} 局），"
          f"写入 {OUT_ARCHIVE}（{time.time()-t0:.1f}s）")


# ── stage 2：读回放补 rewards ─────────────────────────────────────────
def stage_results(limit=0):
    if not os.path.exists(OUT_ARCHIVE):
        raise SystemExit("先跑 --stage index")
    with open(OUT_ARCHIVE, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if limit:
        rows = rows[:limit]
    t0 = time.time()
    done = miss = 0
    for i, r in enumerate(rows):
        p = os.path.join(ROOT, r["file"])
        try:
            d = json.load(open(p))
            rw = d.get("rewards") or []
            if len(rw) == 2:
                # rewards[i] > 0 ⇒ seat i 胜；== 0 可能是平局/未决
                if rw[0] > rw[1]:
                    r["winner"] = 0
                elif rw[1] > rw[0]:
                    r["winner"] = 1
                else:
                    r["winner"] = -1
                r["result_known"] = 1
                done += 1
        except Exception:
            miss += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    with open(OUT_ARCHIVE, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ARCHIVE_FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"补到胜负 {done} 局，失败 {miss}，用时 {time.time()-t0:.0f}s → {OUT_ARCHIVE}")


# ── our own games（带 submission 版本）────────────────────────────────
def stage_ours():
    rows, seen = [], set()
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "episodes*.csv"))):
        tag = os.path.basename(f)[9:-4]          # episodes_<tag>.csv
        for r in csv.DictReader(open(f, encoding="utf-8")):
            ep = r.get("episode")
            if not ep or ep in seen:
                continue
            seen.add(ep)
            our = (r.get("my_deck") or "?").strip()
            opp = (r.get("op_deck") or "?").strip()
            rows.append(dict(replay_id=ep, submission=tag, our_deck=our, opp_deck=opp,
                             opponent=r.get("opponent") or "?", our_seat="", result=int(r.get("won") or 0),
                             n_steps=r.get("steps") or "", seed=r.get("seed") or "",
                             mirror=int(our == opp and our != "?")))
    with open(OUT_OURS, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OURS_FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"我们自己的对局 {len(rows)} 场（镜像 {sum(r['mirror'] for r in rows)}），写入 {OUT_OURS}")
    print("⚠️ our_seat 目前取不到（CSV 没存座位）→ 需要从回放补，见 report 的口径说明")


# ── report ───────────────────────────────────────────────────────────
def report():
    print("===== A. 真实天梯对手档案（1756 局，别人的对局）=====")
    if not os.path.exists(OUT_ARCHIVE):
        print("（还没有 ladder_games.tsv，先跑 --stage index）")
    else:
        rows = list(csv.DictReader(open(OUT_ARCHIVE, encoding="utf-8"), delimiter="\t"))
        known = [r for r in rows if r["result_known"] == "1"]
        print(f"总局数 {len(rows)}，有胜负 {len(known)}，镜像 {sum(int(r['mirror']) for r in rows)}")
        deck = collections.Counter()
        for r in rows:
            deck[r["seat0_deck"]] += 1
            deck[r["seat1_deck"]] += 1
        print(f"覆盖 {len(deck)} 副精确卡组；出现最多的 8 副：")
        for d, c in deck.most_common(8):
            print(f"   {d}  {c:4d} 次")
        if known:
            mir = [r for r in known if r["mirror"] == "1"]
            print(f"\n镜像局 {len(mir)} 场 —— 这是**真人与真人同牌的参考分布**：")
            print("   （镜像里 seat0 与 seat1 是两个不同的人；胜率应接近 0.5，偏离说明水平差异）")
            w0 = sum(1 for r in mir if r["winner"] == "0")
            p, lo, hi = wilson(w0, len(mir))
            print(f"   seat0 胜 {w0}/{len(mir)} = {p:.3f}  Wilson 95% [{lo:.3f},{hi:.3f}]")

    print("\n===== B. 我们自己的对局（按精确卡组看镜像）=====")
    if not os.path.exists(OUT_OURS):
        print("（还没有 our_games.tsv，先跑 --stage ours）")
    else:
        rows = list(csv.DictReader(open(OUT_OURS, encoding="utf-8"), delimiter="\t"))
        print(f"总 {len(rows)} 场（来源 {len(set(r['submission'] for r in rows))} 个提交）")
        print(f"\n{'我方卡组':34s}{'对手卡组':34s}{'场次':>5}{'胜':>4}{'胜率':>7}{'Wilson 95%':>18}{' 镜像'}")
        by = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            k = (r["our_deck"], r["opp_deck"])
            by[k][0] += 1
            by[k][1] += int(r["result"])
        for (a, b), (n, w) in sorted(by.items(), key=lambda kv: -kv[1][0])[:20]:
            p, lo, hi = wilson(w, n)
            mi = "镜像" if a == b else ""
            print(f"{a[:33]:34s}{b[:33]:34s}{n:>5}{w:>4}{p:>7.3f}{f'[{lo:.3f},{hi:.3f}]':>18}  {mi}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["index", "results", "ours"], default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.stage == "index":
        stage_index(a.limit)
    elif a.stage == "results":
        stage_results(a.limit)
    elif a.stage == "ours":
        stage_ours()
    if a.report or not a.stage:
        report()


if __name__ == "__main__":
    main()
