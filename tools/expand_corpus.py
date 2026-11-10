#!/usr/bin/env python3
"""为某副目标牌，找出「哪个队伍的哪个提交在用这副牌」，再批量拉回放。

背景：Kaggle 每个 submission 只对应一副牌，所以不能按队伍抓，必须按**提交**抓。
流程：
  队伍 → 列 submissions → 每个 submission 下 3 局做探针 → 识别卡组 → 命中的再批量抓

用法：
  python3 tools/expand_corpus.py probe --deck 7e39 --teams 17176952,17079521
  python3 tools/expand_corpus.py fetch --submission 57666077 --deck 7e39 --limit 150
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
KAGGLE = os.environ.get("KAGGLE_BIN", "/home/zhang/miniforge3/bin/kaggle")
POOL = ROOT / "decks" / "pool_field"
TARGETS = {"7e39": "7e398437", "0bf5": "0bf5955c", "2617": "2617a1c6", "9d6e": "9d6ea740",
           "0904": "0904e46f", "26c6": "26c6e843", "e1d7": "e1d754a7", "0d86": "0d86f3a5", "a47a": "a47a7971"}


def run(cmd, timeout=300):
    env = dict(os.environ, KAGGLE_CONFIG_DIR=os.path.expanduser("~/.kaggle"))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} 失败: {p.stderr.strip()[:200]}")
    return p.stdout


def _walk(obj, out):
    """递归找出所有「长度 60 的整数列表」——deck-request 的作答。"""
    if isinstance(obj, list):
        if len(obj) == 60 and all(isinstance(x, int) for x in obj):
            out.append(obj); return
        for x in obj: _walk(x, out)
    elif isinstance(obj, dict):
        for v in obj.values(): _walk(v, out)


def owner_decks(files):
    """在同一个 submission 的多局里，找出「每局都出现」的队伍（= 提交者），
    再取它那一侧的卡组。**不能用『第一个 60 张列表』**：那可能是对手的牌
    （实测 Zedan 的提交被误判成 0bf5，实际是 b381）。"""
    per = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        teams = (d.get("info") or {}).get("TeamNames") or []
        out = []; _walk(d, out)
        h = {i: hashlib.sha256(",".join(map(str, sorted(ids))).encode()).hexdigest()[:8]
             for i, ids in enumerate(out[:2])}
        if len(teams) == 2 and h: per.append((teams, h))
    if not per: return None, {}
    cnt = collections.Counter()
    for t, _ in per: cnt.update(t)
    owner = cnt.most_common(1)[0][0]
    deck = collections.Counter()
    for t, h in per:
        if owner in t and t.index(owner) in h: deck[h[t.index(owner)]] += 1
    return owner, dict(deck)

def probe(deck, team_ids, per_sub=3):
    want = TARGETS[deck]
    hits = []
    for tid in team_ids:
        try:
            out = run([KAGGLE, "competitions", "team-submissions", tid], timeout=120)
        except Exception as e:
            print(f"  队伍 {tid} 列提交失败: {e}"); continue
        subs = [l.split()[0] for l in out.splitlines()[1:] if l.strip() and l.split()[0].isdigit()]
        print(f"队伍 {tid}: {len(subs)} 个提交")
        for sid in subs[:4]:
            tmp = ROOT / "_probe_tmp"
            try:
                shutil.rmtree(tmp, ignore_errors=True)
                subprocess.run([sys.executable, str(ROOT / "tools" / "fetch_episodes.py"),
                                "--submission", sid, "--out", "_probe_tmp",
                                "--limit", str(per_sub)],
                               capture_output=True, text=True, timeout=600)
                owner, cnt = owner_decks(sorted((ROOT / "_probe_tmp").glob("*.json")))
                top = sorted(cnt.items(), key=lambda kv: -kv[1])
                ok = bool(top) and top[0][0] == want
                print(f"   {sid}: {owner or '?'} → {cnt}  {'✅ 命中 ' + deck if ok else ''}")
                if ok:
                    hits.append(sid)
            finally:
                shutil.rmtree(ROOT / "_probe_tmp", ignore_errors=True)
    print(f"\n命中 {deck} 的提交：{hits}")
    return hits


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("probe"); p1.add_argument("--deck", required=True); p1.add_argument("--teams", required=True)
    p1.add_argument("--per-sub", type=int, default=3)
    p2 = sub.add_parser("fetch"); p2.add_argument("--submission", required=True)
    p2.add_argument("--deck", required=True); p2.add_argument("--limit", type=int, default=150)
    a = ap.parse_args()
    if a.cmd == "probe":
        probe(a.deck, [t.strip() for t in a.teams.split(",") if t.strip()], a.per_sub)
    else:
        out = ROOT / f"epsoide_{a.deck}_s{a.submission}"
        print(f"抓取 submission {a.submission} → {out.name}")
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "fetch_episodes.py"),
                            "--submission", a.submission, "--out", out.name,
                            "--limit", str(a.limit)], capture_output=True, text=True, timeout=7200)
        print(r.stdout[-500:] or r.stderr[-500:])


if __name__ == "__main__":
    raise SystemExit(main())
