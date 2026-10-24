#!/usr/bin/env python3
"""把若干卡组 hash 落成"端口 + 指定 deck.csv"的目录，供 tools/bo3_eval.py 使用。

卡表来源优先级：① 指定回放目录里出现过的（精确）② data/deck_corpus.csv。
为什么需要它：`/tmp` 会被清理，卡组目录经常丢；重建成本很低但容易忘。
用法：./run.sh tools/stage_decks.py 7e398437 9d6ea740 26c6e843 --replays epsoide_dipam,epsoide_port
"""
import argparse, csv, glob, hashlib, json, os, shutil, sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from framework.replay import decks_of

def h(ids): return hashlib.sha256(",".join(str(x) for x in sorted(ids)).encode()).hexdigest()[:8]

ap=argparse.ArgumentParser()
ap.add_argument("hashes", nargs="+")
ap.add_argument("--bundle", default="build/port_plain")
ap.add_argument("--out", default="/tmp/variants")
ap.add_argument("--replays", default="epsoide_dipam,epsoide_port")
a=ap.parse_args()
want=set(a.hashes); found={}
for d in a.replays.split(","):
    for f in sorted(glob.glob(os.path.join(ROOT, d.strip(), "*.json"))):
        try: ds=decks_of(json.load(open(f)).get("steps") or [])
        except Exception: continue
        for ids in ds.values():
            hh=h(ids)
            if hh in want and hh not in found: found[hh]=ids
corpus={r['deck_hash']: [int(x) for x in r['deck'].split()] for r in csv.DictReader(open(os.path.join(ROOT,'data','deck_corpus.csv')))}
bundle=os.path.join(ROOT, a.bundle)
for hh in a.hashes:
    ids=found.get(hh) or corpus.get(hh)
    if not ids or len(ids)!=60:
        print(f"⚠️ 跳过 {hh}（未找到或张数 {len(ids) if ids else 0}）"); continue
    d=os.path.join(a.out, hh); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for e in os.listdir(bundle):
        if e=='deck.csv': continue
        os.symlink(os.path.join(bundle,e), os.path.join(d,e))
    open(os.path.join(d,'deck.csv'),'w').write("\n".join(map(str,ids))+"\n")
    print(f"✅ {hh}（来源：{'回放' if hh in found else '语料'}）")
