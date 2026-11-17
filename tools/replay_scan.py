#!/usr/bin/env python3
"""**并行扫描回放**（把复盘从"分钟级"压到"秒级"）—— 复盘只走这一个入口。

为什么重写：2026-11-16 复盘时用串行脚本扫 243 局要几分钟、扫全语料直接超时，
原因是每个 episode JSON 有几十 MB、`json.load` + `decks_of` 全在主进程里跑。
这里做三件事：

1. **多进程**：`multiprocessing.Pool(jobs)`，默认 `jobs = min(32, cpu*2)`；
2. **缓存**：每份回放的解析结果按 `(路径, mtime, size)` 存进
   `logs/cache/replay_scan/<sha1>.json`，第二次复盘直接读缓存（秒级）；
3. **唯一口径**：卡表只从 `framework/replay.decks_of()` 取（Kaggle 的 action 记在下一步，
   自己配对会整体错位一位），先手只从 `observation.current.firstPlayer` 取
   （不要拿 agent 槽位当先手：通用语料里槽位 0 的胜率是 50.3%，而按 firstPlayer 也不是槽位）。

用法：
    ./tools/py tools/replay_scan.py --dirs epsoide_subB_f5f8G,epsoide_subB_f5f8S
    ./tools/py tools/replay_scan.py --dirs epsoide_teams2 --jobs 32 --out logs/scan_teams2.jsonl
    ./tools/py tools/replay_scan.py --dirs ... --cache off      # 强制重算

输出 JSONL，每行一局：
    {"file","seat","first","first_is_us","opp_hash","opp_ids","win","steps"}
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from framework.replay import decks_of, our_index  # noqa: E402

CACHE = os.path.join(ROOT, "logs", "cache", "replay_scan")
ME = "zhang1wen2hao3"


def _cache_path(path: str) -> str:
    st = os.stat(path)
    key = hashlib.sha1(f"{path}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:24]
    return os.path.join(CACHE, key + ".json")


def scan_one(path: str):
    """返回一条记录或 None。带缓存：命中就直接回放上次结果。"""
    try:
        cp = _cache_path(path)
        if os.path.exists(cp):
            with open(cp) as fh:
                return json.load(fh)
    except OSError:
        cp = None
    try:
        with open(path) as fh:
            e = json.load(fh)
        steps = e.get("steps") or []
        if len(steps) < 3:
            return None
        names = [(a.get("Name") or "") for a in ((e.get("info") or {}).get("Agents") or [])]
        dk = decks_of(steps)
        me = next((i for i, n in enumerate(names) if ME in n), None)
        fp = -1
        for t in range(min(6, len(steps))):
            for ent in steps[t] or []:
                cur = ((ent or {}).get("observation") or {}).get("current") or {}
                v = cur.get("firstPlayer")
                if isinstance(v, int) and v >= 0:
                    fp = v
                    break
            if fp >= 0:
                break
        rec = {"file": os.path.relpath(path, ROOT), "seat": me, "first": fp,
               "n_agents": len(names), "names": names, "steps": len(steps)}
        if me is not None and len(dk) >= 2:
            rec["our_hash"] = hashlib.sha256(
                ",".join(map(str, sorted(dk[me]))).encode()).hexdigest()[:8]
            opp = next((j for j in dk if j != me), None)
            if opp is not None:
                rec["opp_ids"] = dk[opp]
                rec["opp_hash"] = hashlib.sha256(
                    ",".join(map(str, sorted(dk[opp]))).encode()).hexdigest()[:8]
                rw = e.get("rewards") or []
                rec["win"] = bool(len(rw) > me and rw[me] is not None and rw[me] > 0)
                rec["first_is_us"] = (fp == me) if fp >= 0 else None
                rec["selfmirror"] = len(names) == 2 and all(ME in n for n in names)
        if cp:
            os.makedirs(CACHE, exist_ok=True)
            tmp = cp + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(rec, fh)
            os.replace(tmp, cp)
        return rec
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="逗号分隔的回放目录")
    ap.add_argument("--jobs", type=int, default=0, help="0 = min(32, cpu*2)")
    ap.add_argument("--out", default="logs/replay_scan.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="每个目录最多扫多少局（0=全部）")
    ap.add_argument("--cache", default="on", choices=["on", "off"])
    a = ap.parse_args()
    if a.cache == "off":
        import shutil
        shutil.rmtree(CACHE, ignore_errors=True)
        # 清完缓存后 _cache_path 仍会写新缓存；这里只是强制重算

    files = []
    for d in a.dirs.split(","):
        d = d.strip()
        if not d:
            continue
        fs = sorted(glob.glob(os.path.join(ROOT, d, "*.json")))
        files += fs[: a.limit] if a.limit else fs
    if not files:
        print("没有回放文件"); return 2
    jobs = a.jobs or min(32, (os.cpu_count() or 8) * 2)
    t0 = time.time()
    print(f"扫 {len(files)} 份回放，jobs={jobs}", flush=True)
    recs = []
    with mp.Pool(jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(scan_one, files, chunksize=4), 1):
            if r:
                recs.append(r)
            if i % 200 == 0:
                print(f"  {i}/{len(files)}  {i/(time.time()-t0):.0f} 份/s", flush=True)
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    dt = time.time() - t0
    print(f"完成：{len(recs)}/{len(files)} 条可用，用时 {dt:.1f}s（{len(files)/max(dt,1e-9):.0f} 份/s）→ {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
