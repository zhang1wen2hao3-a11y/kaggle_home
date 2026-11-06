#!/usr/bin/env python3
"""回放去重：同一条 episode 常常被不同批次下载进多个 `epsoide_*` 目录。

实测（2026-11-06）：4277 条回放里 **470 条是重复**，白占 5.9 GB —— 而且下载脚本
每跑一次都可能再产生一批。所以做成工具，定期跑。

    ./tools/py tools/dedupe_replays.py --dry     # 只看会删什么
    ./tools/py tools/dedupe_replays.py           # 真删（保留最先出现的那一份）

判定规则：
  * 键 = 文件名（`episode-<id>-replay.json`），同 id 才可能是同一局
  * 大小必须相同才删；同名不同大小会单独列出来让人看（不自动删）
  * 抽样做 md5 校验，确认真的逐字节相同才动手 —— 只删"确定是副本"的
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只报告，不删")
    ap.add_argument("--samples", type=int, default=8, help="抽样 md5 校验的条数")
    a = ap.parse_args()

    os.chdir(ROOT)
    dirs = sorted(d for d in glob.glob("epsoide_*") if os.path.isdir(d))
    seen, dup, mismatch = {}, [], []
    total = 0
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "episode-*-replay.json")))
        for f in files:
            n = os.path.basename(f)
            sz = os.path.getsize(f)
            total += sz
            if n in seen:
                if seen[n][1] == sz:
                    dup.append((f, seen[n][0], sz))
                else:
                    mismatch.append((f, seen[n][0], sz, seen[n][1]))
            else:
                seen[n] = (f, sz)

    dupsz = sum(x[2] for x in dup)
    print(f"回放目录 {len(dirs)} 个，共 {len(seen) + len(dup)} 条 / {total / 2**30:.1f} GB")
    print(f"重复（同 id 同大小）{len(dup)} 条 → 可省 {dupsz / 2**30:.1f} GB"
          f" ⇒ 去重后 {(total - dupsz) / 2**30:.1f} GB")
    if mismatch:
        print(f"⚠️ 同名但大小不同 {len(mismatch)} 条（**不删**，需人工确认）：")
        for f, keep, s1, s2 in mismatch[:10]:
            print(f"   {f} ({s1}) vs {keep} ({s2})")

    if a.dry or not dup:
        return 0

    ok = True
    for f, keep, _ in random.sample(dup, min(a.samples, len(dup))):
        if md5(f) != md5(keep):
            print(f"✗ {f} 与 {keep} 内容不同，中止（不删任何东西）")
            ok = False
            break
    if not ok:
        return 1

    for f, _, _ in dup:
        os.remove(f)
    print(f"✅ 已删除 {len(dup)} 条重复回放，释放 {dupsz / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
