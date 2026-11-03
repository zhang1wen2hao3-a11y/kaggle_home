#!/usr/bin/env python3
"""阶段 A：头部方案的**卡组盘点**（不训练、不做策略假设，只读回放）。

回答三个问题：
  1. 头部玩家（1400+）实际用了哪些卡组？各打了几局、胜率多少？
  2. 这些卡组在我们现有的 185（pool_field）/ 616（port_repo/decks/pool）池里吗？
  3. 不在池里的，把 60 张卡表导出到 `decks/top/`（可直接当 anchor 用）。

盘点口径
* 卡组 tag = `sha256(",".join(sorted(ids)))[:8]`，与 `decks/pool_field` 文件名后缀同源
  （`dreepy_drakloak__3eaa.csv` ↔ `3eaaf117`）
* 一场对局两个座位各算一次出场；胜负来自 `rewards`
* **不猜原型、不合并变体**：8 位 hash 就是身份，4 位相同的不同 hash 也分开列

用法
    ./tools/py tools/top_deck_inventory.py                      # 默认扫 epsoide_petit epsoide_miwa epsoide_top epsoide_dipam epsoide_teams2
    ./tools/py tools/top_deck_inventory.py --dirs epsoide_miwa --out data/miwa_decks.tsv
    ./tools/py tools/top_deck_inventory.py --players "Petit Canard,やる気元気ミワハルキ 2nd ver."
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from framework.replay import decks_of  # noqa: E402

DEFAULT_DIRS = ["epsoide_petit", "epsoide_miwa", "epsoide_top", "epsoide_dipam",
                "epsoide_teams2"]


def tag(ids):
    return hashlib.sha256(",".join(str(x) for x in sorted(ids)).encode()).hexdigest()[:8]


def pool_index():
    """→ {4 位后缀: 路径}；两个池都收，标注来源。"""
    idx = {}
    for d, src in ((os.path.join(ROOT, "decks", "pool_field"), "field185"),
                   (os.path.join(ROOT, "port_repo", "decks", "pool"), "pool616")):
        for p in glob.glob(os.path.join(d, "*.csv")):
            s = os.path.basename(p)[:-4]
            if "__" in s:
                idx.setdefault(s.rsplit("__", 1)[1], (p, src))
    return idx


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
        return {"file": os.path.basename(os.path.dirname(path)) + "/" +
                        os.path.basename(path),
                "tags": (tag(dk[0]), tag(dk[1])),
                "names": names,
                "ids": (tuple(dk[0]), tuple(dk[1])),
                "win": win}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=DEFAULT_DIRS)
    ap.add_argument("--players", default="", help="只看这些玩家的对局（逗号分隔，子串匹配）")
    ap.add_argument("--out", default="data/top_decks.tsv")
    ap.add_argument("--deck-dir", default="decks/top")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    dirs = [d if os.path.isabs(d) else os.path.join(ROOT, d) for d in a.dirs]
    files = []
    for d in dirs:
        files += sorted(glob.glob(os.path.join(d, "*.json")))
    print(f"扫描 {len(files)} 个回放（{len(dirs)} 个目录）", file=sys.stderr)

    rows = []
    if a.workers <= 1:
        for i, f in enumerate(files, 1):
            r = one(f)
            if r:
                rows.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(files)}", file=sys.stderr)
    else:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(one, files, chunksize=2), 1):
                if r:
                    rows.append(r)
                if i % 100 == 0:
                    print(f"  {i}/{len(files)}", file=sys.stderr)
    print(f"有效对局 {len(rows)} 场", file=sys.stderr)

    want = [p.strip() for p in a.players.split(",") if p.strip()]

    # ---- 按 tag 汇总 ----
    stat = collections.defaultdict(lambda: [0, 0])          # tag → [w, n]
    owners = collections.defaultdict(collections.Counter)
    reps = {}                                               # tag → 第一次见到的卡表
    for r in rows:
        if want and not any(w in n for n in r["names"] for w in want):
            continue
        for i in (0, 1):
            t = r["tags"][i]
            stat[t][1] += 1
            if r["win"] == i:
                stat[t][0] += 1
            owners[t][r["names"][i]] += 1
            reps.setdefault(t, r["ids"][i])

    idx = pool_index()
    os.makedirs(os.path.join(ROOT, a.deck_dir), exist_ok=True)
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    lines = ["deck_hash\tgames\twins\twinrate\tin_pool\tpool_src\tmain_owners\tdeck_file"]
    new_decks = []
    for t, (w, n) in sorted(stat.items(), key=lambda kv: -kv[1][1]):
        hit = idx.get(t[:4])
        src = hit[1] if hit else "-"
        own = ";".join(f"{k}:{v}" for k, v in owners[t].most_common(3))
        deck_file = ""
        if hit:
            deck_file = os.path.relpath(hit[0], ROOT)
        else:
            fn = os.path.join(ROOT, a.deck_dir, f"top__{t}.csv")
            with open(fn, "w") as fh:
                fh.write("\n".join(str(x) for x in sorted(reps[t])) + "\n")
            deck_file = os.path.relpath(fn, ROOT)
            new_decks.append((t, n, w / n, deck_file))
        lines.append(f"{t}\t{n}\t{w}\t{w / n:.4f}\t{'Y' if hit else 'N'}\t{src}\t{own}\t{deck_file}")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\n{'deck':10s}{'局数':>6}{'胜率':>8}  池     主要持有者")
    for t, (w, n) in sorted(stat.items(), key=lambda kv: -kv[1][1])[:25]:
        hit = idx.get(t[:4])
        own = ",".join(k for k, _ in owners[t].most_common(2))
        print(f"{t:10s}{n:>6}{w / max(1, n):>8.3f}  {'池内' if hit else '**新**':6s} {own[:46]}")
    if new_decks:
        print(f"\n不在池里的卡组 {len(new_decks)} 副，已导出到 {a.deck_dir}/：")
        for t, n, wr, f in new_decks[:20]:
            print(f"  {t}  {n} 局  胜率 {wr:.3f}  → {f}")
    print(f"\n表 → {a.out}")


if __name__ == "__main__":
    main()
