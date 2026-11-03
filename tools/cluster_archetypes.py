#!/usr/bin/env python3
"""把卡组池按"原型"聚类，供两件事使用：

1. **选卡组**：每个原型挑代表去跑闸门，而不是 185 副平铺（本机一副约 13 分钟）；
2. **训练采样**：1st place 的方案明确指出——按"出镜频率"加权会**过度采样变体多的原型**
   （他们那边 Dragapult 346 副 vs Zoroark 21 副）。应当**按原型均匀**，再在原型内按权重。

方法（与他们一致的思路）：对 60 张卡做多重集相似度（Jaccard on ids），层次聚类，
相似度低于阈值就分裂。我们的池只有 185 副，用最朴素的 single-linkage 即可。

用法：
    ./run.sh tools/cluster_archetypes.py --pool decks/pool_field --thresh 0.5
"""
import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="decks/pool_field")
    ap.add_argument("--thresh", type=float, default=0.5,
                    help="similarity below this splits (their value: 0.5 cosine)")
    ap.add_argument("--json", default=None, help="把聚类结果写 JSON")
    a = ap.parse_args()

    pool = os.path.join(ROOT, a.pool)
    files = sorted(glob.glob(os.path.join(pool, "*.csv")))
    names, decks = [], []
    for f in files:
        ids = [int(x) for x in open(f).read().split()]
        if len(ids) == 60:
            names.append(os.path.basename(f)[:-4])
            decks.append(ids)
    n = len(names)
    print(f"池 {a.pool}: {n} 副，阈值 {a.thresh}")

    # 并查集 single-linkage
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        si = set(decks[i])
        for j in range(i + 1, n):
            if len(si & set(decks[j])) / len(si | set(decks[j])) >= a.thresh:
                union(i, j)

    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    # 用组内出现最多的 "原型名"（__ 前的部分）命名
    out = []
    for _, idxs in groups.items():
        bases = collections.Counter(names[i].split("__")[0] for i in idxs)
        base = bases.most_common(1)[0][0]
        out.append((base, len(idxs), [names[i] for i in idxs]))
    out.sort(key=lambda x: -x[1])

    print(f"\n聚出 {len(out)} 个原型\n")
    print(f"{'原型':<38}{'副数':>5}   代表（池中文件名）")
    for base, k, mem in out:
        print(f"{base[:36]:<38}{k:>5}   {mem[0]}")

    if a.json:
        with open(os.path.join(ROOT, a.json), "w", encoding="utf-8") as f:
            json.dump([{"archetype": b, "size": k, "members": m}
                       for b, k, m in out], f, ensure_ascii=False, indent=1)
        print(f"\n已写 {a.json}")

    print(f"\n按原型均匀采样时，每个原型权重 = 1/{len(out)}；"
          f"组内再按出镜频率分配。")


if __name__ == "__main__":
    main()
