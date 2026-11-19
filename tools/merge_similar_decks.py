#!/usr/bin/env python3
"""**近似同构卡组合并**：维护表里"只差 ≤4 张"的两副牌其实是同一副牌，不该各占一行/一列，
也不该各养一个驾驶员。

用户 2026-11-19：「9d6e / db04 是同一副牌的两个 2 张差异变体 …… 这两卡组为什么不聚合，
用最强驾驶员 + 卡组？」—— 这条规则对新接的牌已经生效（`add_highscore_decks.py`），
但对**老表**从没回头跑过，于是表里同时躺着 9d6e、db04、0bf5（两两差 2/4/6 张）。

做的事：
  1. 在 `decks/deck_order.txt` 范围内两两算卡表差（60 张多重集差 /2）；
  2. 差 ≤ MAX_DIFF(4) 的聚成一簇；
  3. 每簇选**最强的当代表**（优先 init-vs-init 行均值，其次 17 格面板分，再次投资分）；
  4. 写 `decks/deck_merge.txt`（变体 → 代表），并把变体从 `deck_order.txt` 删掉；
  5. `export_master.py` 读出这张映射后，会把变体的对局数据**并入代表**（n 相加、CI 收窄）。

    ./tools/py tools/merge_similar_decks.py --dry     # 只看会怎么合并
    ./tools/py tools/merge_similar_decks.py
"""
from __future__ import annotations
import argparse, collections, csv, glob, os, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = os.path.join(ROOT, "decks", "pool_field")
ORDER = os.path.join(ROOT, "decks", "deck_order.txt")
MERGE = os.path.join(ROOT, "decks", "deck_merge.txt")
INTABLE = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-in-table.csv")
INV = os.path.join(ROOT, "docs", "卡组天花板与投资分.csv")
MAX_DIFF = 4


def pool_decks():
    out = {}
    for p in glob.glob(os.path.join(POOL, "*.csv")):
        out[os.path.basename(p)[:-4]] = collections.Counter(int(x) for x in open(p) if x.strip())
    return out


def diff(a, b):
    return sum(abs(a[c] - b.get(c, 0)) for c in set(a) | set(b)) // 2


def load_order():
    return [ln.strip() for ln in open(ORDER) if ln.strip() and not ln.startswith("#")]


def strength():
    """代表的选取依据：init-vs-init 行均值（只比牌不比驾驶员），缺则用投资分。"""
    s = {}
    if os.path.exists(INTABLE):
        txt = open(INTABLE, "rb").read().decode("gb18030", errors="replace")
        agg = collections.defaultdict(list)
        for r in csv.DictReader(txt.splitlines()):
            if r.get("驾驶员") != "init" or r.get("对手驾驶员") != "init":
                continue
            try:
                agg[r["挑战者"]].append(float(r["胜率"]))
            except (TypeError, ValueError):
                pass
        s = {d: statistics.mean(v) for d, v in agg.items()}
    inv, lvl = {}, {}
    if os.path.exists(INV):
        for r in csv.DictReader(open(INV, encoding="utf-8-sig")):
            d = (r.get("卡组") or "").strip()
            try:
                inv[d] = float(r.get("投资分") or 0)
            except ValueError:
                pass
            v = (r.get("我方水平%") or "").rstrip("%")
            if v:
                try:
                    lvl[d] = float(v)
                except ValueError:
                    pass
    return s, inv, lvl


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)

    pool = pool_decks()
    order = load_order()
    S, INV, LVL = strength()

    def pname(h):
        for n in pool:
            if n.split("__")[-1].lower() == h.lower():
                return n
        return None

    # 并查集
    parent = {h: h for h in order}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pairs = []
    for i, x in enumerate(order):
        for y in order[i + 1:]:
            px, py = pname(x), pname(y)
            if not px or not py:
                continue
            d = diff(pool[px], pool[py])
            if d <= MAX_DIFF:
                pairs.append((d, x, y))
                union(x, y)
    clusters = collections.defaultdict(list)
    for h in order:
        clusters[find(h)].append(h)

    # 代表怎么选：① 实测面板水平（我方水平%，17 格真实口径）最高者；
    #               ② 没有面板就用 init-vs-init 强度；③ 再平手才看投资分
    def rank(h):
        return (LVL.get(h, 0.0), S.get(h, 0.0), INV.get(h, 0.0))

    merge = {}
    print(f"维护表 {len(order)} 副，检出近似同构对 {len(pairs)} 个：")
    for d, x, y in pairs:
        print(f"  {x} ↔ {y}: 差 {d} 张")
    for root, members in clusters.items():
        if len(members) < 2:
            continue
        rep = max(members, key=rank)
        for m in members:
            if m != rep:
                merge[m] = rep
        detail = ", ".join(f"{m}(面板 {LVL.get(m, 0):.1f}%/强度 {S.get(m, 0):.1%}/投资 {INV.get(m, 0):.1f})" for m in members)
        print(f"  → 合并为 [{rep}]：{detail}")

    if not merge:
        print("· 没有需要合并的卡组")
        return 0
    new_order = [h for h in order if h not in merge]
    print(f"牌序 {len(order)} → {len(new_order)}：{' '.join(new_order)}")
    if a.dry:
        return 0
    with open(MERGE, "w", encoding="utf-8") as fh:
        fh.write("# 近似同构卡组合并表（变体 -> 代表）；export_master.py 会把变体的对局数据并入代表\n")
        for k, v in sorted(merge.items()):
            fh.write(f"{k} {v}\n")
    with open(ORDER, "w", encoding="utf-8") as fh:
        fh.write("# 维护表牌序（build_from_master.py 读这个文件；一行一副，顺序即矩阵列顺序）\n")
        for h in new_order:
            fh.write(h + "\n")
    print(f"✓ 写入 {os.path.relpath(MERGE, ROOT)} 与 {os.path.relpath(ORDER, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
