#!/usr/bin/env python3
"""**卡组标准标签**：以后报卡组一律用这个格式，不要只给 4 位 id。

输出三段：`<卡组全名>`（4 位 id）+`最高分玩家 分数`+`该牌天梯胜率(局数)`+`真实场份额`，
并标注覆盖情况（17 格测试集 / 母表 / 池）。

    ./tools/py tools/deck_label.py 3eaa e1d7 0bf5        # 按 4 位 id
    ./tools/py tools/deck_label.py --all                 # 全部（按分数降序）
"""
from __future__ import annotations
import argparse, collections, csv, glob, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LB_PATH = os.path.join(ROOT, "data", "lb_scores.json")


def pool():
    m = {}
    for p in glob.glob(os.path.join(ROOT, "decks", "pool_field", "*.csv")):
        ids = sorted(int(x) for x in open(p) if x.strip())
        h = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:8]
        m[h] = os.path.basename(p)[:-4]
        m.setdefault(h[:4], os.path.basename(p)[:-4])
    return m


def stats():
    """(h4, team) -> 局数/胜场；h4 -> 局数/胜场"""
    g = collections.Counter(); w = collections.Counter()
    gt = collections.Counter(); wt = collections.Counter()   # (h4, team)
    for r in csv.DictReader(open(os.path.join(ROOT, "data", "ladder_games.tsv")), delimiter="\t"):
        try:
            win = int(r["winner"])
        except (TypeError, ValueError):
            continue
        for i in (0, 1):
            h = (r.get(f"seat{i}_deck") or "")[:4]
            t = (r.get(f"seat{i}_agent") or "").strip()
            if not h:
                continue
            g[h] += 1
            if win == i:
                w[h] += 1
            if t:
                gt[(h, t)] += 1
                if win == i:
                    wt[(h, t)] += 1
    return g, w, gt, wt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min-games", type=int, default=10)
    a = ap.parse_args()
    P = pool(); G, W, GT, WT = stats()
    LB = json.load(open(LB_PATH))
    # 份额（真实场）
    share = {}
    for ln in open(os.path.join(ROOT, "decks", "testset_meta_real.tsv")):
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.rstrip("\n").split("\t")
        if f[0] == "tail_mix":
            continue
        share[f[0].split("_")[-1]] = float(f[4])
    # 覆盖
    ts = set(share); master = set()
    txt = open(os.path.join(ROOT, "docs", "the-pockmon-battle-choice-in-table.csv"), "rb").read().decode("gb18030", "replace")
    for r in csv.DictReader(txt.splitlines()):
        master.add(r["被挑战者"])

    ids = list(a.ids)
    if a.all or not ids:
        ids = sorted(G, key=lambda h: -max((LB.get(t, 0) for (hh, t) in GT if hh == h), default=0))
    print(f"{'卡组（4位）':44s}{'最高分玩家':26s}{'分':>7s}{'该牌胜率':>9s}{'局数':>5s}{'份额':>7s}  覆盖")
    for h4 in ids:
        h4 = h4.lower()
        name = P.get(h4, f"未入库:{h4}")
        # 该牌上局数≥MIN 且胜率≥50% 的最高分玩家
        cands = []
        for (hh, t), n in GT.items():
            if hh != h4 or n < a.min_games:
                continue
            wr = WT[(hh, t)] / n
            cands.append((LB.get(t, 0), t, wr, n))
        cands.sort(reverse=True)
        good = [c for c in cands if c[2] >= 0.5]
        top = (good or cands or [(0, "—", 0, 0)])[0]
        dn, dg = G.get(h4, 0), W.get(h4, 0)
        wr_all = f"{dg/dn*100:.1f}%" if dn else "NA"
        cov = []
        cov.append("17格" if h4 in ts else "—")
        cov.append("母表" if h4 in master else "—")
        label = f"{name}"
        print(f"{label:44s}{top[1][:24]:26s}{top[0]:7.1f}{wr_all:>9s}{dn:5d}"
              f"{(f'{share[h4]:.1f}%' if h4 in share else '—'):>7s}  {'/'.join(cov)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
