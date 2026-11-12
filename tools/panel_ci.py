#!/usr/bin/env python3
"""**面板加权差的置信区间**：用每格的胜场数做蒙特卡洛，判断 +X pp 到底是真提升还是噪声。

    ./tools/py tools/panel_ci.py --base meta_port_9d6e --tag r1A_9d6e --tag r2A_9d6e

为什么需要：`tools/panel_compare.py` 只给点估计。面板每格 n=200（CI ±6.5pp），
加权后到底还有多少把握，得把每格的不确定性按权重传下去才看得见。

模型：每格胜场 ~ Binomial(n, p)，p 的先验取 Beta(1,1)（均匀），后验 Beta(w+1, n-w+1)；
从后验抽 p，按场地权重加权求均值，重复 M 次得到「加权胜率」的分布，
再对「候选 − 基线」的分布取分位数 ⇒ 差值的 CI 与 P(候选更好)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(tag):
    d = os.path.join(ROOT, "logs", "meta", tag)
    s = json.load(open(os.path.join(d, "summary.json")))
    cells = {}
    for f in os.listdir(d):
        if not (f.startswith("vs_") and f.endswith(".json")):
            continue
        o = json.load(open(os.path.join(d, f)))["overall"]
        cells[f[3:-5]] = (float(o["score"]), int(o["n"]))
    return s, cells


def weights():
    w = {}
    for line in open(os.path.join(ROOT, "decks", "testset_meta.tsv")):
        if line.startswith("#") or not line.strip():
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 5:
            w[p[0]] = float(p[4])
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="append", required=True,
                    help="基线 tag，可重复（多个种子会被合并：胜场与场次相加）")
    ap.add_argument("--tag", action="append", default=[],
                    help="候选 tag，可重复；同一个 --tag 只算一个候选")
    ap.add_argument("--pool", action="append", default=[],
                    help="合并多个种子成一个候选，格式 name=tag1,tag2,...（可重复）")
    ap.add_argument("--draws", type=int, default=20000)
    a = ap.parse_args()
    rng = np.random.default_rng(12345)
    w = weights()

    def pool(tags):
        """把多个 tag 的每格胜场/场次相加（等价于同协议多跑几次种子）。"""
        acc = {}
        for t in tags:
            _s, cells = load(t)
            for c, (sc, n) in cells.items():
                k, n0 = acc.get(c, (0.0, 0))
                acc[c] = (k + sc * n, n0 + n)
        return {c: (k / n0, n0) for c, (k, n0) in acc.items()}, "+".join(tags)

    bcells, blabel = pool(a.base)
    cands = [(t, load(t)[1]) for t in a.tag]
    for spec in a.pool:
        name, _, tags = spec.partition("=")
        cells, label = pool([x for x in tags.split(",") if x])
        cands.append((f"{name} [{label}]", cells))

    print(f"基线 {blabel}（权重取自场地占比，champ 格权重为 0）")
    def sample(cells, common):
        tot = sum(w[c] for c in common if w.get(c, 0) > 0)
        acc = np.zeros(a.draws)
        for c in common:
            ww = w.get(c, 0.0)
            if ww <= 0:
                continue
            s, n = cells[c]
            k = int(round(s * n))
            p = rng.beta(k + 1, n - k + 1, size=a.draws)
            acc += ww * p
        return acc / tot

    for t, cells in cands:
        common = sorted(set(bcells) & set(cells))
        b = sample(bcells, common)
        x = sample(cells, common)
        d = (x - b) * 100
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"\n{t}  （{len(common)} 个公共格）")
        print(f"  加权胜率  {b.mean():.4f}  →  {x.mean():.4f}")
        print(f"  Δ = {d.mean():+.2f}pp   95% CI [{lo:+.2f}, {hi:+.2f}]pp   P(更好) = {(d > 0).mean():.1%}")
        verdict = ("显著更好" if lo > 0 else "显著更差" if hi < 0 else "**无法区分（落在噪声里）**")
        print(f"  判定：{verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
