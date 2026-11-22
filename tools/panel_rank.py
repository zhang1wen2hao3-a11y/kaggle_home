#!/usr/bin/env python3
"""**面板分栏排名**：把"加权面板"拆成"除某几格之外"和"那几格"，并和 LB 对照。

## 为什么要这个（2026-11-22 踩到的坑）

强对手 17 格面板里 `ogerpon_7e39` 占 **34.9%** 权重，而这一格由**牌组克制**决定：
端口驾驶员打它是 35~41%、训练后 38~39%，几乎不变；而 9d6e 家族天生赢它 89%。
更关键的是：**我们的 `port_7e39` 打别的牌有 60~65%，比真人拿 7e39 的天梯胜率（33%）高一大截**
⇒ 这一格的面板对手强度**高于真实天梯**，于是对"怕 7e39 的牌"系统性低估。

实测佐证：`m9d6e` 面板 73.8% → LB 995.8；`cand0904R1` 面板 54.4% → LB 1025.3（面板排序和 LB 相反，
差别就在这一格：89% vs 39%）。

## 用法

    ./tools/py tools/panel_rank.py --cands cand9D6ER3 cand0904R1 cand3EAAR1
    ./tools/py tools/panel_rank.py --cands A B --exclude ogerpon_7e39,dunsparce_0bf5
    ./tools/py tools/panel_rank.py --cands A B --lb "A=995.8,B=1025.3"   # 顺便报面板-LB 一致性

输出：每候选的「全格加权」「剔除后加权（按剩余权重重新归一）」「被剔除格在面板里的表现」。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def shares() -> dict:
    out = {}
    p = os.path.join(ROOT, "decks", "testset_meta_real.tsv")
    for ln in open(p):
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.rstrip("\n").split("\t")
        if len(f) >= 5 and f[0] != "tail_mix":
            out[f[0]] = float(f[4])
    return out


def cells(cand: str) -> dict:
    """{格: (胜率, n)} —— 同一格多个种子合并。"""
    agg = collections.defaultdict(lambda: [0.0, 0])
    for p in glob.glob(os.path.join(ROOT, "logs", "meta", f"{cand}_real_s95*", "vs_*.json")):
        cell = os.path.basename(p)[3:-5]
        try:
            o = json.load(open(p)).get("overall") or {}
        except (OSError, ValueError):
            continue
        n = int(o.get("n") or 0)
        if not n:
            continue
        agg[cell][0] += float(o.get("score") or 0) * n
        agg[cell][1] += n
    return {k: (v[0] / v[1], v[1]) for k, v in agg.items() if v[1]}


def weighted(c: dict, w: dict, skip: set) -> tuple:
    tw = ts = 0.0
    for cell, (wr, _) in c.items():
        if cell in skip or cell not in w:
            continue
        tw += w[cell]
        ts += w[cell] * wr
    return (ts / tw * 100 if tw else float("nan"), tw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--exclude", default="ogerpon_7e39", help="逗号分隔的格名（默认排掉 7e39）")
    ap.add_argument("--lb", default="", help="候选=LB 的对照串，如 'cand9D6ER3=995.8'")
    a = ap.parse_args()
    w = shares()
    skip = {x.strip() for x in a.exclude.split(",") if x.strip()}
    lb = {}
    for kv in a.lb.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                lb[k.strip()] = float(v)
            except ValueError:
                pass

    print(f"剔除格：{','.join(sorted(skip)) or '（无）'}（占权重 {sum(w.get(c,0) for c in skip):.1f}%）")
    print(f"{'候选':14s}{'全格加权':>9s}{'剔除后':>9s}{'剔除格表现':>12s}{'格子数':>6s}{'LB':>8s}")
    rows = []
    for cand in a.cands:
        c = cells(cand)
        if not c:
            print(f"{cand:14s}{'—':>9s}")
            continue
        full, _ = weighted(c, w, set())
        excl, tw = weighted(c, w, skip)
        miss = []
        for cell in skip:
            if cell in c:
                miss.append(f"{cell.split('_')[-1]} {c[cell][0]*100:.0f}%")
            else:
                miss.append(f"{cell.split('_')[-1]} —")
        rows.append((excl, cand, full, " ".join(miss), len(c), lb.get(cand)))
    rows.sort(reverse=True)
    for excl, cand, full, miss, nc, l in rows:
        ls = f"{l:8.1f}" if l is not None else "       —"
        print(f"{cand:14s}{full:8.1f}%{excl:8.1f}%{miss:>12s}{nc:6d}{ls}")

    if lb:
        ranked = [r[1] for r in rows if r[5] is not None]
        if len(ranked) >= 2:
            print(f"\n有 LB 的候选（按剔除后面板排序）：{' > '.join(ranked)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
