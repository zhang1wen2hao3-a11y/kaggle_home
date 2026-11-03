#!/usr/bin/env python3
"""**Champion Policy Axis 报告**：同卡组、不同策略的真实天梯对比（G6 vs port@db04）。

分层口径（2026-11-03 用户指定）：
  A. 总胜率 + Wilson 95% CI
  B. **精确镜像**（对手卡表 == 我方卡表）
  C. **同族不同构筑**（同 archetype 前缀、卡表不同，如 dunsparce_abra__0bf5）
  D. 其它卡组**按原型分组**
  E. **座位分层**（seat0 / seat1 = 我们在回放里的座位下标）
  F. **时间分层**（按 episode createTime 或 id 先后切两半）

⚠️ 两条纪律：
  * 我方卡组必须**单一**，否则整份数据作废（防混牌）；
  * 两批数据**不是配对样本**（不同的对局），所以只能逐层比 CI，不能做配对差。

用法：
    ./tools/py tools/policy_axis_report.py --a epsoide_g6 --b epsoide_portdb04 \
        --label-a "G6@db04(1082.1)" --label-b "port@db04(1028.2)"
    # 只要一份也支持（--b 省略）
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = {}


def load_pool():
    if POOL:
        return POOL
    for d in ("decks/pool_field", "port_repo/decks/pool", "port_repo/decks/pool_alt"):
        for f in glob.glob(os.path.join(ROOT, d, "*.csv")):
            try:
                ids = tuple(sorted(int(l) for l in open(f) if l.strip()))
            except ValueError:
                continue
            if len(ids) == 60:
                POOL[ids] = os.path.basename(f)[:-4]
    return POOL


def dh(ids):
    return hashlib.sha256(",".join(str(x) for x in sorted(ids)).encode()).hexdigest()[:8]


def wil(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def load_dir(dirp, me):
    pool = load_pool()
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, dirp, "*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        st = d.get("steps") or []
        if len(st) < 2:
            continue
        ag = [x.get("Name") or "" for x in (d.get("info", {}).get("Agents") or [])]
        i = next((k for k, n in enumerate(ag) if me in n), None)
        if i is None:
            continue
        decks = []
        for k in (0, 1):
            a = st[1][k].get("action")
            decks.append(tuple(sorted(a)) if isinstance(a, list) and len(a) == 60 else None)
        if not all(decks):
            continue
        rew = d.get("rewards") or [0, 0]
        ep = os.path.basename(p).replace("episode-", "").replace("-replay.json", "")
        out.append(dict(ep=ep, seat=i, my=decks[i], opp=decks[1 - i],
                        won=int(rew[i] > rew[1 - i]),
                        my_name=pool.get(decks[i]) or f"池外:{dh(decks[i])}",
                        opp_name=pool.get(decks[1 - i]) or f"池外:{dh(decks[1 - i])}"))
    return out


def summarize(games, tag):
    pool = load_pool()
    if not games:
        print(f"\n### {tag}: 无可用对局")
        return {}
    mine = collections.Counter(g["my_name"] for g in games)
    print(f"\n### {tag}")
    print(f"对局 {len(games)} 局；我方卡组：{dict(mine)}")
    if len(mine) > 1:
        print("   ⚠️ 我方卡组不唯一 ⇒ 该批数据不可用于对位结论（先查是否混了 ref）")
    mydeck = mine.most_common(1)[0][0]
    myarch = mydeck.split("__")[0].split(":")[0]
    layers = collections.defaultdict(lambda: [0, 0])

    def add(key, g):
        layers[key][0] += 1
        layers[key][1] += g["won"]

    for g in games:
        o = g["opp_name"]
        if g["opp"] == g["my"]:
            add("B 精确镜像", g)
        elif o.split("__")[0] == myarch:
            add("C 同族不同构筑", g)
        elif o.startswith("池外"):
            add("D 池外（未收录卡表）", g)
        else:
            add(f"D {o.split('__')[0][:24]}", g)
        add(f"E seat{g['seat']}", g)
    # 时间分层：按 episode id 排序切两半（id 单调递增）
    gs = sorted(games, key=lambda x: int(x["ep"]) if x["ep"].isdigit() else 0)
    mid = len(gs) // 2
    for g in gs[:mid]:
        add("F 前半段", g)
    for g in gs[mid:]:
        add("F 后半段", g)

    tot_n = len(games)
    tot_w = sum(g["won"] for g in games)
    p, lo, hi = wil(tot_w, tot_n)
    print(f"{'层':26s}{'局':>5}{'胜':>5}{'胜率':>7}{'Wilson 95%':>18}")
    print(f"{'A 总计':26s}{tot_n:>5}{tot_w:>5}{p:>7.3f}{f'[{lo:.3f},{hi:.3f}]':>18}")
    for k in sorted(layers):
        n, w = layers[k]
        p, lo, hi = wil(w, n)
        print(f"{k:26s}{n:>5}{w:>5}{p:>7.3f}{f'[{lo:.3f},{hi:.3f}]':>18}")
    return dict(total=(tot_w, tot_n), layers=layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default=None)
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--me", default="zhang1wen2hao3")
    a = ap.parse_args()

    A = load_dir(a.a, a.me)
    sa = summarize(A, a.label_a or a.a)
    sb = None
    if a.b:
        B = load_dir(a.b, a.me)
        sb = summarize(B, a.label_b or a.b)

    if sb:
        print("\n===== 两批数据逐层对照（无配对，仅比 CI）=====")
        keys = sorted(set(sa["layers"]) | set(sb["layers"]))
        print(f"{'层':26s}{'A 胜率 (n)':>18}{'B 胜率 (n)':>18}{'差':>8}")
        for k in keys:
            na, wa = sa["layers"].get(k, [0, 0])
            nb, wb = sb["layers"].get(k, [0, 0])
            if na == 0 or nb == 0:
                continue
            pa, pb = wa / na, wb / nb
            print(f"{k:26s}{f'{pa:.3f} ({na})':>18}{f'{pb:.3f} ({nb})':>18}{100*(pa-pb):>+7.1f}pp")
        wa, na = sa["total"]
        wb, nb = sb["total"]
        print(f"{'A 总计':26s}{f'{wa/na:.3f} ({na})':>18}{f'{wb/nb:.3f} ({nb})':>18}"
              f"{100*(wa/na-wb/nb):>+7.1f}pp")


if __name__ == "__main__":
    main()
