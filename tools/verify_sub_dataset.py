#!/usr/bin/env python3
"""**提交数据集校验器**：抓完回放后，先证明"这批数据是同卡组、同策略、可分组"，再拿去算胜率。

为什么必须先校验（2026-11-03 用户要求）：
  * 回放**不揭示策略权重**，只能揭示卡组 ⇒ "ref → 策略"是**外部信息**，必须在报告里标注来源；
  * 如果同一批里出现多副我方卡组，说明抓错 ref 或 agent 名匹配错 ⇒ 胜率不可用；
  * 座位必须能确定，否则无法做座位分层。

输出四块：
  ① 我方 agent 名与座位映射（能否稳定识别）
  ② **我方卡组哈希分布**（应当只有 1 个占绝对多数）
  ③ 对手卡组分布 + 对每个原型的胜率（Wilson CI）
  ④ 数据健康：异常局、缺卡表局、座位分布

用法：
    ./tools/py tools/verify_sub_dataset.py --dir epsoide_champ --label "port@db04(1071.4)"
    ./tools/py tools/verify_sub_dataset.py --dir epsoide_champ --me zhang1wen2hao3
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


def load_pools():
    if POOL:
        return POOL
    for d in ("decks/pool_field", "port_repo/decks/pool", "port_repo/decks/pool_alt"):
        for f in glob.glob(os.path.join(ROOT, d, "*.csv")):
            try:
                ids = tuple(sorted(int(l) for l in open(f) if l.strip()))
            except ValueError:
                continue
            if len(ids) == 60:
                POOL.setdefault(ids, os.path.basename(f)[:-4])
    return POOL


def dhash(ids):
    return hashlib.sha256(",".join(str(x) for x in sorted(ids)).encode()).hexdigest()[:8]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--me", default="zhang1wen2hao3", help="我方 agent 名字（子串匹配）")
    a = ap.parse_args()

    pool = load_pools()
    files = sorted(glob.glob(os.path.join(ROOT, a.dir, "*.json")))
    if not files:
        raise SystemExit(f"✗ {a.dir} 里没有回放")

    games, bad = [], 0
    for p in files:
        try:
            d = json.load(open(p))
        except Exception:
            bad += 1
            continue
        st = d.get("steps") or []
        if len(st) < 2:
            bad += 1
            continue
        agents = [x.get("Name") or "" for x in (d.get("info", {}).get("Agents") or [])]
        decks = []
        for k in (0, 1):
            act = st[1][k].get("action")
            decks.append(tuple(sorted(act)) if isinstance(act, list) and len(act) == 60 else None)
        rew = d.get("rewards") or [0, 0]
        games.append(dict(ep=os.path.basename(p), agents=agents, decks=decks,
                          winner=0 if rew[0] > rew[1] else (1 if rew[1] > rew[0] else -1)))

    print(f"===== {a.label or a.dir} =====")
    print(f"回放 {len(files)} 个：可用 {len(games)}，损坏/缺步骤 {bad}")

    # ① 我方 agent 识别
    idx = []
    for g in games:
        i = next((k for k, n in enumerate(g["agents"]) if a.me in n), None)
        idx.append(i)
    found = sum(1 for i in idx if i is not None)
    print(f"\n① 我方 agent 识别：{found}/{len(games)} 局命中 '{a.me}'")
    if found != len(games):
        names = collections.Counter(n for g in games for n in g["agents"])
        print("   出现的 agent 名（前 6）：")
        for n, c in names.most_common(6):
            print(f"     {n[:40]:42s}{c}")
        print("   ⚠️ 若有未命中，用 --me 指定正确的名字")

    seat_cnt = collections.Counter(i for i in idx if i is not None)
    print(f"② 我方座位分布：{dict(seat_cnt)}")

    # ③ 我方卡组哈希
    mine = collections.Counter()
    for g, i in zip(games, idx):
        if i is None:
            continue
        dk = g["decks"][i]
        if dk:
            mine[dhash(dk)] += 1
    print(f"\n③ **我方卡组哈希**（应当只有 1 个占绝对多数）：")
    for h, c in mine.most_common():
        nm = next((v for k, v in pool.items() if dhash(k) == h), "池外")
        print(f"   {h}  {c:4d} 局  {nm}")
    if len(mine) > 1:
        print("   ⚠️ 出现多个我方卡组 ⇒ 这批数据混了不同 ref/卡组，不能直接合并算胜率")
    else:
        print("   ✓ 单一卡组，可安全分组")

    # ④ 对手分布 + 胜率
    fam = collections.defaultdict(lambda: [0, 0])
    opp_decks = collections.Counter()
    for g, i in zip(games, idx):
        if i is None:
            continue
        j = 1 - i
        od = g["decks"][j]
        if not od:
            continue
        h = dhash(od)
        opp_decks[h] += 1
        nm = next((v for k, v in pool.items() if dhash(k) == h), None)
        key = (nm or f"池外:{h}").split("__")[0]
        fam[key][0] += 1
        if g["winner"] == i:
            fam[key][1] += 1
    print(f"\n④ 对手分布与胜率（我方视角）：")
    print(f"{'对手原型/卡表':26s}{'局':>4}{'胜':>4}{'胜率':>7}{'Wilson 95%':>18}")
    tot_n = tot_w = 0
    for k, (n, w) in sorted(fam.items(), key=lambda kv: -kv[1][0]):
        p, lo, hi = wilson(w, n)
        tot_n += n
        tot_w += w
        print(f"{k[:25]:26s}{n:>4}{w:>4}{p:>7.3f}{f'[{lo:.3f},{hi:.3f}]':>18}")
    if tot_n:
        p, lo, hi = wilson(tot_w, tot_n)
        print(f"{'合计':26s}{tot_n:>4}{tot_w:>4}{p:>7.3f}{f'[{lo:.3f},{hi:.3f}]':>18}")

    print(f"\n⚠️ 口径提醒：策略权重**无法**从回放验证 —— 'ref → 策略' 属于外部信息，"
          f"报告里必须标注来源（提交记录），不能说成回放证据。")


if __name__ == "__main__":
    main()
