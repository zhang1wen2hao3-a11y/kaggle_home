#!/usr/bin/env python3
"""**配对面板比较**（2026-11-23）：两个候选在同一批比赛种子上对打同一批对手 ⇒ 用配对检验。

为什么需要
----------
面板每格 n=150 局、±8pp，直接比两个候选的"独立置信区间"几乎没有检出力：
我们实测同配方两次跑的差异就有 −1.8 ~ +3.5pp（R5c vs R4、R4 vs R3），**比效应本身还大**。

但面板结果里存了 `overall.per_match`（种子 → 得分），**两个候选面对的是完全相同的 150 个种子**
（已验证：150/150 相同）⇒ 可以逐局配对，用差值做检验。配对能把"对手抽牌/种子"这层方差抵掉，
同样的局数下检出力高得多。

口径
----
* 每格：d_i = score_A(seed_i) − score_B(seed_i)，报告 mean(d)、SE = sd(d)/√n、z = mean/SE
* 汇总：按真实场权重 w 加权，mean = Σ w·mean_d / Σ w，SE = √(Σ w²·SE²) / Σ w（格间独立）
* 判定：只报"配对差"和它的 95% CI；**默认阈值 +3pp**（低于它一律按噪声）

用法
----
    # A = 新候选，B = 对照
    ./tools/py tools/panel_paired.py --a cand9D6ED1 --b cand9D6ER4 --tag v2
    ./tools/py tools/panel_paired.py --a cand9D6ED1B --b cand9D6ER4 --tag v2 --threshold 0.03
    ./tools/py tools/panel_paired.py --a X --b Y --tag v2 --cells grookey_2617,dunsparce_9d6e
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def weights() -> dict:
    w = {}
    for ln in open(os.path.join(ROOT, "decks", "testset_meta_real.tsv")):
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.rstrip("\n").split("\t")
        if f[0] != "tail_mix":
            w[f[0]] = float(f[4])
    return w


def per_match(cand: str, tag: str, cell: str) -> dict:
    """把同一格的多个种子目录合并成 {match_seed: score}。"""
    out = {}
    for p in glob.glob(os.path.join(ROOT, "logs", "meta", f"{cand}_{tag}_s95*", f"vs_{cell}.json")):
        try:
            o = json.load(open(p)).get("overall") or {}
        except (OSError, ValueError):
            continue
        for k, v in (o.get("per_match") or {}).items():
            out.setdefault(str(k), []).append(float(v))
    return {k: sum(v) / len(v) for k, v in out.items()}


def paired(a: dict, b: dict):
    keys = sorted(set(a) & set(b))
    if len(keys) < 10:
        return None
    d = [a[k] - b[k] for k in keys]
    n = len(d)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return dict(n=n, mean=m, se=se, z=(m / se if se else 0.0),
                wins=sum(1 for x in d if x > 0), losses=sum(1 for x in d if x < 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="新候选")
    ap.add_argument("--b", required=True, help="对照")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--cells", default="", help="只看这些格（逗号分隔，默认全部）")
    ap.add_argument("--threshold", type=float, default=0.03, help="判有用的最小配对差（默认 0.03=3pp）")
    a = ap.parse_args()

    W = weights()
    cells = [c.strip() for c in a.cells.split(",") if c.strip()] or list(W)
    print(f"A={a.a}  B={a.b}  口径={a.tag}  阈值 {a.threshold*100:+.1f}pp（配对检验，同种子）\n")
    print(f"{'格':18s}{'权重':>6s}{'n':>5s}{'A':>7s}{'B':>7s}{'配对差':>9s}{'SE':>7s}{'z':>7s}{'A胜/负局':>10s}")
    tot_w = tot_m = tot_v = 0.0
    rows = []
    for c in sorted(cells, key=lambda x: -W.get(x, 0)):
        pm_a, pm_b = per_match(a.a, a.tag, c), per_match(a.b, a.tag, c)
        st = paired(pm_a, pm_b)
        if not st:
            continue
        w = W.get(c, 0.0)
        wr_a = sum(pm_a[k] for k in set(pm_a) & set(pm_b)) / st["n"]
        wr_b = sum(pm_b[k] for k in set(pm_a) & set(pm_b)) / st["n"]
        rows.append((c, w, st))
        print(f"{c:18s}{w:5.1f}%{st['n']:5d}{wr_a*100:7.1f}{wr_b*100:7.1f}"
              f"{st['mean']*100:+9.1f}{st['se']*100:7.1f}{st['z']:+7.2f}"
              f"{str(st['wins'])+'/'+str(st['losses']):>10s}")
        if w:
            tot_w += w
            tot_m += w * st["mean"]
            tot_v += (w * st["se"]) ** 2
    if not tot_w:
        print("✗ 没有可配对的格")
        return 2
    mean = tot_m / tot_w
    se = math.sqrt(tot_v) / tot_w
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    print(f"\n加权配对差 = {mean*100:+.1f}pp   SE {se*100:.1f}pp   95% CI [{lo*100:+.1f}, {hi*100:+.1f}]pp")
    verdict = "✅ 有用（超过阈值且 CI 不含 0）" if (mean >= a.threshold and lo > 0) else \
              ("⚠️ 方向为正但不显著/不到阈值" if mean > 0 else "❌ 没有优势")
    print(f"判定（阈值 {a.threshold*100:+.1f}pp）：{verdict}")
    return 0 if verdict.startswith("✅") else 3


if __name__ == "__main__":
    sys.exit(main())
