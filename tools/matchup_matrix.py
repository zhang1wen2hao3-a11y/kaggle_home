#!/usr/bin/env python3
"""League Research · **对位矩阵 + 双排行榜**（2026-11-01 与用户对齐的口径）。

两件独立的事，评价标准不混：

  A. **League Rating**：Agent 在训练对手池里的综合能力
     → 矩阵的**行平均**（对每个对手的胜率取平均，不按天梯分布加权）
  B. **Ladder EV**：在真实天梯对手分布下的预期收益
     → Σ P(对手) × W(agent, 对手)，P 来自我们自己的实地对局分布

并且强制报告四个"防止只擅长一个对手"的指标：
  1. 加权期望胜率（Ladder EV）
  2. **最低对位胜率**（min matchup）
  3. **80% 覆盖范围内的平均胜率**（按份额降序累加到 ≥80% 的那几个原型）
  4. **低于 40% 的严重劣势对位数量**

统一协议（不可改，否则格子之间不可比）：
  * 单位 = Bo3 match；座位交替 `seat = mid % 2`
  * 全部格子共用**同一批种子基准** ⇒ 可逐场配对
  * 每格场数相同

用法：
    # 跑矩阵（只跑还没跑的格子）
    ./tools/py tools/matchup_matrix.py --agents "db04=build/agents/port_db04,abra=build/agents/c_abra" \
        --matches 200 --opponents ogerpon=build/agents/port_7e39,...
    # 只看报告（从已有的 logs/matrix2/*.json）
    ./tools/py tools/matchup_matrix.py --report
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX_DIR = os.path.join(ROOT, "logs", "matrix2")
OUT_TSV = os.path.join(ROOT, "data", "matchup_matrix.tsv")

# 真实天梯对手分布（来自我们 640 场实地对局；`?` = 池外长尾，不参与逐格计算但会提示）
FIELD = {
    "teal_mask_ogerpon_ex_applin": 0.356,
    "abra_kadabra": 0.149,
    "grookey_thwackey": 0.129,
    "dunsparce_abra": 0.113,
    "dreepy_drakloak": 0.096,
    "shuppet_dhelmise": 0.055,
}
SEVERE = 0.40          # 低于这个胜率算"严重劣势对位"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def parse_pairs(spec):
    out = []
    for item in [x.strip() for x in spec.split(",") if x.strip()]:
        name, _, d = item.partition("=")
        out.append((name.strip(), os.path.abspath(d.strip())))
    return out


def run_cells(agents, opponents, matches, seed, workers):
    os.makedirs(MATRIX_DIR, exist_ok=True)
    allp = agents + [o for o in opponents if o[0] not in {a[0] for a in agents}]
    done = 0
    for an, ad in agents:
        for on, od in allp:
            if an == on:
                continue
            out = os.path.join(MATRIX_DIR, f"{an}__vs__{on}.json")
            if os.path.exists(out):
                print(f"[skip] {an} vs {on}")
                continue
            print(f"########## {an}  vs  {on} ##########", flush=True)
            cmd = [sys.executable, os.path.join(ROOT, "ptcg_league", "eval_pair.py"),
                   "--cand", ad, "--opp", f"{on}={od}", "--matches", str(matches),
                   "--workers", str(workers), "--seed-base", str(seed),
                   "--out", out, "--logdir", os.path.join(MATRIX_DIR, f"{an}__{on}")]
            subprocess.run(cmd, cwd=ROOT)
            done += 1
    print(f"新跑 {done} 格")


def load_matrix():
    cells = {}
    for f in sorted(glob.glob(os.path.join(MATRIX_DIR, "*.json"))):
        base = os.path.basename(f)[:-5]
        if "__vs__" not in base:
            continue
        a, b = base.split("__vs__", 1)
        o = json.load(open(f))["overall"]
        cells[(a, b)] = o
    return cells


def report(field=None, write_tsv=True):
    cells = load_matrix()
    if not cells:
        raise SystemExit("还没有矩阵数据（logs/matrix2/*.json）")
    rows = sorted({a for a, _ in cells})
    cols = sorted({b for _, b in cells})
    print("===== 对位矩阵（行=agent，列=对手；值=agent 胜率，括号=场次）=====")
    hdr = f"{'':16s}" + "".join(f"{c[:14]:>17s}" for c in cols)
    print(hdr)
    for a in rows:
        line = f"{a[:15]:16s}"
        for c in cols:
            o = cells.get((a, c))
            line += f"{'—':>17s}" if not o else f"{o['score']:>8.3f}({o['n']:>3d})".rjust(17)
        print(line)

    if write_tsv:
        with open(OUT_TSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["agent", "opponent", "win_rate", "ci_low", "ci_high", "n", "errors"])
            for (a, b), o in sorted(cells.items()):
                w.writerow([a, b, f"{o['score']:.4f}", f"{o['ci_low']:.4f}",
                            f"{o['ci_high']:.4f}", o["n"], o["errors"]])
        print(f"\n→ {OUT_TSV}")

    # 四个指标 + 双排行榜
    field = field or FIELD
    print("\n===== 双排行榜 =====")
    print(f"{'agent':16s}{'LeagueRating':>13}{'LadderEV':>10}{'最低对位':>9}{'80%档均值':>10}{'<40%对数':>9}")
    rows_out = []
    for a in rows:
        mine = {b: cells[(a, b)] for (x, b) in list(cells.keys()) if x == a}
        if not mine:
            continue
        league = sum(o["score"] for o in mine.values()) / len(mine)
        # Ladder EV：只算 field 里有的对手，按份额归一
        ev_num = ev_den = 0.0
        for b, o in mine.items():
            key = None
            for k in field:
                if b.startswith(k.split("_")[0]) or k.startswith(b):
                    key = k
                    break
            if key is None:
                continue
            p = field[key]
            ev_num += p * o["score"]
            ev_den += p
        ev = ev_num / ev_den if ev_den else float("nan")
        worst = min(o["score"] for o in mine.values())
        n_severe = sum(1 for o in mine.values() if o["score"] < SEVERE)
        # 80% 覆盖：按场份额降序累加到 ≥80%
        acc, cov = 0.0, []
        for k, p in sorted(field.items(), key=lambda kv: -kv[1]):
            if acc >= 0.80:
                break
            o = None
            for y, v in mine.items():
                if y.startswith(k.split("_")[0]) or k.startswith(y):
                    o = v
                    break
            if o is not None:
                cov.append((o["score"], p))
            acc += p
        cov_avg = (sum(s * p for s, p in cov) / sum(p for _, p in cov)) if cov else float("nan")
        rows_out.append((a, league, ev, worst, cov_avg, n_severe))
    for a, league, ev, worst, cov_avg, n_sev in sorted(rows_out, key=lambda r: -r[1]):
        print(f"{a[:15]:16s}{league:>13.3f}{ev:>10.3f}{worst:>9.3f}{cov_avg:>10.3f}{n_sev:>9d}")

    print("\n注：LeagueRating 与 LadderEV 可能给出不同冠军，这是**预期**的：")
    print("     前者衡量'在训练池里综合强'，后者衡量'在当前天梯分布下收益高'。")
    print("     选择提交对象时看 LadderEV；评估研究进展时看 LeagueRating + 最低对位。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="", help="NAME=DIR, ...  （矩阵的行）")
    ap.add_argument("--opponents", default="", help="NAME=DIR, ...（矩阵的列）")
    ap.add_argument("--matches", type=int, default=200)
    ap.add_argument("--seed", type=int, default=930000)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.agents and a.opponents:
        run_cells(parse_pairs(a.agents), parse_pairs(a.opponents), a.matches, a.seed, a.workers)
    report()


if __name__ == "__main__":
    main()
