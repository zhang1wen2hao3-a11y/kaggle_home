#!/usr/bin/env python3
"""**阶段 B：卡组横向筛选**（统一 port 策略，只换卡组）。

## 为什么需要它

对位数字必须成矩阵才有意义：单独说"某副牌对某副牌 0.86"没有决策价值
（我们就是因为这个把 0904 判错过一次 —— 头对头赢 0.86，按真实场次加权却输 10pp）。
本工具一次跑出 **候选卡组 × 对手卡组** 的完整矩阵，并给出两个汇总口径：

1. `plain`：对全部对手等权
2. `field`：按 `--weights` 给的对手出场占比加权（默认读 `tools/../data/top_decks.tsv`
   的局数分布，也可手写 `名字=占比`）

## 协议（与 ptcg_league/eval_pair.py 一致）
Bo3、座位交替、同 `--seed` 配对，因此**格子之间、候选之间都可配对比较**。

## 用法

    # 只看阶段 B：8 副候选 × 3 副头部对手，每格 200 场
    ./tools/py tools/deck_screen.py \\
      --cand "0904=decks/top/top__0904e46f.csv" --cand "e1d7=decks/top/top__e1d754a7.csv" \\
      --opp  "0904=decks/top/top__0904e46f.csv" --opp  "e1d7=decks/top/top__e1d754a7.csv" \\
      --matches 200 --seed-base 970000 --out data/deck_matrix_top.tsv

    # 用 G6 权重当策略（阶段 C：同一批卡组换策略）
    ./tools/py tools/deck_screen.py ... --policy models/ppo/g6_db04.pt

产物
* `data/deck_matrix_top.tsv`：`cand_opponent` 长表（胜率 / CI / n）
* 终端打印矩阵 + 两种加权的排序
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKELETON = os.path.join(ROOT, "build", "port_plain")
LINKS = ["EN_Card_Data.csv", "cg", "data", "main.py", "ptcg", "r2e_manifest.json"]


def deck_tag(path):
    ids = [int(x.strip()) for x in open(path) if x.strip()]
    return hashlib.sha256(",".join(str(x) for x in sorted(ids)).encode()).hexdigest()[:8]


def file_tag(path):
    """权重文件的短指纹（用于给临时 agent 目录去重，避免两个策略共用同名目录）。"""
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:8]


def make_agent(name, deck_path, policy_pt, outdir):
    """在 outdir 里铺一个标准 agent 目录（骨架软链 + deck.csv + model.pt）。"""
    dst = os.path.join(outdir, name)
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for l in LINKS:
        os.symlink(os.path.join(SKELETON, l), os.path.join(dst, l))
    shutil.copy(deck_path, os.path.join(dst, "deck.csv"))
    shutil.copy(policy_pt, os.path.join(dst, "model.pt"))
    return dst


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n
    dd = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, (c - h) / dd), min(1.0, (c + h) / dd)


def parse_pairs(items, flag):
    out = []
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"{flag} 需要 name=path，收到 {it!r}")
        n, p = it.split("=", 1)
        p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        if not os.path.exists(p):
            raise SystemExit(f"{flag} {n}: 文件不存在 {p}")
        out.append((n, p))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cand", action="append", required=True, help="候选卡组 name=deck.csv")
    ap.add_argument("--opp", action="append", required=True, help="对手卡组 name=deck.csv")
    ap.add_argument("--policy", default=os.path.join(ROOT, "build", "port_plain", "model.pt"),
                    help="统一策略权重（默认端口权重）")
    ap.add_argument("--matches", type=int, default=200, help="**每个对手**跑多少场 Bo3")
    ap.add_argument("--seed-base", type=int, default=970000)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--weights", default="", help="对手权重 name=占比,…（默认按出场局数）")
    ap.add_argument("--out", default="data/deck_matrix.tsv")
    ap.add_argument("--workdir", default="build/screen")
    ap.add_argument("--logdir", default="logs/screen")
    ap.add_argument("--reuse", action="store_true", help="已有 json 就跳过（断点续跑）")
    a = ap.parse_args()

    cands = parse_pairs(a.cand, "--cand")
    opps = parse_pairs(a.opp, "--opp")
    workdir = os.path.join(ROOT, a.workdir) if not os.path.isabs(a.workdir) else a.workdir
    logdir = os.path.join(ROOT, a.logdir) if not os.path.isabs(a.logdir) else a.logdir
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)) or ".", exist_ok=True)

    print(f"策略权重 {a.policy}")
    print("候选卡组:")
    for n, p in cands:
        print(f"  {n:12s} {deck_tag(p)}  {os.path.relpath(p, ROOT)}")
    print("对手卡组:")
    for n, p in opps:
        print(f"  {n:12s} {deck_tag(p)}  {os.path.relpath(p, ROOT)}")
    total = a.matches * len(opps) * len(cands)
    print(f"\n协议：{len(cands)} 候选 × {len(opps)} 对手 × {a.matches} 场/格 = **{total} 场 Bo3**"
          f"（种子 {a.seed_base}）\n")

    # 铺 agent
    tag = f"_{file_tag(a.policy)[:4]}"
    cand_dirs = {n: make_agent(f"c_{n}{tag}", p, a.policy, workdir) for n, p in cands}
    opp_dirs = {n: make_agent(f"o_{n}{tag}", p, a.policy, workdir) for n, p in opps}

    rows = []
    for cn, _ in cands:
        out = os.path.join(logdir, f"{cn}.json")
        if a.reuse and os.path.exists(out):
            print(f"[skip] {cn}（已有 {out}）")
        else:
            cmd = [os.path.join(ROOT, "tools", "py"), os.path.join(ROOT, "ptcg_league", "eval_pair.py"),
                   "--cand", cand_dirs[cn]]
            for on, _ in opps:
                cmd += ["--opp", f"{on}={opp_dirs[on]}"]
            # ⚠️ eval_pair 的 --matches 是**候选的总场次**，会平均分给所有对手；
            # 这里按"每格 a.matches"折算成总量，否则 6 个对手只剩 1/6 样本。
            cmd += ["--matches", str(a.matches * len(opps)), "--workers", str(a.workers),
                    "--seed-base", str(a.seed_base), "--out", out,
                    "--logdir", os.path.join(logdir, cn)]
            print(f"########## {cn} ##########", flush=True)
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            tail = (r.stdout or "").strip().splitlines()[-1:] + (r.stderr or "").strip().splitlines()[-1:]
            print("  " + " | ".join(tail), flush=True)
        d = json.load(open(out))
        for on, _ in opps:
            v = d["per_opp"][on]
            rows.append({"cand": cn, "opp": on, "score": v["score"], "lo": v["ci_low"],
                         "hi": v["ci_high"], "n": v["n"], "err": v.get("errors", 0)})

    with open(os.path.join(ROOT, a.out), "w") as fh:
        fh.write("cand\topp\tscore\tci_low\tci_high\tn\terrors\n")
        for r in rows:
            fh.write(f"{r['cand']}\t{r['opp']}\t{r['score']:.4f}\t{r['lo']:.4f}\t"
                     f"{r['hi']:.4f}\t{r['n']}\t{r['err']}\n")

    # ---- 汇总 ----
    w = {n: 1.0 for n, _ in opps}
    if a.weights:
        w = {}
        for it in a.weights.split(","):
            k, v = it.split("=")
            w[k.strip()] = float(v)
    tw = sum(w.values())

    print(f"\n{'候选':14s}" + "".join(f"{on:>10s}" for on, _ in opps) +
          f"{'等权':>9}{'按场次加权':>12}")
    ranked = []
    for cn, _ in cands:
        cells = {r["opp"]: r for r in rows if r["cand"] == cn}
        plain = sum(c["score"] for c in cells.values()) / len(cells)
        wt = sum(w.get(k, 1.0) * c["score"] for k, c in cells.items()) / tw
        ranked.append((wt, plain, cn))
        print(f"{cn:14s}" + "".join(f"{cells[on]['score']:>10.3f}" for on, _ in opps) +
              f"{plain:>9.3f}{wt:>12.3f}")
    print(f"\n按加权胜率排序：")
    for wt, plain, cn in sorted(ranked, reverse=True):
        print(f"  {cn:14s} {wt:.4f}（等权 {plain:.4f}）")
    print(f"\n矩阵 → {a.out}")


if __name__ == "__main__":
    main()
