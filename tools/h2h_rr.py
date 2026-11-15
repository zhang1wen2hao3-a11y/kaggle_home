#!/usr/bin/env python3
"""**决赛圈循环赛（round-robin）：候选驾驶员两两对打，3 种子，可多机分片**

为什么单独做一个工具：到收尾阶段反复要回答的问题只有一个——
"这几个候选里到底谁最强"，而这个问题只能靠**同一把尺**两两对打来答。
散着用 eval_pair 手工跑会踩三个坑：
  ① 名字写岔（`r1A` 的 agent 目录是 `r1A_26c6`），跑出空面板还不报错
  ② 每次只跑一半方向（对位是不对称的：卡组不同+网络不同），结论偏
  ③ 输出文件名随手起，事后谁也说不清哪份是哪份

约定（写死）：
  · 配对 = 有序对 (cand, opp)，i≠j 全跑（20 个方向）
  · 每个方向 3 个种子（950000/950001/950002）× 200 场 = 600 场池化
  · 输出 `data/opp_matrix/rr_<cand>_vs_<opp>_s<seed>.json`，已存在且 n≥200 的行自动跳过
  · 分片按"未完成行数"轮转（`--shard i/n`），机器掉线后重算即可

    ./tools/py tools/h2h_rr.py plan                 # 看还剩哪些方向/种子
    ./tools/py tools/h2h_rr.py run --shard 0/2      # 本机跑第 0 片
    ./tools/py tools/h2h_rr.py report               # 池化后打对位矩阵
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import driver_registry as REG                     # noqa: E402

# 决赛圈：本地评测里进过前五名的驾驶员（名字用 driver_registry 的唯一名）
FINALS = ["candf5f8S", "cand7e39S", "cand26c6M", "cand26c6S", "r1A"]
SEEDS = [950000, 950001, 950002]
OUTDIR = os.path.join(ROOT, "data", "opp_matrix")

# 早先零散跑过的、协议完全相同（200 场 / seed-base 95000x）的对位文件：
# 认它们，避免为了"文件改名"重跑一遍机器时间。键 = rr 名，值 = 旧文件名。
LEGACY = {
    "rr_cand7e39S_vs_candf5f8S_s950000.json": "h2h_cand7e39S_vs_candf5f8S.json",
    "rr_cand26c6M_vs_cand26c6S_s950000.json": "h2h_cand26c6M_vs_cand26c6S.json",
    "rr_cand26c6M_vs_cand26c6S_s950001.json": "cand26c6M_vs_cand26c6S_s950001.json",
    "rr_cand26c6M_vs_r1A_s950000.json": "cand26c6M_vs_r1A.json",
    "rr_cand26c6S_vs_r1A_s950001.json": "cand26c6S_vs_r1A_s950001.json",
    "rr_cand26c6S_vs_r1A_s950002.json": "cand26c6S_vs_r1A_s950002.json",
    "rr_candf5f8S_vs_cand26c6M_s950000.json": "f5f8S_vs_cand26c6M_s950000.json",
    "rr_candf5f8S_vs_cand26c6M_s950001.json": "f5f8S_vs_cand26c6M_s950001.json",
    "rr_candf5f8S_vs_r1A_s950000.json": "f5f8S_vs_r1A_s950000.json",
    "rr_candf5f8S_vs_r1A_s950001.json": "f5f8S_vs_r1A_s950001.json",
}


def agent(name: str) -> str:
    return REG.agent_dir(name)


def read_overall(path: str):
    """读新名文件；没有就退回协议等价的旧名文件。返回 overall 字典或 None。"""
    for p in (path, os.path.join(OUTDIR, LEGACY.get(os.path.basename(path), "\0"))):
        try:
            d = json.load(open(p))
            o = d.get("overall") or {}
            if int(o.get("n") or 0) >= 200:
                return o
        except Exception:
            continue
    return None


def done(path: str) -> bool:
    return read_overall(path) is not None


def tasks(finals=None, seeds=None):
    """全部 (cand, opp, seed) 任务，含 out 路径与是否已完成。"""
    finals = finals or FINALS
    seeds = seeds or SEEDS
    out = []
    for a in finals:
        for b in finals:
            if a == b:
                continue
            for s in seeds:
                p = os.path.join(OUTDIR, f"rr_{a}_vs_{b}_s{s}.json")
                out.append((a, b, s, p, done(p)))
    return out


def cmd_plan(args):
    ts = tasks(args.finals, args.seeds)
    left = [t for t in ts if not t[4]]
    print(f"总任务 {len(ts)}，已完成 {len(ts) - len(left)}，待跑 {len(left)}")
    for a, b, s, p, _ in left:
        print(f"  {a:12s} vs {b:12s} s={s}")
    return 0


def cmd_run(args):
    ts = [t for t in tasks(args.finals, args.seeds) if not t[4]]
    if args.cand:
        # 多机时**按候选名显式分派**，不要靠"剩余行数取模"：各机器本地已完成的
        # 旧文件不一样，取模会算出不同分片 ⇒ 重复跑 / 漏跑。
        ts = [t for t in ts if t[0] in args.cand]
    if not ts:
        print("没有待跑任务")
        return 0
    mine = [t for i, t in enumerate(ts) if i % args.nshard == args.shard]
    print(f"[shard {args.shard}/{args.nshard}] 本片 {len(mine)} 个方向×种子", flush=True)
    for a, b, s, p, _ in mine:
        if done(p):
            print(f"skip {os.path.basename(p)}", flush=True)
            continue
        cmd = [sys.executable or "python3", os.path.join(ROOT, "ptcg_league", "eval_pair.py"),
               "--cand", agent(a), "--opp", f"{b}={agent(b)}",
               "--matches", str(args.matches), "--workers", str(args.workers),
               "--seed-base", str(s), "--out", p]
        print("→", " ".join(cmd[1:]), flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"!! 失败 rc={r.returncode}（继续下一个）", flush=True)
    return 0


def cmd_report(args):
    """池化每个方向：wr = cand 的胜率；同时给出对称方向的互补值供交叉验证。"""
    import collections
    agg = collections.defaultdict(lambda: [0, 0.0])
    for a, b, s, p, _ in tasks(args.finals, args.seeds):
        o = read_overall(p)
        if not o:
            continue
        n = int(o.get("n") or 0)
        wr = float(o.get("wr", o.get("score", 0)))
        agg[(a, b)][0] += n
        agg[(a, b)][1] += wr * n
    if not agg:
        print("还没有结果")
        return 0
    print(f"{'候选':12s} {'对手':12s} {'n':>4s} {'胜率':>7s}")
    for (a, b), (n, w) in sorted(agg.items()):
        print(f"{a:12s} {b:12s} {n:4d} {w / n * 100:6.1f}%")
    print("\n互补校验（同一对位两个方向的胜率应≈互补，偏差>10pp 说明卡组/座位有系统差）：")
    for a, b in sorted({(a, b) for (a, b) in agg if a < b}):
        if (a, b) in agg and (b, a) in agg:
            p1 = agg[(a, b)][1] / agg[(a, b)][0]
            p2 = agg[(b, a)][1] / agg[(b, a)][0]
            print(f"  {a} vs {b}: {p1 * 100:.1f}% / 反向 {p2 * 100:.1f}% (互补应 {100 - p1 * 100:.1f}%)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "report"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--matches", type=int, default=200)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--finals", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--cand", nargs="*", default=None, help="只跑这些候选当 cand 的方向（多机分派用）")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    return {"plan": cmd_plan, "run": cmd_run, "report": cmd_report}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
