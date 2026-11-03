#!/usr/bin/env python3
"""生成与 1st-place writeup 对应的表格（只造我们现有数据能支撑的）。

对应关系：
  表 8  原型统计      <- decks/weights_*.txt + decks/pool_field/
  表 11 竞技场排名    <- logs/gate_*.json（gate 的全池加权分 + n）
  表 12 对位矩阵      <- logs/matchup_cpu*.json（多机平均）
  表 13a 原型内部循环赛 <- 需要跑（本脚本只列变体清单与建议赛程）
  表 17/18/19 真实对局 <- data/episodes_*.csv（scrape_episodes.py 产出）

用法：python3 tools/field_tables.py [--band ours] [--out 表格_对照.md]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_weights(band: str) -> dict[str, float]:
    w: dict[str, float] = {}
    for line in (ROOT / "decks" / f"weights_{band}.txt").read_text().splitlines():
        p = line.split()
        if len(p) == 2 and p[0] not in ("version", "default"):
            w[p[0]] = float(p[1])
    return w


def table8(band: str) -> list[str]:
    w = load_weights(band)
    tot = sum(w.values())
    variants = collections.Counter()
    for f in (ROOT / "decks" / "pool_field").glob("*.csv"):
        variants[f.stem.split("__")[0]] += 1
    agg = collections.defaultdict(float)
    for k, v in w.items():
        agg[k.split("__")[0]] += v
    out = [
        f"### 表 8 等价 —— 野区原型统计（`{band}` 档，{len(w)} 副卡组）\n",
        "| 原型 | 池内变体 | 野区占比 | 最大单副 | 该副占比 | 累计 |",
        "|---|---:|---:|---|---:|---:|",
    ]
    cum = 0.0
    for a, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        if 100 * v / tot < 0.5:
            continue
        best = max((k for k in w if k.split("__")[0] == a), key=lambda k: w[k])
        cum += v / tot
        out.append(
            f"| `{a}` | {variants[a]} | {100*v/tot:.1f}% | `{best.split('__')[1]}` "
            f"| {100*w[best]/tot:.1f}% | {100*cum:.1f}% |"
        )
    return out


def table11() -> list[str]:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "logs" / "gate_*.json"))):
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        if "score" not in d or "matches" not in d:
            continue
        rows.append((pathlib.Path(f).stem.replace("gate_", ""), d["score"], d["matches"],
                     d.get("ci_low"), d.get("ci_high"), d.get("band", "")))
    rows.sort(key=lambda r: -r[1])
    out = ["### 表 11 等价 —— 竞技场排名（gate 全池加权，按分数降序）\n",
           "| # | agent / 卡组 | 档 | 分数 | 95% CI | n |", "|---:|---|---|---:|---|---:|"]
    for i, (nm, sc, n, lo, hi, band) in enumerate(rows, 1):
        ci = f"[{lo:.3f}, {hi:.3f}]" if lo is not None and hi is not None else "-"
        out.append(f"| {i} | `{nm}` | {band} | **{sc:.4f}** | {ci} | {n} |")
    return out


def table12() -> list[str]:
    """多机平均的对位矩阵（cells = 各机均值）"""
    files = sorted(glob.glob(str(ROOT / "logs" / "matchup_cpu*.json")))
    if not files:
        return ["### 表 12 等价 —— 对位矩阵\n", "_无数据_"]
    acc: dict[str, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in files:
        d = json.load(open(f))
        for row, cols in d.items():
            if row.startswith("_"):
                continue
            for c, v in cols.items():
                acc[row][c].append(v["score"] if isinstance(v, dict) else v)
    out = ["### 表 12 等价 —— 对位矩阵（多机均值；行=我方，列=对手）\n"]
    for row, cols in acc.items():
        out.append(f"\n**`{row}`**（{len(acc)} 机平均）\n")
        out.append("| 对手 | 分数 | 机器数 |")
        out.append("|---|---:|---:|")
        for c, vs in sorted(cols.items(), key=lambda kv: -st.mean(kv[1])):
            out.append(f"| `{c}` | **{st.mean(vs):.3f}** | {len(vs)} |")
    return out


def table13_plan() -> list[str]:
    variants = collections.defaultdict(list)
    for f in (ROOT / "decks" / "pool_field").glob("*.csv"):
        variants[f.stem.split("__")[0]].append(f.stem)
    out = ["### 表 13a/b 等价 —— 原型内部循环赛（**尚未跑**，先列赛程）\n",
           "| 原型 | 变体数 | 两两配对数 | 变体 |", "|---|---:|---:|---|"]
    for a, vs in sorted(variants.items(), key=lambda kv: -len(kv[1]))[:8]:
        n = len(vs)
        names = ", ".join(sorted(v.split("__")[1] for v in vs)[:8])
        out.append(f"| `{a}` | {n} | {n*(n-1)//2} | {names}{'…' if n > 8 else ''} |")
    return out


def table17() -> list[str]:
    files = sorted(glob.glob(str(ROOT / "data" / "episodes_*.csv")))
    if not files:
        return ["### 表 17/18/19 等价 —— 真实天梯对局记录\n", "_尚无数据（跑 `tools/scrape_episodes.py`）_"]
    import csv as _csv
    rows = []
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            rows += list(_csv.DictReader(fh))
    out = [f"### 表 17/18/19 等价 —— 真实天梯对局（{len(rows)} 局，来自我们自己的提交）\n",
           "| 对手原型 | 场次 | 胜 | 胜率 | 野区占比 |", "|---|---:|---:|---:|---:|"]
    w = load_weights("ours")
    tot = sum(w.values())
    aw = collections.defaultdict(float)
    for k, v in w.items():
        aw[k.split("__")[0]] += v / tot
    agg = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        a = r["op_deck"].split("__")[0] if r["op_deck"] != "?" else "（不在池中）"
        agg[a][0] += int(r["won"])
        agg[a][1] += int(r["won"]) + int(r["loss"])
    for a, (win, n) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        share = f"{100*aw[a]:.1f}%" if a in aw else "-"
        out.append(f"| `{a}` | {n} | {win} | {win/max(n,1):.3f} | {share} |")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="ours")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    parts = ["# 与 1st-place writeup 对照的表格（自动生成）\n",
             f"> band = `{args.band}`\n"]
    parts += table8(args.band) + [""]
    parts += table11() + [""]
    parts += table12() + [""]
    parts += table13_plan() + [""]
    parts += table17() + [""]
    text = "\n".join(parts)
    if args.out:
        (ROOT / args.out).write_text(text, encoding="utf-8")
        print(f"写入 {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
