#!/usr/bin/env python3
"""agent 资产分级与归档（Tier A/B/C），三条判据固化在此。

判据（与 docs/pool_plan.md §10 一致）：
  ① 角色必需 —— 面板 7 对手（覆盖评估基线）+ 当前代工作模型
  ② 代码引用 —— 被 tools/・ptcg_league/・decks/ 的**活代码**引用（logs/ 与 docs 不算）
  ③ 真实天梯 —— data/lb_ours.json 里记录的我们自己提交过的分数
     · ≥ KEEP_LB(1010) → 无条件 Tier A（最高分 - 50）
     · 其余有分数 → 至少 Tier B（模型可复原但已被超越）

分级：
  Tier A  留在 build/agents/（评估与训练直接取用）
  Tier B  移到 models/archive/（被分析工具引用 / 有历史天梯分 / 近冠军）
  Tier C  移到 models/archive/（无引用、无分数、已被取代）

用法：
    python3 tools/agent_tier.py            # 只打印分级
    python3 tools/agent_tier.py --apply    # 执行归档（幂等：已归档的不再移动）
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENTS = ROOT / "build" / "agents"
ARCHIVE = ROOT / "models" / "archive"
INDEX = ARCHIVE / "index.tsv"
LB_FILE = ROOT / "data" / "lb_ours.json"

# 我们自己提交过的真实天梯分（一名 agent = 一次提交）
LB_FALLBACK = {
    "g6_db04": 1060.3, "port_db04": 1042.3, "g6_9d6e": 992.1,
    "g6_7e39": 880.9, "g14a_e1d7": 875.5, "g8_bc_parent": 779.6,
}

# ① 角色必需
PANEL = {"port_7e39", "opp_dunsparce_0bf5", "c_grookey", "opp_shuppet_human",
         "port_on_9d6e", "c_abra", "opp_3eaa"}
WORKING = {"g6_db04", "port_db04", "g8_bc_parent", "g8_shuppet", "bc_3eaa", "bc_26c6"}

# ② 扫描范围（活代码；logs/、docs、data/ 的元数据不算引用）
SCAN_DIRS = ["tools", "ptcg_league", "decks", "framework", "bots", "cg"]
SCAN_EXT = {".sh", ".py", ".tsv", ".yaml"}

KEEP_LB = 1010.0


def load_lb() -> dict:
    if LB_FILE.exists():
        try:
            return {k: float(v) for k, v in json.loads(LB_FILE.read_text()).items()}
        except Exception:
            pass
    return dict(LB_FALLBACK)


def scan_refs(names: list[str]) -> dict[str, list[str]]:
    files: list[str] = []
    for d in SCAN_DIRS:
        p = ROOT / d
        if not p.exists():
            continue
        files += [str(f.relative_to(ROOT)) for f in p.rglob("*")
                  if f.is_file() and f.suffix in SCAN_EXT and "/logs/" not in str(f)]
    files += [f.name for f in ROOT.glob("*.sh")] + [f.name for f in ROOT.glob("*.py")]
    texts = {}
    for f in files:
        p = ROOT / f
        try:
            if p.stat().st_size < 3_000_000:
                texts[f] = p.read_text(errors="ignore")
        except Exception:
            pass
    refs = {}
    for n in names:
        pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(n) + r"(?![A-Za-z0-9_])")
        refs[n] = sorted(f for f, t in texts.items() if pat.search(t))
    return refs


def classify() -> list[dict]:
    names = sorted(d.name for d in AGENTS.iterdir() if d.is_dir())
    for d in ARCHIVE.iterdir() if ARCHIVE.exists() else []:
        if d.is_dir() and d.name not in names:
            names.append(d.name)
    names.sort()
    lb = load_lb()
    refs = scan_refs(names)
    rows = []
    for n in names:
        loc = "build/agents" if (AGENTS / n).exists() else (
            "models/archive" if (ARCHIVE / n).exists() else "-")
        sz = 0
        base = ROOT / loc / n
        if base.exists():
            sz = sum(f.stat().st_size for f in base.rglob("*") if f.is_file())
        score = lb.get(n)
        if n in PANEL:
            tier, why = "A", "① 面板对手（覆盖评估基线）"
        elif n in WORKING:
            tier, why = "A", "① 当前代工作模型"
        elif score is not None and score >= KEEP_LB:
            tier, why = "A", f"③ 天梯 {score:.1f} ≥ {KEEP_LB:.0f}"
        elif score is not None:
            tier, why = "B", f"③ 历史天梯 {score:.1f}（已被超越，dist/ 可复原）"
        elif refs[n]:
            tier = "B"
            why = "② 活代码引用：" + ", ".join(sorted({f.split("/")[1]
                  if "/" in f else f for f in refs[n]})[:3])
        else:
            tier, why = "C", "无引用 / 无天梯分 / 已被取代"
        rows.append(dict(name=n, tier=tier, why=why, loc=loc, size=sz,
                         lb=score, refs=refs[n]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="执行归档移动")
    a = ap.parse_args()
    rows = classify()
    print(f"{'tier':<5}{'agent':<24}{'MB':>7}{'lb':>8}  {'位置':<15}理由")
    print("-" * 100)
    for r in rows:
        lbs = f"{r['lb']:.1f}" if r["lb"] is not None else "-"
        print(f"{r['tier']:<5}{r['name']:<24}{r['size']/1e6:>7.1f}{lbs:>8}  {r['loc']:<15}{r['why']}")
    agg = {}
    for r in rows:
        k = (r["tier"], r["loc"])
        agg[k] = agg.get(k, [0, 0])
        agg[k][0] += 1
        agg[k][1] += r["size"]
    print()
    for (t, loc), (c, s) in sorted(agg.items()):
        print(f"  Tier {t} @ {loc:<14} {c:>3} 个  {s/1e6:>6.0f} MB")
    moved = [r for r in rows if r["tier"] in ("B", "C") and r["loc"] == "build/agents"]
    if moved:
        print(f"\n待归档 {len(moved)} 个，共 {sum(r['size'] for r in moved)/1e6:.0f} MB")
    if not a.apply:
        print("\n（只打印。执行归档加 --apply）")
        return 0
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    n_moved = 0
    for r in moved:
        src, dst = AGENTS / r["name"], ARCHIVE / r["name"]
        if dst.exists():
            print(f"  ! 目标已存在，跳过 {r['name']}")
            continue
        shutil.move(str(src), str(dst))
        n_moved += 1
    with INDEX.open("w") as f:
        f.write("name\ttier\tloc\tsize_mb\tlb\twhy\n")
        for r in rows:
            loc = "models/archive" if (ARCHIVE / r["name"]).exists() or r["tier"] != "A" else "build/agents"
            lbs = f"{r['lb']:.1f}" if r["lb"] is not None else ""
            f.write(f"{r['name']}\t{r['tier']}\t{loc}\t{r['size']/1e6:.1f}\t{lbs}\t{r['why']}\n")
    print(f"\n✓ 归档 {n_moved} 个 → {ARCHIVE.relative_to(ROOT)}；索引 {INDEX.relative_to(ROOT)}")
    print(f"  build/agents 剩余 {len(list(AGENTS.iterdir()))} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
