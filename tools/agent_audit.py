#!/usr/bin/env python3
"""按格子选冠军：一个 agent 只有在**某一格**上是最好（或接近最好）才值得留。

判据（不是"有多少 agent"，而是"每格谁最强"）：
    对每一格 j，取所有候选里 max 的胜率；某 agent 只要在至少一格上
    与冠军的差距 ≤ MARGIN（默认 3pp），就认为它有不可替代的价值。

数据来源：
  * data/scheduler/matrix_full.csv   —— 7 个端口对手 × 7 格（49 格）
  * data/opp_matrix/panel_*.json     —— 各 learner 的 7 对手面板
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data" / "scheduler" / "matrix_full.csv"
PANELS = ROOT / "data" / "opp_matrix"

# 面板 JSON 里的对手名 → 统一的格名
OPP2CELL = {
    "ogerpon_7e39": "7e39", "dunsparce_0bf5": "0bf5", "grookey_2617": "2617",
    "shuppet_0d86": "shuppet", "dunsparce_9d6e": "9d6e", "abra_f5f8": "f5f8",
    "dreepy_3eaa": "3eaa",
}
CELLS = ["7e39", "0bf5", "2617", "shuppet", "9d6e", "f5f8", "3eaa"]
FIELD = {"7e39": 0.4559, "0bf5": 0.1454, "2617": 0.1322, "shuppet": 0.1079,
         "9d6e": 0.0815, "f5f8": 0.0463, "3eaa": 0.0308}


def load_agents():
    """返回 {agent_label: {cell: win_rate}}"""
    out = {}
    # ① 端口对手（矩阵行）
    rows = {}
    if MATRIX.exists():
        for r in csv.DictReader(open(MATRIX)):
            rows.setdefault(r["cand_deck"], {})[r["opp_deck"]] = float(r["cand_win"])
    for deck, cells in rows.items():
        out[f"port@{deck}"] = {OPP2CELL.get(k, k): v for k, v in cells.items()}
    # ② 面板结果（含 g6_db04 与各 learner）
    for p in sorted(PANELS.glob("panel_*.json")):
        tag = p.stem.replace("panel_", "")
        try:
            j = json.load(open(p))
        except Exception:
            continue
        per = {}
        for opp, v in (j.get("per_opp") or {}).items():
            pm = v.get("per_match")
            val = sum(pm.values()) / len(pm) if isinstance(pm, dict) else pm
            per[OPP2CELL.get(opp, opp)] = val
        if per:
            out[tag] = per
    return out


# 审计标签 → 调度器里的候选名（权重指纹只认后者）
LABEL2CAND = {
    "shbc": "g8_bc_parent", "db04": "g6_db04", "ph3": "ph3_db04_v1",
    "d0904": "port_0904", "d3eaa": "port_3eaa", "ph3_2617": "ph3_2617_v1",
    "ph3_9d6e": "ph3_9d6e_v1",
}


def _cand(label):
    if label in LABEL2CAND:
        return LABEL2CAND[label]
    if label.startswith("port@"):
        return "port_" + label.split("@", 1)[1]
    return label


def agent_similarity(n1, n2):
    """两个 agent 的策略相似度（权重指纹的相对 L2，映射到 0~1）。拿不到权重返回 None。"""
    try:
        sys.path.insert(0, str(ROOT / "tools"))
        import bc_scheduler as B
        v1, v2 = B.weight_fingerprint(_cand(n1)), B.weight_fingerprint(_cand(n2))
        if v1 is None or v2 is None or v1.numel() != v2.numel():
            return None
        d = float((v1 - v2).norm()) / max(float(v2.norm()), 1e-9)
        return max(0.0, 1.0 - min(1.0, d / 0.05))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.03,
                    help="与冠军的允许差距（默认 3pp）：多接近才算“性能接近”")
    ap.add_argument("--keep-if-lb-above", type=float, default=1010.0,
                    help="判据③：真实天梯分 ≥ 此值则无条件保留（默认 1010 = 最高分-50）")
    ap.add_argument("--sim-threshold", type=float, default=0.95,
                    help="相似度低于此值视为策略不同（默认 0.95）")
    a = ap.parse_args()
    agents = load_agents()
    cells = [c for c in CELLS if any(c in v for v in agents.values())]
    # 打印矩阵
    print(f"{'agent':<18}" + "".join(f"{c:>9}" for c in cells) + f"{'加权':>9}")
    print("-" * (18 + 9 * len(cells) + 9))
    for n, per in sorted(agents.items()):
        vals = [per.get(c) for c in cells]
        wsum = sum(FIELD[c] * v for c, v in zip(cells, vals) if v is not None)
        wtot = sum(FIELD[c] for c, v in zip(cells, vals) if v is not None)
        w = wsum / wtot if wtot else 0
        print(f"{n:<18}" + "".join(f"{v:>9.3f}" if v is not None else f"{'-':>9}" for v in vals) + f"{w:>9.3f}")
    # 每格冠军
    print(f"\n=== 每格冠军（允许差 {a.margin*100:.0f}pp）===")
    champs = {}
    for c in cells:
        cand = [(n, per[c]) for n, per in agents.items() if c in per]
        if not cand:
            continue
        best = max(cand, key=lambda t: t[1])
        near = [n for n, v in cand if best[1] - v <= a.margin]
        champs[c] = (best[0], best[1], near)
        print(f"  {c:<8} 冠军 {best[0]:<18} {best[1]:.3f}   接近的: {', '.join(near)}")
    # 复合判据：① 格子冠军（含 margin）② 性能接近冠军 且 与冠军策略不同
    keep = set()
    reasons = {}
    for c, (champ, best, near) in champs.items():
        keep.update(near)
        for n in near:
            reasons.setdefault(n, f"{c} 格冠军/接近")
    all_agents = set(agents)
    print(f"\n=== 判据②：性能接近 + 策略不同（相似度 < {a.sim_threshold}）===")
    for c, (champ, best, _) in champs.items():
        cand = [(n, per[c]) for n, per in agents.items() if c in per and n != champ]
        for n, v in cand:
            if best - v > a.margin:      # 性能不够接近
                continue
            sim = agent_similarity(n, champ)
            if sim is None:
                continue
            tag = "✅ 保留" if sim < a.sim_threshold else "✗ 太像冠军"
            print(f"  {c:<8} {n:<18} {v:.3f} vs 冠军 {champ}({best:.3f})  相似度 {sim:.3f}  {tag}")
            if sim < a.sim_threshold:
                keep.add(n); reasons.setdefault(n, f"{c} 格与冠军不同策略(相似度{sim:.2f})")
    # 判据③：历史天梯分
    LB = {"db04": 1060.3, "port_db04": 1042.3, "ph3": 600.0, "shbc": 779.6}
    for n, v in LB.items():
        if n in all_agents and v >= a.keep_if_lb_above:
            keep.add(n); reasons.setdefault(n, f"判据③ 天梯 {v:.1f}")
    print(f"\n=== 最终保留（{len(keep)}/{len(all_agents)}）===")
    for n in sorted(keep):
        print(f"  ✓ {n:<20} {reasons.get(n,'')}")
    print(f"\n=== 可退休（{len(all_agents-keep)}）===")
    print("  " + ", ".join(sorted(all_agents - keep)))


if __name__ == "__main__":
    raise SystemExit(main())
