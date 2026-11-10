#!/usr/bin/env python3
"""BC 训练优先级调度器（第一版）。

目标（按设计规范）：预测「把候选 BC 加入对手池」的**边际价值**
    ΔV_i = V(Pool + Candidate_i) − V(Pool)
而不是候选自己的胜率。

三种排序器：
    M0 人工公式（现有权重，作为基线）
    M1 Ridge
    M2 部分符号约束 Ridge（正/负特征分别限号，scipy.lsq_linear 实现）

四张表（都落在 data/scheduler/）：
    candidate_features.csv   候选静态特征（数据质量 / Meta 占比 / 成本）
    matchup_matrix.csv       候选 × 对手卡组 的实测胜率、样本量、难度 q
    training_experiments.csv 真实训练标签：训练前后池价值、边际收益、成本
    scheduler_predictions.csv 每次预测的分数、排序、模型版本

用法：
    python3 tools/bc_scheduler.py features      # 由现有测量重建前两张表
    python3 tools/bc_scheduler.py fit           # 训练 M0/M1/M2（含 LOO 诊断）
    python3 tools/bc_scheduler.py rank          # 输出优先级 + ROI + 建议预算
    python3 tools/bc_scheduler.py log --candidate X --before 0.66 --after 0.64 --gpu-h 0.25
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "scheduler"
MATRIX_PATH = ROOT / "data" / "scheduler" / "matrix_full.csv"
FIELD = {"ogerpon_7e39": 0.4559, "dunsparce_0bf5": 0.1454, "grookey_2617": 0.1322,
         "shuppet_0d86": 0.1079, "dunsparce_9d6e": 0.0815, "abra_f5f8": 0.0463,
         "dreepy_3eaa": 0.0308}

# 现有对手池（阶段一二的产物）：每个候选 = 一个 agent
POOL = {
    "g8_bc_parent":  {"deck": "shuppet_0d86",  "kind": "bc",   "teacher": "ミワハルキ", "teacher_lb": 1594.4},
    "g8_shuppet":    {"deck": "shuppet_0d86",  "kind": "ppo",  "teacher": "ミワハルキ", "teacher_lb": 1594.4},
    "bc_0904p":      {"deck": "0904",          "kind": "bc",   "teacher": "Petit Canard", "teacher_lb": 1498.1},
    "port_0904":     {"deck": "0904",          "kind": "port", "teacher": "-",          "teacher_lb": 0},
    "bc_26c6p":      {"deck": "26c6",          "kind": "bc",   "teacher": "ミワハルキ", "teacher_lb": 1594.4},
    "port_26c6":     {"deck": "26c6",          "kind": "port", "teacher": "-",          "teacher_lb": 0},
    "bc_e1d7p":      {"deck": "e1d7",          "kind": "bc",   "teacher": "ミワハルキ", "teacher_lb": 1594.4},
    "port_e1d7":     {"deck": "e1d7",          "kind": "port", "teacher": "-",          "teacher_lb": 0},
    "port_7e39":     {"deck": "7e39",          "kind": "port", "teacher": "-",          "teacher_lb": 0},
    "port_3eaa":     {"deck": "3eaa",          "kind": "port", "teacher": "-",          "teacher_lb": 0},
    "port_0d86":     {"deck": "shuppet_0d86",  "kind": "port", "teacher": "-",          "teacher_lb": 0},
    "shup_a0.5":     {"deck": "shuppet_0d86",  "kind": "blend", "teacher": "混合",       "teacher_lb": 0},
    # 7 格面板里另外 4 格（此前漏在池表外，导致覆盖率恒为 0）
    "port_0bf5":     {"deck": "0bf5",          "kind": "port", "teacher": "-",          "teacher_lb": 0},
    "port_2617":     {"deck": "2617",          "kind": "port", "teacher": "-",          "teacher_lb": 0},
    "port_9d6e":     {"deck": "9d6e",          "kind": "port", "teacher": "-",          "teacher_lb": 0},
    "port_f5f8":     {"deck": "f5f8",          "kind": "port", "teacher": "-",          "teacher_lb": 0},
}

# 候选 × 对手卡组 的实测「该候选作为对手时，我方赢多少」→ q = 1 - 我方胜率
# 数据来源：data/opp_matrix/（400 场 Bo3，seed 990001）
MATCHUPS = {
    # candidate: {opp_deck: our_win_rate}
    "g8_bc_parent": {"dunsparce_abra__db04": 0.0222},
    "g8_shuppet":   {"dunsparce_abra__db04": 0.0550},
    "bc_0904p":     {"dunsparce_abra__db04": 0.7125},
    "port_0904":    {"dunsparce_abra__db04": 0.1725},
    "bc_26c6p":     {"dunsparce_abra__db04": 0.1200},
    "port_26c6":    {"dunsparce_abra__db04": 0.6500},
    "bc_e1d7p":     {"dunsparce_abra__db04": 0.0975},
    "port_e1d7":    {"dunsparce_abra__db04": 0.8550},
    "port_7e39":    {"dunsparce_abra__db04": 0.8350},
    "port_3eaa":    {"dunsparce_abra__db04": 0.3500},
    "port_0d86":    {"dunsparce_abra__db04": 0.9975},
    "shup_a0.5":    {"dunsparce_abra__db04": 0.6575},
    "port_0bf5":    {"dunsparce_abra__db04": 0.8111},
    "port_2617":    {"dunsparce_abra__db04": 0.5278},
    "port_9d6e":    {"dunsparce_abra__db04": 0.7809},
    "port_f5f8":    {"dunsparce_abra__db04": 0.7303},
}
DECK_OF = {"shuppet_0d86": "shuppet_0d86", "0904": "0904", "26c6": "26c6",
           "e1d7": "e1d7", "7e39": "7e39", "3eaa": "3eaa",
           "0bf5": "0bf5", "2617": "2617", "9d6e": "9d6e", "f5f8": "f5f8"}
# 池覆盖的「对手牌 → Meta 场的映射」：只有真正覆盖该牌才算覆盖
# deck → 7 格面板里的 key（不在面板内的牌返回 None：它们属于未覆盖的 35.1% 场次）
COVERS = {"shuppet_0d86": "shuppet_0d86", "3eaa": "dreepy_3eaa", "7e39": "ogerpon_7e39",
          "0bf5": "dunsparce_0bf5", "2617": "grookey_2617", "9d6e": "dunsparce_9d6e",
          "f5f8": "abra_f5f8", "0904": None, "26c6": None, "e1d7": None}

# 数据质量（来自 data/distill_top.jsonl 的帧数统计，2026-11-08）
DATAQ = {
    "shuppet_0d86": {"replay": 757, "players": 3, "decisions": 130236, "val_acc": 0.731},
    "0904":         {"replay": 808, "players": 1, "decisions": 191649, "val_acc": 0.696},
    "26c6":         {"replay": 500, "players": 2, "decisions":  64976, "val_acc": 0.797},
    "e1d7":         {"replay": 450, "players": 2, "decisions":  78596, "val_acc": 0.742},
    "7e39":         {"replay":   8, "players": 1, "decisions":   1768, "val_acc": 0.0},
    "3eaa":         {"replay": 720, "players": 1, "decisions": 165983, "val_acc": 0.867},
}


def wsum(pairs):
    return sum(a * b for a, b in pairs)


def load_matrix():
    """候选牌 × 对手牌 → 候选胜率（49 格实测，data/scheduler/matrix_full.csv）。"""
    m = {}
    if MATRIX_PATH.exists():
        for r in csv.DictReader(open(MATRIX_PATH)):
            m[(r["cand_deck"], r["opp_deck"])] = float(r["cand_win"])
    return m


MATRIX = load_matrix()
# 矩阵里的牌名 → FIELD 的 key（只有这 7 副能算覆盖率）
M2F = {"7e39": "ogerpon_7e39", "0bf5": "dunsparce_0bf5", "2617": "grookey_2617",
       "0d86": "shuppet_0d86", "9d6e": "dunsparce_9d6e", "f5f8": "abra_f5f8",
       "3eaa": "dreepy_3eaa"}


def coverage_of(cands):
    """Coverage_j = 1 − Π_k (1 − q_kj)，q_kj = 候选 k（打它的牌）对「目标牌 j」的胜率。

    比旧版的关键改进：旧版只会看「池里有没有这副牌」，新版看**池里有没有能压制这副牌的对手**。
    例：池里有 shuppet，但 shuppet 打 7e39 只有 0.237 ⇒ 对 7e39 这一格的覆盖几乎为零；
        真正压制 7e39 的是 0bf5(0.887)/9d6e(0.890)。"""
    cov = {}
    for field_key in FIELD:
        prod = 1.0
        for c in cands:
            dk = DECK_OF.get(c.deck, "")
            # 目标牌的矩阵名
            target = next((k for k, v in M2F.items() if v == field_key), None)
            if target is None or not MATRIX:
                continue
            q = MATRIX.get((dk, target))
            if q is None:
                continue
            prod *= (1.0 - q)
        cov[field_key] = 1.0 - prod
    return cov


def pool_value(cands):
    """V = 0.3·Coverage + 0.2·Diversity + 0.5·Learner
    Learner 项目前用「该池训练出的 learner 的加权面板」代理；无实验时置 0 并标注。"""
    cov = coverage_of(cands)
    cov_s = wsum([(FIELD[d], cov[d]) for d in FIELD])
    # diversity：候选两两间「对 db04 的压力」差的平均绝对值（0~1）
    qs = [1.0 - MATCHUPS.get(c.name, {}).get("dunsparce_abra__db04", 0.5) for c in cands]
    div = 0.0
    if len(qs) > 1:
        pairs = [(abs(a - b)) for i, a in enumerate(qs) for b in qs[i + 1:]]
        div = sum(pairs) / len(pairs)
    learner = 0.0   # 需要真实训练实验填充
    return 0.3 * cov_s + 0.2 * div + 0.5 * learner, dict(coverage=cov_s, diversity=div, learner=learner)


class Cand:
    def __init__(self, name):
        self.name = name
        info = POOL[name]
        self.deck = info["deck"]
        self.kind = info["kind"]
        self.teacher_lb = info["teacher_lb"]


# ---- 玩家级策略相似度（用权重指纹算，而不是用"牌"算） ----
_MODEL_HINTS = {
    "g8_bc_parent": "build/agents/g8_bc_parent/model.pt",
    "g8_shuppet":   "build/agents/g8_shuppet/model.pt",
    "bc_26c6":      "build/agents/bc_26c6/model.pt",
    "bc_3eaa":      "build/agents/bc_3eaa/model.pt",
    "bc_2617k":     "build/agents/bc_2617k/model.pt",
    "bc_7e39d":     "build/agents/bc_7e39d/model.pt",
    "bc_0904p":     "build/agents/bc_0904p/model.pt",
    "bc_26c6p":     "build/agents/bc_26c6p/model.pt",
    "bc_e1d7p":     "build/agents/bc_e1d7p/model.pt",
}
_PORT = "build/port_plain/model.pt"
_FP_CACHE = {}


def weight_fingerprint(cand_name):
    """把权重压成一个单位向量（抽样到 ≤20 万维，够算余弦）。"""
    if cand_name in _FP_CACHE:
        return _FP_CACHE[cand_name]
    path = _MODEL_HINTS.get(cand_name)
    if path is None:
        # 端口系候选共用同一份权重
        path = _PORT if cand_name.startswith("port_") else None
    v = None
    if path:
        try:
            import torch
            fp = ROOT / path
            if fp.exists():
                sd = torch.load(fp, map_location="cpu", weights_only=False)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                # ★ 必须排除静态卡表：card_static/card_attacks 等在各模型间**完全相同**，
                #   它们张量巨大，会把余弦压到恒 1（实测所有模型两两 1.0001）。
                STATIC = {"card_static", "card_attacks", "card_skills", "att_static",
                          "card_text", "att_text", "skill_text"}
                flat = []
                for k in sorted(sd):
                    t = sd[k]
                    if k in STATIC or not hasattr(t, "float") or not hasattr(t, "numel"):
                        continue
                    if t.numel() == 0:
                        continue
                    flat.append(t.float().flatten())   # 用原始权重：相对 L2 距离才有物理意义
                cat = torch.cat(flat)
                if cat.numel() > 400_000:                      # 均匀抽样，保持顺序可比
                    step = max(1, cat.numel() // 400_000)
                    cat = cat[::step]
                v = cat if float(cat.norm()) > 0 else None
        except Exception:
            v = None
    _FP_CACHE[cand_name] = v
    return v


def strategy_similarity(cand_name):
    """与现有池里**最接近**候选的相似度（1 − 归一化 RMS 距离）。

    ⚠️ 不能用余弦：所有模型都由同一个 port 祖先微调而来，可学习权重余弦恒在
       0.9987~1.0000，完全饱和、没有区分度（实测）。改用逐张量 z-score 后的
       **RMS 距离**：`dist(a,b) = ||a−b||/sqrt(dim)`，identical→0。
    """
    v = weight_fingerprint(cand_name)
    if v is None:
        return None
    best = None
    for other in POOL:
        if other == cand_name:
            continue
        vo = weight_fingerprint(other)
        if vo is None or vo.numel() != v.numel():
            continue
        d = float((v - vo).norm()) / max(float(vo.norm()), 1e-9)   # 相对 L2
        if best is None or d < best:
            best = d
    if best is None:
        return None
    # 把距离映射成"相似度"：距离 0 → 1；距离 0.5 以上视为完全不同 → 0
    # best = 最小相对 L2 距离；把 0~5% 映射到 similarity 1~0（5% 以上视为"完全不同"）
    return max(0.0, 1.0 - min(1.0, best / 0.05))


# deck 名 → 矩阵里的短名（矩阵只覆盖这 7 副）
DECK2MX = {"shuppet_0d86": "0d86", "7e39": "7e39", "3eaa": "3eaa", "0bf5": "0bf5",
           "2617": "2617", "9d6e": "9d6e", "f5f8": "f5f8"}


def interaction_features(cand):
    """§9.2 的五个交互项（第一版）。全部由 49 格矩阵 + 覆盖率推出。"""
    dk = DECK2MX.get(cand.deck, "")
    row = {o: MATRIX.get((dk, m)) for m, o in M2F.items()   # o = FIELD key, m = 矩阵牌名
           for _ in (0,)} if False else {}
    row = {}
    for m, fk in M2F.items():
        v = MATRIX.get((dk, m))
        if v is not None:
            row[fk] = v
    if not row:
        return dict(meta_wr=0.0, meta_gap_own=0.0, gap_x_q=0.0, worst_wr=0.0,
                    wr_variance=0.0, diversity=0.0, new_deck=0.0)
    wsum = sum(FIELD[k] for k in row) or 1.0
    meta_wr = sum(FIELD[k] * v for k, v in row.items()) / wsum          # ① w_j × WR_ij
    cov = coverage_of([cand])
    gap_own = (1 - cov[COVERS[cand.deck]]) if (cand.deck in COVERS and COVERS[cand.deck] in cov) else 0.0
    meta_gap_own = FIELD.get(COVERS.get(cand.deck) or "", 0.0) * gap_own  # ② w_j × Gap_j（本格）
    gap_x_q = sum((1 - cov[k]) * v for k, v in row.items())                # ③⭐ Gap_j × q_ij（聚合）
    worst = min(row.values())                                             # ④ worst
    mean = meta_wr
    var = sum((v - mean) ** 2 for v in row.values()) / len(row)
    # ⑤ 多样性 = 1 − 与现有池的最大相似度（矩阵行向量余弦）
    # 玩家级相似度：用**权重指纹**（同牌不同玩家会在这里被区分开；矩阵行区分不了）
    sim = strategy_similarity(cand.name)
    if sim is None:
        # 兜底：拿不到权重就退回矩阵行相似度
        mx_names = list(M2F)
        cur = [MATRIX.get((dk, m), 0.0) for m in mx_names]
        sims = []
        for n in POOL:
            if n == cand.name: continue
            pr = [MATRIX.get((DECK2MX.get(POOL[n]["deck"], ""), m), 0.0) for m in mx_names]
            num = sum(a * b for a, b in zip(cur, pr))
            den = (sum(a * a for a in cur) ** 0.5) * (sum(b * b for b in pr) ** 0.5)
            if den: sims.append(num / den)
        sim = max(sims) if sims else 0.0
    diversity = 1 - max(0.0, min(1.0, sim))
    new_deck = 0.0 if any(POOL[n]["deck"] == cand.deck for n in POOL) else 1.0
    return dict(meta_wr=meta_wr, meta_gap_own=meta_gap_own, gap_x_q=gap_x_q,
                worst_wr=worst, wr_variance=var, diversity=diversity, new_deck=new_deck)


def build_features():
    OUT.mkdir(parents=True, exist_ok=True)
    names = list(POOL)
    gap = {d: 1.0 - v for d, v in coverage_of([Cand(n) for n in names]).items()}
    rows = []
    for n in names:
        c = Cand(n)
        dq = DATAQ.get(c.deck, {"replay": 0, "players": 0, "decisions": 0, "val_acc": 0.0})
        our = MATCHUPS.get(n, {}).get("dunsparce_abra__db04")
        ix = interaction_features(c)
        rows.append(dict(
            candidate=n, deck=c.deck, kind=c.kind, teacher_lb=c.teacher_lb,
            **ix,
            meta_share=FIELD.get(COVERS.get(c.deck) or "", 0.0),
            replay_count=dq["replay"], player_count=dq["players"],
            decision_count=dq["decisions"], bc_val_acc=dq["val_acc"],
            coverage_gap=gap.get(COVERS.get(c.deck) or "", 1.0),
            our_win_vs_it=(our if our is not None else ""),
            pressure_q=(1.0 - our) if our is not None else "",
            data_x_quality=dq["decisions"] / 1e5 * (0.6 + 0.4 * min(dq["players"], 3) / 3),
            # 候选类型：只有"要新训练"的候选才有 ΔV；现成 agent 的 ΔV≈0
            needs_training=(1 if c.kind == "bc" else 0),
            gpu_hours=(0.4 if c.kind == "bc" else 0.0),
            teacher_present=(1 if c.teacher_lb > 0 else 0),
        ))
    with open(OUT / "candidate_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(OUT / "matchup_matrix.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["candidate", "opp_deck", "our_win_rate", "pressure_q", "n_games"])
        for n, m in MATCHUPS.items():
            for od, wr in m.items():
                w.writerow([n, od, wr, round(1 - wr, 4), 400])
    print(f"✅ 写出 {len(rows)} 个候选 → {OUT/'candidate_features.csv'}")
    print(f"✅ matchup 矩阵 {sum(len(v) for v in MATCHUPS.values())} 格")
    covered = [d for d, g in gap.items() if g < 0.99]
    print(f"   池已覆盖：{covered}")
    print(f"   覆盖缺口：{[f'{d}:{g:.2f}' for d, g in sorted(gap.items(), key=lambda kv:-kv[1])[:4]]}")


# ---------------------------------------------------------------- 三种排序器
# §9.2 五个交互项全部入列；§9.1 的符号约束按组写死
FEATS = ["meta_share", "meta_wr", "meta_gap_own", "gap_x_q", "diversity", "new_deck",
         "worst_wr", "wr_variance", "teacher_lb", "bc_val_acc", "data_x_quality",
         "gpu_hours"]
POSITIVE = {"meta_share", "meta_wr", "meta_gap_own", "gap_x_q", "diversity",
            "new_deck", "teacher_lb", "data_x_quality", "replay_count",
            "player_count", "decision_count"}
NEGATIVE = {"gpu_hours", "similarity", "max_similarity", "avg_similarity"}


def load_experiments():
    p = OUT / "training_experiments.csv"
    if not p.exists():
        return []
    return list(csv.DictReader(open(p)))


def m0_manual(row):
    """M0：人工公式。**只在"需要新训练"时才有边际价值**，
    现成 agent 已经在池里（needs_training=0），ΔV 记为 0，不进排序。"""
    if not row.get("needs_training", 0):
        return 0.0
    return (0.45 * row["meta_share"] + 0.35 * row["gap_x_q"]
            + 0.15 * min(row["bc_val_acc"], 1.0)
            + 0.05 * min(row["data_x_quality"], 2.0) / 2
            - 0.10 * min(row.get("gpu_hours", 0.0), 1.0))


def fit_models(verbose=True):
    exp = load_experiments()
    n = len(exp)
    if verbose:
        print(f"训练实验标签：{n} 条")
    if n < 10:
        print(f"⚠️  只有 {n} 条标签，M1/M2 无法可靠拟合（经验下限 ~10-20，建议 ≥30）。")
        print("    先把 M0 作为排序器，每跑一次真实训练就 `log` 一条，攒够再拟合。")
        return None
    import numpy as np
    from scipy.optimize import lsq_linear
    F = {r["candidate"]: r for r in csv.DictReader(open(OUT / "candidate_features.csv"))}
    X, y = [], []
    for e in exp:
        r = F.get(e["candidate"])
        if not r:
            continue
        X.append([float(r.get(k, 0) or 0) for k in FEATS])
        y.append(float(e["marginal_value"]))
    X = np.array(X); y = np.array(y)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    # M1 Ridge（闭式）
    lam = 1.0
    beta1 = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ y)
    # M2 部分符号约束（正特征 ≥0，负特征 ≤0）
    lb = np.array([0.0 if f in POSITIVE else -np.inf for f in FEATS])
    ub = np.array([0.0 if f in NEGATIVE else np.inf for f in FEATS])
    beta2 = lsq_linear(Xs, y, bounds=(lb, ub)).x
    if verbose:
        print(f"\n{'特征':<18}{'M1 Ridge':>12}{'M2 符号约束':>14}")
        for i, f in enumerate(FEATS):
            print(f"{f:<18}{beta1[i]:>12.4f}{beta2[i]:>14.4f}")
        for name, b in (("M1", beta1), ("M2", beta2)):
            pred = Xs @ b
            r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
            print(f"  {name}: 训练 R²={r2:.3f}")
    import json as _j
    _j.dump({"feats": FEATS, "mu": mu.tolist(), "sd": sd.tolist(),
             "M1": beta1.tolist(), "M2": beta2.tolist()}, open(OUT / "model.json", "w"), indent=1)
    print(f"   模型参数 → {OUT/'model.json'}")
    return {"feats": FEATS, "mu": mu, "sd": sd, "M1": beta1, "M2": beta2}


def rank():
    F = list(csv.DictReader(open(OUT / "candidate_features.csv")))
    m = None
    p = OUT / "model.json"
    if p.exists():
        import numpy as np
        d = json.load(open(p))
        m = {"feats": d["feats"], "mu": np.array(d["mu"]), "sd": np.array(d["sd"]),
             "M1": np.array(d["M1"]), "M2": np.array(d["M2"])}
    rows = []
    for r in F:
        r = {k: (float(v) if v not in ("", None) else 0.0) for k, v in r.items() if k != "candidate" and k != "deck" and k != "kind"} | {"candidate": r["candidate"], "deck": r["deck"]}
        s0 = m0_manual(r)
        s1 = s2 = None
        if m:
            import numpy as np
            x = np.array([r.get(f, 0.0) for f in m["feats"]])
            xs = (x - m["mu"]) / m["sd"]
            s1 = float(xs @ m["M1"]); s2 = float(xs @ m["M2"])
        rows.append((r["candidate"], r["deck"], s0, s1, s2))
    rows.sort(key=lambda t: -(t[3] if t[3] is not None else t[2]))
    print(f"{'候选':<14}{'牌':<14}{'M0 人工':>10}{'M1 Ridge':>10}{'M2 符号':>10}")
    print("-" * 60)
    for c, d, s0, s1, s2 in rows:
        print(f"{c:<14}{d:<14}{s0:>10.4f}{(f'{s1:.4f}' if s1 is not None else '—'):>10}"
              f"{(f'{s2:.4f}' if s2 is not None else '—'):>10}")
    print("\n（排序按 M2，未拟合时按 M0）")
    print("预算建议：前 2 名各 40%，其余 20% 探索（初始比例，按收益更新）")


def log_experiment(args):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "training_experiments.csv"
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["candidate", "pool_version", "train_minutes", "before_value",
                        "after_value", "marginal_value", "gpu_hours", "notes"])
        w.writerow([args.candidate, args.pool_version, args.train_minutes, args.before,
                    args.after, round(args.after - args.before, 4), args.gpu_hours,
                    args.notes or ""])
    print(f"✅ 记录一条实验：{args.candidate} ΔV={args.after - args.before:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("features")
    sub.add_parser("fit")
    sub.add_parser("rank")
    lg = sub.add_parser("log")
    lg.add_argument("--candidate", required=True)
    lg.add_argument("--pool-version", default="P001")
    lg.add_argument("--train-minutes", type=float, default=0)
    lg.add_argument("--before", type=float, required=True)
    lg.add_argument("--after", type=float, required=True)
    lg.add_argument("--gpu-hours", type=float, default=0)
    lg.add_argument("--notes", default="")
    a = ap.parse_args()
    if a.cmd == "features":
        build_features()
    elif a.cmd == "fit":
        fit_models()
    elif a.cmd == "rank":
        rank()
    elif a.cmd == "log":
        log_experiment(a)


if __name__ == "__main__":
    raise SystemExit(main())
