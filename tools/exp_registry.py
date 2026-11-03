#!/usr/bin/env python3
"""**实验注册表**：每次评估自动追加一行，用来回答"这次提升来自换牌、换模型，还是两者的交互"。

字段（2026-11-02 与用户对齐，缺一个就没法归因）：
    policy_id            策略标识（port / g6_db04 / bc_abra76k …）
    policy_sha256        权重文件哈希（防止"同名不同权重"）
    deck_id              精确卡组名
    deck_hash            卡组 60 张的哈希（与 distill_index 的 8 位同源）
    training_decks       该策略训练时学习者的卡组（空=未训练/通用）
    training_opponents   该策略训练时的 bank 列表
    opponent_policy_id   对手策略
    opponent_deck_id     对手卡组
    matchup_win_rate     本格胜率
    ci_low / ci_high     95% CI
    n_matches            场数（Bo3 match 为单位）
    errors               异常数
    evaluation_seed      种子基准（**不同种子不可配对**）
    seat_distribution    座位分布（可解析则填，否则 n/a）
    arena                这个格子属于哪一类：policy_axis | deck_axis | cross
                          —— policy_axis = 同卡组不同策略（唯一能归因到策略的格子）
    note

用法：
    # 从一次 eval_pair 的 json 追加一行
    ./tools/py tools/exp_registry.py --json logs/xdeck/g6_on_7e39.json \
        --policy g6_db04 --opponent-policy port \
        --arena policy_axis --seed 900000

    # 看汇总（按 arena 分类，避免把"牌好"误读成"策略好"）
    ./tools/py tools/exp_registry.py --report
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TSV = os.path.join(ROOT, "data", "experiments_registry.tsv")

FIELDS = ["when", "arena", "policy_id", "policy_sha256", "deck_id", "deck_hash",
          "training_decks", "training_opponents", "opponent_policy_id",
          "opponent_deck_id", "matchup_win_rate", "ci_low", "ci_high",
          "n_matches", "errors", "evaluation_seed", "seat_distribution", "note"]

MODELS = {  # policy_id -> 权重路径（用于算 sha 和训练配方）
    "port": "build/port_plain/model.pt",
    "g5_db04": "models/ppo/g5_db04.pt",
    "g6_db04": "models/ppo/g6_db04.pt",
    "g9_db04": "models/ppo/g9_db04.pt",
    "g10_db04_banks": "models/ppo/g10_db04_banks.pt",
    "g7a_db04_shuppet": "models/ppo/g7a_db04_shuppet.pt",
    "g8_shuppet_bc": "models/ppo/g8_shuppet_bc.pt",
    "g8_bc_parent": "bc_pool/deck_0d86_.bin",
    "bc_abra2": "logs/abra2&dreepy2/abra2.pt",
    "bc_abra76k": "logs/abra76k/abra76k.pt",
    "bc_grookey3": "logs/grookey3/grookey3.pt",
    "bc_dreepy2": "logs/abra2&dreepy2/dreepy2.pt",
}


def sha_of(path, n=16):
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(p):
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:n]


def deck_hash_of(deck_id):
    for base in ("decks/pool_field", "port_repo/decks/pool_field"):
        p = os.path.join(ROOT, base, deck_id + ".csv")
        if os.path.exists(p):
            ids = sorted(int(l) for l in open(p) if l.strip())
            if len(ids) == 60:
                return hashlib.sha256(",".join(str(x) for x in ids).encode()).hexdigest()[:8]
    return ""


def recipe_of(policy_id):
    """从 models/ppo/<x>.json 读训练配方。"""
    for cand in (policy_id, policy_id.replace("_db04", "")):
        p = os.path.join(ROOT, "models", "ppo", cand + ".json")
        if os.path.exists(p):
            try:
                m = json.load(open(p))
            except Exception:
                continue
            rec = m.get("recipe", {})
            return (rec.get("anchor") or "",
                    ",".join(os.path.basename(str(b)) for b in (rec.get("banks") or [])))
    return "", ""


def append(a):
    d = json.load(open(a.json))
    o = d["overall"]
    cand_path = d.get("cand") or ""
    pol_deck = ""
    # 从 cand 目录推卡组名（stage_ckpt 会把卡组写成 deck.csv）
    cand = cand_path
    if cand and not os.path.isabs(cand):
        cand = os.path.join(ROOT, cand)
    dj = os.path.join(cand, "deck.csv") if cand else ""
    if dj and os.path.exists(dj):
        ids = sorted(int(l) for l in open(dj) if l.strip())
        if len(ids) == 60:
            pol_deck = hashlib.sha256(",".join(str(x) for x in ids).encode()).hexdigest()[:8]
    td, to = recipe_of(a.policy)
    row = dict(
        when=__import__("time").strftime("%Y-%m-%d %H:%M"),
        arena=a.arena,
        policy_id=a.policy,
        policy_sha256=sha_of(MODELS.get(a.policy, "")),
        deck_id=a.deck,
        deck_hash=deck_hash_of(a.deck) or pol_deck,
        training_decks=td,
        training_opponents=to,
        opponent_policy_id=a.opponent_policy,
        opponent_deck_id=a.opponent_deck,
        matchup_win_rate=f"{o['score']:.4f}",
        ci_low=f"{o['ci_low']:.4f}",
        ci_high=f"{o['ci_high']:.4f}",
        n_matches=o["n"],
        errors=o["errors"],
        evaluation_seed=a.seed,
        seat_distribution=o.get("seat_distribution") or "n/a",
        note=a.note,
    )
    newfile = not os.path.exists(TSV)
    with open(TSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
        if newfile:
            w.writeheader()
        w.writerow(row)
    print(f"✓ 追加到 {TSV}")
    print(f"  arena={row['arena']}  {a.policy}@{a.deck} vs {a.opponent_policy}@{a.opponent_deck}: "
          f"{row['matchup_win_rate']} [{row['ci_low']},{row['ci_high']}] n={row['n_matches']}")


def report():
    if not os.path.exists(TSV):
        raise SystemExit("还没有注册表；先跑 --json 追加几行")
    rows = list(csv.DictReader(open(TSV, encoding="utf-8"), delimiter="\t"))
    print(f"共 {len(rows)} 条\n")
    import collections
    by = collections.Counter(r["arena"] for r in rows)
    print("按 arena 分类（**只有 policy_axis 能归因到策略**）：")
    for k, v in by.most_common():
        print(f"  {k:14s}{v:>4}")
    print(f"\n{'when':17s}{'arena':14s}{'policy@deck':34s}{'vs':22s}{'胜率':>7}{'CI':>18}{'seed':>8}")
    for r in rows[-25:]:
        pd = f"{r['policy_id']}@{r['deck_id'][:18]}"
        vs = f"{r['opponent_policy_id']}@{r['opponent_deck_id'][:10]}"
        print(f"{r['when']:17s}{r['arena']:14s}{pd[:34]:34s}{vs[:22]:22s}"
              f"{float(r['matchup_win_rate']):>7.3f}"
              f"{'['+r['ci_low']+','+r['ci_high']+']':>18}{r['evaluation_seed']:>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="一次 eval_pair 的 output json")
    ap.add_argument("--policy", default="")
    ap.add_argument("--deck", default="")
    ap.add_argument("--opponent-policy", default="")
    ap.add_argument("--opponent-deck", default="")
    ap.add_argument("--arena", default="cross", choices=["policy_axis", "deck_axis", "cross"])
    ap.add_argument("--seed", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.json:
        if not (a.policy and a.deck):
            raise SystemExit("--json 需要同时给 --policy 和 --deck")
        append(a)
    if a.report or not a.json:
        report()


if __name__ == "__main__":
    main()
