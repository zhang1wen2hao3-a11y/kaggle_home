#!/usr/bin/env python3
"""League Research · **卡组与策略分离**的联赛表（P1 数据结构）。

按用户 2026-11-01 定的字段导出 `data/league_agents.tsv`：

    agent_id          池内唯一 id（如 opp:g5_db04）
    deck_id           精确卡组名（pool_field 里的文件名主干）
    deck_hash         卡组 60 张的哈希（前 8 位，与 distill_index 的 deck 字段同源：4 位后缀+4 位区分）
    policy_checkpoint 策略权重文件（.bin/.pt 路径）
    parent_checkpoint 父权重
    training_opponents 该代训练时用的 bank（来自 models/ppo/*.json 的 recipe.banks）
    rating_league     对位矩阵的**行平均**（在训练对手池里的综合强度）
    matchup_profile   逐对手胜率（json 字符串）
    source/status/deck_archetype/p_i（学习者对它的胜率）

设计目的：**同一副卡组可以有多个策略、同一策略架构也可以配不同卡组**，
所以"这代的优势来自牌还是来自权重"必须能分别查。

用法：./tools/py tools/league_export.py
"""
import collections
import csv
import glob
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def deck_hash(path):
    try:
        ids = sorted(int(l) for l in open(path) if l.strip())
    except Exception:
        return ""
    if len(ids) != 60:
        return ""
    return hashlib.md5(" ".join(map(str, ids)).encode()).hexdigest()[:8]


def main():
    reg = {}
    for line in open(os.path.join(ROOT, "ptcg_league", "state", "registry.jsonl"), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            reg[r["agent_id"]] = r
    state = json.load(open(os.path.join(ROOT, "ptcg_league", "state", "league_state.json")))
    p_i = state.get("pfsp", {}).get("p", {})

    # 训练配方（models/ppo/*.json）里的 bank
    banks = {}
    for f in glob.glob(os.path.join(ROOT, "models", "ppo", "*.json")):
        try:
            m = json.load(open(f))
        except Exception:
            continue
        aid = m.get("agent_id")
        if aid:
            banks[aid] = m.get("recipe", {}).get("banks") or []

    # 对位矩阵（若有）
    prof = collections.defaultdict(dict)
    mfile = os.path.join(ROOT, "data", "matchup_matrix.tsv")
    if os.path.exists(mfile):
        for r in csv.DictReader(open(mfile, encoding="utf-8"), delimiter="\t"):
            prof[r["agent"]][r["opponent"]] = round(float(r["win_rate"]), 4)

    rows = []
    for aid, r in sorted(reg.items()):
        dp = r.get("deck_path") or ""
        m = prof.get(aid) or prof.get(aid.split(":")[-1]) or {}
        league = round(sum(m.values()) / len(m), 4) if m else ""
        rows.append(dict(
            agent_id=aid,
            deck_id=r.get("deck_id") or "",
            deck_hash=deck_hash(dp) if dp and os.path.exists(dp) else "",
            policy_checkpoint=r.get("checkpoint_path") or "",
            parent_checkpoint=parent_of(aid, banks),
            training_opponents=",".join(banks.get(aid, [])),
            rating_league=league,
            matchup_profile=json.dumps(m, ensure_ascii=False) if m else "",
            source=r.get("source") or "",
            status=r.get("status") or "",
            learner_win_vs_it=p_i.get(aid, ""),
        ))
    out = os.path.join(ROOT, "data", "league_agents.tsv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"→ {out}（{len(rows)} 个 agent）")
    decks = collections.Counter(r["deck_id"] for r in rows)
    print(f"  覆盖 {len(decks)} 副卡组；其中多策略共存的卡组：")
    for d, n in decks.most_common():
        if n > 1:
            print(f"    {d:34s} {n} 个策略")


def parent_of(aid, banks):
    m = os.path.join(ROOT, "models", "ppo",
                     aid.split(":")[-1].replace("g7a_db04_shuppet", "g7a_db04_shuppet") + ".json")
    for f in glob.glob(os.path.join(ROOT, "models", "ppo", "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("agent_id") == aid:
            return d.get("parent", "")
    return ""


if __name__ == "__main__":
    main()
