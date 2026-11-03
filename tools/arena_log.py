#!/usr/bin/env python3
"""**Arena per-match 账本**：把每次对局的**单场**记录（含座位、胜负、策略、卡组哈希）落表。

为什么不能只存 win_rate（2026-11-02 与用户对齐）：
  * 只有聚合胜率时，无法事后做**配对差**（同种子逐场对减），也无法按座位分层；
  * 无法区分"策略变强"和"卡组变强"——除非每场都记住 **policy_hash × deck_hash**；
  * 无法检查座位分布是否失衡（我们踩过跨机器 ±5–8pp 的坑）。

字段（严格按用户清单 + 我们需要的两个）：
    when, arena, agent_id, policy_id, policy_hash, deck_id, deck_hash,
    opponent_policy_id, opponent_policy_hash, opponent_deck_id, opponent_deck_hash,
    seed(match_id), seat, winner, round_results, error, source_log

数据来源：`logs/**/shard-*.jsonl`（eval_pair 的面板输出，每行一个 Bo3 match）。
策略识别：从 `cand`/`opp` 的目录名反推，再查 `MODELS` 表拿 policy_hash 与 model_hash。

用法：
    ./tools/py tools/arena_log.py --harvest        # 扫描所有 logdir，追加到 data/arena_matches.tsv
    ./tools/py tools/arena_log.py --report         # 按 (策略,卡组,对手) 聚合 + 座位分布
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import re
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TSV = os.path.join(ROOT, "data", "arena_matches.tsv")
FIELDS = ["when", "arena", "agent_id", "policy_id", "policy_hash", "deck_id", "deck_hash",
          "opponent_policy_id", "opponent_policy_hash", "opponent_deck_id",
          "opponent_deck_hash", "seed", "seat", "winner", "round_results", "error",
          "source_log"]

# 目录名 → policy_id（按我们实际用过的命名）
POLICY_PATTERNS = [
    (r"^port_db04$|^port_7e39$|^port_shuppet$|^port_on_9d6e$", "port"),
    (r"^g5_db04$|^ag_g5_db04$", "g5_db04"), (r"^g6_db04$|^ag_g6_db04$", "g6_db04"),
    (r"^g7a_db04_shuppet$|^ag_g7a.*$", "g7a_db04_shuppet"), (r"^g9_db04$|^ag_g9.*$", "g9_db04"),
    (r"^g10_db04_banks$", "g10_db04_banks"), (r"^g7a_db04_shuppet$", "g7a_db04_shuppet"),
    (r"^g8_shuppet$|^opp_g8_shuppet$", "g8_shuppet_bc"),
    (r"^g8_bc_parent$|^opp_shuppet_human$", "g8_bc_parent"),
    (r"^bc_abra2$|^bc_abra$", "bc_abra2"), (r"^bc_abra76k$", "bc_abra76k"),
    (r"^bc_grookey3$", "bc_grookey3"), (r"^bc_dreepy2$", "bc_dreepy2"),
    (r"^g6_db04_on_7e39$", "g6_db04"), (r"^g6_db04_on_9d6e$", "g6_db04"),
    (r"^c_abra$|^abra$", "port"), (r"^c_grookey$|^grookey$", "port"),
    (r"^c_dreepy$|^dreepy$", "port"), (r"^c_slowking$", "port"),
    (r"^opp_g4a$|^opp_g4b$|^opp_g4c$|^opp_g0001$|^^opp_g0003$", "ppo_g"),
]
DECK_PATTERNS = [
    (r"db04", "dunsparce_abra__db04"), (r"9d6e", "dunsparce_abra__9d6e"),
    (r"7e39", "teal_mask_ogerpon_ex_applin__7e39"), (r"0d86|shuppet", "shuppet_dhelmise__0d86"),
    (r"f5f8|abra", "abra_kadabra__f5f8"), (r"2617|grookey", "grookey_thwackey__2617"),
    (r"ea39|dreepy", "dreepy_drakloak__ea39"), (r"26c6|slowking", "slowpoke_slowking__26c6"),
]

CACHE = {}


def sha_file(p, n=16):
    if not p or not os.path.exists(p):
        return ""
    if p in CACHE:
        return CACHE[p]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    CACHE[p] = h.hexdigest()[:n]
    return CACHE[p]


def classify(name, kind):
    table = POLICY_PATTERNS if kind == "policy" else DECK_PATTERNS
    for pat, val in table:
        if re.search(pat, name):
            return val
    return name


def agent_dir(tag_left):
    """`0Aport_db04` → (host_dir, agent_tag)。"""
    m = re.match(r"^\d+[AB](.+)$", tag_left or "")
    return (m.group(1) if m else (tag_left or ""))


def dirmap():
    """目录名 → 绝对路径（用于算 policy_hash / deck_hash）。"""
    out = {}
    for pat in ("build/agents/*", "ptcg_league/pools/league/*", "ptcg_league/pools/init/*"):
        for d in glob.glob(os.path.join(ROOT, pat)):
            out[os.path.basename(d)] = d
    return out


def harvest():
    dirs_ = dirmap()
    seen = set()
    if os.path.exists(TSV):
        for r in csv.DictReader(open(TSV, encoding="utf-8"), delimiter="\t"):
            seen.add((r["source_log"], r["seed"], r["agent_id"], r["opponent_policy_id"], r["seat"]))
    rows, added = [], 0
    now = time.strftime("%Y-%m-%d %H:%M")
    for shard in glob.glob(os.path.join(ROOT, "logs/**/shard-*.jsonl"), recursive=True):
        logdir = os.path.dirname(shard)
        # 从同目录的 config 猜 arena：policy_axis 需要"同卡组不同策略"——这里由调用者事后标注，
        # 默认先记 cross，report 时再按 deck_hash 是否相同重新分类。
        for line in open(shard, encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            la, ra = agent_dir(d.get("left")), agent_dir(d.get("right"))
            # ⚠️ eval_pair 里 cand 的 AgentProcess 名字借用的是**对手名**，
            #    所以不能按进程名判策略。改用同前缀的 --out json 里的 `cand` 路径。
            run_json = logdir.rstrip("/") + ".json"
            cand_dir = ""
            if os.path.exists(run_json):
                try:
                    cand_dir = json.load(open(run_json)).get("cand") or ""
                except Exception:
                    cand_dir = ""
            pol_a = classify(os.path.basename(cand_dir.rstrip("/")), "policy") if cand_dir else classify(la, "policy")
            dk_a = classify(os.path.basename(cand_dir.rstrip("/")), "deck") if cand_dir else classify(la, "deck")
            pa = cand_dir if cand_dir else dirs_.get(la, "")
            pb = dirs_.get(os.path.basename(ra), "")
            pol_b = classify(ra, "policy")
            dk_b = classify(ra, "deck")
            dj_a, dj_b = os.path.join(pa, "deck.csv"), os.path.join(pb, "deck.csv")
            deck_h_a = sha_file(dj_a, 8) if os.path.exists(dj_a) else ""
            deck_h_b = sha_file(dj_b, 8) if os.path.exists(dj_b) else ""
            key = (os.path.basename(logdir), str(d.get("match_id")), la, ra, d.get("seat"))
            if key in seen:
                continue
            rows.append(dict(
                when=now,
                arena=("policy_axis" if (dk_a == dk_b and dk_a) or
                       (deck_h_a and deck_h_a == deck_h_b) else "cross"),
                agent_id=la, policy_id=pol_a,
                policy_hash=sha_file(os.path.join(pa, "model.pt")),
                deck_id=dk_a, deck_hash=deck_h_a,
                opponent_policy_id=pol_b,
                opponent_policy_hash=sha_file(os.path.join(pb, "model.pt")),
                opponent_deck_id=dk_b, opponent_deck_hash=deck_h_b,
                seed=d.get("match_id"), seat=d.get("seat"),
                winner=d.get("winner"),
                round_results=",".join(str(w) for w in (d.get("wins") or [])),
                error=(d.get("error") or "")[:80],
                source_log=os.path.basename(logdir),
            ))
            added += 1
    if rows:
        newfile = not os.path.exists(TSV)
        with open(TSV, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
            if newfile:
                w.writeheader()
            w.writerows(rows)
    print(f"新增 {added} 场 → {TSV}")
    return rows


def report():
    if not os.path.exists(TSV):
        raise SystemExit("还没有账本；先 --harvest")
    rows = list(csv.DictReader(open(TSV, encoding="utf-8"), delimiter="\t"))
    print(f"共 {len(rows)} 场\n")
    agg = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for r in rows:
        k = (r["policy_id"], r["deck_id"], r["opponent_policy_id"], r["opponent_deck_id"], r["arena"])
        try:
            w = int(r["winner"]) if r["winner"] not in ("", "None") else None
        except ValueError:
            w = None
        a = agg[k]
        a[0] += 1
        if w == 0:
            a[1] += 1                     # left = agent_id 座位
        a[2][r["seat"]] += 1
    print(f"{'策略@卡组':38s}{'vs':30s}{'arena':12s}{'场':>5}{'胜':>5}{'胜率':>7}  座位分布")
    for k, (n, w, seats) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        pol, dk, opol, odk, arena = k
        sd = ",".join(f"s{s}:{c}" for s, c in sorted(seats.items()))
        print(f"{(pol+'@'+dk)[:37]:38s}{(opol+'@'+odk)[:29]:30s}{arena:12s}{n:>5}{w:>5}{w/n:>7.3f}  {sd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.harvest:
        harvest()
    if a.report or not a.harvest:
        report()


if __name__ == "__main__":
    main()
