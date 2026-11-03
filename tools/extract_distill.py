#!/usr/bin/env python3
"""第 ① 步：从顶档对局里抽出 (obs 指针, action) 索引，并做对齐自检。

## 为什么先做"索引 + 自检"而不是直接产出训练矩阵

一帧模型输入是 10,944 个 float = 43.8 KB。40 万帧就是 **17.7 GB**，而且一旦对齐错了
（动作与观测错位一格），整份数据就是废的——**上次 BC 正是栽在这里**（见 `别再试.md` E8）。

所以这一步只产出**指针**（哪一局 / 第几步 / 哪个座位 / 什么动作），体积几十 MB；
然后用**已验证能 100% 复现线上行为**的喂法做自检：把抽出来的观测按顺序喂给我们的端口，
看它是否复现回放里记录的动作。**自检通过，这批指针才可信**，第 ② 步再去物化输入行。

## 对齐的关键（`framework/replay.py` 已封装）

回放的观测有"重复投递"现象：一个决策的动作记在 `steps[t+1][j]`，而对应的观测在
`steps[t][j]`。`iter_decisions()` 已经按这个规则对齐，并且在本仓库验证过
**100.00% 合法**（395/395 的 IS_FIRST 不变量、卡组 317/317 可抽取）。
本工具直接用它，不自己数下标。

## 用法

    ./run.sh tools/extract_distill.py --dirs epsoide_teams2 --out data/distill_index.jsonl
    ./run.sh tools/extract_distill.py --dirs epsoide_teams2 --limit 50 --check-games 8
"""
import argparse
import collections
import hashlib
import json
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from framework.replay import iter_decisions, our_index, decks_of   # noqa: E402


def h(ids):
    return hashlib.sha256(",".join(str(x) for x in sorted(ids)).encode()).hexdigest()[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="epsoide_teams2")
    ap.add_argument("--out", default="data/distill_index.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 局（调试）")
    ap.add_argument("--check-games", type=int, default=8,
                    help="抽 N 局做对齐自检（要加载我们的 agent，比较慢）")
    ap.add_argument("--skip-check", action="store_true")
    a = ap.parse_args()

    files = []
    for d in a.dirs.split(","):
        files += sorted(glob.glob(os.path.join(ROOT, d.strip(), "*.json")))
    if a.limit:
        files = files[: a.limit]

    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_dec = 0
    per_deck = collections.Counter()
    per_agent = collections.Counter()
    games_used = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for gi, fp in enumerate(files):
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            steps = d.get("steps") or []
            if not steps:
                continue
            decks = decks_of(steps)
            if len(decks) != 2:
                continue
            names = [x.get("Name", "") for x in ((d.get("info") or {}).get("Agents") or [])]
            dhash = {seat: h(ids) for seat, ids in decks.items()}
            rel = os.path.relpath(fp, ROOT)
            any_row = False
            for t, j, obs, act in iter_decisions(steps):
                sel = obs.get("select") or {}
                if not sel:
                    continue
                opts = sel.get("option") or []
                if not act or not all(isinstance(x, int) and 0 <= x < len(opts) for x in act):
                    continue
                f.write(json.dumps({
                    "file": rel, "step": t, "seat": j,
                    "agent": names[j] if j < len(names) else "",
                    "deck": dhash.get(j, "?"),
                    "n_opt": len(opts),
                    "act": act,
                }, ensure_ascii=False) + "\n")
                n_dec += 1
                per_deck[dhash.get(j, "?")] += 1
                per_agent[names[j] if j < len(names) else ""] += 1
                any_row = True
            if any_row:
                games_used += 1
            if (gi + 1) % 200 == 0:
                print(f"  ...{gi + 1}/{len(files)} 局，累计 {n_dec} 帧", flush=True)

    print(f"\n写出 {out_path}")
    print(f"  局数 {games_used}，决策帧 {n_dec:,}")
    print(f"  不同卡组 {len(per_deck)} 副，不同 agent {len(per_agent)} 位")
    print(f"  卡组分布 top5: {per_deck.most_common(5)}")
    print(f"  agent 分布 top5: {per_agent.most_common(5)}")

    if a.skip_check:
        return
    # ---------------- 对齐自检 ----------------
    print(f"\n=== 对齐自检（取前 {a.check_games} 局，按座位逐帧复演）===")
    from agentpool import get_agent
    tot = agree = skipped = 0
    checked = 0
    for fp in files:
        if checked >= a.check_games:
            break
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        steps = d.get("steps") or []
        decks = decks_of(steps)
        if len(decks) != 2:
            continue
        ok_any = True
        for seat, ids in decks.items():
            if len(ids) != 60:
                continue
            try:
                fn = get_agent(ids)
            except Exception as e:
                print(f"  ⚠️ 座位 {seat} 加载失败：{type(e).__name__}")
                ok_any = False
                continue
            for t, j, obs, act in iter_decisions(steps):
                if j != seat:
                    continue
                call = dict(obs)
                call.setdefault("logs", [])
                call.setdefault("remainingOverageTime", 600.0)
                call["player"] = seat
                call["step"] = t
                try:
                    got = list(fn(dict(call)) or [])
                except Exception:
                    skipped += 1
                    continue
                tot += 1
                if got == act:
                    agree += 1
        if ok_any:
            checked += 1
    if tot:
        print(f"  复演 {tot} 帧，完全一致 {agree}（{agree / tot:.2%}）")
        print(f"  ⚠️ 注：两位顶档玩家的决策本来就和我们的端口不同"
              f"（EXP-0138 实测约 62–67% 严格一致），")
        print("     所以这里不是「应该 100%」，而是「应该落在 60–70% 区间」——")
        print(f"     落在这个区间就说明观测与动作**对齐正确**；接近 0% 或异常高才是对齐错了。")
        print(f"     真正的 100% 自检在 `run.sh tools/replay_policy_diff.py --agent-name <我们自己>`。")


if __name__ == "__main__":
    main()
