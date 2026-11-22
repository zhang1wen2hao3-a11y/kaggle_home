#!/usr/bin/env python3
"""**D0：把"指定变体打指定格子"的对局录成蒸馏数据**（2026-11-22）

用户方案：同一副牌里，**逐对位最强变体**的走法蒸馏进一个模型（先把 4.55M 验通，再谈 20M）。

这个工具只负责**造数据**：
  1. 用 `tools/bo3_eval.py` 的 `AgentProcess` + `play_match(..., trace_path=)` 让
     「教师变体」打「指定对手（格子）」若干场，落 trace（每行一条 `{round, player, observation, action}`）；
  2. 把 trace 转成**回放格式**（`steps / info.Agents / rewards`），
     这样后面能直接复用现成流水线：`tools/bc_from_replays.sh --dir <out> --deck-prefix <8hex>`；
  3. 打印每个 trace 的决策数/胜负，便于确认数据可用。

只有教师座位会被写成帧（对手座位也写，但 BC 侧按 deck-prefix 过滤）。

用法
----
    ./tools/py tools/dump_teacher_games.py \
        --teacher build/agents/cand9D6ER4 --opp build/agents/g6_db04_on_2617 \
        --matches 10 --out data/distill/9d6e_vs_2617

    # 之后：
    bash tools/bc_from_replays.sh --dir data/distill/9d6e_vs_2617 --deck-prefix <教师牌表8hex> --tag 9d6e_vs2617
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def deck8_of(agent_dir: str) -> str:
    ids = [l.strip() for l in open(os.path.join(agent_dir, "deck.csv")) if l.strip()]
    return hashlib.sha256(",".join(str(x) for x in sorted(map(int, ids))).encode()).hexdigest()[:8]


def trace_to_replay(trace_path: str, names, rewards) -> dict:
    """trace(jsonl) → 回放结构。关键约定（与 framework/replay.iter_decisions 一致）：
    同一座位的一次决策，obs 在第 t 步、action 在第 t+1 步 ⇒ 这里把每条记录拆成两步。
    """
    # 布局必须和**真实回放**一致，否则 `framework.replay.decks_of` 取不到卡组：
    #   steps[0] 起始 → steps[1] 收到"交卡组"请求（observation.select 为 None）
    #   → steps[2] 卡组 action（60 个 id）→ 之后才是决策帧
    req = {"observation": {"select": None, "current": None, "logs": [], "step": 0}}
    steps = [[{}, {}], [dict(req), dict(req)], [{}, {}]]
    for ln in open(trace_path, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        p = int(r["player"])
        obs, act = r.get("observation"), r.get("action")
        row = [{}, {}]
        row[p] = {"observation": obs}
        steps.append(row)
        row2 = [{}, {}]
        row2[p] = {"action": act}
        steps.append(row2)
    return {"steps": steps, "info": {"Agents": [{"Name": names[0]}, {"Name": names[1]}]},
            "rewards": rewards}


def _run_batch(teacher, opp, mids, tdir, out_dir, logdir):
    """一个 worker：跑给定的场次并落回放，返回 (局数, 决策数)。"""
    from bo3_eval import AgentProcess, play_match
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    tdeck = [int(x) for x in open(os.path.join(teacher, "deck.csv")) if x.strip()]
    odeck = [int(x) for x in open(os.path.join(opp, "deck.csv")) if x.strip()]
    t8 = deck8_of(teacher)
    left = AgentProcess(f"t{os.getpid()}", teacher, logdir)
    right = AgentProcess(f"o{os.getpid()}", opp, logdir)
    games = dec = 0
    try:
        for mid in mids:
            tp = os.path.join(tdir, f"trace-{mid:03d}.jsonl")
            res = play_match(left, right, mid, trace_path=tp)
            if res.get("error"):
                continue
            per_round = {}
            for ln in open(tp, encoding="utf-8"):
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                per_round.setdefault(int(r.get("round", 1)), []).append(r)
            for rnd, recs in sorted(per_round.items()):
                tmp = os.path.join(tdir, f"tmp-{mid:03d}-{rnd}.jsonl")
                with open(tmp, "w", encoding="utf-8") as f:
                    for r in recs:
                        f.write(json.dumps(r) + "\n")
                w = next((x for x in res["rounds"] if x["round"] == rnd), {}).get("winner")
                args = trace_to_replay(tmp, ["teacher:" + os.path.basename(teacher),
                                             "opp:" + os.path.basename(opp)],
                                       [1.0 if w == 0 else 0.0, 1.0 if w == 1 else 0.0])
                args["steps"][2] = [{"action": tdeck}, {"action": odeck}]
                tagpart = (a.tag + "-") if a.tag else ""
                json.dump(args, open(os.path.join(out_dir, f"replay-{t8}-{tagpart}{mid:03d}-{rnd}.json"), "w"))
                games += 1
                dec += len(recs)
                os.remove(tmp)
    finally:
        left.close()
        right.close()
    return games, dec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="教师 agent 目录（含 main.py/model.pt/deck.csv）")
    ap.add_argument("--opp", required=True, help="对手 agent 目录（= 该格子当前的强驾驶员）")
    ap.add_argument("--matches", type=int, default=10, help="bo3 场数（每场 1~3 局）")
    ap.add_argument("--out", required=True, help="输出目录（写 replay-*.json）")
    ap.add_argument("--trace-dir", default="", help="trace 存放目录（默认 <out>/trace）")
    ap.add_argument("--workers", type=int, default=1, help="（已停用，见文件头注释）")
    ap.add_argument("--start", type=int, default=0, help="起始场次（配合 --step 做外部并发）")
    ap.add_argument("--tag", default="", help="写进文件名，避免**不同格子同名覆盖**（强烈建议给）")
    ap.add_argument("--step", type=int, default=1, help="场次步长（配合 --start 做外部并发）")
    a = ap.parse_args()

    from bo3_eval import AgentProcess, play_match        # noqa: E402
    from concurrent.futures import ProcessPoolExecutor

    os.makedirs(a.out, exist_ok=True)
    tdir = a.trace_dir or os.path.join(a.out, "trace")
    os.makedirs(tdir, exist_ok=True)
    t8, o8 = deck8_of(a.teacher), deck8_of(a.opp)
    print(f"教师 {a.teacher}（deck8 {t8}） vs 对手 {a.opp}（deck8 {o8}），{a.matches} 场")
    if t8 == o8:
        print("⚠️ 两边牌表相同（镜像对局）——蒸馏数据仍然可用，但决策会更接近")

    logdir = os.path.join(a.out, "agentlogs")
    if a.workers > 1:
        # ★ 2026-11-22：不能在进程池里再起 AgentProcess（池 worker 是 daemonic，fork 子进程会报
        #   "daemonic processes are not allowed to have children"）。并发请**从外部起多个进程**：
        #   for i in 0 1 2 3; do ... --start $i --step 4 & done
        print("⚠️ --workers 已停用：请用外部多进程并发（见 --start/--step），内部不再 fork")
    left = AgentProcess("teacher", a.teacher, logdir)
    right = AgentProcess("opp", a.opp, logdir)
    tot_games = tot_dec = 0
    try:
        for mid in range(a.start, a.start + a.matches * a.step, a.step):
            tp = os.path.join(tdir, f"trace-{mid:03d}.jsonl")
            res = play_match(left, right, mid, trace_path=tp)
            if res.get("error"):
                print(f"  场 {mid}: ✗ {res['error'].splitlines()[-1][:120]}")
                continue
            # 每局各自写一个回放（BC 侧按整局切 train/val）
            # trace 里带 round 字段 ⇒ 按 round 分组
            per_round = {}
            for ln in open(tp, encoding="utf-8"):
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                per_round.setdefault(int(r.get("round", 1)), []).append(r)
            for rnd, recs in sorted(per_round.items()):
                tmp = os.path.join(tdir, f"tmp-{mid:03d}-{rnd}.jsonl")
                with open(tmp, "w", encoding="utf-8") as f:
                    for r in recs:
                        f.write(json.dumps(r) + "\n")
                w = next((x for x in res["rounds"] if x["round"] == rnd), {}).get("winner")
                rewards = [1.0 if w == 0 else 0.0, 1.0 if w == 1 else 0.0]
                rep = trace_to_replay(tmp, ["teacher:" + os.path.basename(a.teacher),
                                            "opp:" + os.path.basename(a.opp)], rewards)
                # 卡组行
                tdeck = [int(x) for x in open(os.path.join(a.teacher, "deck.csv")) if x.strip()]
                odeck = [int(x) for x in open(os.path.join(a.opp, "deck.csv")) if x.strip()]
                rep["steps"][2] = [{"action": tdeck}, {"action": odeck}]
                tagpart = (a.tag + "-") if a.tag else ""
                out = os.path.join(a.out, f"replay-{t8}-{tagpart}{mid:03d}-{rnd}.json")
                json.dump(rep, open(out, "w", encoding="utf-8"))
                nd = len(recs)
                tot_games += 1
                tot_dec += nd
                os.remove(tmp)
            print(f"  场 {mid}: 胜 {res['wins']}，局 {len(per_round)}，决策 {sum(len(v) for v in per_round.values())}")
    finally:
        left.close()
        right.close()
    print(f"\n✓ 写出 {tot_games} 局回放（决策 {tot_dec} 条）→ {a.out}")
    print(f"  下一步：bash tools/bc_from_replays.sh --dir {a.out} --deck-prefix {t8} --tag <标签>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
