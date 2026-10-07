#!/usr/bin/env python3
"""自对弈生成训练数据。

产出 JSONL，一行一个决策点：
    状态特征 + 每个候选动作的特征 + 实际动作 + 终局胜负

关键设计：
  * **特征只用公开信息**（framework.features 保证），标签可以用隐藏信息
  * **按对局切分**（framework.dataset.split_by_game），杜绝同局泄漏
  * 可选记录 oracle 隐藏信息（--oracle），用于将来训练对手推断模型
    ⚠️ oracle 依赖 visualize_data()，仅本地可用；用不用它生成训练数据请自行核对比赛规则

用法：
    python gen_data.py --games 200 --out data/selfplay_v1.jsonl
    python gen_data.py --games 50 --oracle --out data/selfplay_oracle.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from framework import features as F  # noqa: E402
from framework.dataset import describe, write_records  # noqa: E402
from framework.memory import GameMemory  # noqa: E402
from local_match import load_deck  # noqa: E402


def load_agent(path: str):
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    if d not in sys.path:
        sys.path.append(d)          # append 而非 insert：避免遮蔽标准库（踩过这个坑）
    spec = importlib.util.spec_from_file_location(
        "gen_agent_" + os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "agent"):
        raise SystemExit(f"{path} 里没有 agent 函数")
    return mod.agent


def record_oracle(obs: dict) -> dict | None:
    """用 visualize_data() 读取真实隐藏信息（仅本地可用）。"""
    try:
        from cg.game import visualize_data
        import json
        replay = json.loads(visualize_data())
        if not replay:
            return None
        cur = obs.get("current") or {}
        me = int(cur.get("yourIndex", 0))
        for rec in reversed(replay):
            c = rec.get("current")
            if not c:
                continue
            ps = c.get("players") or []
            if len(ps) < 2:
                continue
            op = ps[1 - me]

            def ids(cards):
                return [x.get("id") for x in (cards or []) if x]

            oh = ids(op.get("hand"))
            if len(oh) == int(op.get("handCount", 0)):
                return {"opponent_hand": oh,
                        "opponent_deck": ids(op.get("deck")),
                        "opponent_prize": ids(op.get("prize"))}
    except Exception:
        return None
    return None


def play_and_record(agent0, agent1, deck0, deck1, game_idx: int,
                    reverse: bool, oracle: bool, max_steps: int = 3000) -> list[dict]:
    """打一局并记录每一步。"""
    obs, sd = battle_start(deck0, deck1, reverse_player=reverse)
    if obs is None:
        return []

    mems = {0: GameMemory(), 1: GameMemory()}
    recs: list[dict] = []
    steps = 0
    try:
        while steps < max_steps:
            cur, sel = obs.get("current"), obs.get("select")
            if cur is None or sel is None or cur.get("result", -1) != -1:
                break
            me = int(cur["yourIndex"])
            agents = {0: agent0, 1: agent1}
            mem = mems[me]
            mem.observe(obs)                    # 累积增量日志

            opts = [F.option_features(obs, i, mem) for i in range(len(sel["option"]))]
            rec = {
                "game": game_idx,
                "step": steps,
                "turn": int(cur.get("turn", 0)),
                "player": me,
                "features": F.state_features(obs, mem),
                "options": opts,
                "action": None,
                "result": None,
            }
            if oracle:
                o = record_oracle(obs)
                if o:
                    rec["oracle"] = o

            try:
                act = agents[me](obs)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                break
            if not isinstance(act, list):
                break
            rec["action"] = act
            recs.append(rec)
            obs = battle_select(act)
            steps += 1

        result = (obs.get("current") or {}).get("result", -1)
        for r in recs:
            r["result"] = None if result == -1 else result
    finally:
        battle_finish()
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--agent", default="main.py", help="玩家0 的 agent")
    ap.add_argument("--opponent", default="agents/random_baseline.py", help="玩家1 的 agent")
    ap.add_argument("--deck", default="deck.csv")
    ap.add_argument("--deck-opponent", default=None, help="对手卡组（默认同 --deck）")
    ap.add_argument("--out", default="data/selfplay.jsonl")
    ap.add_argument("--oracle", action="store_true",
                    help="额外记录真实隐藏信息（仅本地，用途见文件头说明）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--game-offset", type=int, default=0,
                    help="局号偏移，便于把多次生成的数据合并成一个数据集而不撞号")
    ap.add_argument("--max-records", type=int, default=0, help="达到该记录数即停（0=不限）")
    args = ap.parse_args()

    A0 = load_agent(args.agent)
    A1 = load_agent(args.opponent)
    deck0 = load_deck(args.deck if os.path.isabs(args.deck) else os.path.join(HERE, args.deck))
    d1p = args.deck_opponent or args.deck
    deck1 = load_deck(d1p if os.path.isabs(d1p) else os.path.join(HERE, d1p))

    print(f"玩家0 = {args.agent} ({len(deck0)} 张)")
    print(f"玩家1 = {args.opponent} ({len(deck1)} 张)")
    print(f"目标 {args.games} 局 | oracle={'开' if args.oracle else '关'}\n")

    all_recs: list[dict] = []
    t0 = time.time()
    started = failed = 0
    for g in range(args.games):
        recs = play_and_record(A0, A1, deck0, deck1, g + args.game_offset,
                               bool(g % 2), args.oracle)
        if recs:
            all_recs.extend(recs)
            started += 1
        else:
            failed += 1
        if not args.oracle and (g + 1) % 10 == 0:
            dt = time.time() - t0
            print(f"  {g+1}/{args.games} 局  记录 {len(all_recs)} 条  "
                  f"{dt:.1f}s  ({(g+1)/dt:.1f} 局/秒)")
        if args.max_records and len(all_recs) >= args.max_records:
            print(f"  达到 max-records={args.max_records}，提前停止")
            break

    dt = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    n = write_records(args.out, all_recs)

    print()
    print("=" * 58)
    print(f"写盘 {n} 条 -> {args.out}")
    print(f"成功开局 {started} 局 / 失败 {failed} 局 | 耗时 {dt:.1f}s")
    st = describe(all_recs)
    for k, v in st.items():
        print(f"  {k:22s} {v}")
    if all_recs and "oracle" in all_recs[0]:
        with_o = sum(1 for r in all_recs if "oracle" in r)
        print(f"  {'含 oracle 记录':22s} {with_o}/{len(all_recs)}")
    print("=" * 58)


if __name__ == "__main__":
    main()
