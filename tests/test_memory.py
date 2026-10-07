#!/usr/bin/env python3
"""GameMemory 的回归测试。

重点测【跨局污染】—— 这是最阴险的一类 bug：
    Kaggle 很可能复用进程跨对局跑 agent。如果记忆不重置，
    第二局会带着第一局的对手信息，而且**本地单局测试根本发现不了**。

运行：python tests/test_memory.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from framework.memory import GameMemory  # noqa: E402
from local_match import load_deck  # noqa: E402

FAILED = 0


def check(cond: bool, msg: str) -> None:
    global FAILED
    print(f"  {'✅' if cond else '❌'} {msg}")
    if not cond:
        FAILED += 1


def play_game(deck, mem: GameMemory, max_steps: int = 3000):
    """用同一个 memory 对象跑完一局，返回 (步数, 终局结果)。"""
    obs, sd = battle_start(deck, deck)
    if obs is None:
        return 0, None
    steps = 0
    try:
        while steps < max_steps:
            cur, sel = obs.get("current"), obs.get("select")
            if cur is None or sel is None or cur.get("result", -1) != -1:
                break
            mem.observe(obs)
            pick = list(range(sel.get("minCount", 1))) or [0]
            obs = battle_select(pick)
            steps += 1
        return steps, (obs.get("current") or {}).get("result", -1)
    finally:
        battle_finish()


def main() -> None:
    deck = load_deck()
    print("=" * 60)
    print("GameMemory 回归测试")
    print("=" * 60)

    # ---------------- 1. 新局检测
    print("\n[1] 新局检测（本地 self-play 路径）")
    mem = GameMemory()
    steps1, _ = play_game(deck, mem)          # 跑完整局，让回合推进
    check(mem.max_turn_seen > 0, f"第一局推进到 turn={mem.max_turn_seen}")
    t1 = mem.turns_seen

    obs2, _ = battle_start(deck, deck)
    is_new = mem.is_new_game(obs2)
    check(is_new, f"新对局的 obs 被判为新局（turn 0 < max_turn {mem.max_turn_seen}）")
    mem.observe(obs2)
    check(mem.turns_seen == 1,
          f"重置后 turns_seen 归 1（实际 {mem.turns_seen}，上一局是 {t1}）")
    battle_finish()

    # ---------------- 2. select is None（线上开局信号）也判新局
    print("\n[2] 线上开局信号（select is None）")
    fake = {"select": None, "current": None, "logs": []}
    check(mem.is_new_game(fake), "select is None 判为新局")
    mem.observe(fake)
    check(mem.turns_seen == 0 and not mem.opp_seen,
          "observe(开局obs) 后记忆被清空")

    # ---------------- 3. ★ 跨局污染
    print("\n[3] ★ 跨局污染（同一个 memory 连跑多局）")
    mem = GameMemory()
    snaps = []
    for g in range(4):
        steps, res = play_game(deck, mem)
        snaps.append(dict(
            turns=mem.turns_seen,
            seen_top=len(mem.opp_seen),
            seen_total=sum(mem.opp_seen.values()),
            prizes=mem.opp_prizes_taken,
            searches=mem.opp_searches,
            hand_samples=len(mem.opp_hand_history),
        ))
        print(f"    第{g+1}局: {steps} 步 result={res} -> 记忆快照 {snaps[-1]}")

    # 每局的记忆规模应该在同一量级，而不是逐局累加
    turns = [s["turns"] for s in snaps]
    seen = [s["seen_total"] for s in snaps]
    check(max(turns) <= 60 and all(t > 0 for t in turns),
          f"每局 turns_seen 独立且合理 {turns}（若累加会看到 1,2,3,4... 或爆炸）")
    check(max(seen) < 500,
          f"每局对手已见卡独立 {seen}（若污染会单调累加）")
    check(snaps[-1]["turns"] < sum(turns),
          "第4局的 turns_seen 不等于四局之和 => 记忆确实被重置")

    # ---------------- 4. 特征可用性
    print("\n[4] 记忆特征")
    mem = GameMemory()
    play_game(deck, mem)
    f = mem.features(0)
    check(len(f) >= 10, f"features() 产出 {len(f)} 个特征")
    check(all(isinstance(v, float) for v in f.values()), "全部为 float")
    print("    " + ", ".join(list(f)[:8]) + " ...")

    print("\n" + "=" * 60)
    if FAILED:
        print(f"❌ {FAILED} 项失败")
        sys.exit(1)
    print("✅ 全部通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
