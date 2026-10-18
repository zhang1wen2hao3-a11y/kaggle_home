#!/usr/bin/env python3
"""卡组合法性的回归测试。

这个 bug 极其阴险：**is_legal 说合法、引擎却拒绝**，于是对局根本开不起来，
表现成"胜率 0.000"而不是报错。根因是 PTCG 的 4 张上限按【卡名】算而不是按 cardId
（卡池里同名不同 ID 的卡很多，两个 ID 各 4 张 = 同名 8 张，引擎 errorType=2）。

所以这里最核心的一条是：**is_legal 必须与引擎判定 100% 一致**。

运行：./run.sh tests/test_deck.py
"""
from __future__ import annotations

import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from cg.game import battle_start, battle_finish  # noqa: E402
from framework.deck import (CardPool, crossover, is_legal, mutate, repair,  # noqa: E402
                            random_deck, structural_mutate)

FAILED = 0


def check(cond: bool, msg: str) -> None:
    global FAILED
    print(f"  {'✅' if cond else '❌'} {msg}")
    if not cond:
        FAILED += 1


def engine_accepts(deck: list[int]) -> tuple[bool, int]:
    obs, sd = battle_start(deck, deck)
    if obs is not None:
        battle_finish()
        return True, -1
    return False, sd.errorType


def main() -> None:
    pool = CardPool()
    rng = random.Random(0)
    print("=" * 64)
    print("卡组合法性回归测试")
    print("=" * 64)

    # ---------------- 1. 规则细节
    print("\n[1] 规则细节")
    base = pool.deck_from_file(os.path.join(HERE, "deck.csv"))
    ok, _ = is_legal(pool, base)
    check(ok, "样例卡组合法")

    # 同名不同 ID：两个不同 cardId 但同名的卡各 4 张 —— 应判非法
    by_name: dict[str, list[int]] = {}
    for cid, c in pool.by_id.items():
        if int(c.cardType) == 0 and not pool.is_basic_energy(cid):
            by_name.setdefault(c.name, []).append(cid)
    dup = next((ids for ids in by_name.values() if len(ids) >= 2), None)
    if dup:
        d = [dup[0]] * 4 + [dup[1]] * 4 + pool.basic_energy[:1] * 52
        ok, why = is_legal(pool, d)
        acc, et = engine_accepts(repair(pool, d, rng) if False else d)
        check(not ok and not acc,
              f"同名不同ID各4张（{pool.by_id[dup[0]].name}）判非法，引擎也拒绝(errorType={et})")

    d = [c for c in base if not pool.is_basic_pokemon(c)]
    while len(d) < 60:
        d.append(pool.basic_energy[0])
    ok, why = is_legal(pool, d)
    check(not ok, f"无基础宝可梦判非法（{why}）")

    if pool.ace_spec:
        ace = sorted(pool.ace_spec)[:2]
        d = [pool.basic_pokemon[0]] * 4 + ace + [pool.basic_energy[0]] * (60 - 4 - len(ace))
        ok, why = is_legal(pool, d)
        check(not ok, f"两张 ACE SPEC 判非法（{why}）")

    # ---------------- 2. ★ is_legal 与引擎一致
    print("\n[2] ★ is_legal 与引擎判定一致性（最关键）")
    agree = disagree = 0
    for _ in range(120):
        d = random_deck(pool, rng)
        mine, _ = is_legal(pool, d)
        acc, _et = engine_accepts(d)
        if mine == acc:
            agree += 1
        else:
            disagree += 1
            if disagree <= 3:
                print(f"    不一致: is_legal={mine} engine={acc}  {pool.describe(d, 5)}")
    check(disagree == 0, f"随机卡组 120 副：一致 {agree}/120")

    # 变异/交叉后也必须一致
    print("\n[3] 算子产出的合法性")
    for name, fn in [
        ("mutate", lambda: mutate(pool, base, rng)),
        ("structural", lambda: structural_mutate(pool, base, rng)),
        ("crossover", lambda: crossover(pool, base, random_deck(pool, rng), rng)),
    ]:
        okc = sum(1 for _ in range(30) if is_legal(pool, fn())[0])
        check(okc == 30, f"{name:<11} 合法 {okc}/30")

    # ---------------- 4. 能量匹配
    print("\n[4] 能量匹配（避免造出打不出招式的废组）")
    def energy_covered(d) -> bool:
        need: set[int] = set()
        for cid in set(d):
            c = pool.by_id.get(cid)
            if c is not None and int(c.cardType) == 0 and (getattr(c, "attacks", None) or []):
                need |= pool.energy_types_needed(cid)
        have = {int(pool.by_id[c].energyType) for c in d if pool.is_basic_energy(c)}
        return not need or need <= have

    okc = sum(1 for _ in range(30) if energy_covered(random_deck(pool, rng, n_lines=2)))
    check(okc >= 28, f"random_deck(2条线) 能量覆盖 {okc}/30")
    okc = sum(1 for _ in range(30) if energy_covered(structural_mutate(pool, base, rng)))
    check(okc >= 28, f"structural_mutate 能量覆盖 {okc}/30")

    print("\n" + "=" * 64)
    if FAILED:
        print(f"❌ {FAILED} 项失败")
        sys.exit(1)
    print("✅ 全部通过")
    print("=" * 64)


if __name__ == "__main__":
    main()
