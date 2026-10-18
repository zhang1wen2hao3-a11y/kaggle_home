#!/usr/bin/env python3
"""B1：手工构造多套不同 archetype 的卡组，作为对手池的卡组维度。

为什么需要这一步：
    V 的学习曲线在 600 局就饱和（AUC 0.71），说明**数据量不是瓶颈**；
    而"混合不同强度对手"会让 AUC 掉到 0.68 —— 说明**对手多样性**才是。
    所以先把这个维度补齐。

设计原则：
    1. **能量必须匹配招式需求**（踩过坑：不匹配会让卡组直接废掉）
    2. **每个 archetype 一条清晰的攻击手**，不用弱卡凑数
    3. 训练师用一组通用的"已知能用"的卡（取自样例卡组），
       不追求最优 —— 目标是**多样性**，不是最优性
    4. 全部通过 is_legal + 引擎校验后才能落盘

用法：
    ./run.sh make_decks.py                 # 生成到 decks/
    ./run.sh make_decks.py --verify        # 生成后跑交叉对战验证多样性
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from framework.deck import CardPool, energy_cards_for, is_legal, repair  # noqa: E402

ENERGY_NAME = {0: "无色", 1: "草", 2: "火", 3: "水", 4: "雷", 5: "超",
               6: "斗", 7: "恶", 8: "钢", 9: "龙"}

# 目标 archetype：(显示名, 主攻击手卡名, 备用攻击手卡名 or None)
ARCHETYPES = [
    ("water_abomasnow",  "Mega Abomasnow ex",      "Snover"),          # 官方样例同款
    ("colorless_ursaluna", "Bloodmoon Ursaluna ex", None),              # 无色需求，任何能量
    ("fire_gouging",     "Gouging Fire ex",        None),               # 火
    ("water_kyurem",     "Black Kyurem ex",        None),               # 水
    ("metal_mawile",     "Mega Mawile ex",         None),               # 钢
    ("fighting_zygarde", "Mega Zygarde ex",        None),               # 斗
    ("dark_absol",       "Mega Absol ex",          None),               # 恶
    ("psychic_floette",  "Mega Floette ex",        None),               # 超
    ("grass_heracross",  "Mega Heracross ex",      None),               # 草
    ("dragon_latias",    "Mega Latias ex",         None),               # 龙（双色）

    # ---- 第二批：补齐能量覆盖（原 10 套缺口最多的是【雷】，无色只有 1 套）----
    ("lightning_zeraora",    "Mega Zeraora ex",        None),           # 雷
    ("lightning_pikachu",    "Pikachu ex",             None),           # 雷
    ("lightning_ironthorns", "Iron Thorns ex",         None),           # 雷
    ("colorless_kangaskhan", "Mega Kangaskhan ex",     None),           # 无色
    ("psychic_mewtwo",       "Team Rocket's Mewtwo ex", None),          # 超
    ("psychic_miraidon",     "Miraidon ex",            None),           # 超/雷
    ("metal_zacian",         "Zacian ex",              None),           # 钢
    ("metal_genesect",       "Genesect ex",            None),           # 钢
    ("fighting_koraidon",    "Koraidon ex",            None),           # 斗
    ("fighting_hawlucha",    "Mega Hawlucha ex",       None),           # 斗
    ("dark_darkrai",         "Mega Darkrai ex",        None),           # 恶
    ("fire_hooh",            "Ethan's Ho-Oh ex",       None),           # 火
    ("water_dondozo",        "Dondozo ex",             None),           # 水
    ("grass_ironleaves",     "Iron Leaves ex",         None),           # 草
]

# 通用训练师（取自样例卡组，已实测可用）
DEFAULT_TRAINERS = [
    ("Mega Signal", 4),
    ("Lillie's Determination", 4),
    ("Waitress", 4),
    ("Cyrano", 2),
    ("Maximum Belt", 1),
]


def _stable_hash(s: str) -> int:
    """与进程无关的字符串散列（内置 hash() 会随 PYTHONHASHSEED 变化）。"""
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


def find_card(pool: CardPool, name: str):
    ids = pool._name_to_ids.get(name)
    return ids[0] if ids else None


def build_archetype(pool: CardPool, attacker_name: str, backup_name: str | None,
                    trainers=DEFAULT_TRAINERS) -> tuple[list[int] | None, str]:
    """按一个攻击手构造卡组。返回 (deck, 说明)。"""
    atk_id = find_card(pool, attacker_name)
    if atk_id is None:
        return None, f"找不到卡：{attacker_name}"

    chain = pool.evolution_chain(atk_id)
    # 确保攻击手在链里（有些卡 evolution_chain 只返回同族）
    if atk_id not in chain:
        chain = list(chain) + [atk_id]

    deck: list[int] = []
    # 1) 主攻击手线：每阶 4 张（按卡名限 4，不同阶名字不同所以不冲突）
    for cid in chain:
        deck += [cid] * 4

    # 2) 备用攻击手（提升前期站场率）
    if backup_name:
        b = find_card(pool, backup_name)
        if b is not None and b not in chain:
            deck += [b] * 4

    # 3) 副攻击手线（只带 4 张宝可梦的话，主攻手一倒就空场）
    #    ⚠️ 必须【能量兼容】：否则会把 archetype 的能量稀释成 3~4 色。
    #       之前直接取 strong_basics[0]（永远是 Mega Latias，需火+超），
    #       结果每套卡组都被拖成三色，主策略全废。
    main_needs: set[int] = set()
    for cid in chain:
        main_needs |= pool.energy_types_needed(cid)
    same_base = pool.base_name(atk_id)
    # 优先：能量需求是主攻手需求的子集；退而求其次：只要无色能量
    sec_cands = [c for c in pool.strong_basics
                 if pool.base_name(c) != same_base
                 and pool.energy_types_needed(c) and pool.energy_types_needed(c) <= main_needs]
    if not sec_cands:
        sec_cands = [c for c in pool.strong_basics
                     if pool.base_name(c) != same_base
                     and not pool.energy_types_needed(c)]
    if sec_cands:
        # 在同分档里按名字散开，避免所有卡组都用同一张副攻手
        # ⚠️ 不能用内置 hash()：Python 的字符串 hash 每个进程随机（PYTHONHASHSEED），
        #    会导致同名 archetype 每次生成出不同的卡组，破坏可复现性。
        sec = sec_cands[_stable_hash(attacker_name) % len(sec_cands)]
        for cid in pool.evolution_chain(sec):
            deck += [cid] * 4

    # 4) 训练师
    for name, n in trainers:
        cid = find_card(pool, name)
        if cid is not None:
            deck += [cid] * n

    deck = deck[:60]
    # 5) 能量：按卡组里【所有攻击手】的需求配，并只补这些属性
    energy = energy_cards_for(pool, deck)
    if not energy:
        return None, "无法确定能量类型"
    deck = repair(pool, deck, random.Random(0), fill=energy)
    while len(deck) < 60:
        deck.append(energy[0] if len(energy) == 1 else energy[len(deck) % len(energy)])

    # 6) 校验
    ok, why = is_legal(pool, deck)
    if not ok:
        return None, f"非法：{why}"

    need = set()
    for cid in set(deck):
        c = pool.by_id.get(cid)
        if c is not None and int(c.cardType) == 0 and (getattr(c, "attacks", None) or []):
            need |= pool.energy_types_needed(cid)
    etypes = sorted(ENERGY_NAME.get(int(pool.by_id[c].energyType), "?")
                    for c in deck if pool.is_basic_energy(c))
    return deck, f"能量={sorted(set(etypes))} 需求={sorted(ENERGY_NAME.get(int(x),'?') for x in need)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="decks")
    ap.add_argument("--verify", action="store_true", help="生成后跑交叉对战矩阵")
    ap.add_argument("--games", type=int, default=120, help="验证时每对局数")
    ap.add_argument("--skip-existing", action="store_true",
                    help="已存在的 decks/<key>.csv 不重新生成（保护既有卡组不被改写）")
    args = ap.parse_args()

    pool = CardPool()
    out_dir = os.path.join(HERE, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("B1：构造 archetype 卡组")
    print("=" * 70)

    made: dict[str, list[int]] = {}
    for key, atk, backup in ARCHETYPES:
        path = os.path.join(out_dir, f"{key}.csv")
        if args.skip_existing and os.path.exists(path):
            print(f"  ⏭  {key:<20} 已存在，跳过")
            continue
        deck, note = build_archetype(pool, atk, backup)
        if deck is None:
            print(f"  ❌ {key:<20} {note}")
            continue
        # 引擎校验
        from cg.game import battle_start, battle_finish
        obs, sd = battle_start(deck, deck)
        ok_engine = obs is not None
        if obs:
            battle_finish()
        if not ok_engine:
            print(f"  ❌ {key:<20} 引擎拒绝 errorType={sd.errorType}")
            continue
        made[key] = deck
        path = os.path.join(out_dir, f"{key}.csv")
        with open(path, "w") as f:
            f.write("\n".join(str(c) for c in deck) + "\n")
        st = pool.deck_stats(deck)
        print(f"  ✅ {key:<20} {note}")
        print(f"       {pool.describe(deck, 5)}")
        print(f"       宝可梦{st['pokemon']} 训练师{st['trainers']} 能量{st['basic_energy']}")

    # 把官方样例也放进去当基准
    sample = pool.deck_from_file(os.path.join(HERE, "deck.csv"))
    made = {"sample_official": sample, **made}
    with open(os.path.join(out_dir, "sample_official.csv"), "w") as f:
        f.write("\n".join(str(c) for c in sample) + "\n")

    print(f"\n共生成 {len(made)} 套 -> {out_dir}/")
    if not args.verify:
        print("\n加 --verify 可跑交叉对战矩阵验证多样性")
        return

    # ---------------- 交叉验证
    print("\n" + "=" * 70)
    print(f"交叉对战矩阵（每格 {args.games} 局，座位交替）")
    print("=" * 70)
    import importlib.util
    from local_match import play_match
    sys.path.append(os.path.join(HERE, "agents"))
    spec = importlib.util.spec_from_file_location("g", os.path.join(HERE, "agents/greedy.py"))
    ag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ag)
    agent = ag.agent

    keys = list(made)
    short = {k: k[:9] for k in keys}

    def ev(a, b):
        w = l = d = 0
        for g in range(args.games):
            seat, rev = g % 2, (g // 2) % 2
            r = (play_match(agent, agent, made[a], made[b], reverse=bool(rev)) if seat == 0
                 else play_match(agent, agent, made[b], made[a], reverse=bool(rev)))
            res = r.get("result")
            if res is None:
                continue
            if res == 2:
                d += 1
            elif res == seat:
                w += 1
            else:
                l += 1
        p = w + l + d
        return (w + 0.5 * d) / p if p else 0.0

    print(f"{'':<20}" + "".join(f"{short[k]:>10}" for k in keys) + f"{'平均':>9}")
    avgs = {}
    for a in keys:
        row = [None if a == b else ev(a, b) for b in keys]
        vals = [v for v in row if v is not None]
        avgs[a] = sum(vals) / len(vals)
        print(f"{short[a]:<20}" + "".join(f"{'—':>10}" if v is None else f"{v:>10.3f}"
                                          for v in row) + f"{avgs[a]:>9.3f}")

    import math
    v = list(avgs.values())
    span = max(v) - min(v)
    se = math.sqrt(0.25 / args.games)
    print(f"\n平均胜率跨度 {min(v):.3f} ~ {max(v):.3f} = {span:.3f}")
    print(f"每格 {args.games} 局 SE≈{se:.3f}，2SE={2*se:.3f} => "
          f"{'多样性充足 ✅' if span > 4 * se else '差异偏小，可能 archetype 不够分明'}")


if __name__ == "__main__":
    main()
