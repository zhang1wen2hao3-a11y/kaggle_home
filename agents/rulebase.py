"""参数化的规则策略：把"决策顺序"变成可实验的超参。

背景：PTCG 的 MAIN 阶段是**顺序动作**，而"攻击"会直接结束回合。
所以优先级排错，agent 就会变成"只会平A不会铺场"的废柴
（实测：ATTACK 排第一时，对 random 胜率仅 0.10）。

于是把顺序做成命名配置，用 evaluate.py / sweep.py 实测比较。
"""
from cg.api import OptionType, SelectType, SelectContext, all_attack

# attackId -> (基础伤害, 能量需求数)。
# 用来在【同一只宝可梦有多个招式】时择优：规则表按类型排序，所有 ATTACK 同级，
# 原本只会取列表里第一个 —— 而第一个常常是有条件的废招，例如
#   Mega Floette ex : Gentle Light(伤害0,还给对手回血) / Eternity Bloom(200)
#   Mega Mawile ex  : Gobble Down(伤害0) / Huge Bite(260)
# 实测 psychic_floette 因此只有 0.40（且 20/24 的败局是"牌库抽空"）。
_ATTACK_INFO = {}
try:
    for _a in all_attack():
        _ATTACK_INFO[_a.attackId] = (int(getattr(_a, "damage", 0) or 0),
                                     len(getattr(_a, "energies", None) or []))
except Exception:
    pass

# 顺序即优先级：列表越靠前越先做
ORDERS = {
    # 错误示范：攻击优先 -> 每回合只平A，从不铺场（保留作为对照）
    "attack_first": [
        OptionType.ATTACK, OptionType.EVOLVE, OptionType.ABILITY, OptionType.ATTACH,
        OptionType.PLAY, OptionType.RETREAT, OptionType.DISCARD, OptionType.END,
    ],
    # 先铺场、再进化、后攻击
    "develop_first": [
        OptionType.PLAY, OptionType.EVOLVE, OptionType.ABILITY, OptionType.ATTACH,
        OptionType.RETREAT, OptionType.DISCARD, OptionType.ATTACK, OptionType.END,
    ],
    # 先充能（能量是攻击的前提）
    "attach_first": [
        OptionType.ATTACH, OptionType.PLAY, OptionType.EVOLVE, OptionType.ABILITY,
        OptionType.RETREAT, OptionType.DISCARD, OptionType.ATTACK, OptionType.END,
    ],
    # 铺场优先但不撤退（撤退浪费回合）
    "no_retreat": [
        OptionType.PLAY, OptionType.ABILITY, OptionType.EVOLVE, OptionType.ATTACH,
        OptionType.DISCARD, OptionType.ATTACK, OptionType.RETREAT, OptionType.END,
    ],
    # RETREAT 挪到 ATTACK 之后 —— 撤退会浪费本回合攻击机会
    "retreat_late": [
        OptionType.PLAY, OptionType.EVOLVE, OptionType.ABILITY, OptionType.ATTACH,
        OptionType.DISCARD, OptionType.ATTACK, OptionType.RETREAT, OptionType.END,
    ],
    # 进化优先（实测 EVOLVE 只占 0.3%，可能被 PLAY 挤掉了）
    "evolve_first": [
        OptionType.EVOLVE, OptionType.PLAY, OptionType.ABILITY, OptionType.ATTACH,
        OptionType.ATTACK, OptionType.RETREAT, OptionType.DISCARD, OptionType.END,
    ],
    # 纯平A但把 END 排最后（对照：确认 END 不是问题所在）
    "attack_last_only": [
        OptionType.ATTACH, OptionType.EVOLVE, OptionType.PLAY, OptionType.ABILITY,
        OptionType.ATTACK, OptionType.END,
    ],
    # ★ 规范配置：当前 main.py / agents/greedy.py 用的就是它。
    #   与 attack_last_only 行为相同（MAIN 阶段只有 DISCARD/RETREAT 是未列出项），
    #   但把 DISCARD/RETREAT 显式列在最后，使三处优先级表**逐位一致**，
    #   便于 tests/test_priority_sync.py 做强校验。
    #
    #   为什么 RETREAT 必须排在 ATTACK 之后：见本文件开头与 agents/greedy.py 的说明。
    #   8 副卡组 vs random 平均：旧顺序 0.2725 -> 本配置 0.7612。
    "safe_v1": [
        OptionType.ATTACH, OptionType.EVOLVE, OptionType.PLAY, OptionType.ABILITY,
        OptionType.ATTACK, OptionType.END,
        OptionType.DISCARD, OptionType.RETREAT,
    ],
}

# 选择多个选项时，这些 context 取满 maxCount（铺满后备 = 抗打击）
FILL_CONTEXTS = {
    SelectContext.SETUP_BENCH_POKEMON,
    SelectContext.TO_BENCH,
}


def make_agent(order="develop_first", go_first=True, bench_full=True,
               attack_pick="first"):
    """构造一个规则 agent。

    Args:
        order: ORDERS 里的配置名
        go_first: 开局是否选择先手
        bench_full: 铺后备时是否选满 maxCount
    """
    if order not in ORDERS:
        raise KeyError(f"未知 order: {order}，可选 {list(ORDERS)}")
    prio = {t: i for i, t in enumerate(ORDERS[order])}

    def agent(obs_dict):
        sel = obs_dict.get("select")
        if sel is None:
            return _deck()

        opts = sel["option"]
        n = len(opts)
        lo = sel["minCount"]
        hi = min(sel["maxCount"], n)
        if n == 0 or hi <= 0:
            return []
        stype = SelectType(sel["type"])
        ctx = SelectContext(sel["context"])

        # 开局选先后手
        if stype == SelectType.YES_NO and ctx == SelectContext.IS_FIRST:
            want = OptionType.YES if go_first else OptionType.NO
            for i, o in enumerate(opts):
                if o["type"] == want:
                    return [i]
            return [0]

        # 主阶段：按优先级挑第一个动作（一次一个）
        if stype == SelectType.MAIN:
            if attack_pick == "first":
                best = min(range(n), key=lambda i: prio.get(OptionType(opts[i]["type"]), 99))
                return [best]
            # 多招式择优：优先级仍主导，同类 ATTACK 内部再按伤害降序
            def _key(i):
                t = OptionType(opts[i]["type"])
                p = prio.get(t, 99)
                if t != OptionType.ATTACK:
                    return (p, 0, 0)
                info = _ATTACK_INFO.get(opts[i].get("attackId"))
                dmg = info[0] if info else 0
                cost = info[1] if info else 9
                if attack_pick == "maxdmg":
                    return (p, -dmg, cost)
                if attack_pick == "maxdmg_eff":       # 伤害/能量
                    return (p, -(dmg / max(cost, 1)), cost)
                return (p, -dmg, cost)
            return [min(range(n), key=_key)]

        # 铺后备：选满，提高抗打击能力
        if bench_full and ctx in FILL_CONTEXTS:
            return list(range(hi))[:hi]

        # 是/否：默认选"是"
        if stype == SelectType.YES_NO:
            for i, o in enumerate(opts):
                if o["type"] == OptionType.YES:
                    return [i]
            return [0]

        # 其余：满足下限即可，避免乱选牌
        k = max(lo, 1 if hi >= 1 else 0)
        return list(range(min(k, hi)))

    return agent


def _deck():
    import os
    path = "deck.csv" if os.path.exists("deck.csv") else "/kaggle_simulations/agent/deck.csv"
    with open(path) as f:
        return [int(x) for x in f.read().split() if x.strip()][:60]
