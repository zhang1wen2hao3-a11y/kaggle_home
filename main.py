"""提交入口：PTCG AI Battle Challenge。

提交时上传的就是 本文件 + cg/ + deck.csv，所以这里必须自包含，
不能 import 本仓库的 agents/ 目录。

agent() 的约定：
  * 开局 obs.select is None  -> 必须返回 60 张卡组
  * 其余情况返回 option 下标列表，长度在 [minCount, maxCount] 之间，且不能重复
"""
import os

from cg.api import (OptionType, SelectType, SelectContext,
                    to_observation_class, all_attack)

# attackId -> (基础伤害, 能量需求数)。
# 用途：同一只宝可梦常有多个招式，而规则表按【动作类型】排序，所有 ATTACK 同级，
# 于是原本只会取列表里第一个 —— 而第一个常常是有条件的废招：
#   Mega Floette ex : Gentle Light(伤害0, 还给对手回血 30) / Eternity Bloom(200)
#   Mega Mawile ex  : Gobble Down(伤害0) / Huge Bite(260)
#   Mega Absol ex   : Terminal Period(伤害0, 需恰好6个伤害指示物) / Claw of Darkness(200)
# 实测（8 副卡组 vs random 的平均）：
#   取第一个招式 0.7312  ->  按伤害择优 0.9675
#   psychic_floette 0.37 -> 0.98，metal_mawile 0.54 -> 0.98
_ATTACK_INFO = {}
try:
    for _a in all_attack():
        _ATTACK_INFO[_a.attackId] = (int(getattr(_a, "damage", 0) or 0),
                                     len(getattr(_a, "energies", None) or []))
except Exception:
    pass

# 顺序即优先级。切勿把 ATTACK 排前面 —— 攻击会结束回合，
# 攻击优先会让 agent 每回合只平A、从不铺场（实测对 random 胜率仅 0.17）。
# 必须"先铺场/进化/充能，最后攻击"。此顺序与 agents/rulebase.py 的 develop_first 一致。
# 顺序即优先级。核心纪律：
#   ATTACK 绝不能排前面 —— 攻击会结束回合，攻击优先会让 agent 每回合只平A、
#   从不铺场（实测对 random 胜率仅 0.17 vs 0.85）。
#
# ⚠️ 另一个反直觉结果：把 RETREAT 挪到 ATTACK 之后（retreat_late）能在
#    **头对头**上赢本配置（0.5483, n=600），但**绝对强度更差**：
#        develop_first vs random = 0.8150
#        retreat_late  vs random = 0.7600
#    即"能打赢某个对手" ≠ "更强"（非传递性）。评估必须用统一对手做绝对比较。
PRIORITY = {
    OptionType.ATTACH: 0,
    OptionType.EVOLVE: 1,
    OptionType.PLAY: 2,
    OptionType.ABILITY: 3,
    OptionType.ATTACK: 4,
    OptionType.END: 5,
    # ⚠️ RETREAT / DISCARD 压到最后。历史上它们排在 ATTACK 之前，是**严重 bug**：
    #    有可撤退目标时 agent 会撤退而不攻击，白白浪费回合的攻击机会。
    #    8 副卡组 vs random 的平均胜率：旧顺序 0.2725 -> 修复后 0.7612；
    #    最极端的 fire_gouging 从 0.23 变成 0.89。
    #    之所以长期没发现，是因为所有评估都用 deck.csv(sample_official)，
    #    而它恰好是唯一不受该 bug 影响的卡组（两版都是 0.78）。
    OptionType.DISCARD: 90,
    OptionType.RETREAT: 91,
}
GO_FIRST = True


def read_deck_csv() -> list[int]:
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path) as f:
        return [int(x) for x in f.read().split() if x.strip()][:60]


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()

    opts = obs_dict["select"]["option"]
    n = len(opts)
    lo = obs.select.minCount
    hi = min(obs.select.maxCount, n)
    if hi <= 0 or n == 0:
        return []

    if obs.select.type == SelectType.YES_NO and obs.select.context == SelectContext.IS_FIRST:
        want = OptionType.YES if GO_FIRST else OptionType.NO
        for i, o in enumerate(opts):
            if o["type"] == want:
                return [i]
        return [0]

    if obs.select.type == SelectType.MAIN:
        def _key(i):
            t = OptionType(opts[i]["type"])
            p = PRIORITY.get(t, 99)
            if t != OptionType.ATTACK:
                return (p, 0, 0)
            dmg, cost = _ATTACK_INFO.get(opts[i].get("attackId"), (0, 9))
            return (p, -dmg, cost)      # ATTACK 内部按伤害降序，再按能量需求升序
        return [min(range(n), key=_key)]

    if obs.select.type == SelectType.YES_NO:
        for i, o in enumerate(opts):
            if o["type"] == OptionType.YES:
                return [i]
        return [0]

    # 铺设后备时选满：主力被击倒后 Active 空场会直接判负
    if obs.select.context in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH):
        return list(range(hi))

    k = max(lo, 1 if hi >= 1 else 0)
    return list(range(min(k, hi)))
