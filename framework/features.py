"""特征提取 —— 所有模型共享的特征层。

这是"模型融合"的接口：任何模型（线性 / LightGBM / XGBoost / 神经网络）
都消费同一份扁平特征字典，所以换模型不需要改特征代码。

三条设计纪律：
  1. **只用公开信息**。隐藏信息（对手手牌内容、牌库顺序、里侧奖赏卡）
     绝对不能进特征 —— 线上拿不到，用了模型无法迁移。
     隐藏信息只能用于【生成训练标签】。
  2. **特征名稳定**。用带前缀的字符串键（st_/opt_/mem_），便于跨版本对比重要性。
  3. **纯 Python 零依赖**。不引入 numpy，保证框架在任何环境可跑。

用法：
    from framework import features
    f = features.state_features(obs_dict, memory)        # V 用
    f = features.option_features(obs_dict, i)            # P 用
    f = features.combined_features(obs_dict, i, memory)  # P 用（状态+选项）
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

# ---------------------------------------------------------------- 卡牌数据缓存
_CARD_CACHE: dict[int, Any] = {}
_ATTACK_CACHE: dict[int, Any] = {}


def _cards() -> dict[int, Any]:
    """惰性加载卡牌数据（需要引擎已初始化）。"""
    if not _CARD_CACHE:
        try:
            from cg.api import all_card_data, all_attack
            for c in all_card_data():
                _CARD_CACHE[c.cardId] = c
            for a in all_attack():
                _ATTACK_CACHE[a.attackId] = a
        except Exception:
            pass
    return _CARD_CACHE


def _card(card_id: int | None):
    return _cards().get(card_id) if card_id is not None else None


# ---------------------------------------------------------------- 选项类型枚举
# 与 cg.api.OptionType 对齐（此处硬编码，避免特征层依赖引擎导入）
OPTION_TYPES = [
    "NUMBER", "YES", "NO", "CARD", "TOOL_CARD", "ENERGY_CARD", "ENERGY",
    "PLAY", "ATTACH", "EVOLVE", "ABILITY", "DISCARD", "RETREAT", "ATTACK",
    "END", "SKILL", "SPECIAL_CONDITION",
]
_OPT_IDX = {i: n for i, n in enumerate(OPTION_TYPES)}


# ---------------------------------------------------------------- 场面辅助
def _hp_sum(pokemons: list) -> int:
    return sum((p.get("hp") or 0) for p in pokemons if p)


def _energy_count(pokemon: dict | None) -> int:
    if not pokemon:
        return 0
    return len(pokemon.get("energies") or [])


@lru_cache(maxsize=8192)
def _sibling_attack_max(attack_ids: tuple):
    """一个决策里所有选项的最大招式伤害 + 攻击选项个数。

    用来生成"相对兄弟选项"的特征 —— GBDT 无法跨行比较，这类特征必须显式给。
    """
    dmg, n = 0.0, 0
    for aid in attack_ids:
        a = _ATTACK_CACHE.get(aid) if aid is not None else None
        if a is None:
            continue
        n += 1
        dmg = max(dmg, float(a.damage or 0))
    return dmg, n


def _energy_count_of(pk) -> float:
    if not pk:
        return 0.0
    e = pk.get("energies")
    if e:
        return float(len(e))
    return float(len(pk.get("energyCards") or []))


def _best_attack_stats(card_id):
    """某张宝可梦卡：最强招式伤害 + 最便宜招式能量需求。"""
    c = _card(card_id)
    if c is None:
        return 0.0, 0.0
    max_d, min_c = 0.0, 0.0
    for aid in (getattr(c, "attacks", None) or []):
        a = _ATTACK_CACHE.get(aid)
        if a is None:
            continue
        max_d = max(max_d, float(a.damage or 0))
        cst = float(len(a.energies or []))
        if cst > 0 and (min_c == 0.0 or cst < min_c):
            min_c = cst
    return max_d, min_c


def _target_pokemon(player: dict, area, index):
    """把选项的 area/index 解析成【我方某只宝可梦】。area=4 出场 / 5 后备。"""
    try:
        if area == 4:
            arr = player.get("active") or []
        elif area == 5:
            arr = player.get("bench") or []
        else:
            return None
        arr = [x for x in arr if x]
        i = int(index) if index is not None else -1
        return arr[i] if 0 <= i < len(arr) else None
    except Exception:
        return None


def _active_of(player: dict) -> dict | None:
    act = player.get("active") or []
    return act[0] if act and act[0] else None


def _has_special_condition(player: dict) -> int:
    return sum(1 for k in ("poisoned", "burned", "asleep", "paralyzed", "confused")
               if player.get(k))


# ---------------------------------------------------------------- 状态特征
def state_features(obs: dict, memory=None) -> dict[str, float]:
    """从【当前行动方】视角提取状态特征。

    视角约定：所有 *_diff 特征都是 (我方 - 对手)，正值表示我方占优。
    这是不完美信息博弈的标准做法——同一份局面换个视角，V 的值应该取反。

    Args:
        obs: 引擎给的原始 obs（dict）
        memory: 可选的 GameMemory，提供对手已见卡等信息
    """
    f: dict[str, float] = {}
    cur = obs.get("current")
    if not cur:
        return _empty_state(f)

    me_idx = int(cur.get("yourIndex", 0))
    players = cur.get("players") or []
    if len(players) < 2:
        return _empty_state(f)
    me, op = players[me_idx], players[1 - me_idx]

    # ---- 回合与阶段
    # ⚠️ 必须用「回合轮数」而不是绝对回合数。
    #    绝对回合在两人之间交替（1,2,3,4...），它的**奇偶**等价于编码了"谁在行动"，
    #    而视角特征又是相对的（*_diff = 我方-对手），两者叠加会让模型学到
    #    "奇数回合的行动方更可能输"这种伪相关。
    #    实测：全特征时按回合奇偶分组的平均预测差 0.068（偶0.530/奇0.462）；
    #    这个偏差会直接毒化搜索——因为搜索里"结束回合"的动作会让 yourIndex 翻转，
    #    翻转后的估值被这个偏差系统性抬高，导致 agent 96% 的决策都选 END。
    _turn = int(cur.get("turn", 0))
    f["st_round"] = float(_turn // 2)
    f["st_turn_action"] = float(cur.get("turnActionCount", 0))
    f["st_is_first_player"] = 1.0 if cur.get("firstPlayer") == me_idx else 0.0

    # ---- 奖赏卡进度（越少越好，所以取 对手-我方 = 我方领先量）
    my_prize, op_prize = len(me.get("prize") or []), len(op.get("prize") or [])
    f["st_prize_me"] = float(my_prize)
    f["st_prize_adv"] = float(op_prize - my_prize)          # 正 = 我领先
    f["st_prize_progress"] = float(6 - my_prize) / 6.0

    # ---- 资源
    f["st_deck_me"] = float(me.get("deckCount", 0))
    f["st_deck_diff"] = float(me.get("deckCount", 0) - op.get("deckCount", 0))
    f["st_hand_me"] = float(me.get("handCount", 0))
    f["st_hand_diff"] = float(me.get("handCount", 0) - op.get("handCount", 0))
    f["st_discard_me"] = float(len(me.get("discard") or []))
    f["st_discard_diff"] = float(len(me.get("discard") or []) - len(op.get("discard") or []))
    f["st_deck_low"] = 1.0 if me.get("deckCount", 60) <= 5 else 0.0   # 快抽光了 = 危险

    # ---- 场面
    my_active, op_active = _active_of(me), _active_of(op)
    f["st_bench_me"] = float(len(me.get("bench") or []))
    f["st_bench_diff"] = float(len(me.get("bench") or []) - len(op.get("bench") or []))
    f["st_bench_max"] = float(me.get("benchMax", 0))
    f["st_active_missing"] = 1.0 if my_active is None else 0.0      # 空场 = 直接判负风险
    f["st_op_active_missing"] = 1.0 if op_active is None else 0.0

    my_hp, op_hp = _hp_sum([my_active] + list(me.get("bench") or [])), \
                   _hp_sum([op_active] + list(op.get("bench") or []))
    f["st_hp_total_me"] = float(my_hp)
    f["st_hp_total_diff"] = float(my_hp - op_hp)
    f["st_active_hp"] = float((my_active or {}).get("hp") or 0)
    f["st_active_hp_ratio"] = (
        float((my_active or {}).get("hp") or 0) / max((my_active or {}).get("maxHp") or 1, 1)
    )
    f["st_op_active_hp"] = float((op_active or {}).get("hp") or 0)
    f["st_op_active_hp_ratio"] = (
        float((op_active or {}).get("hp") or 0) / max((op_active or {}).get("maxHp") or 1, 1)
    )

    # ---- 能量
    my_e = _energy_count(my_active) + sum(_energy_count(p) for p in (me.get("bench") or []))
    op_e = _energy_count(op_active) + sum(_energy_count(p) for p in (op.get("bench") or []))
    f["st_energy_me"] = float(my_e)
    f["st_energy_diff"] = float(my_e - op_e)
    f["st_active_energy"] = float(_energy_count(my_active))
    f["st_op_active_energy"] = float(_energy_count(op_active))

    # ---- 本回合已用资源
    f["st_supporter_played"] = 1.0 if cur.get("supporterPlayed") else 0.0
    f["st_energy_attached"] = 1.0 if cur.get("energyAttached") else 0.0
    f["st_stadium_played"] = 1.0 if cur.get("stadiumPlayed") else 0.0
    f["st_retreated"] = 1.0 if cur.get("retreated") else 0.0
    f["st_stadium_in_play"] = float(len(cur.get("stadium") or []))

    # ---- 特殊状态
    f["st_special_me"] = float(_has_special_condition(me))
    f["st_special_op"] = float(_has_special_condition(op))
    f["st_special_diff"] = float(_has_special_condition(me) - _has_special_condition(op))

    # ---- 能否攻击
    f["st_can_attack"] = 1.0 if _can_attack(my_active) else 0.0
    f["st_op_can_attack"] = 1.0 if _can_attack(op_active) else 0.0

    # ---- KO 威胁（核心战术量：我打对手几下能 KO / 对手打我几下）
    am, ao = _attack_info(my_active), _attack_info(op_active)
    my_hp_now = float((my_active or {}).get("hp") or 0)
    op_hp_now = float((op_active or {}).get("hp") or 0)
    # 注意：**不要**直接暴露原始伤害值（dmg_now/dmg_max）。
    # 实测（pool_v3_v，同数据集消融）：
    #   加 KO压力(比值)  -> AUC 0.8247 -> 0.8286  ✅
    #   加 回合到KO      -> AUC 0.8247 -> 0.8284  ✅
    #   加 原始伤害值    -> AUC 0.8247 -> 0.8172  ❌
    # 原因：原始伤害是绝对值（0~300），与已有 HP 特征共线，纯增噪；
    #       而归一化后的比值（伤害/血量）才是真正的战术信号。
    f["st_energy_gap_me"] = am["gap"]            # 还差几个能量能攻击
    f["st_energy_gap_op"] = ao["gap"]
    f["st_active_hp_gap"] = my_hp_now - op_hp_now # 双方 Active 血量差
    # KO 压力：伤害 / 对手血量。>1 表示这回合能直接击倒
    f["st_ko_pressure_me"] = (am["dmg_now"] / op_hp_now) if op_hp_now > 0 else 0.0
    f["st_ko_pressure_op"] = (ao["dmg_now"] / my_hp_now) if my_hp_now > 0 else 0.0
    # 还需要几回合才能 KO 对手（向下取整 +1），0 表示本来就能 KO
    # ⚠️ 必须显式 float()：max(0.0, int) 在 int 更大时会返回 int，
    #    导致同一特征在不同样本上出现 int/float 混型 -> parquet 跨 part schema 不一致。
    f["st_turns_to_ko_me"] = float(
        max(0.0, math.ceil(op_hp_now / am["dmg_now"]) - 1)) if am["dmg_now"] > 0 else 99.0
    f["st_turns_to_ko_op"] = float(
        max(0.0, math.ceil(my_hp_now / ao["dmg_now"]) - 1)) if ao["dmg_now"] > 0 else 99.0
    # 威胁差：谁能更快达成 KO
    f["st_ko_race"] = f["st_turns_to_ko_op"] - f["st_turns_to_ko_me"]
    # 奖赏卡还需要几次 KO 才能赢
    f["st_prize_need_me"] = float(my_prize)
    f["st_prize_need_op"] = float(op_prize)

    # ---- 记忆特征（对手已见信息）
    if memory is not None:
        f.update(memory.features(me_idx))

    return f


def _attack_info(pokemon: dict | None) -> dict:
    """算一个宝可梦的攻防数值 —— 卡牌游戏最核心的战术量。

    Returns:
        dmg_now  : 当前能量下**实际能打出**的最大伤害（付不起的招式不算）
        dmg_max  : 所有招式的最大伤害（理论值，用于衡量潜在威胁）
        min_cost : 最便宜招式的能量需求
        gap      : 还差几个能量才能攻击（0 表示已经能攻击）
        have     : 当前已附能量数
    """
    out = {"dmg_now": 0.0, "dmg_max": 0.0, "min_cost": 0.0, "gap": 0.0, "have": 0.0}
    if not pokemon:
        return out
    card = _card(pokemon.get("id"))
    if card is None:
        return out
    have = len(pokemon.get("energies") or [])
    out["have"] = float(have)
    costs, dmgs_now = [], []
    for aid in (card.attacks or []):
        atk = _ATTACK_CACHE.get(aid)
        if atk is None:
            continue
        cost = len(atk.energies or [])
        dmg = float(atk.damage or 0)
        costs.append(cost)
        out["dmg_max"] = max(out["dmg_max"], dmg)
        if cost <= have:
            dmgs_now.append(dmg)
    out["dmg_now"] = max(dmgs_now) if dmgs_now else 0.0
    if costs:
        out["min_cost"] = float(min(costs))
        out["gap"] = max(0.0, out["min_cost"] - have)
    return out


def _can_attack(pokemon: dict | None) -> bool:
    """当前 Active 是否付得起至少一个招式的能量。"""
    return _attack_info(pokemon)["gap"] == 0.0 and _attack_info(pokemon)["min_cost"] > 0


def _empty_state(f: dict) -> dict[str, float]:
    """current 缺失时的占位特征（保证特征维度一致）。"""
    for k in STATE_FEATURE_KEYS:
        f.setdefault(k, 0.0)
    return f


# state_features 产出的全部键（缺失时补 0，保证维度对齐）
STATE_FEATURE_KEYS = [
    "st_round", "st_turn_action", "st_is_first_player",
    "st_prize_me", "st_prize_adv", "st_prize_progress",
    "st_deck_me", "st_deck_diff", "st_hand_me", "st_hand_diff",
    "st_discard_me", "st_discard_diff", "st_deck_low",
    "st_bench_me", "st_bench_diff", "st_bench_max",
    "st_active_missing", "st_op_active_missing",
    "st_hp_total_me", "st_hp_total_diff", "st_active_hp", "st_active_hp_ratio",
    "st_op_active_hp", "st_op_active_hp_ratio",
    "st_energy_me", "st_energy_diff", "st_active_energy", "st_op_active_energy",
    "st_supporter_played", "st_energy_attached", "st_stadium_played",
    "st_retreated", "st_stadium_in_play",
    "st_special_me", "st_special_op", "st_special_diff",
    "st_can_attack", "st_op_can_attack",
    # KO 威胁（核心战术量，只用归一化后的比值）
    "st_energy_gap_me", "st_energy_gap_op", "st_active_hp_gap",
    "st_ko_pressure_me", "st_ko_pressure_op",
    "st_turns_to_ko_me", "st_turns_to_ko_op", "st_ko_race",
    "st_prize_need_me", "st_prize_need_op",
]


# ---------------------------------------------------------------- 选项特征
def option_features(obs: dict, idx: int, memory=None) -> dict[str, float]:
    """提取第 idx 个候选动作的特征（P / LTR 模型用）。

    含 deck-awareness 特征：能看出"这张能量是不是我招式需要的"。
    """
    f: dict[str, float] = {}
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    if idx >= len(opts):
        return _empty_option(f)

    cur = obs.get("current") or {}
    me_idx = int(cur.get("yourIndex", 0))
    players = cur.get("players") or [{}, {}]
    me = players[me_idx] if len(players) > me_idx else {}

    o = opts[idx]
    otype = int(o.get("type", 0))
    for i, name in _OPT_IDX.items():
        f[f"opt_t_{name}"] = 1.0 if otype == i else 0.0
    f["opt_n_options"] = float(len(opts))
    f["opt_min"] = float(sel.get("minCount", 0))
    f["opt_max"] = float(sel.get("maxCount", 0))
    f["opt_select_type"] = float(sel.get("type", 0))
    f["opt_context"] = float(sel.get("context", 0))

    # 源卡反查：从 area/index 找到具体是哪张卡
    src_id = _option_source_card_id(obs, o)
    f["opt_has_source"] = 1.0 if src_id is not None else 0.0
    if src_id is not None:
        src = _card(src_id)
        f["opt_src_is_energy"] = 1.0 if (src is not None and int(src.cardType) == 5) else 0.0
        f["opt_src_is_pokemon"] = 1.0 if (src is not None and int(src.cardType) == 0) else 0.0
        f["opt_src_is_supporter"] = 1.0 if (src is not None and int(src.cardType) == 3) else 0.0
        # ★ deck-awareness: 这张能量是不是我 Active 需要的属性
        f["opt_src_energy_matches"] = _energy_matches_active(src, _active_of(me))
    else:
        f["opt_src_is_energy"] = 0.0
        f["opt_src_is_pokemon"] = 0.0
        f["opt_src_is_supporter"] = 0.0
        f["opt_src_energy_matches"] = 0.0

    # ---------------- 攻击复合特征
    # 设计意图：基础的"伤害"已有，但真正影响决策的是【相对量】——
    # 相对对手血量（能不能打死/溢出多少）、相对能量（还差几张）、
    # 相对同决策的其它攻击选项（是不是最优选择）。GBDT 跨行比较不了，必须显式给。
    _my_active = _active_of(me)
    _op_active = _active_of(players[1 - me_idx] if len(players) > 1 else {})
    _dmg = _cost = 0.0
    _has_atk = False
    _aid = o.get("attackId")
    if _aid is not None and _aid in _ATTACK_CACHE:
        _a = _ATTACK_CACHE[_aid]
        _dmg = float(_a.damage or 0)
        _cost = float(len(_a.energies or []))
        _has_atk = True
    _op_hp = float((_op_active or {}).get("hp") or 0)
    f["opt_attack_overkill"] = (_dmg - _op_hp) if _op_hp else 0.0
    f["opt_attack_gap"] = max(0.0, _op_hp - _dmg) if _op_hp else 0.0
    f["opt_attack_dmg_per_energy"] = (_dmg / _cost) if _cost > 0 else 0.0
    f["opt_attack_energy_gap"] = (max(0.0, _cost - _energy_count_of(_my_active))
                                  if _has_atk else 0.0)
    _maxd, _natk = _sibling_attack_max(tuple(x.get("attackId") for x in opts))
    f["opt_n_attack_options"] = float(_natk)
    f["opt_attack_dmg_gap_best"] = (_dmg - _maxd) if _has_atk else 0.0
    f["opt_attack_dmg_frac_best"] = (_dmg / _maxd) if (_has_atk and _maxd > 0) else 0.0
    f["opt_attack_is_best"] = 1.0 if (_has_atk and _dmg >= _maxd and _natk >= 2) else 0.0

    # 目标状态特征已按实测删除（见 OPTION_FEATURE_KEYS 后的说明）

    # 攻击选项：伤害 / 能否 KO
    attack_id = o.get("attackId")
    if attack_id is not None and attack_id in _ATTACK_CACHE:
        atk = _ATTACK_CACHE[attack_id]
        dmg = float(atk.damage or 0)
        f["opt_attack_damage"] = dmg
        f["opt_attack_cost"] = float(len(atk.energies or []))
        op_active = _active_of(players[1 - me_idx] if len(players) > 1 else {})
        f["opt_attack_can_ko"] = 1.0 if (op_active and dmg >= (op_active.get("hp") or 0)) else 0.0
        f["opt_attack_is_ko_pressure"] = 1.0 if (op_active and op_active.get("hp") and
                                                 dmg >= 0.5 * op_active["hp"]) else 0.0
    else:
        f["opt_attack_damage"] = 0.0
        f["opt_attack_cost"] = 0.0
        f["opt_attack_can_ko"] = 0.0
        f["opt_attack_is_ko_pressure"] = 0.0

    if memory is not None:
        f.update(memory.option_features(obs, idx))

    return f


def _option_source_card_id(obs: dict, option: dict) -> int | None:
    """从 option 的 area/index 反查源卡的 cardId。

    ATTACH/PLAY/EVOLVE 等选项用 (area, index) 指代卡牌，
    HAND 区域可以直接从 obs 里的手牌查到。
    """
    cur = obs.get("current") or {}
    me_idx = int(cur.get("yourIndex", 0))
    players = cur.get("players") or []
    if len(players) <= me_idx:
        return None
    me = players[me_idx]
    area, index = option.get("area"), option.get("index")
    if area == 2 and index is not None:                 # AreaType.HAND
        hand = me.get("hand") or []
        if 0 <= index < len(hand):
            return hand[index].get("id")
    if option.get("cardId") is not None:
        return option["cardId"]
    return None


def _energy_matches_active(src_card, active: dict | None) -> float:
    """★ 核心的 deck-awareness 特征：这张能量是否匹配当前 Active 的招式需求。"""
    if src_card is None or active is None:
        return 0.0
    active_card = _card(active.get("id"))
    if active_card is None:
        return 0.0
    src_type = int(getattr(src_card, "energyType", 0))
    for aid in (active_card.attacks or []):
        atk = _ATTACK_CACHE.get(aid)
        if atk and src_type in [int(e) for e in (atk.energies or [])]:
            return 1.0
    return 0.0


def _empty_option(f: dict) -> dict[str, float]:
    for k in OPTION_FEATURE_KEYS:
        f.setdefault(k, 0.0)
    return f


OPTION_FEATURE_KEYS = (
    [f"opt_t_{n}" for n in OPTION_TYPES] +
    ["opt_n_options", "opt_min", "opt_max", "opt_select_type", "opt_context",
     "opt_has_source", "opt_src_is_energy", "opt_src_is_pokemon",
     "opt_src_is_supporter", "opt_src_energy_matches",
     "opt_attack_damage", "opt_attack_cost", "opt_attack_can_ko",
     "opt_attack_is_ko_pressure",
     # ---- 攻击复合（跨局面 / 跨兄弟选项）★ GBDT 自己构造不出来的相对量
     "opt_attack_overkill", "opt_attack_gap", "opt_attack_dmg_per_energy",
     "opt_attack_energy_gap", "opt_attack_dmg_gap_best",
     "opt_attack_dmg_frac_best", "opt_n_attack_options", "opt_attack_is_best"]
)
# ⚠️ 曾加过 9 个"目标状态"特征（opt_tgt_hp / opt_tgt_energy / opt_tgt_energy_gap
#     / opt_tgt_max_dmg / opt_tgt_is_active / opt_tgt_weak_to_op / opt_opp_weak_to_tgt …），
#     意图让模型学会"贴能量给哪只 / 换谁上场"。
#     但**逐折配对实测显示它们有害**（data/cf_q_v3.jsonl，非 MAIN 决策一致率）：
#         有目标特征 55.46%   vs   无目标特征 57.18%    =>  -1.72pp
#     MAIN 决策上也无收益（61.86% vs 61.41%，噪声内）。
#     推测原因：这些特征在 MAIN 决策上恒为 0、只在目标决策上非零，等于塞进一个
#     "这是不是目标决策"的标识位，助长过拟合。
#     结论：已删除。要重做需先解决"MAIN 与目标决策混训"这个结构问题。


def combined_features(obs: dict, idx: int, memory=None) -> dict[str, float]:
    """状态 + 选项（P / LTR 模型用）。"""
    f = state_features(obs, memory)
    f.update(option_features(obs, idx, memory))
    return f


# ---------------------------------------------------------------- 矩阵化
def feature_keys(rows: list[dict]) -> list[str]:
    """收集所有出现过的特征名（排序保证稳定）。"""
    keys: set[str] = set()
    for r in rows:
        keys.update(r.keys())
    return sorted(keys)


def to_matrix(rows: list[dict], keys: list[str] | None = None):
    """把特征字典列表转成 (特征名, 二维列表)。

    这是给模型层的统一入口：线性模型、LightGBM、XGBoost、sklearn 都能直接吃。
    不依赖 numpy —— 需要时由适配器自己转。
    """
    if keys is None:
        keys = feature_keys(rows)
    idx = {k: i for i, k in enumerate(keys)}
    mat = [[0.0] * len(keys) for _ in rows]
    for r, row in enumerate(rows):
        for k, v in row.items():
            j = idx.get(k)
            if j is not None:
                mat[r][j] = float(v)
    return keys, mat
