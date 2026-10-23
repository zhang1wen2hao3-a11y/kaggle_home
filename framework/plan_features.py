"""牌组计划特征：补上"这张牌在这套牌里是什么角色、这条进化线还缺哪一环"。

## 为什么要单独一层

`framework/features.py` 的 48+39 维特征是为**规则 agent** 设计的（类型优先级、HP、能量、伤害）。
行为克隆的实测说明它表达不了强 agent 的策略：

    强 agent 与我们规则的一致率只有 31.8%
    BC 留出局 top-1 只有 30.4%（随机 17.9%），而且 train 38.3% > holdout ⇒ 连拟合都吃力

缺的正是这一层信息：

* **卡牌角色**（检索 / 抽卡 / 回收 / 加速 / 有代价 / 条件 / 掷币 / 换位）
* **进化线的位置**（这是缺的那一环吗？前置在不在手上/场上？）
* **牌组计划**（这套牌的主攻手是谁、能量要贴给谁、现在离能攻击差几张）
* **冗余度**（同一张牌手上还有几张 —— 多余的才敢打出去/弃掉）

## 用法

    from framework.plan_features import build_plan, option_plan_features, state_plan_features
    plan = build_plan(observed_card_ids)      # 一局一次
    f = option_plan_features(obs, i, plan)    # 每个选项
    g = state_plan_features(obs, plan)        # 每个状态

`observed_card_ids` 用该 agent 在这场里**见过的所有自己的牌**（回放里自己的手牌是可见的，
所以并集足够近似它的牌组构成）。
"""
from __future__ import annotations

import collections

from framework.rollout import _load_card_stats, _roles_of, _get

ROLE_NAMES = ("search", "draw", "recover", "accel", "conditional",
              "discard_cost", "coin", "switch", "heal", "cannot_next")

OPTION_PLAN_KEYS = (
    [f"ob_role_{r}" for r in ROLE_NAMES] +
    ["ob_is_attacker", "ob_on_line", "ob_completes_line", "ob_is_pre_of_line",
     "ob_card_hp", "ob_card_min_cost", "ob_card_max_dmg", "ob_card_dpe",
     "ob_card_is_ex", "ob_card_stage", "ob_card_is_basic", "ob_card_is_energy",
     "ob_copies_in_hand", "ob_energy_matches_attacker",
     # ATTACH 目标（inPlayArea/inPlayIndex 指向我方宝可梦）
     "ob_tgt_has", "ob_tgt_is_attacker", "ob_tgt_energy_gap", "ob_tgt_hp_ratio"]
)

STATE_PLAN_KEYS = (
    ["obp_attacker_in_play", "obp_attacker_ready", "obp_attacker_energy_gap",
     "obp_line_piece_in_hand", "obp_basic_in_play", "obp_basic_in_hand",
     "obp_seen_types", "obp_attacker_hp_ratio"]
)


def build_plan(observed_ids) -> dict:
    """从"这场里见过的自己的牌"推断牌组计划。"""
    stats = _load_card_stats()
    ids = [int(c) for c in (observed_ids or []) if c]
    best = None
    for cid in ids:
        st = stats.get(cid)
        if not st or st["max_dmg"] <= 0 or st["min_cost"] <= 0:
            continue
        score = st["max_dmg"] / st["min_cost"]
        if best is None or score > best[0]:
            best = (score, cid)
    attacker = best[1] if best else None
    att_name = None
    pre_names = set()
    line_ids = set()
    if attacker:
        a = stats[attacker]
        line_ids.add(attacker)
        pre_names = {a["evolves_from"]} if a["evolves_from"] else set()
        # 谁进化成它 / 它进化成谁，都算同一条线
        for cid in ids:
            st = stats.get(cid)
            if not st or not st["evolves_from"]:
                continue
            if st["evolves_from"] in pre_names or cid == attacker:
                line_ids.add(cid)
    return {"attacker": attacker, "line_ids": line_ids,
            "pre_names": pre_names, "all_ids": set(ids)}


def _pokemon_names_on_board(obs, me):
    """我方场上/手牌里出现过的宝可梦名字（判断"缺的那一环在不在"）。"""
    st = _load_card_stats()
    names = set()
    p = (obs.get("current") or {}).get("players") or []
    if me >= len(p) or not p[me]:
        return names
    d = p[me]
    for pk in list(d.get("active") or []) + list(d.get("bench") or []):
        if pk:
            names.add(_name_of(_get(pk, "id")))
    for c in (d.get("hand") or []):
        names.add(_name_of(_get(c, "id")))
    names.discard(None)
    return names


_NAME_CACHE: dict = {}


def _name_of(cid):
    if cid is None:
        return None
    cid = int(cid)
    if cid in _NAME_CACHE:
        return _NAME_CACHE[cid]
    nm = None
    try:
        from cg.api import all_card_data          # noqa: F401  (只在有引擎时可用)
        global _ID2NAME
        try:
            _ID2NAME
        except NameError:
            _ID2NAME = {int(c.cardId): c.name for c in all_card_data()}
        nm = _ID2NAME.get(cid)
    except Exception:
        nm = None
    _NAME_CACHE[cid] = nm
    return nm


def _resolve_opt_card(obs, opt):
    """option → cardId。MAIN 的 PLAY/EVOLVE 没有 area，隐含是手牌。"""
    area, idx = opt.get("area"), opt.get("index")
    if area is None:
        if int(opt.get("type", -1)) in (7, 8, 9):
            area = 2
        else:
            return None
    if idx is None:
        return None
    cur = obs.get("current") or {}
    me = int(cur.get("yourIndex", 0))
    pls = cur.get("players") or []
    if area == 1:
        arr = (obs.get("select") or {}).get("deck") or []
        return arr[idx].get("id") if 0 <= idx < len(arr) and arr[idx] else None
    if me >= len(pls) or not pls[me]:
        return None
    arr = {2: pls[me].get("hand"), 3: pls[me].get("discard"),
           4: pls[me].get("active"), 5: pls[me].get("bench"),
           6: pls[me].get("prize"), 12: cur.get("looking")}.get(area) or []
    return arr[idx].get("id") if 0 <= idx < len(arr) and arr[idx] else None


def option_plan_features(obs: dict, i: int, plan: dict) -> dict:
    f = {k: 0.0 for k in OPTION_PLAN_KEYS}
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    if not (0 <= i < len(opts)):
        return f
    o = opts[i]
    cid = _resolve_opt_card(obs, o)
    if cid is not None:
        cid = int(cid)
        st = _load_card_stats().get(cid)
        for r in _roles_of(cid):
            k = f"ob_role_{r}"
            if k in f:
                f[k] = 1.0
        if st:
            f["ob_card_hp"] = st["hp"] / 350.0
            f["ob_card_min_cost"] = st["min_cost"] / 5.0
            f["ob_card_max_dmg"] = st["max_dmg"] / 300.0
            if st["min_cost"] > 0:
                f["ob_card_dpe"] = (st["max_dmg"] / st["min_cost"]) / 100.0
            f["ob_card_is_ex"] = 1.0 if st["ex"] else 0.0
            f["ob_card_stage"] = (2.0 if st.get("stage2") else
                                  1.0 if st.get("stage1") else 0.0)
            f["ob_card_is_basic"] = 1.0 if st["basic"] else 0.0
            f["ob_card_is_energy"] = 1.0 if st["type"] == 5 else 0.0
        att = plan.get("attacker")
        if att is not None:
            f["ob_is_attacker"] = 1.0 if cid == att else 0.0
            f["ob_on_line"] = 1.0 if cid in plan.get("line_ids", ()) else 0.0
            st_att = _load_card_stats().get(att) or {}
            st_c = _load_card_stats().get(cid) or {}
            if st_c.get("evolves_from") and st_att:
                # 这张是不是主攻手的进化形态之一
                f["ob_completes_line"] = 1.0 if cid == att else 0.0
            if st_c.get("evolves_from") in plan.get("pre_names", ()):
                f["ob_completes_line"] = 1.0
            if st_att.get("evolves_from") and \
                    _name_of(cid) == st_att.get("evolves_from"):
                f["ob_is_pre_of_line"] = 1.0
            # 能量属性是否匹配主攻手所需
            if st_c.get("type") == 5 and st_att:
                f["ob_energy_matches_attacker"] = 1.0
        # 手上同名张数（冗余度）
        cur = obs.get("current") or {}
        me = int(cur.get("yourIndex", 0))
        pls = cur.get("players") or []
        hand = (pls[me].get("hand") if me < len(pls) and pls[me] else []) or []
        f["ob_copies_in_hand"] = min(
            sum(1 for c in hand if _get(c, "id") == cid), 4) / 4.0

    # ---- ATTACH 的目标
    ta, ti = o.get("inPlayArea"), o.get("inPlayIndex")
    if ta is not None and ti is not None:
        cur = obs.get("current") or {}
        me = int(cur.get("yourIndex", 0))
        pls = cur.get("players") or []
        if me < len(pls) and pls[me]:
            arr = {4: pls[me].get("active"), 5: pls[me].get("bench")}.get(ta) or []
            if 0 <= ti < len(arr) and arr[ti]:
                pk = arr[ti]
                f["ob_tgt_has"] = 1.0
                tid = _get(pk, "id")
                f["ob_tgt_is_attacker"] = 1.0 if tid == plan.get("attacker") else 0.0
                st = _load_card_stats().get(int(tid)) if tid else {}
                need = (st or {}).get("min_cost", 0) or 0
                n_e = len(pk.get("energyCards") or [])
                f["ob_tgt_energy_gap"] = max(0.0, need - n_e) / 3.0
                hp, mx = _get(pk, "hp") or 0, _get(pk, "maxHp") or 1
                f["ob_tgt_hp_ratio"] = float(hp) / max(float(mx), 1.0)
    return f


def state_plan_features(obs: dict, plan: dict) -> dict:
    f = {k: 0.0 for k in STATE_PLAN_KEYS}
    cur = obs.get("current") or {}
    me = int(cur.get("yourIndex", 0))
    pls = cur.get("players") or []
    if me >= len(pls) or not pls[me]:
        return f
    d = pls[me]
    stats = _load_card_stats()
    att = plan.get("attacker")
    in_play = [pk for pk in (list(d.get("active") or []) +
                             list(d.get("bench") or [])) if pk]
    hand = d.get("hand") or []
    f["obp_basic_in_play"] = min(sum(
        1 for pk in in_play if (stats.get(int(_get(pk, "id") or 0)) or {}).get("basic")), 6) / 6.0
    f["obp_basic_in_hand"] = min(sum(
        1 for c in hand if (stats.get(int(_get(c, "id") or 0)) or {}).get("basic")), 6) / 6.0
    names = _pokemon_names_on_board(obs, me)
    f["obp_line_piece_in_hand"] = 1.0 if any(
        n in names for n in plan.get("pre_names", ())) else 0.0
    if att is not None and stats.get(att):
        need = stats[att]["min_cost"] or 0
        for pk in in_play:
            if int(_get(pk, "id") or 0) == att:
                n_e = len(pk.get("energyCards") or [])
                f["obp_attacker_in_play"] = 1.0
                f["obp_attacker_ready"] = 1.0 if n_e >= need else 0.0
                f["obp_attacker_energy_gap"] = max(0.0, need - n_e) / 3.0
                hp, mx = _get(pk, "hp") or 0, _get(pk, "maxHp") or 1
                f["obp_attacker_hp_ratio"] = float(hp) / max(float(mx), 1.0)
                break
    types = set()
    for pk in in_play + [c for c in hand]:
        st = stats.get(int(_get(pk, "id") or 0)) if pk else None
        if st and st["type"] == 5:
            types.add(_get(pk, "id"))
    f["obp_seen_types"] = min(len(types), 4) / 4.0
    return f
