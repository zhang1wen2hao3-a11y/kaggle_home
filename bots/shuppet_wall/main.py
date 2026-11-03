"""shuppet 墙/引擎 bot —— **战术测试器**（由 tools/make_shuppet_bot.py 生成）。

骨架照 `dist/ptcg_prefix.tar.gz` 里那份能跑的规则 agent（开局 ABI：
`select.type/context` 分流、铺后备要铺满 hi），只把 MAIN 分支换成 shuppet 引擎的优先级。

要测的东西：我们的 7e39 靠 `Powerful Hand`（**放置伤害指示物=招式效果**）赢；
shuppet 族 `Hide 'n' Sneak` 是**被动免疫**，效果打不上去。
所以"面对一个真的在用 HnS 免疫 + Vengeful Anchor(170) 反打的对手，我们剩多少胜率"。
"""
import json
import os

from cg.api import (OptionType, SelectType, SelectContext,
                    to_observation_class, all_attack)

DECK = [5, 5, 12, 19, 19, 19, 19, 44, 56, 66, 66, 272, 272, 305, 305, 305, 1086, 1086, 1097, 1097, 1121, 1121, 1121, 1121, 1122, 1122, 1152, 1152, 1152, 1152, 1174, 1182, 1182, 1182, 1191, 1227, 1227, 1227, 1227, 1340, 1348, 1348, 1349, 1374, 1374, 1374, 1374, 1375, 1375, 1375, 1380, 1380, 1380, 1380, 1428, 1428, 1428, 1430, 1430, 1430]

HNS = {1374, 1375, 1348, 1349}          # Shuppet/Banette/Poltchageist/Sinistcha
DHELMISE, SINISTCHA = 1380, 1349
GWYNN, LILLIE = 1428, 1227
SEARCH = {1121, 1152, 1122, 1086}
ENERGY = {5, 19, 12}

# PLAY/EVOLVE/ATTACH 内部再按卡牌偏好排序（越小越先）
CARD_PREF = {GWYNN: 0, LILLIE: 1, DHELMISE: 2, SINISTCHA: 3,
             **{c: 5 for c in HNS}, **{c: 20 for c in SEARCH}, **{c: 30 for c in ENERGY}}

PRIORITY = {
    OptionType.PLAY: 0,
    OptionType.EVOLVE: 1,
    OptionType.ABILITY: 2,
    OptionType.ATTACH: 3,
    OptionType.ATTACK: 4,
    OptionType.RETREAT: 90,
    OptionType.DISCARD: 91,
}
GO_FIRST = True                          # 引擎卡组吃能量，先手更稳
_STATS = os.environ.get("WALL_STATS", "stats.jsonl")


def _ids(zone):
    return [(c.get("id") or c.get("cardId")) for c in (zone or []) if c]


def _log(reason, extra):
    try:
        with open(_STATS, "a") as f:
            f.write(json.dumps(dict(reason=reason, **extra)) + "\n")
    except OSError:
        pass


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return list(DECK)

    opts = obs_dict["select"]["option"]
    n = len(opts)
    lo = obs.select.minCount
    hi = min(obs.select.maxCount, n)
    if hi <= 0 or n == 0:
        return []

    # ① 选先后手
    if obs.select.type == SelectType.YES_NO and obs.select.context == SelectContext.IS_FIRST:
        want = OptionType.YES if GO_FIRST else OptionType.NO
        for i, o in enumerate(opts):
            if o["type"] == want:
                return [i]
        return [0]

    # ② 主决策
    if obs.select.type == SelectType.MAIN:
        cur = obs_dict["current"]
        me = cur["yourIndex"]
        pl = cur["players"][me]
        hand = _ids(pl.get("hand"))
        disc = _ids(pl.get("discard"))
        act = (pl.get("active") or [None])[0]
        act_id = (act or {}).get("id") or (act or {}).get("cardId")
        hns = sum(1 for c in disc if c in HNS)

        # 引擎阈值达标 + 前排是 Dhelmise ⇒ Vengeful Anchor（30+140=170）
        if hns >= 4 and act_id == DHELMISE:
            for i, o in enumerate(opts):
                if o["type"] == OptionType.ATTACK:
                    _log("vengeful170", dict(hns=hns, active=act_id, n=n))
                    return [i]

        # 否则：先发展（PLAY/EVOLVE/ABILITY/ATTACH 内按卡牌偏好），最后才攻击
        def key(i):
            o = opts[i]
            t = o["type"]
            p = PRIORITY.get(t, 50)
            cid = o.get("cardId") or 0
            if t == OptionType.ATTACK:
                # 有 4 只 HnS 时优先 Dhelmise；否则随便（避免无意义拖延）
                p = 4 if cid == DHELMISE and hns >= 4 else 6
            return (p, CARD_PREF.get(cid, 10), i)

        best = min(range(n), key=key)
        _log("main", dict(hns=hns, active=act_id, hand=len(hand),
                          pick_type=int(opts[best]["type"]),
                          pick_id=opts[best].get("cardId") or 0))
        return [best]

    # ③ 其余 YES_NO 一律答应（引擎需要能量/抽牌推进）
    if obs.select.type == SelectType.YES_NO:
        for i, o in enumerate(opts):
            if o["type"] == OptionType.YES:
                return [i]
        return [0]

    # ④ 铺后备/放场地：**必须铺满**（Active 空场直接判负）
    if obs.select.context in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH):
        return list(range(hi))

    # ⑤ 其它多选：按卡牌偏好取 lo（弃牌类会多拿，靠偏好保证先弃 HnS）
    order = sorted(range(n), key=lambda i: (CARD_PREF.get(opts[i].get("cardId") or 0, 40), i))
    k = max(lo, 1 if hi >= 1 else 0)
    return order[:min(k, hi)]
