#!/usr/bin/env python3
"""生成 **shuppet 墙/引擎 bot**：一个**可重复、可验证的战术测试器**，不是"更强的对手"。

## 它要测什么

我们的主卡组（9d6e / 7e39）赢牌手段是 `Alakazam` 的 `Powerful Hand`（**放置伤害指示物** = 招式效果）。
shuppet 族的 `Hide 'n' Sneak`（ID 1374/1375/1348/1349）是**被动特性**：
"prevent all effects of your opponent's Pokémon's attacks and Abilities done to **this Pokémon**"
⇒ 只要场上全是 HnS 宝可梦，`Powerful Hand` **一点伤害都打不出来**。

真实天梯对 shuppet 是 **0/9 ~ 0/16**，但我们的 BC 克隆当对手时学习者赢 **0.76–0.99** ——
因为 BC 克隆不会把这个引擎跑起来。本 bot 用规则**100% 执行**它，所以能测出：

> 面对一个"真正在用 HnS 免疫 + Vengeful Anchor 反打"的对手，我们到底剩多少胜率？

## 这副牌的引擎（`shuppet_dhelmise__0d86`，查卡表得出）

| 卡 | 效果 | 角色 |
|---|---|---|
| `Dhelmise`(1380) `Vengeful Anchor` | 30 伤害；**弃牌堆有 ≥4 只 HnS 时 +140** | **主攻（170 真伤害）** |
| `Sinistcha`(1349) `Matcha Spin` | ≥6 只 HnS 时，对手每只宝可梦 4 个指示物 | 群伤 |
| `Spiritomb`(1376) `Spiritual End` | ≥13 只时，2 只宝可梦指示物 ×4 | 爆发 |
| `Gwynn`(1428) | 弃最多 2 只无 Rule Box 宝可梦，每弃 1 抽 3 | **往弃牌堆塞 HnS 的主要手段** |
| `Lillie's Determination`(1227) | 洗手抽 6/8 | 抽牌 |
| `Ultra Ball`(1121)/`Poké Pad`(1152)/`Poké Gear`(1122)/`Buddy Poffin`(1086) | 检索 | 找 Dhelmise / HnS |

## 验证口径（写死在 run 脚本里）

1. `异常 = 0`、`非法动作 = 0`（bo3_eval 自带计数）
2. 对端口@7e39 的胜率 —— 目标：**显著低于 0.76**（BC 克隆的水平），方向朝真实 0.043
3. 不靠拖延：`超过 4000 决策上限` 的局数必须为 0
4. 同种子重跑结果逐位相同（bot 是确定性的）

用法：
    ./tools/py tools/make_shuppet_bot.py --out bots/shuppet_wall
    ./tools/py ptcg_league/eval_pair.py --cand bots/shuppet_wall \\
        --opp port=<端口@7e39> --matches 200 --workers 0 --out /tmp/wall.json
"""
import argparse
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECK = "decks/pool_field/shuppet_dhelmise__0d86.csv"

MAIN_PY = r'''#!/usr/bin/env python3
"""shuppet 墙/引擎 bot（由 tools/make_shuppet_bot.py 生成，勿手改）。

规则优先级（自上而下，取第一个可执行项）：
  1. HnS 弃牌数 ≥4 且能打 Vengeful Anchor      → 打（170 伤害）
  2. 手上有 Gwynn 且手上有无 Rule Box 的宝可梦   → 打 Gwynn（之后弃 HnS 进弃牌堆）
  3. 有 Lillie's Determination 且手牌 ≤4         → 打（洗手抽牌）
  4. 检索卡（Ultra Ball / Poké Pad / Poké Gear / Buddy Poffin）→ 打
  5. 给前排贴能量
  6. 能进化（Shuppet→Banette / Poltchageist→Sinistcha）→ 进化
  7. 能打任何攻击 → 打
  8. 否则结束回合

多选（送死/弃牌/检索结果）用 CARD_RANK 排序：优先把 HnS 宝可梦送进弃牌堆。
"""
import os

# ---- 卡 id ----
HNS = {1374, 1375, 1348, 1349}          # Shuppet / Banette / Poltchageist / Sinistcha
DHELMISE, SINISTCHA, SPIRITOMB = 1380, 1349, 1376
GWYNN, LILLIE = 1428, 1227
SEARCH_ITEMS = {1121, 1152, 1122, 1086}  # Ultra Ball / Poké Pad / Poké Gear / Buddy Poffin
ENERGY = {5, 19, 12}                    # Basic {P} / Telepath Psychic / Legacy
ATTACK_VENGEFUL = 1380                  # attack 归属用 option 的 cardId 判断

# 多选时的偏好：越小越先选
CARD_RANK = {
    GWYNN: 0, LILLIE: 1, DHELMISE: 2, SINISTCHA: 3,
    **{c: 5 for c in HNS},
    **{c: 20 for c in SEARCH_ITEMS},
    **{c: 30 for c in ENERGY},
}

OT_PLAY, OT_ATTACK, OT_END = 7, 13, 14


def _cid(o):
    return o.get("cardId") or 0


def _hand(cur, me):
    return [c for c in (cur["players"][me].get("hand") or []) if c]


def _discard_hns(cur, me):
    d = cur["players"][me].get("discard") or []
    return sum(1 for c in d if c and (c.get("id") or c.get("cardId")) in HNS)


def _active(cur, me):
    a = cur["players"][me].get("active") or []
    return a[0] if a else None


def _has_rule_box(cid):
    """简易判定：ex/大怪按卡表 id 白名单更差，这里只把已知的 ex 排除。"""
    return cid in {44, 272}   # Bloodmoon Ursaluna ex / Lillie's Clefairy ex


_STATS = os.environ.get("WALL_STATS", "stats.jsonl")
_STEP = [0]


def _log(cur, me, act_id, hns_disc, picked, reason):
    """可验证：逐决策落盘（对手盘面/弃牌堆 HnS 数/选了什么/为什么）。"""
    try:
        _STEP[0] += 1
        with open(_STATS, "a") as f:
            f.write('{"step":%d,"my_active":%s,"hns_discard":%d,"hand":%d,"picked":%s,"reason":"%s",'
                    '"n_opt":%d,"lo":%d,"hi":%d,"types":%s}\n'
                    % (_STEP[0], act_id, hns_disc, len(_hand(cur, me)),
                       str(picked), reason, _SEL[0], _SEL[1], _SEL[2], str(_SEL[3])))
    except OSError:
        pass


_SEL = [0, 0, 0, []]
_DBG = []


def choose(obs):
    sel = obs["select"]
    opts = sel["option"]
    n = len(opts)
    lo, hi = sel["minCount"], sel["maxCount"]
    _SEL[:] = [n, lo, hi, sorted({o.get("type") for o in opts})]
    cur = obs["current"]
    me = cur["yourIndex"]
    hand = _hand(cur, me)
    hand_ids = [(c.get("id") or c.get("cardId")) for c in hand]
    hns_disc = _discard_hns(cur, me)
    act = _active(cur, me)
    act_id = (act or {}).get("id") or (act or {}).get("cardId")

    # ---- 单一动作（maxCount == 1）----
    if hi == 1:
        # 1) Vengeful Anchor（170）
        if hns_disc >= 4 and act_id == DHELMISE:
            for j, o in enumerate(opts):
                if o.get("type") == OT_ATTACK and _cid(o) == DHELMISE:
                    _log(cur, me, act_id, hns_disc, [j], "vengeful170")
                    return [j]
        # 2) Gwynn：手上有无 Rule Box 宝可梦才值得打
        if GWYNN in hand_ids and any(c not in HNS and not _has_rule_box(c) for c in hand_ids):
            for j, o in enumerate(opts):
                if o.get("type") == OT_PLAY and _cid(o) == GWYNN:
                    _log(cur, me, act_id, hns_disc, [j], "gwynn")
                    return [j]
        # 3) Lillie：手牌少时洗手抽牌
        if LILLIE in hand_ids and len(hand) <= 4:
            for j, o in enumerate(opts):
                if o.get("type") == OT_PLAY and _cid(o) == LILLIE:
                    return [j]
        # 4) 检索卡
        for j, o in enumerate(opts):
            if o.get("type") == OT_PLAY and _cid(o) in SEARCH_ITEMS:
                return [j]
        # 5) 贴能量给前排
        if act is not None:
            for j, o in enumerate(opts):
                if o.get("type") == OT_PLAY and _cid(o) in ENERGY:
                    return [j]
        # 6) 进化
        for j, o in enumerate(opts):
            if o.get("type") == 10:            # 进化/上场
                return [j]
        # 7) 任何攻击
        for j, o in enumerate(opts):
            if o.get("type") == OT_ATTACK:
                return [j]
        # 8) 结束回合
        for j, o in enumerate(opts):
            if o.get("type") == OT_END:
                return [j]
        return [0]

    # ---- 多选：按 CARD_RANK 取前 k 个 ----
    order = sorted(range(n), key=lambda j: (CARD_RANK.get(_cid(opts[j]), 40), j))
    k = max(lo, min(hi, n))
    _log(cur, me, act_id, hns_disc, order[:k], "multipick%d" % k)
    return order[:k]


# ⚠️ 评测框架用 `exec(compile(main.py))` 加载 agent，**没有 `__file__`**（tools/bo3_eval.py 的约定），
#    所以卡组必须内联在这里，不能 os.path.dirname(__file__)。
DECK = {deck}

_DECK = None


def deck_ids():
    global _DECK
    if _DECK is None:
        _DECK = list(DECK)
    return _DECK


def agent(obs):
    """评测框架（tools/bo3_eval.py）取命名空间里最后一个 callable。"""
    if not obs or obs.get("select") is None:
        return deck_ids()
    if os.environ.get("WALL_DBG") and len(_DBG) <= 8:
        try:
            with open("dbg.jsonl", "a") as f:
                f.write(__import__("json").dumps(_DBG[-1] if _DBG else {}, default=str) + "\n")
        except OSError:
            pass
    return choose(obs)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "bots", "shuppet_wall"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    ids_ = [int(x) for x in open(os.path.join(ROOT, DECK)).read().split()]
    with open(os.path.join(out, "main.py"), "w", encoding="utf-8") as f:
        f.write(MAIN_PY.replace("{deck}", repr(ids_)))
    # 变量名冲突：MAIN_PY 里的 DECK 常量名与模块级 DECK（路径）同名，这里用的是模板替换
    print(f"   卡组内联 {len(ids_)} 张")
    ids = [int(x) for x in open(os.path.join(ROOT, DECK)).read().split()]
    assert len(ids) == 60, len(ids)
    with open(os.path.join(out, "deck.csv"), "w") as f:
        f.write("\n".join(str(x) for x in ids) + "\n")
    print(f"✅ {out}\n   main.py + deck.csv（{len(ids)} 张，shuppet_dhelmise__0d86）")


if __name__ == "__main__":
    main()
