#!/usr/bin/env python3
"""把**规则自己用的信息**显式做成特征 —— 用来诊断"特征够不够"。

## 为什么要这个模块

今天量出一条硬事实（`logs/bc_rule_diag.log`）：

    让 MLP 去模仿**我们自己的手写规则**（规则是特征的确定性函数，理论上可学到 100%）
    最优 holdout = index 65.5% / 语义 71.3% / 语义 top-3 90.5%
    而 train 能到 86.9% 还在涨  ⇒  卡住的是**特征**，不是数据、也不是容量

也就是说：神经网络看不见规则决策所依赖的那部分信息。规则依赖而 `option_features`
没给的东西，主要是四类：

1. **优先级位次**（`PRIORITY` 表：ATTACH 0 → EVOLVE 1 → PLAY 2 → … → DISCARD 90）
   —— 决定了"现在该做哪一类动作"，是决策的第一性信息
2. **规则自己选了哪个**（`is_rule_choice` 锚）—— 学"何时偏离规则"比从零学便宜得多；
   这也是 RL 那边唯一有效的结构性锚（无锚时策略崩掉 76%，加锚后 13.4%）
3. **卡牌静态属性**（最高伤害 / 最低能量需求 / HP / 进化段位）—— 规则通过
   `rollout._load_card_stats()` 用它们判断"这张牌值不值得打"
4. **卡牌角色**（检索 / 抽牌 / 加速能量 …）—— 已在 `plan_features` 里覆盖，本模块不重复

## 用法

    from framework.rule_features import OPTION_RULE_KEYS, rule_context, option_rule_features
    ctx = rule_context(obs, sel)                     # 每个决策算一次（O(k) 次规则调用）
    f = option_rule_features(obs, i, ctx)            # 每个选项一次

`BC_NO_ANCHOR=1` 环境变量会把两个"锚"特征恒置 0，用于做消融
（区分"锚让模型抄规则" 与 "特征真的带来了信息"）。
"""
import os

NO_ANCHOR = os.environ.get("BC_NO_ANCHOR", "") == "1"

OPTION_RULE_KEYS = [
    "rl_is_choice",        # 这个选项就是规则会选的（锚）
    "rl_same_type",        # 与规则所选同类（锚的软版本）
    "rl_prio_rank",        # 本选项类型的优先级位次（越小越先做）
    "rl_is_top_prio",      # 本选项类型是当前可选里优先级最高的
    "rl_prio_x_nopts",     # 位次 × 选项数（交互项，帮助区分"该动还是该结束"）
    "rl_card_maxdmg",      # 卡牌最高招式伤害 / 100
    "rl_card_mincost",     # 卡牌最便宜招式的能量数
    "rl_card_hp",          # 卡牌 HP / 100
    "rl_card_stage",       # 0 基础 / 1 一阶 / 2 二阶 / 3 其他
    "rl_card_is_ex",
]


def _prio_rank(opt_type, prio=None):
    """选项类型 → 优先级位次。未知类型给 50（排在 END(5) 之后、DISCARD(90) 之前）。"""
    from framework.rollout import PRIORITY
    from cg.api import OptionType
    table = prio or PRIORITY
    try:
        t = OptionType(int(opt_type))
    except Exception:
        return 50
    return int(table.get(t, 50))


def _card_stats_of(obs, idx):
    """选项 idx 指向的卡牌静态属性（借 `resolve_cid` 处理"隐含手牌"的坑）。"""
    from framework.rollout import _load_card_stats
    from train_bc import resolve_cid
    cid = resolve_cid(obs, (obs.get("select") or {}).get("option", [])[idx])
    if cid is None:
        return None
    return _load_card_stats().get(int(cid))


def rule_context(obs, sel, prio=None):
    """算一次、给整个决策的选项复用。"""
    from framework.rollout import rule_action
    opts = (sel or {}).get("option") or []
    ranks = [_prio_rank(o.get("type", -1), prio) for o in opts]
    try:
        act = rule_action(sel, obs)
        choice = int(act[0]) if act else -1
    except Exception:
        choice = -1
    choice_type = opts[choice].get("type", -1) if 0 <= choice < len(opts) else -1
    top = min(ranks) if ranks else 50
    return {"ranks": ranks, "choice": choice, "choice_type": choice_type,
            "top_rank": top, "n": len(opts)}


def option_rule_features(obs, idx, ctx=None):
    """第 idx 个选项的规则信息特征。"""
    from framework.rollout import _load_card_stats  # noqa: F401  (确保卡库已加载)
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    if not (0 <= idx < len(opts)):
        return {k: 0.0 for k in OPTION_RULE_KEYS}
    if ctx is None:
        ctx = rule_context(obs, sel)
    opt = opts[idx]
    rank = ctx["ranks"][idx] if idx < len(ctx["ranks"]) else 50
    f = {
        "rl_is_choice": 0.0 if NO_ANCHOR else float(idx == ctx["choice"]),
        "rl_same_type": 0.0 if NO_ANCHOR else float(
            opt.get("type", -2) == ctx["choice_type"]),
        "rl_prio_rank": rank / 10.0,
        "rl_is_top_prio": float(rank == ctx["top_rank"]),
        "rl_prio_x_nopts": rank * ctx["n"] / 100.0,
        "rl_card_maxdmg": 0.0, "rl_card_mincost": 0.0, "rl_card_hp": 0.0,
        "rl_card_stage": 3.0, "rl_card_is_ex": 0.0,
    }
    st = _card_stats_of(obs, idx)
    if st:
        f["rl_card_maxdmg"] = float(st.get("max_dmg", 0)) / 100.0
        f["rl_card_mincost"] = float(st.get("min_cost", 0))
        f["rl_card_hp"] = float(st.get("hp", 0)) / 100.0
        f["rl_card_stage"] = (0.0 if st.get("basic") else
                              1.0 if st.get("stage1") else
                              2.0 if st.get("stage2") else 3.0)
        f["rl_card_is_ex"] = float(bool(st.get("ex")))
    return f
