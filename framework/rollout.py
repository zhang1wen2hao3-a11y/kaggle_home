"""rollout.py —— 用**规则策略打到终局**来评估动作，不依赖价值函数。

## 为什么不用 V

前面三轮实验的结论：位置价值函数 V 判断不了动作。

    动作引起的胜率变化  ≈ 0.02
    V 的预测误差        ≈ 0.05      ← 比信号还大

所以"用 V 当叶子评估"的搜索必然比规则策略差（实测 MCTS 0.39、1-ply V 0.20）。

## 为什么 rollout 行

把"评估局面"换成"把这个动作执行下去、双方都用规则策略打完、看谁赢"：

    实测跨动作信号 0.361   vs   rollout 噪声 0.273      ← 信噪比 > 1

不需要任何模型。代价是慢（每次评估要打完整局），所以必须靠
**公共随机数（CRN）** 把候选之间的比较变成配对比较，噪声相互抵消。

## CRN 的关键前提：一个 world 内没有随机性

`search_begin` 传入确定的隐藏信息后，整局是**确定性**的 —— 同一个 world 里
重复 rollout 会得到完全相同的结果。所以：

    n_worlds  就是   rollout 采样次数

真正提供方差的是 **determinization（隐藏信息采样）**，而不是引擎内部随机。
所以 CRN 是天然的：所有候选动作共享同一批 world。

## 回合结构注意事项

`ATTACK` / `END` 会立刻结束本回合（turn+1、换人）；`PLAY`/`ATTACH`/`EVOLVE`/
`ABILITY`/`RETREAT` 不结束回合。而且一个回合内玩家会连续做约 5 个决策。
所以"一步"≠"一回合" —— rollout 必须一路打到终局，不能只多走一步。
`ATTACK` 还有 49.8% 的情况会在同一回合内进入后续选择（招式效果），
这里也不需要特殊处理：规则策略会自然把这些选完。
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cg.api import (  # noqa: E402
    search_begin, search_step, search_end, to_observation_class,
    OptionType, SelectType, SelectContext,
)

# 与 main.py / agents/rulebase.py 的 develop_first 完全一致。
# ATTACK 绝不能排前面：攻击会结束回合，攻击优先会变成"只会平A、从不铺场"
# （实测对 random 胜率 0.17 vs 0.85）。
PRIORITY = {
    OptionType.PLAY: 0,
    OptionType.EVOLVE: 1,
    OptionType.ABILITY: 2,
    OptionType.ATTACH: 3,
    OptionType.RETREAT: 4,
    OptionType.DISCARD: 5,
    OptionType.ATTACK: 6,
    OptionType.END: 9,
}

FILL_CONTEXTS = (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH)
GO_FIRST = True


# ============================================================ 规则策略
def _sel_fields(sel):
    """归一化 SelectData —— 同时支持 dict（提交入口）和 dataclass（搜索状态）。"""
    if isinstance(sel, dict):
        opts = sel.get("option") or []
        return (sel.get("type"), sel.get("context"),
                [o.get("type") for o in opts],
                int(sel.get("minCount", 1)), int(sel.get("maxCount", 1)))
    opts = getattr(sel, "option", None) or []
    return (getattr(sel, "type", None), getattr(sel, "context", None),
            [getattr(o, "type", None) for o in opts],
            int(getattr(sel, "minCount", 1)), int(getattr(sel, "maxCount", 1)))


def _decide(stype, ctx, opt_types, lo, hi, n_opts=None):
    """规则决策核心。返回选项下标列表。与 main.py 逐行等价。"""
    n = len(opt_types) if n_opts is None else n_opts
    hi = min(hi, n)
    if n == 0 or hi <= 0:
        return []

    # 开局选先后手
    if stype == SelectType.YES_NO and ctx == SelectContext.IS_FIRST:
        want = OptionType.YES if GO_FIRST else OptionType.NO
        for i, t in enumerate(opt_types):
            if t == want:
                return [i]
        return [0]

    # 主阶段：按优先级取第一个（一次一个动作）
    if stype == SelectType.MAIN:
        best = min(range(n), key=lambda i: PRIORITY.get(opt_types[i], 99))
        return [best]

    # 是/否：默认"是"
    if stype == SelectType.YES_NO:
        for i, t in enumerate(opt_types):
            if t == OptionType.YES:
                return [i]
        return [0]

    # 铺后备：选满（主力倒了空场会直接判负）
    if ctx in FILL_CONTEXTS:
        return list(range(hi))

    # 其余：满足下限即可
    k = max(lo, 1 if hi >= 1 else 0)
    return list(range(min(k, hi)))


def rule_action(sel):
    """给定 SelectData（dict 或 dataclass），返回规则策略会选的选项下标列表。"""
    return _decide(*_sel_fields(sel))


def is_main(sel) -> bool:
    stype, *_ = _sel_fields(sel)
    return stype == SelectType.MAIN


def has_type(sel, want) -> bool:
    _, _, opt_types, _, _ = _sel_fields(sel)
    return want in opt_types


def index_of_type(sel, want):
    _, _, opt_types, _, _ = _sel_fields(sel)
    for i, t in enumerate(opt_types):
        if t == want:
            return i
    return None


# ============================================================ rollout
def rollout_to_end(sid, obs, my_index: int, max_steps: int = 800):
    """从搜索节点一路用规则策略打到终局。

    Returns:
        1.0 我赢 / 0.0 我输 / 0.5 平局 / None 异常（无法判定）
    """
    for _ in range(max_steps):
        cur = getattr(obs, "current", None)
        if cur is None:
            return None
        res = getattr(cur, "result", -1)
        if res != -1:
            if res == 2:
                return 0.5
            return 1.0 if res == my_index else 0.0
        sel = getattr(obs, "select", None)
        if sel is None:
            return None
        act = rule_action(sel)
        try:
            st = search_step(sid, act)
        except Exception:
            return None
        sid = st.searchId
        obs = st.observation
    return None


# ============================================================ 动作评估
def evaluate_actions(obs: dict, actions, *, filler, my_deck,
                     n_worlds: int = 10, max_steps: int = 800,
                     memory=None):
    """用 CRN 评估一组候选动作。

    Args:
        obs: 真实局面的观测（dict）。必须是单选取决策（MAIN）。
        actions: 候选动作下标列表，如 [0, 3, 7]
        filler: framework.filler.Filler，负责采样隐藏信息
        n_worlds: determinization 采样数（= 每个动作的 rollout 次数）

    Returns:
        (outcomes, info)
            outcomes: {action: [每个 world 的结果或 None]}
            info: {"worlds": int, "fail": int, "my_index": int}
    """
    cur = obs.get("current") or {}
    my_index = int(cur.get("yourIndex", 0))
    actions = list(actions)
    outcomes: dict[int, list] = {a: [] for a in actions}
    info = {"worlds": 0, "fail": 0, "my_index": my_index}

    for _ in range(n_worlds):
        try:
            world = filler.fill(obs, memory, my_deck)
        except Exception:
            world = None
        if not world:
            info["fail"] += 1
            continue
        root = None
        try:
            root = search_begin(
                agent_observation=to_observation_class(obs), **world
            ).searchId
        except Exception:
            root = None
        if root is None:
            info["fail"] += 1
            continue
        info["worlds"] += 1
        try:
            # 所有候选从【同一个 root】分叉 => 共享这一批隐藏信息（CRN）
            for a in actions:
                try:
                    st = search_step(root, [a])
                except Exception:
                    outcomes[a].append(None)
                    continue
                outcomes[a].append(
                    rollout_to_end(st.searchId, st.observation, my_index, max_steps)
                )
        finally:
            try:
                search_end()
            except Exception:
                pass
    return outcomes, info


def _stats(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    m = sum(v) / len(v)
    if len(v) > 1:
        var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
        sd = math.sqrt(var)
    else:
        sd = 0.0
    return {"mean": m, "sd": sd, "n": len(v)}


def choose_action(obs: dict, *, filler, my_deck, n_worlds: int = 10,
                  margin: float = 1.0, min_pairs: int = 4,
                  max_steps: int = 800, memory=None):
    """决定这一手该下哪个动作。

    安全设计：**默认就是规则策略的选择**，只有当某个候选在配对比对上
    显著更好（差值 > margin 个标准误）时才偏离。这样最坏情况退化为 greedy，
    不会因为 rollout 噪声而系统性地变差。

    Returns:
        (action: list[int], info: dict)
    """
    sel = obs.get("select")
    base = rule_action(sel)
    info = {"base": base, "chosen": base, "reason": "rule",
            "cands": [], "stats": {}, "pairs": {}, "worlds": 0}
    if not sel or len(base) != 1:
        return base, info
    if not is_main(sel):
        return base, info

    base_i = base[0]
    cands = [base_i]
    for want in (OptionType.ATTACK, OptionType.END):
        i = index_of_type(sel, want)
        if i is not None and i not in cands:
            cands.append(i)
    if len(cands) < 2:                  # 没有可比的备选
        return base, info

    outcomes, ev = evaluate_actions(obs, cands, filler=filler, my_deck=my_deck,
                                   n_worlds=n_worlds, max_steps=max_steps,
                                   memory=memory)
    info["worlds"] = ev["worlds"]
    info["cands"] = cands
    info["stats"] = {a: _stats(outcomes[a]) for a in cands}

    if ev["worlds"] < min_pairs:
        info["reason"] = "rollout 失败，回退规则"
        return base, info

    base_vals = outcomes[base_i]
    best_a, best_z = None, -1e9
    for a in cands:
        if a == base_i:
            continue
        d = [x - y for x, y in zip(outcomes[a], base_vals)
             if x is not None and y is not None]
        if len(d) < min_pairs:
            continue
        m = sum(d) / len(d)
        if len(d) > 1:
            sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
        else:
            sd = 0.0
        se = sd / math.sqrt(len(d)) if sd > 0 else 0.0
        z = (m / se) if se > 0 else (float("inf") if m > 0 else 0.0)
        info["pairs"][a] = {"mean_diff": m, "se": se, "z": z, "n": len(d)}
        if z > best_z:
            best_a, best_z = a, z

    if best_a is not None and best_z >= margin:
        info["chosen"] = [best_a]
        info["reason"] = f"rollout 更优 (z={best_z:.2f} >= {margin})"
        return [best_a], info
    info["reason"] = ("与规则持平" if best_a is None
                      else f"未超过阈值 (最优 z={best_z:.2f} < {margin})")
    return base, info
