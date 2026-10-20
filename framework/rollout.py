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
    OptionType, SelectType, SelectContext, all_attack,
)

# attackId -> (基础伤害, 能量需求数)，用于"同一只宝可梦有多个招式"时择优。
# 规则表按【动作类型】排序，所有 ATTACK 同级；不做这个二次排序就只有"取第一个"，
# 而第一个常常是有条件的废招（伤害0、甚至给对手回血）。详见 main.py 的注释。
_ATTACK_INFO: dict = {}
try:
    for _a in all_attack():
        _ATTACK_INFO[_a.attackId] = (int(getattr(_a, "damage", 0) or 0),
                                     len(getattr(_a, "energies", None) or []))
except Exception:
    pass

# cardId -> 该宝可梦【最便宜招式】需要的能量数（用来判断"离能攻击还差几张能量"）
_MIN_ATK_COST: dict = {}
try:
    from cg.api import all_card_data as _all_card_data
    for _c in _all_card_data():
        costs = [_ATTACK_INFO[a][1] for a in (getattr(_c, "attacks", None) or [])
                 if a in _ATTACK_INFO and _ATTACK_INFO[a][1] > 0]
        if costs:
            _MIN_ATK_COST[_c.cardId] = min(costs)
    del _all_card_data
except Exception:
    pass

# 与 main.py / agents/rulebase.py 的 develop_first 完全一致。
# ATTACK 绝不能排前面：攻击会结束回合，攻击优先会变成"只会平A、从不铺场"
# （实测对 random 胜率 0.17 vs 0.85）。
# ⚠️ 顺序曾经是 PLAY>EVOLVE>ABILITY>ATTACH>RETREAT>DISCARD>ATTACK>END，
# 即 RETREAT 排在 ATTACK 之前 —— 这是个**严重 bug**：
# 只要场上有可撤退的目标，agent 就会撤退而不攻击。
# 实测（8 副卡组 vs random 的平均）：
#   RETREAT 在 ATTACK 之前（旧）  = 0.2725   ← 7 副卡组上接近随机
#   RETREAT 完全移除（现）        = 0.7612
# 最极端的是 fire_gouging：0.23 -> 0.89。
#
# 之前一直没发现，是因为所有评估都用 deck.csv(sample_official)，
# 而那副快攻牌恰好是唯一不受此 bug 影响的卡组（两个配置都是 0.78）。
PRIORITY = {
    OptionType.ATTACH: 0,
    OptionType.EVOLVE: 1,
    OptionType.PLAY: 2,
    OptionType.ABILITY: 3,
    OptionType.ATTACK: 4,
    OptionType.END: 5,
    # 撤退与弃牌压到最后：撤退会浪费本回合攻击机会，实测严格有害
    OptionType.DISCARD: 90,
    OptionType.RETREAT: 91,
}

FILL_CONTEXTS = (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH)
GO_FIRST = True


# ============================================================ 规则策略
def _sel_fields(sel):
    """归一化 SelectData —— 同时支持 dict（提交入口）和 dataclass（搜索状态）。

    返回 8 元组：(type, context, option_types, minCount, maxCount,
                attack_ids, areas, indices)
    attack_ids 用于同一宝可梦多招式时按伤害择优；
    areas/indices 用于把子决策（"贴给哪只""换谁上场"）的选项解析回实际宝可梦。
    """
    if isinstance(sel, dict):
        opts = sel.get("option") or []
        return (sel.get("type"), sel.get("context"),
                [o.get("type") for o in opts],
                int(sel.get("minCount", 1)), int(sel.get("maxCount", 1)),
                [o.get("attackId") for o in opts],
                [o.get("area") for o in opts],
                [o.get("index") for o in opts])
    opts = getattr(sel, "option", None) or []
    return (getattr(sel, "type", None), getattr(sel, "context", None),
            [getattr(o, "type", None) for o in opts],
            int(getattr(sel, "minCount", 1)), int(getattr(sel, "maxCount", 1)),
            [getattr(o, "attackId", None) for o in opts],
            [getattr(o, "area", None) for o in opts],
            [getattr(o, "index", None) for o in opts])


# ============================================================ 策略子决策
# 背景：非 MAIN 决策占全部决策的 37.5%（CARD 20.7% + ENERGY 15.6%），
# 而原来的兜底是 `return list(range(k))` —— **永远取列表第一个**，
# 等于让"引擎的选项排列顺序"替我们做决定。
#
# 这里先只做【可解析且量最大】的一类：贴能量时选哪只宝可梦（ATTACH_FROM，
# 实测约 113 次 / 1200 局）。选项带 area(4=ACTIVE/5=BENCH)+index，
# 能解析回实际宝可梦，从而看到 HP、已有能量、离能攻击还差几张。

def _get(o, key, default=None):
    """同时支持 dict 与 dataclass 的取值。"""
    if o is None:
        return default
    if isinstance(o, dict):
        v = o.get(key, default)
    else:
        v = getattr(o, key, default)
    return default if v is None else v


def _players(obs):
    """返回 (players 列表, 我的下标) —— 同时支持 dict 与 dataclass 的 observation。"""
    cur = _get(obs, "current")
    if cur is None:
        return None, 0
    ps = _get(cur, "players") or []
    me = int(_get(cur, "yourIndex", 0))
    return ps, me


def _pokemon_at(ps, me, area, index):
    """把 area+index 解析成宝可梦对象；解析不了返回 None。"""
    try:
        if area == 4:          # AreaType.ACTIVE
            arr = _get(ps[me], "active") or []
        elif area == 5:        # AreaType.BENCH
            arr = _get(ps[me], "bench") or []
        else:
            return None
        arr = [x for x in arr if x]
        return arr[int(index)] if index is not None and 0 <= int(index) < len(arr) else None
    except Exception:
        return None


def _energy_count(pk):
    n = _get(pk, "energies")
    if n:
        return len(n)
    return len(_get(pk, "energyCards") or [])


def _score_pokemon(pk, is_active: bool):
    """给"该不该把能量投给它"打分，越大越优先。

    设计依据：能量是稀缺资源，应该投给【马上能打出来】且【活得下去】的那只。
      · 差 1 张就能满足最便宜招式  -> 最高优先级（立刻转化为攻击）
      · 已有能量进度                -> 越接近能攻击越好
      · 血量                          -> 太薄的血量投入容易打水漂
      · 出场位                        -> 能立刻攻击，略加分
    """
    hp = float(_get(pk, "hp", 0) or 0)
    max_hp = float(_get(pk, "maxHp", hp) or hp or 1)
    cur = _energy_count(pk)
    need = _MIN_ATK_COST.get(_get(pk, "id"), 0)
    score = 0.0
    if need > 0:
        prog = min(cur / need, 1.0)
        score += 2.0 * prog
        if cur < need and need - cur == 1:
            score += 3.0                 # 再贴一张就能攻击
        if cur >= need:
            score += 1.0                 # 已经能攻击，边际价值略低
    if is_active:
        score += 2.0
    score += 1.5 * (hp / max_hp)
    if max_hp > 0 and hp / max_hp < 0.3:
        score -= 3.0                     # 残血，别浪费
    return score


# 这些 context 的选项指向"我的某只宝可梦"，可以用上面的打分挑目标
POKEMON_TARGET_CONTEXTS = None   # 延迟初始化，避免 SelectContext 未导入


def pick_pokemon_target(obs, areas, indices):
    """在"选择我的宝可梦"这类子决策里挑最优下标；挑不了返回 None。"""
    ps, me = _players(obs)
    if not ps or len(ps) <= me:
        return None
    best_i, best_s = None, -1e9
    for i, (a, idx) in enumerate(zip(areas, indices)):
        pk = _pokemon_at(ps, me, a, idx)
        if pk is None:
            continue
        s = _score_pokemon(pk, is_active=(a == 4))
        if s > best_s:
            best_i, best_s = i, s
    return best_i


def _decide(stype, ctx, opt_types, lo, hi, attack_ids=None, areas=None,
            indices=None, obs=None, n_opts=None, prio=None):
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
        table = prio if prio is not None else PRIORITY

        def _key(i):
            p = table.get(opt_types[i], 99)
            if opt_types[i] != OptionType.ATTACK or not attack_ids:
                return (p, 0, 0)
            dmg, cost = _ATTACK_INFO.get(attack_ids[i], (0, 9))
            return (p, -dmg, cost)     # ATTACK 内部按伤害降序，再按能量需求升序
        return [min(range(n), key=_key)]

    # 是/否：默认"是"
    if stype == SelectType.YES_NO:
        for i, t in enumerate(opt_types):
            if t == OptionType.YES:
                return [i]
        return [0]

    # 铺后备：选满（主力倒了空场会直接判负）
    if ctx in FILL_CONTEXTS:
        return list(range(hi))

    # ---- 策略子决策：选"我的哪只宝可梦"
    # 原先是 `return list(range(k))`，即永远取列表第一个 —— 把决定权
    # 交给了引擎的选项排列顺序。这里改成按"离能攻击的进度 + 血量"打分。
    if ctx in _POKEMON_TARGET_CTX and obs is not None and areas and indices and hi >= 1:
        best = pick_pokemon_target(obs, areas, indices)
        if best is not None:
            return [best]

    # 其余：满足下限即可
    k = max(lo, 1 if hi >= 1 else 0)
    return list(range(min(k, hi)))


# 子决策总开关。默认关闭：实测它作为【部署策略】是 0.4637（略差于 0.5），
# 且只改变 2.2% 的行为。但 rollout 的续弈模拟曾经单独启用它，造成
# "模拟用的策略 ≠ 实际打的策略"，估值会有偏差 —— 所以这里统一成一个开关。
SMART_SUB = False


# 哪些 context 的选项指向"我的宝可梦"（可用 HP/能量打分挑目标）
_POKEMON_TARGET_CTX = frozenset({
    SelectContext.ATTACH_FROM,          # 贴能量给哪只
    SelectContext.SWITCH,               # 换哪只上场
    SelectContext.TO_ACTIVE,            # 把后备提上场
    SelectContext.SETUP_ACTIVE_POKEMON,  # 开局选出场
})


def rule_action(sel, obs=None, prio=None):
    """给定 SelectData（dict 或 dataclass），返回规则策略会选的选项下标列表。

    Args:
        sel: SelectData
        obs: 完整 observation（可选）。给了才能做策略子决策
             （如"贴能量给哪只宝可梦"），否则退化为"取第一个选项"。
    """
    return _decide(*_sel_fields(sel), obs=obs, prio=prio)


def is_main(sel) -> bool:
    stype, *_ = _sel_fields(sel)
    return stype == SelectType.MAIN


def has_type(sel, want) -> bool:
    _, _, opt_types, *_ = _sel_fields(sel)
    return want in opt_types


def index_of_type(sel, want):
    _, _, opt_types, *_ = _sel_fields(sel)
    for i, t in enumerate(opt_types):
        if t == want:
            return i
    return None


# ============================================================ rollout
def rollout_to_end(sid, obs, my_index: int, max_steps: int = 800, playout=None):
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
        # 续弈策略：默认与兜底一致（rule_action）。传入 playout 可换一套 ——
        # 这是"评估器偏差"的唯一单变量入口：模拟用的策略 ≠ 实际打的策略时，
        # 估值会有系统偏差。注意它同时作用于双方。
        act = (playout(sel, obs) if playout is not None
               else rule_action(sel, obs if SMART_SUB else None))
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
                     memory=None, playout=None):
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
                    rollout_to_end(st.searchId, st.observation, my_index,
                                   max_steps, playout)
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
                  max_steps: int = 800, memory=None,
                  candidate_mode: str = "trio", margin_k: float = 0.0,
                  playout=None):
    """决定这一手该下哪个动作。

    安全设计：**默认就是规则策略的选择**，只有当某个候选在配对比对上
    显著更好（差值 > margin 个标准误）时才偏离。这样最坏情况退化为 greedy，
    不会因为 rollout 噪声而系统性地变差。

    Returns:
        (action: list[int], info: dict)
    """
    sel = obs.get("select")
    base = rule_action(sel, obs if SMART_SUB else None)
    info = {"base": base, "chosen": base, "reason": "rule",
            "cands": [], "stats": {}, "pairs": {}, "worlds": 0}
    if not sel or len(base) != 1:
        return base, info
    if not is_main(sel):
        return base, info

    base_i = base[0]
    n_opts = len(sel.get("option") or [])

    # 候选集：
    #   trio —— [baseline, ATTACK, END]，实测约 12 个选项里只看 2~3 个（默认）
    #   all  —— 全部选项。成本约 4 倍，但覆盖完整
    #
    # ⚠️ 候选变多后有个统计问题：我们从 k 个候选里取"最大 z"，k 越大这个最大值
    #    天然越大（multipel comparison），所以 margin 必须同步调高，否则会
    #    在纯噪声上做出偏离。经验上 k 从 3 增到 12，阈值应从 ~1.0 提到 ~2.0+。
    if candidate_mode == "all":
        cands = [base_i] + [i for i in range(n_opts) if i != base_i]
    else:
        cands = [base_i]
        for want in (OptionType.ATTACK, OptionType.END):
            i = index_of_type(sel, want)
            if i is not None and i not in cands:
                cands.append(i)
    if len(cands) < 2:                  # 没有可比的备选
        return base, info

    # 多重比较校正：我们从 k 个候选里取"最大 z"，k 越大这个最大值天然越大
    # （k 个标准正态的最大值 ≈ √(2 ln k)）。所以阈值必须随 k 上升，
    # 否则候选一多就会在纯噪声上做出假偏离。
    #   margin_k=0     → 关闭（保持旧的平坦阈值，向后兼容）
    #   margin_k=0.7   → k=3 时 ×1.00，k=12 时 ×1.97，k=122 时 ×3.5
    if margin_k:
        k_eff = max(len(cands), 3)
        margin = margin * (1.0 + margin_k * math.log(k_eff / 3.0))
        info["margin_used"] = margin
        info["n_cands"] = len(cands)

    outcomes, ev = evaluate_actions(obs, cands, filler=filler, my_deck=my_deck,
                                   n_worlds=n_worlds, max_steps=max_steps,
                                   memory=memory, playout=playout)
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
