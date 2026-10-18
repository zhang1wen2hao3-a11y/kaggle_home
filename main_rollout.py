"""提交入口：rollout 版。用规则策略把候选动作打到底，再决定走哪手。

依赖 `framework/rollout.py` 与 `framework/filler.py`，两个文件随包一起上传。

## ⚠️ 时间预算保护（本文件最重要的部分）

rollout 比纯规则慢约 165 倍，实测：快攻卡组 0.5~1.5 s/局，长局卡组可达 15 s/局。
而**我们不知道 Kaggle 沙箱的单局时限**。一旦超时，整局算失败，比打输更糟。

所以这里加了硬保险丝：

    PTCG_BUDGET_S（默认 45 秒）—— 累计耗时超过预算后，本局剩下的决策
    全部退化为纯规则策略（每步约 0.05 ms）

效果：最坏情况是"这一局变成纯规则策略"，而不是"这一局被判定失败"。
预算从每次开局交卡组（`select is None`）时重置。

## 环境变量

    PTCG_ROLLOUT_MODE    attack（默认）| all | off
    PTCG_ROLLOUT_WORLDS  每个局面的 determinization 采样数（默认 10）
    PTCG_ROLLOUT_MARGIN  偏离阈值，单位是标准误（默认 1.0）
    PTCG_BUDGET_S        单局耗时预算（默认 45）
    PTCG_ROLLOUT_DECK    卡组文件（默认 deck.csv）
"""
from __future__ import annotations

import os
import sys
import time

# ⚠️ 不能用 __file__！Kaggle 是用 exec() 执行提交代码的（kaggle_environments/agent.py），
# 那时 __file__ 未定义。实测报错：
#     NameError: name '__file__' is not defined
# 直接导致 status=ERROR、"Validation Episode failed"。官方样例从不使用 __file__。
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = "/kaggle_simulations/agent"

# 把可能的模块搜索路径都加上（append 而非 insert(0)：避免包内模块遮蔽标准库）
for _p in (_HERE, "/kaggle_simulations/agent", os.getcwd()):
    if _p and _p not in sys.path:
        sys.path.append(_p)

from cg.api import OptionType, SelectType, SelectContext, to_observation_class  # noqa: E402

# ⚠️ 提交环境里 framework/ 可能因为任何原因不可用（路径、缺文件、cg 版本差异…）。
# 模块顶层 import 一旦抛异常，整个 agent 就加载不了 —— 平台会直接判
# "Validation Episode failed"，拿不到任何分数。所以这里【必须】容错：
# 失败就退化为纯规则策略（仍然是一个能拿 500 分左右的 agent）。
try:
    from framework.filler import make_filler as _make_filler
    from framework.rollout import (
        choose_action as _choose_action,
        has_type as _has_type,
        is_main as _is_main,
        rule_action as _rule_action,
    )
    _ROLLOUT_OK = True
    _IMPORT_ERR = ""
except Exception as _e:                      # pragma: no cover
    _ROLLOUT_OK = False
    _IMPORT_ERR = f"{type(_e).__name__}: {_e}"

ENV = os.environ
MODE = ENV.get("PTCG_ROLLOUT_MODE", "attack")
WORLDS = int(ENV.get("PTCG_ROLLOUT_WORLDS", "10"))
MARGIN = float(ENV.get("PTCG_ROLLOUT_MARGIN", "1.5"))
# 候选集：trio=[base,ATTACK,END] | all=全部选项
# 实测（跨 5 卡组，对手=最强规则 agent，各 150 局）：
#   trio + margin1.0 = 0.490
#   all  + margin1.5 = 0.733   ← 当前采用
#   all  + margin4.0 = 0.593
CANDS = ENV.get("PTCG_ROLLOUT_CANDS", "all")
# 单局耗时预算。超预算后本局剩余决策全部退化为纯规则 —— 宁可弱，也不要超时判负。
# 实测：sample_official 约 0.4 s/局，长局卡组可达 2~3 s/局；线上时限未知，取保守值。
BUDGET_S = float(ENV.get("PTCG_BUDGET_S", "20"))


def _read_deck() -> list[int]:
    for p in (ENV.get("PTCG_ROLLOUT_DECK"), os.path.join(_HERE, "deck.csv"),
              "deck.csv", "/kaggle_simulations/agent/deck.csv"):
        if p and os.path.exists(p):
            with open(p) as f:
                return [int(x) for x in f.read().split() if x.strip()][:60]
    raise SystemExit("找不到 deck.csv")


_DECK = _read_deck()

# ---- 纯规则兜底：与 main.py 完全一致的优先级表 + 多招式择优
# framework 不可用时用它，保证平台上永远能给出合法动作而不是崩掉。
_FB_PRIORITY = {
    OptionType.ATTACH: 0, OptionType.EVOLVE: 1, OptionType.PLAY: 2,
    OptionType.ABILITY: 3, OptionType.ATTACK: 4, OptionType.END: 5,
    OptionType.DISCARD: 90, OptionType.RETREAT: 91,
}
_FB_ATTACK = {}
try:
    from cg.api import all_attack as _fb_all_attack
    for _a in _fb_all_attack():
        _FB_ATTACK[_a.attackId] = (int(getattr(_a, "damage", 0) or 0),
                                   len(getattr(_a, "energies", None) or []))
except Exception:
    pass


def _fallback_rule(sel_dict: dict) -> list:
    """不依赖 framework 的规则决策（和 main.py 等价）。"""
    opts = sel_dict.get("option") or []
    n = len(opts)
    if n == 0:
        return []
    hi = min(int(sel_dict.get("maxCount", 1)), n)
    lo = int(sel_dict.get("minCount", 1))
    st = int(sel_dict.get("type", -1))
    ctx = int(sel_dict.get("context", -1))
    if st == int(SelectType.YES_NO) and ctx == int(SelectContext.IS_FIRST):
        for i, o in enumerate(opts):
            if o.get("type") == int(OptionType.YES):
                return [i]
        return [0]
    if st == int(SelectType.MAIN):
        def _k(i):
            t = opts[i].get("type")
            p = _FB_PRIORITY.get(OptionType(t), 99) if t is not None else 99
            if t != int(OptionType.ATTACK):
                return (p, 0, 0)
            dmg, cost = _FB_ATTACK.get(opts[i].get("attackId"), (0, 9))
            return (p, -dmg, cost)
        return [min(range(n), key=_k)]
    if st == int(SelectType.YES_NO):
        for i, o in enumerate(opts):
            if o.get("type") == int(OptionType.YES):
                return [i]
        return [0]
    if ctx in (int(SelectContext.SETUP_BENCH_POKEMON), int(SelectContext.TO_BENCH)):
        return list(range(hi))
    k = max(lo, 1 if hi >= 1 else 0)
    return list(range(min(k, hi)))


_FILLER = None
if _ROLLOUT_OK:
    try:
        _FILLER = _make_filler("uniform", my_deck=_DECK)
    except Exception as _e:
        _ROLLOUT_OK = False
        _IMPORT_ERR = f"filler 初始化失败: {_e}"

# 单局计时起点。每次开局交卡组时重置。
_T0 = [time.monotonic()]
_OVER_BUDGET = [False]


def agent(obs_dict: dict) -> list[int]:
    sel = obs_dict.get("select")
    if sel is None:                       # 开局交卡组 => 新的一局，重置预算
        _T0[0] = time.monotonic()
        _OVER_BUDGET[0] = False
        return list(_DECK)

    # ---- 纯规则兜底先算好：它永远可用，不依赖 framework
    base = _fallback_rule(sel)
    if not _ROLLOUT_OK or MODE == "off":
        return base

    # ---- 时间预算保险丝：超预算后退化为纯规则，宁可弱也不要超时判负
    if not _OVER_BUDGET[0] and (time.monotonic() - _T0[0]) > BUDGET_S:
        _OVER_BUDGET[0] = True

    try:
        if _OVER_BUDGET[0] or not _is_main(sel):
            return base
        if MODE == "attack" and not _has_type(sel, OptionType.ATTACK):
            return base
        action, _info = _choose_action(
            obs_dict, filler=_FILLER, my_deck=_DECK,
            n_worlds=WORLDS, margin=MARGIN, memory=None,
            candidate_mode=CANDS,
        )
        return action
    except Exception:
        return base            # 任何异常都不许让整局崩掉
