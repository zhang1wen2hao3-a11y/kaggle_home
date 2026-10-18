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

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    # append 而不是 insert(0)：避免包内模块遮蔽标准库（本仓库踩过这个坑）
    sys.path.append(_HERE)

from cg.api import OptionType  # noqa: E402
from framework.filler import make_filler  # noqa: E402
from framework.rollout import (  # noqa: E402
    choose_action, has_type, is_main, rule_action,
)

ENV = os.environ
MODE = ENV.get("PTCG_ROLLOUT_MODE", "attack")
WORLDS = int(ENV.get("PTCG_ROLLOUT_WORLDS", "10"))
MARGIN = float(ENV.get("PTCG_ROLLOUT_MARGIN", "1.0"))
BUDGET_S = float(ENV.get("PTCG_BUDGET_S", "45"))


def _read_deck() -> list[int]:
    for p in (ENV.get("PTCG_ROLLOUT_DECK"), os.path.join(_HERE, "deck.csv"),
              "deck.csv", "/kaggle_simulations/agent/deck.csv"):
        if p and os.path.exists(p):
            with open(p) as f:
                return [int(x) for x in f.read().split() if x.strip()][:60]
    raise SystemExit("找不到 deck.csv")


_DECK = _read_deck()
_FILLER = make_filler("uniform", my_deck=_DECK)

# 单局计时起点。每次开局交卡组时重置。
_T0 = [time.monotonic()]
_OVER_BUDGET = [False]


def agent(obs_dict: dict) -> list[int]:
    sel = obs_dict.get("select")
    if sel is None:                       # 开局交卡组 => 新的一局，重置预算
        _T0[0] = time.monotonic()
        _OVER_BUDGET[0] = False
        return list(_DECK)

    base = rule_action(sel)
    if MODE == "off":
        return base

    # ---- 时间预算保险丝：超预算后退化为纯规则，宁可弱也不要超时判负
    if not _OVER_BUDGET[0] and (time.monotonic() - _T0[0]) > BUDGET_S:
        _OVER_BUDGET[0] = True

    try:
        if _OVER_BUDGET[0] or not is_main(sel):
            return base
        if MODE == "attack" and not has_type(sel, OptionType.ATTACK):
            return base
        action, _info = choose_action(
            obs_dict, filler=_FILLER, my_deck=_DECK,
            n_worlds=WORLDS, margin=MARGIN, memory=None,
        )
        return action
    except Exception:
        return base            # 任何异常都不许让整局崩掉
