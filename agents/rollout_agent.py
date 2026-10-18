"""rollout_agent.py —— 用规则策略 rollout 评估动作，决定是否偏离 greedy。

这是"能不能超过 greedy"的**决定性实验**。它不需要任何训练好的模型。

## 做法

对每个高价值决策点（默认：MAIN 且 `ATTACK` 可选）：

    候选 = greedy 的选择 + ATTACK(若有) + END(若有)
    对每个 world（determinization 采样）：
        search_begin(同一个 world)              <- 所有候选共享，CRN
        对每个候选：search_step 后一路用规则策略打到终局
    逐候选配对比较胜率差（d_i = v_i - v_greedy）
    ★ 只有 均值差 > margin × 标准误 时才偏离，否则保持 greedy

## 为什么这样设计

上一代"用 V 评估叶子"的搜索实测只有 0.39 —— 因为 V 的误差(0.05)比动作
带来的真实差异(0.02)还大。换成 rollout 后，实测跨动作信号 0.361 > 噪声 0.273。

安全阀：默认值就是规则策略的选择，只有显著更好才换。最坏情况退化为 greedy，
不会因为 rollout 噪声而系统性变差。

## 环境变量

    PTCG_ROLLOUT_MODE     attack(默认) | all | off
                          attack: 只在 ATTACK 可选时评估（约 42 次/局）
                          all:    每个 MAIN 决策都评估（约 136 次/局，慢 3 倍）
                          off:    纯 greedy（用于等价性自检）
    PTCG_ROLLOUT_WORLDS   determinization 采样数（默认 10）
    PTCG_ROLLOUT_MARGIN   偏离阈值，单位是标准误（默认 1.0）
    PTCG_ROLLOUT_DECK     卡组文件（默认 deck.csv）

## 用法

    ./run.sh evaluate.py --agent agents/rollout_agent.py \
                         --opponent agents/greedy.py --games 400
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.append(HERE)          # append 而非 insert：避免遮蔽标准库

from cg.api import OptionType, SelectType  # noqa: E402
from framework.rollout import (  # noqa: E402
    choose_action, has_type, is_main, rule_action,
)
from framework.filler import make_filler  # noqa: E402
from framework.memory import GameMemory  # noqa: E402

ENV = os.environ
MODE = ENV.get("PTCG_ROLLOUT_MODE", "attack")
WORLDS = int(ENV.get("PTCG_ROLLOUT_WORLDS", "10"))
MARGIN = float(ENV.get("PTCG_ROLLOUT_MARGIN", "1.0"))


def _read_deck() -> list[int]:
    for p in (ENV.get("PTCG_ROLLOUT_DECK"), "deck.csv", "/kaggle_simulations/agent/deck.csv"):
        if p and os.path.exists(p):
            with open(p) as f:
                return [int(x) for x in f.read().split() if x.strip()][:60]
    raise SystemExit("找不到 deck.csv")


_DECK = _read_deck()
_FILLER = make_filler("uniform", my_deck=_DECK)
_MEM = GameMemory()

# 诊断计数（跑完可以 dump 出来看偏离率）
STATS = {"calls": 0, "main": 0, "evaluated": 0, "switched": 0, "worlds": 0}


def agent(obs_dict: dict) -> list[int]:
    sel = obs_dict.get("select")
    if sel is None:                 # 开局交卡组
        return list(_DECK)

    STATS["calls"] += 1
    try:
        _MEM.observe(obs_dict)
    except Exception:
        pass

    base = rule_action(sel)
    if MODE == "off":
        return base

    try:
        if not is_main(sel):
            return base
        STATS["main"] += 1
        if MODE == "attack" and not has_type(sel, OptionType.ATTACK):
            return base

        STATS["evaluated"] += 1
        action, info = choose_action(
            obs_dict, filler=_FILLER, my_deck=_DECK,
            n_worlds=WORLDS, margin=MARGIN, memory=_MEM,
        )
        STATS["worlds"] += info.get("worlds", 0)
        if action != base:
            STATS["switched"] += 1
        return action
    except Exception:
        return base            # 任何异常都不能让整局崩掉
