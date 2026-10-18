"""纯 Q 推理 agent —— 不做任何模拟，直接对全部选项打分选最优（门 3a）。

## 用途

验证「Q 能不能替代 rollout 的终局模拟」。rollout 的做法是：
    对每个候选动作 → 用规则策略打到终局 → 数胜负
本 agent 的做法是：
    对每个选项 → Q(s,a) 直接预测胜率 → 取最大

如果两者对打接近 0.5，说明 Q 是个合格的叶子评估器 —— 那么
「截断 rollout + Q 叶子」乃至 depth-2 PIMC 才有意义（否则每层都要打到终局，深搜不可能）。

如果 Q 明显落后（< 0.4），说明 Q 的信息量不足以替代真实模拟，应放弃 ML 方向。

## 安全阀

默认动作仍是规则策略的选择，只有当 Q 认为别的选项明显更好（差值 > margin）才偏离。
margin 用概率尺度（Q 输出是 0~1 的胜率），默认 0.02。

## 环境变量

    PTCG_Q_MODEL   模型路径（默认 models/cf_q_spread.json）
    PTCG_Q_MARGIN  偏离阈值（概率尺度，默认 0.02；0=纯 argmax）
    PTCG_Q_MODE    main（默认，只在 MAIN 决策用 Q）| off（纯规则对照）
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)                    # append 而非 insert(0)：避免遮蔽标准库
sys.path.append(os.path.dirname(HERE))   # 仓库根，供 framework 导入

from cg.api import OptionType, SelectType  # noqa: E402
from framework import features as F  # noqa: E402
from framework.memory import GameMemory  # noqa: E402
from framework.models.base import create as _create_model  # noqa: E402
from framework.rollout import rule_action  # noqa: E402

ENV = os.environ
MODEL_PATH = ENV.get("PTCG_Q_MODEL", os.path.join(HERE, "..", "models", "cf_q_spread.json"))
MARGIN = float(ENV.get("PTCG_Q_MARGIN", "0.02"))
QMODE = ENV.get("PTCG_Q_MODE", "main")


def _read_deck() -> list[int]:
    for p in (ENV.get("PTCG_Q_DECK"), "deck.csv", "/kaggle_simulations/agent/deck.csv"):
        if p and os.path.exists(p):
            with open(p) as f:
                return [int(x) for x in f.read().split() if x.strip()][:60]
    raise SystemExit("找不到 deck.csv")


_DECK = _read_deck()
# ⚠️ BaseModel.load 是 classmethod，返回【新实例】；必须接住返回值，
#    否则拿到的是未加载的空模型（feature_keys 为空）。
_MODEL = _create_model("lightgbm").load(MODEL_PATH)
_KEYS = list(_MODEL.feature_keys)
_STATE_KEYS = [k for k in _KEYS if not k.startswith("opt_")]
_OPT_KEYS = [k for k in _KEYS if k.startswith("opt_")]
_MEM = GameMemory()


def _predict_rows(obs, mem, n_opts):
    """给每个选项拼出完整特征行（状态特征 + 该选项的选项特征）。"""
    sf = F.state_features(obs, mem)
    base = [float(sf.get(k, 0.0) or 0.0) for k in _STATE_KEYS]
    rows = []
    for i in range(n_opts):
        of = F.option_features(obs, i, mem)
        rows.append(base + [float(of.get(k, 0.0) or 0.0) for k in _OPT_KEYS])
    return rows


def agent(obs_dict: dict) -> list[int]:
    sel = obs_dict.get("select")
    if sel is None:
        return list(_DECK)

    base = rule_action(sel, obs_dict)
    if QMODE == "off":
        return base
    if int(sel.get("type", -1)) != int(SelectType.MAIN):
        return base

    try:
        _MEM.observe(obs_dict)
        n = len(sel.get("option") or [])
        if n < 2:
            return base
        preds = _MODEL.predict_arrays(_predict_rows(obs_dict, _MEM, n))
        preds = [float(p) for p in preds]
        best = max(range(n), key=lambda i: preds[i])
        # 安全阀：只有明显更好才偏离规则策略
        if best != base[0] and preds[best] - preds[base[0]] > MARGIN:
            return [best]
        return base
    except Exception:
        return base
