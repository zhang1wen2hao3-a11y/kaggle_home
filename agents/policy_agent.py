"""RL 策略 agent：**MAIN 决策用训练出来的策略网络，其余交给规则**。

与 `train_rl.py` 里采样用的 agent 完全同构，只是不开探索（温度更低、可选纯 argmax），
所以训练出来的东西能直接被 `evaluate.py` 评估。

环境变量：
    PTCG_RL_MODEL   模型路径（默认 models/rl_policy.pt）
    PTCG_RL_TEMP    温度（默认 0.0 = 纯 argmax；>0 则按 π 采样）
    PTCG_RL_MODE    main（默认）| off（退回纯规则，用于对照）
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np  # noqa: E402

ENV = os.environ
MODEL_PATH = ENV.get("PTCG_RL_MODEL", os.path.join(ROOT, "models", "rl_policy.pt"))
TEMP = float(ENV.get("PTCG_RL_TEMP", "0.0"))
BONUS = float(ENV.get("PTCG_RL_BONUS", "4.0"))
MODE = ENV.get("PTCG_RL_MODE", "main")


def _read_deck():
    for p in (os.path.join(ROOT, "deck.csv"),
              "/kaggle_simulations/agent/deck.csv"):
        if os.path.exists(p):
            with open(p) as f:
                return [int(x) for x in f.read().split() if x.strip()][:60]
    return []


_DECK = _read_deck()

_MODEL = None
_OK = False
_ERR = ""
if MODE != "off":
    try:
        import torch
        from framework.features import (
            OPTION_FEATURE_KEYS, STATE_FEATURE_KEYS, option_features, state_features,
        )
        from framework.policy_nn import OptionScorer, load_weights
        from framework.rollout import is_main, rule_action

        _M = OptionScorer(len(STATE_FEATURE_KEYS), len(OPTION_FEATURE_KEYS))
        if os.path.exists(MODEL_PATH):
            load_weights(_M, MODEL_PATH)
            _M.eval()
            _MODEL = _M
            _OK = True
        else:
            _ERR = f"模型不存在: {MODEL_PATH}"
    except Exception as e:                       # 任何异常都退回规则
        _ERR = f"{type(e).__name__}: {e}"


def agent(obs_dict: dict) -> list:
    sel = obs_dict.get("select")
    if sel is None:
        return list(_DECK)
    base = rule_action(sel)
    if not _OK or MODE == "off" or not sel or not is_main(sel) or len(base) != 1:
        return base
    try:
        opts = sel.get("option") or []
        if len(opts) < 2:
            return base
        import torch
        with torch.no_grad():
            sf = np.array([state_features(obs_dict).get(k, 0.0)
                           for k in STATE_FEATURE_KEYS], dtype=np.float32)
            ofs = np.stack([
                np.array([option_features(obs_dict, i).get(k, 0.0)
                          for k in OPTION_FEATURE_KEYS], dtype=np.float32)
                for i in range(len(opts))
            ])
            sc = _MODEL.scores(torch.from_numpy(sf), torch.from_numpy(ofs),
                               rule_idx=int(base[0]), rule_bonus=BONUS)
            if TEMP > 0:
                probs = torch.softmax(sc / TEMP, dim=0)
                return [int(torch.multinomial(probs, 1).item())]
            return [int(torch.argmax(sc).item())]
    except Exception:
        return base


agent.reset = lambda *a, **k: None
agent.model_ok = _OK
agent.model_err = _ERR
