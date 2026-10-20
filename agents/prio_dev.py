"""rollout 变体：**只换优先级表**（核心策略） = develop_first：先出牌/铺场，ATTACH 往后，ATTACK 靠尾

基线 safe_v1：ATTACH → EVOLVE → PLAY → ABILITY → ATTACK → END → DISCARD → RETREAT

⚠️ 这里**同时**换 base 和 playout：优先级表是 agent 的核心策略，它既是安全阀、
又是搜索的基准候选、也是续弈策略。只换一处会造成"模拟策略 ≠ 实际策略"的偏差，
那就不是单变量实验了。

历史：8 种顺序只在 2026-10-18 用 n=200 两两测过一次，而且是在 BUG-002 修复前
（当时 worker 里 rollout 累计 20 秒后会静默退化），误差 5pp 以上，测不出 2~4pp。
现在单局 0.6 秒、n=600 的 SE 是 2.0pp，可以认真测了。
"""
import importlib.util as _ilu
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("_mr_prio_dev", _os.path.join(_HERE, "..", "main_rollout.py"))
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)

import framework.rollout as _R
from cg.api import OptionType as _OT

_ORDER = [_OT.PLAY, _OT.EVOLVE, _OT.ABILITY, _OT.ATTACH, _OT.RETREAT, _OT.DISCARD, _OT.ATTACK, _OT.END]
_PRIO = {t: i for i, t in enumerate(_ORDER)}


def _play(sel, obs):
    return _R.rule_action(sel, None, prio=_PRIO)


_m.MODE = "attack"
_m.WORLDS = 10
_m.MARGIN = 1.5
_m.CANDS = "all"
_m.BASE_PRIO = _PRIO      # 搜索的锚（基准候选）
_m.PLAYOUT = _play        # 续弈策略用同一张表

agent = _m.agent
agent.reset = _m.agent.reset
