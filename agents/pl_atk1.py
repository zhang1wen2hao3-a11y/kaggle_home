"""rollout 变体：**只换续弈（playout）策略** = attack_first：**正对照**，明显更差的续弈策略；若它与基线无差别，说明 playout 没接上

除了 playout 之外，与 `agents/ro_attack_w10.py` 完全一致。
被测的是"评估器偏差"：判断哪个动作更好靠打到终局的模拟，而模拟里双方都用
这条规则打 —— 如果模拟的策略和实际打的策略差太远，估值就会有系统偏差。
"""
import importlib.util as _ilu
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("_mr_pl_atk1", _os.path.join(_HERE, "..", "main_rollout.py"))
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)

_m.MODE = "attack"
_m.WORLDS = 10
_m.MARGIN = 1.5
_m.CANDS = "all"

import framework.rollout as _R
from cg.api import OptionType as _OT

_ORDER = [_OT.ATTACK, _OT.EVOLVE, _OT.ABILITY, _OT.ATTACH, _OT.PLAY, _OT.RETREAT, _OT.DISCARD, _OT.END]
_PRIO = {t: i for i, t in enumerate(_ORDER)}


def _playout(sel, obs):
    return _R.rule_action(sel, None, prio=_PRIO)


_m.PLAYOUT = _playout

agent = _m.agent
agent.reset = _m.agent.reset
