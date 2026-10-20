"""rollout 变体：**只换优先级表** = 只在 EVOLVE 之前插 PLAY（ATTACH 仍第一），ATTACK 位置不动

顺序：ATTACH → PLAY → EVOLVE → ABILITY → ATTACK → END → DISCARD → RETREAT
基线 safe_v1：ATTACH → EVOLVE → PLAY → ABILITY → ATTACK → END → DISCARD → RETREAT

⚠️ 与 `agents/prio_dev.py` 的区别：那个用的是 rulebase 里的 develop_first，
它的 **RETREAT(4)/DISCARD(5) 排在 ATTACK(6) 之前** —— 那是当年那个大 bug 的形状
（agent 会撤退而不是攻击）。所以 prio_dev 测出的 0.4167 混了"先出牌"和"先撤退"
两个因素。本文件把 ATTACK 钉在 END/DISCARD/RETREAT 之前，只动 PLAY 的位置。
"""
import importlib.util as _ilu
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("_mr_prio_play2", _os.path.join(_HERE, "..", "main_rollout.py"))
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)

import framework.rollout as _R
from cg.api import OptionType as _OT

_ORDER = [_OT.ATTACH, _OT.PLAY, _OT.EVOLVE, _OT.ABILITY, _OT.ATTACK, _OT.END, _OT.DISCARD, _OT.RETREAT]
_PRIO = {t: i for i, t in enumerate(_ORDER)}


def _play(sel, obs):
    return _R.rule_action(sel, None, prio=_PRIO)


_m.MODE = "attack"
_m.WORLDS = 10
_m.MARGIN = 1.5
_m.CANDS = "all"
_m.BASE_PRIO = _PRIO
_m.PLAYOUT = _play

agent = _m.agent
agent.reset = _m.agent.reset
