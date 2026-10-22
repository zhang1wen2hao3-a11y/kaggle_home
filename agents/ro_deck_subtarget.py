"""rollout 变体：卡组由 PTCG_RO_DECK 指定 + **smart_sub=True**（能量/换人目标选择）。

与 `agents/ro_deck_base.py` 的唯一区别就是 SMART_SUB。
⚠️ SMART_SUB 是 `framework.rollout` 的模块级全局，而 A/B 两侧在同一进程里共用它，
所以必须**每次调用都设**，不能在 import 时设一次。
"""
import importlib.util as _ilu
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("_mr_deck_subtarget", _os.path.join(_HERE, "..", "main_rollout.py"))
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)

import framework.rollout as _R

_m.MODE = "attack"
_m.WORLDS = 10
_m.MARGIN = 1.5
_m.CANDS = "all"

_p = _os.environ.get("PTCG_RO_DECK", "").strip()
if _p:
    _ids = [int(x) for x in open(_p).read().split() if x.strip()][:60]
    _m._DECK = _ids
    _m._FILLER = _m._make_filler("uniform", my_deck=_ids)


def agent(obs_dict):
    _R.SMART_SUB = True
    return _m.agent(obs_dict)


agent.reset = _m.agent.reset
