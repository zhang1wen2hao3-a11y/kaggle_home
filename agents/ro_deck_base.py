"""rollout 变体：卡组由 PTCG_RO_DECK 指定，**子决策不改**（对照组）。"""
import importlib.util as _ilu
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("_mr_deck_base", _os.path.join(_HERE, "..", "main_rollout.py"))
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


agent = _m.agent
agent.reset = _m.agent.reset
