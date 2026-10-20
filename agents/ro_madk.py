"""rollout 变体：margin=1.5, margin_k=0.7（其余与 ro_attack_w10 完全一致）。

margin_k>0 时阈值随候选数上升：margin_eff = 1.5 × (1 + 0.7·ln(k/3))。
"""
import importlib.util as _ilu
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("_mr_madk", _os.path.join(_HERE, "..", "main_rollout.py"))
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)

_m.MODE = "attack"
_m.WORLDS = 10
_m.MARGIN = 1.5
_m.MARGIN_K = 0.7
_m.CANDS = "all"

agent = _m.agent
agent.reset = _m.agent.reset
