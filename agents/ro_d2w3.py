"""rollout 变体：**depth-2**（reply_width=3），其余与 ro_attack_w10 完全一致。

depth-1 里"对手怎么打"由我们自己的规则策略决定，所以评估的是"这步对规则对手
好不好"。depth-2 换成枚举对手回应并取最坏 —— 估值才反映"对手会挑最坑我的那手"。
"""
import importlib.util as _ilu
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("_mr_d2w3", _os.path.join(_HERE, "..", "main_rollout.py"))
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)

_m.MODE = "attack"
_m.WORLDS = 10
_m.MARGIN = 1.5
_m.CANDS = "all"
_m.REPLY_DEPTH = 1
_m.REPLY_WIDTH = 3

agent = _m.agent
agent.reset = _m.agent.reset
