"""rollout 变体：子决策开关 `SMART_SUB=False`，其余完全相同。

⚠️ 为什么必须在**每次调用**里设，而不是 import 时设一次：
   `framework.rollout` 是进程级单例模块，A/B 两侧共用同一个对象。
   import 时设一次的话，谁后加载谁生效，另一侧被静默污染。
"""
import importlib.util as _ilu
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("_mr_sub_off", _os.path.join(_HERE, "..", "main_rollout.py"))
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)

import framework.rollout as _R

_m.MODE = "attack"
_m.WORLDS = 10
_m.MARGIN = 1.5
_m.CANDS = "all"

_WANT = False


def agent(obs_dict):
    _R.SMART_SUB = _WANT          # 每次调用都设，避免两侧互相污染
    return _m.agent(obs_dict)


agent.reset = _m.agent.reset
