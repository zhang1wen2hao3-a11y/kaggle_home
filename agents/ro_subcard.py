"""rollout 变体：子决策改法（SUB_CARD_CHOICE=1, SKIP_OPTIONAL=0），其余与基线一致。

- SUB_CARD_CHOICE：`TO_HAND`/`LOOK` 这类"选一张牌"的决策，按**卡牌价值**挑
  （补进化线 > 能立刻攻击的基础怪 > 手牌少时的抽卡支援者 > 缺能量时的能量），
  而不是取引擎枚举的第一个。
- SKIP_OPTIONAL：`minCount == 0` 时，若最好的那张牌打分 <= 0 就跳过不选。
"""
import importlib.util as _ilu
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("_mr_subcard", _os.path.join(_HERE, "..", "main_rollout.py"))
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)

import framework.rollout as _R

_m.MODE = "attack"
_m.WORLDS = 10
_m.MARGIN = 1.5
_m.CANDS = "all"

_R.SUB_CARD_CHOICE = True
_R.SKIP_OPTIONAL = False

agent = _m.agent
agent.reset = _m.agent.reset
