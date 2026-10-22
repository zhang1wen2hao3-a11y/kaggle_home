"""rollout 变体：卡组由 PTCG_RO_DECK 指定 + **PLAY_SEMANTICS=True**（卡牌角色语义）。

与 `agents/ro_deck_base.py` 的唯一区别：
  · PLAY 选项按**卡牌角色**打分排序（检索/抽卡/回收/加速/代价/条件），
    不再只按"类型优先级 + 引擎枚举顺序"
  · ATTACK 里"下回合不能攻击"的招式降权（Palafin ex 崩到 0.14 的元凶）
角色标签从卡面文本机械抽取（`cg.api.CardData.skills[].text`），可审可复现。
"""
import importlib.util as _ilu
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("_mr_deck_sem", _os.path.join(_HERE, "..", "main_rollout.py"))
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
    _R.PLAY_SEMANTICS = True
    _R.SMART_SUB = False
    return _m.agent(obs_dict)


agent.reset = _m.agent.reset
