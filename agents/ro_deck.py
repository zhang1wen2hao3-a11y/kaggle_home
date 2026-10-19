"""rollout 变体：卡组由环境变量 `PTCG_RO_DECK` 指定，其余配置与已提交的版本一致。

用途：**卡组挑战赛** —— 挑战者用这个文件（每场换 `PTCG_RO_DECK`），
擂主固定用 `agents/ro_attack_w10.py`（它烘的就是项目根 `deck.csv` = `sample_official`）。

为什么只让一侧读环境变量：`evaluate.py` 的 worker 进程里两侧 agent 共享
`os.environ`，两个都读就会互相污染。这里**只有挑战者**读，擂主是烘死的，所以安全。

⚠️ 必须同时重建 `_FILLER`：填充器的 `my_deck` 决定了 determinsation 时
"我方牌库还剩什么"的采样分布 —— 不换它的话，挑战者的 rollout 会按
`sample_official` 的牌库下界去猜自己的牌库，等于带着错误的先验比赛。
`_DECK` 本身只用于开局交卡组（本地对局由 `play_match` 直接传卡组，用不到），
但为了两边一致还是一起换掉。
"""
import importlib.util as _ilu
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("_mr_deck", _os.path.join(_HERE, "..", "main_rollout.py"))
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)

_m.MODE = "attack"
_m.WORLDS = 10
_m.MARGIN = 1.5
_m.CANDS = "all"

_DECK_PATH = _os.environ.get("PTCG_RO_DECK", "").strip()
if _DECK_PATH:
    with open(_DECK_PATH) as _f:
        _ids = [int(x) for x in _f.read().split() if x.strip()][:60]
    if len(_ids) != 60:
        raise SystemExit(f"{_DECK_PATH} 不是 60 张（读到 {len(_ids)} 张）")
    _m._DECK = _ids
    _m._FILLER = _m._make_filler("uniform", my_deck=_ids)
    _m._DECK_SOURCE = _DECK_PATH

agent = _m.agent
