"""rollout + 审计守栏：与 ro_attack_w10 完全一致，只多一条
「不偏离到纯伤害更低的招式」。

依据（`tools/audit_decisions.py`，317 局真实线上对局）：
  LOW_DAMAGE_ATTACK 在胜局 9.9‰ / 败局 15.3‰（1.54×），226 次 / 135 局，
  且"规则自己犯"= 0 —— 全部来自 rollout 偏离。

A/B 判据（跑之前定）：对 ro_attack_w10 镜像 2000 局，
  胜率 ≥ 0.53 且配对 z ≥ 2 才算赢；0.47~0.53 视为噪声；≤0.47 视为有害。
"""
import importlib.util as _ilu
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("_mr_guard",
                                     _os.path.join(_HERE, "..", "main_rollout.py"))
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)

_m.MODE = "attack"
_m.WORLDS = 10
_m.MARGIN = 1.5
_m.CANDS = "all"

# 打开守栏（framework.rollout 是 main_rollout 已经 import 的同一个模块对象）
import sys as _sys
_sys.path.insert(0, _os.path.join(_HERE, ".."))
import framework.rollout as _ro  # noqa: E402
_ro.GUARD_PLAIN_ATTACK = True

agent = _m.agent
