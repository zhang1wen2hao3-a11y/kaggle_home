"""rollout 变体：候选集回到 trio（[base, ATTACK, END]），其余与 ro_attack_w10 完全一致。

用途：复测「候选集 all vs trio」。历史上的 `cands_ab.log`（2026-10-18）测出
all 比 trio 高 +0.110（n=600，5.5σ），但那是在 **BUG-002 修复之前**跑的 ——
当时 worker 里的 rollout 累计 20 秒后会静默退化成纯规则，所以那个数字不可信。

这里把 margin 固定成和基线一样的 1.5，隔离出"候选集"这一个变量
（历史那次是 all(m1.5) vs trio(m1.0)，混了 margin；而 margin 1.0 vs 1.5
 实测是噪声，所以影响不大，但没必要再混一次）。
"""
import importlib.util as _ilu
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("_mr_trio", _os.path.join(_HERE, "..", "main_rollout.py"))
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)

_m.MODE = "attack"
_m.WORLDS = 10
_m.MARGIN = 1.5
_m.CANDS = "trio"

agent = _m.agent
agent.reset = _m.agent.reset
