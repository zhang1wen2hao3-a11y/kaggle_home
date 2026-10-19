"""rollout 变体：determinization 换成 **OracleFiller**（读 visualize_data 的真实隐藏信息）。

⚠️ **只用于本地诊断，不能提交。**
   它依赖 `Battle.battle_ptr`；线上提交的 agent 自己不调 `battle_start()`，
   所以 `visualize_data()` 拿不到数据，这条路在线上是断的 —— 不是作弊通道。

它回答一个问题：**我们把隐藏信息猜错，代价有多大？**

    现行 rollout（UniformFiller 猜隐藏卡）  vs  Oracle（用真实隐藏卡）
    差值 = 完美信息的价值 = 对手推断这条路的天花板

判据（跑之前定好，n=300，SE≈0.029）：
    ≥0.58（z≥2.7） → determinization 是瓶颈，**值得**投入对手卡组推断
    0.44~0.56      → 不是瓶颈，去查 playout 策略 / 搜索深度
    <0.44          → 说明 OracleFiller 有实现问题（完美信息不该更弱），先查 bug

⚠️ 注意这个差值是**上限**：真实推断只能从「已见的卡」去推，恢复不了全部信息。
   经验上能回收 20%~40% 就算很成功。
"""
import importlib.util as _ilu
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("_mr_uni", _os.path.join(_HERE, "..", "main_rollout.py"))
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)

_m.MODE = "attack"
_m.WORLDS = 10
_m.MARGIN = 1.5
_m.CANDS = "all"

# 只换填充器，其余完全不变 —— 这样任何差异都只能归因于 determinsation 质量
_m._FILLER = _m._make_filler("uniform")

# 诊断用计数器：确认 oracle 真的成功了，而不是静默失败退回规则
STATS = {"calls": 0, "worlds": 0, "fail": 0, "deviated": 0, "reason": {}}
_orig_choose = _m._choose_action


def _counted_choose(*a, **k):
    action, info = _orig_choose(*a, **k)
    STATS["calls"] += 1
    STATS["worlds"] += info.get("worlds", 0)
    STATS["fail"] += info.get("fail", 0)
    if info.get("chosen") != info.get("base"):
        STATS["deviated"] += 1
    r = info.get("reason", "?")
    STATS["reason"][r] = STATS["reason"].get(r, 0) + 1
    return action, info


_m._choose_action = _counted_choose

agent = _m.agent
agent.STATS = STATS        # 方便诊断脚本通过 agent 函数拿到计数器
