"""ε-greedy 探索器 —— 为训练 Q(s,a) 制造"动作方差"。

## 为什么需要它

Q(s,a) 要求：**同一个局面下，不同动作被实际试过**。
但 greedy 是确定性策略（给定局面，动作唯一确定），所以：
    a = f(s)  =>  P(win | s, a) == P(win | s)
动作不提供任何额外信息，Q 必然退化成 V。

实测印证：用 greedy 的轨迹训 Q，**选项特征只占 0.4% 的重要性**，
模型完全无视动作。

所以采集数据时必须引入随机性：以概率 ε 随机选一个合法动作，
其余时候用 greedy。这样同一类局面会对应到不同动作，Q 才可识别。

## 用法

    # 生成探索数据（ε=0.3）
    PTCG_EPSILON=0.3 ./run.sh gen_data.py --agent agents/explorer.py ...

注意：`agents/explorer.py` 默认 ε=0.25，可用 PTCG_EPSILON 覆盖。
"""
import os
import random

from rulebase import make_agent

EPS = float(os.environ.get("PTCG_EPSILON", "0.25"))
_rng = random.Random(int(os.environ.get("PTCG_EPSILON_SEED", "0")))

_greedy = make_agent(order="retreat_late" if False else "develop_first",
                     go_first=True, bench_full=True)


def agent(obs_dict):
    """以 EPS 概率随机选合法动作，否则走 greedy。

    随机分支只在"单选取"决策上生效（maxCount==1）；多选取仍交给 greedy，
    因为随机组合选择会大量产生非法/荒谬的组合。
    """
    sel = obs_dict.get("select")
    if sel is None:
        return _greedy(obs_dict)

    opts = sel.get("option") or []
    hi = int(sel.get("maxCount", 1))
    lo = int(sel.get("minCount", 1))

    if opts and hi == 1 and lo <= 1 and _rng.random() < EPS:
        return [_rng.randrange(len(opts))]
    return _greedy(obs_dict)
