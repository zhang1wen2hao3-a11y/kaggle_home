"""最小可跑 agent：随机合法动作。

用途：作为本地评估的基准对手。
  胜率(你的 agent vs random) 是最低门槛指标 —— 如果连随机都打不过，先别提交。
"""
import random


def agent(obs, config=None):
    """obs 的结构取决于 Kaggle 环境；这里给的是防御式写法。

    先跑通一次对局、把 obs print 出来，再按真实结构改成自己的选牌逻辑。
    """
    if isinstance(obs, dict):
        select = obs.get("select")
        if select is not None:
            options = select.get("option") if isinstance(select, dict) else None
            if options:
                return random.randrange(len(options))
        action = obs.get("action")
        if isinstance(action, list) and action:
            return random.randrange(len(action))
    return 0
