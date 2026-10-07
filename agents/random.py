"""随机基线 agent：本地评估用的最低门槛对手。

签名为提交版签名 agent(obs_dict) -> list[int]，可直接用于本地对局。
"""
import random


def agent(obs_dict):
    sel = obs_dict.get("select")
    if sel is None:
        # 开局选牌：必须返回 60 张卡组
        return _deck()
    n = len(sel["option"])
    lo, hi = sel["minCount"], min(sel["maxCount"], n)
    if hi <= 0:
        return []
    k = random.randint(lo, hi)
    return random.sample(range(n), k)


def _deck():
    import os
    path = "deck.csv" if os.path.exists("deck.csv") else "/kaggle_simulations/agent/deck.csv"
    with open(path) as f:
        return [int(x) for x in f.read().split() if x.strip()][:60]
