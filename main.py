"""提交入口：PTCG AI Battle Challenge。

提交时上传的就是 本文件 + cg/ + deck.csv，所以这里必须自包含，
不能 import 本仓库的 agents/ 目录。

agent() 的约定：
  * 开局 obs.select is None  -> 必须返回 60 张卡组
  * 其余情况返回 option 下标列表，长度在 [minCount, maxCount] 之间，且不能重复
"""
import os

from cg.api import OptionType, SelectType, SelectContext, to_observation_class

# 顺序即优先级。切勿把 ATTACK 排前面 —— 攻击会结束回合，
# 攻击优先会让 agent 每回合只平A、从不铺场（实测对 random 胜率仅 0.17）。
# 必须"先铺场/进化/充能，最后攻击"。此顺序与 agents/rulebase.py 的 develop_first 一致。
PRIORITY = {
    OptionType.PLAY: 0,
    OptionType.EVOLVE: 1,
    OptionType.ABILITY: 2,
    OptionType.ATTACH: 3,
    OptionType.RETREAT: 4,
    OptionType.DISCARD: 5,
    OptionType.ATTACK: 6,
    OptionType.END: 9,
}
GO_FIRST = True


def read_deck_csv() -> list[int]:
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path) as f:
        return [int(x) for x in f.read().split() if x.strip()][:60]


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()

    opts = obs_dict["select"]["option"]
    n = len(opts)
    lo = obs.select.minCount
    hi = min(obs.select.maxCount, n)
    if hi <= 0 or n == 0:
        return []

    if obs.select.type == SelectType.YES_NO and obs.select.context == SelectContext.IS_FIRST:
        want = OptionType.YES if GO_FIRST else OptionType.NO
        for i, o in enumerate(opts):
            if o["type"] == want:
                return [i]
        return [0]

    if obs.select.type == SelectType.MAIN:
        best = min(range(n), key=lambda i: PRIORITY.get(OptionType(opts[i]["type"]), 99))
        return [best]

    if obs.select.type == SelectType.YES_NO:
        for i, o in enumerate(opts):
            if o["type"] == OptionType.YES:
                return [i]
        return [0]

    # 铺设后备时选满：主力被击倒后 Active 空场会直接判负
    if obs.select.context in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH):
        return list(range(hi))

    k = max(lo, 1 if hi >= 1 else 0)
    return list(range(min(k, hi)))
