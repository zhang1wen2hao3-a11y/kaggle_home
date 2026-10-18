"""MCTS 搜索 agent —— 用搜索替代纯规则出招。

这是"模型能不能打赢规则"的验证载体。组件全部来自 framework：
    GameMemory  -> 累积增量日志，供特征和填充器用
    filler      -> determinization，给 search_begin 提供被抹掉的隐藏信息
    MCTS        -> UCB 选点 + 价值回传 + 视角翻转
    ValueFn     -> 叶节点评估（启发式 或 训练好的 LightGBM V）

## 性能上的关键取舍

一次搜索约 30~120ms（取决于 V 和模拟次数），而一局有 40~200 步。
所以必须做**剪枝优化**：
  · 只有一个候选动作时直接返回，不搜索（这类决策占比很高）
  · maxCount > 1 的多选决策回退到规则策略（MCTS 这里是按单个下标翻牌的）
  · 模拟次数/深度可调，先用小值验证方向，再放大

## 配置（环境变量，方便配合 evaluate.py 直接测）

    PTCG_SEARCH_VALUE=heuristic|model   叶评估用哪种（默认 heuristic）
    PTCG_SEARCH_MODEL=path              model 时的权重路径
    PTCG_SEARCH_SIMS=40                每步模拟次数
    PTCG_SEARCH_DEPTH=6                搜索深度上限
    PTCG_SEARCH_WORLDS=1               determinization 次数
    PTCG_SEARCH_DECK=deck.csv          使用的卡组
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from framework.filler import UniformFiller  # noqa: E402
from framework.memory import GameMemory  # noqa: E402
from cg.api import search_begin, search_end, search_step  # noqa: E402
from framework.search import MCTS, HeuristicValue, ModelValue  # noqa: E402
from framework.search import _observation_to_dict as _obs_to_dict  # noqa: E402

ENV = os.environ


def _load_value_fn():
    kind = ENV.get("PTCG_SEARCH_VALUE", "heuristic").lower()
    if kind == "model":
        path = ENV.get("PTCG_SEARCH_MODEL", os.path.join(HERE, "models/value_pool_v1.json"))
        if not os.path.exists(path):
            print(f"[search_agent] 模型不存在 {path}，回退到启发式", file=sys.stderr)
            return HeuristicValue(), "heuristic(fallback)"
        from framework.models.lightgbm_model import LGBMModel
        m = LGBMModel.load(path)
        return ModelValue(m), f"model({os.path.basename(path)})"
    return HeuristicValue(), "heuristic"


def _load_deck() -> list[int]:
    path = ENV.get("PTCG_SEARCH_DECK", "deck.csv")
    if not os.path.isabs(path):
        path = os.path.join(HERE, path)
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path) as f:
        return [int(x) for x in f.read().split() if x.strip()][:60]


class SearchAgent:
    """带记忆 + 搜索的 agent。一个实例对应一局（或复用，靠 is_new_game 重置）。"""

    def __init__(self, value_fn=None, deck: list[int] | None = None,
                 n_simulations: int = 40, max_depth: int = 6, n_worlds: int = 1,
                 fallback_agent=None):
        self.deck = deck or _load_deck()
        if value_fn is None:
            value_fn, self.vname = _load_value_fn()
        else:
            self.vname = "custom"
        self.value_fn = value_fn
        self.memory = GameMemory()
        self.mcts = MCTS(value_fn=value_fn,
                         filler=UniformFiller(my_deck=self.deck,
                                              opponent_deck_prior=self.deck, seed=0),
                         n_simulations=n_simulations, max_depth=max_depth,
                         n_worlds=n_worlds)
        self.fallback = fallback_agent or self._rule_fallback
        self.stats = {"calls": 0, "searched": 0, "trivial": 0, "fallback": 0}

    # ---------------------------------------------------------- 规则回退
    def _rule_fallback(self, obs: dict) -> list[int]:
        """多选/异常时回退：按 option.type 优先级取前 k 个（与 main.py 同逻辑）。"""
        from cg.api import OptionType, SelectType
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        lo = int(sel.get("minCount", 1))
        hi = min(int(sel.get("maxCount", 1)), len(opts))
        if not opts:
            return []
        prio = {OptionType.PLAY: 0, OptionType.EVOLVE: 1, OptionType.ABILITY: 2,
                OptionType.ATTACH: 3, OptionType.RETREAT: 4, OptionType.DISCARD: 5,
                OptionType.ATTACK: 6, OptionType.END: 9}
        if int(sel.get("type", -1)) == int(SelectType.MAIN):
            order = sorted(range(len(opts)),
                           key=lambda i: prio.get(OptionType(opts[i]["type"]), 99))
            return order[:max(lo, 1)]
        k = max(lo, 1)
        return list(range(min(k, hi)))

    # ---------------------------------------------------------- 1-ply（诊断用）
    def _choose_1ply(self, obs: dict) -> int | None:
        """只用 V 做一步前瞻：对每个候选走一步，用 V 评估，取最大。

        这是把「V 好不好用」和「MCTS 实现对不对」分开的关键诊断。
        MCTS 里 UCB 探索、访问计数、回传任何一处有问题，1-ply 都不会受影响。
        """
        from cg.api import to_observation_class
        cur = obs.get("current")
        me = int(cur.get("yourIndex", 0))
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        if not opts:
            return None
        world = self.mcts.filler.fill(obs, self.memory, self.deck) if self.mcts.filler else None
        if world is None:
            return None
        try:
            st = search_begin(agent_observation=to_observation_class(obs),
                              your_deck=world.get("your_deck") or [],
                              your_prize=world.get("your_prize") or [],
                              opponent_deck=world.get("opponent_deck") or [],
                              opponent_prize=world.get("opponent_prize") or [],
                              opponent_hand=world.get("opponent_hand") or [],
                              opponent_active=world.get("opponent_active") or [])
        except Exception:
            return None
        root_sid = st.searchId
        best_a, best_v = None, -1.0
        try:
            for a in range(len(opts)):
                try:
                    child = search_step(root_sid, [a])
                except Exception:
                    continue
                c_obs = child.observation
                c_cur = getattr(c_obs, "current", None)
                res = getattr(c_cur, "result", -1) if c_cur is not None else -1
                if res != -1:                       # 直接终局
                    v = 0.5 if res == 2 else (1.0 if res == me else 0.0)
                elif c_cur is None:
                    continue
                else:
                    d = _obs_to_dict(c_obs)
                    try:
                        raw = float(self.value_fn(d))     # 从 c_cur.yourIndex 视角
                    except Exception:
                        continue
                    v = raw if int(getattr(c_cur, "yourIndex", me)) == me else 1.0 - raw
                if v > best_v:
                    best_v, best_a = v, a
        finally:
            try:
                search_end()
            except Exception:
                pass
        return best_a

    # ---------------------------------------------------------- 主入口
    def __call__(self, obs: dict) -> list[int]:
        self.stats["calls"] += 1

        # 开局选牌
        if obs.get("select") is None:
            self.memory = GameMemory()
            return list(self.deck)

        self.memory.observe(obs)

        cur = obs.get("current")
        sel = obs.get("select") or {}
        if cur is None or not sel:
            return self.fallback(obs)

        opts = sel.get("option") or []
        hi = int(sel.get("maxCount", 1))
        lo = int(sel.get("minCount", 1))

        # 剪枝 1：只有一个候选 -> 不用搜
        if len(opts) <= 1:
            self.stats["trivial"] += 1
            return list(range(lo)) if lo > 0 else [0]

        # 剪枝 2：多选决策 -> 回退规则（MCTS 按单下标展开，不适用于组合选择）
        if hi > 1:
            self.stats["fallback"] += 1
            return self.fallback(obs)

        mode = ENV.get("PTCG_SEARCH_MODE", "mcts").lower()
        if mode == "1ply":
            a = self._choose_1ply(obs)
            if a is None:
                self.stats["fallback"] += 1
                return self.fallback(obs)
            self.stats["searched"] += 1
            return [a]

        try:
            res = self.mcts.search(obs, self.memory, self.deck)
        except Exception as e:
            print(f"[search_agent] 搜索异常，回退规则: {type(e).__name__}: {e}",
                  file=sys.stderr)
            self.stats["fallback"] += 1
            return self.fallback(obs)

        if not res.visits:
            self.stats["fallback"] += 1
            return self.fallback(obs)

        self.stats["searched"] += 1
        return [res.best_action()]


# ---------------------------------------------------------------- 模块级实例
# evaluate.py / local_match.py 通过 `--agent agents/search_agent.py` 直接加载，
# 需要模块级 `agent` 函数。
_INSTANCE: SearchAgent | None = None


def _get_instance() -> SearchAgent:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = SearchAgent(
            n_simulations=int(ENV.get("PTCG_SEARCH_SIMS", 40)),
            max_depth=int(ENV.get("PTCG_SEARCH_DEPTH", 6)),
            n_worlds=int(ENV.get("PTCG_SEARCH_WORLDS", 1)),
        )
    return _INSTANCE


def agent(obs_dict: dict) -> list[int]:
    return _get_instance()(obs_dict)


def reset() -> None:
    """强制重建实例（跨对局干净起步，测试用）。"""
    global _INSTANCE
    _INSTANCE = None
