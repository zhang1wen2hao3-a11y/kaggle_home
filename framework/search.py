"""MCTS 搜索闭环 —— 让 V 真正派上用场。

引擎给的是"前向模拟器"（search_begin/search_step 只返回状态，不含价值），
所以 UCB 选点、价值回传、树管理、determinization 全部要自己实现。本模块就是这些。

## 视角约定（最容易写错的地方）

所有价值都以 **"该节点当前行动方"** 的视角存储：
    node.W / node.N = 该局面下【node.to_move 这个玩家】的平均胜率

回传时逐层翻转。**注意不能假设双方交替行动** —— 同一玩家在自己的回合里
会连续做多次选择（实测确认），而且搜索中途 yourIndex 会变。
所以翻转必须按"父节点与子节点的 to_move 是否相同"来判断，而不是简单地隔层取反。
写错的典型症状是"搜索越深越差"，且极难 debug。

## 用法

    from framework.search import MCTS, HeuristicValue
    mcts = MCTS(value_fn=HeuristicValue(), filler=filler, n_simulations=120, max_depth=12)
    result = mcts.search(obs_dict, memory, my_deck)
    action = result.best_action()
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cg.api import search_begin, search_step, search_end  # noqa: E402


# ============================================================ 价值函数
class ValueFn:
    """价值函数接口：给一个 obs，返回【当前行动方】的胜率估计（0~1）。"""

    name = "base"

    def __call__(self, obs: dict) -> float:
        raise NotImplementedError


class HeuristicValue(ValueFn):
    """手工评估函数 —— 零训练成本的启动基线。

    只用公开信息，且刻意保持简单：它的作用是**让搜索闭环先跑起来**，
    同时充当"学习模型必须超过"的下限。
    """

    name = "heuristic"

    def __init__(self, w_prize: float = 1.1, w_hp: float = 0.9,
                 w_bench: float = 0.35, w_energy: float = 0.15,
                 w_canattack: float = 0.5, w_deckout: float = 0.8,
                 w_noactive: float = 2.0, w_hand: float = 0.08):
        self.w = dict(w_prize=w_prize, w_hp=w_hp, w_bench=w_bench,
                      w_energy=w_energy, w_canattack=w_canattack,
                      w_deckout=w_deckout, w_noactive=w_noactive, w_hand=w_hand)

    def __call__(self, obs: dict) -> float:
        cur = obs.get("current")
        if not cur:
            return 0.5
        res = cur.get("result", -1)
        me_idx = int(cur.get("yourIndex", 0))
        if res != -1:
            if res == 2:
                return 0.5
            return 1.0 if res == me_idx else 0.0

        ps = cur.get("players") or []
        if len(ps) < 2:
            return 0.5
        me, op = ps[me_idx], ps[1 - me_idx]
        w = self.w

        def hp_total(p):
            tot = 0
            for pk in list(p.get("active") or []) + list(p.get("bench") or []):
                if pk:
                    tot += pk.get("hp") or 0
            return tot

        def energy_total(p):
            tot = 0
            for pk in list(p.get("active") or []) + list(p.get("bench") or []):
                if pk:
                    tot += len(pk.get("energies") or [])
            return tot

        # 奖赏卡：越少越好。对手剩得多 = 我领先
        prize_adv = len(op.get("prize") or []) - len(me.get("prize") or [])
        hp_adv = (hp_total(me) - hp_total(op)) / 400.0
        bench_adv = len(me.get("bench") or []) - len(op.get("bench") or [])
        energy_adv = (energy_total(me) - energy_total(op)) / 8.0
        hand_adv = (me.get("handCount", 0) - op.get("handCount", 0)) / 7.0

        my_active = (me.get("active") or [None])[0]
        op_active = (op.get("active") or [None])[0]

        z = (w["w_prize"] * prize_adv
             + w["w_hp"] * hp_adv
             + w["w_bench"] * bench_adv
             + w["w_energy"] * energy_adv
             + w["w_hand"] * hand_adv)
        if my_active is None:
            z -= w["w_noactive"]
        if op_active is None:
            z += w["w_noactive"]
        if me.get("deckCount", 60) <= 3:
            z -= w["w_deckout"]
        return _sigmoid(z)


class ModelValue(ValueFn):
    """用训练好的模型（BaseModel）评估局面。

    特征来自 framework.features.state_features —— 与训练时严格一致，
    否则线上表现会莫名其妙地差。
    """

    name = "model"

    def __init__(self, model, memory=None, use_memory: bool = True):
        self.model = model
        self.memory = memory
        self.use_memory = use_memory
        self._cache: dict[int, float] = {}

    def __call__(self, obs: dict) -> float:
        from framework import features as F
        row = F.state_features(obs, self.memory if self.use_memory else None)
        return float(self.model.predict([row])[0])


def _sigmoid(z: float) -> float:
    if z >= 20:
        return 1.0
    if z <= -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


# ============================================================ MCTS
class Node:
    __slots__ = ("sid", "to_move", "n_options", "children", "untried",
                 "N", "W", "terminal", "parent")

    def __init__(self, sid: int, to_move: int, n_options: int,
                 terminal: bool = False, parent=None):
        self.sid = sid                  # 引擎里的 searchId（用于 fork）
        self.to_move = to_move          # 该节点由哪个玩家决策
        self.n_options = n_options
        self.children: dict[int, "Node"] = {}
        self.untried: list[int] = list(range(n_options))
        self.N = 0
        self.W = 0.0                    # 从 to_move 视角累计的胜率
        self.terminal = terminal
        self.parent = parent

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.5

    def ucb(self, c_puct: float, parent_N: int) -> float:
        """标准 UCB1（从本节点 to_move 视角，越大越好）。"""
        if self.N == 0:
            return float("inf")
        return self.Q + c_puct * math.sqrt(math.log(max(parent_N, 1)) / self.N)


class SearchResult:
    def __init__(self, visits: dict[int, int], values: dict[int, float]):
        self.visits = visits            # 根动作 -> 访问次数
        self.values = values            # 根动作 -> 从根玩家视角的估值

    def best_action(self, by: str = "visits") -> int:
        if not self.visits:
            return 0
        if by == "value":
            return max(self.values, key=lambda a: self.values[a])
        return max(self.visits, key=lambda a: self.visits[a])

    def top(self, k: int = 3) -> list[tuple[int, int, float]]:
        order = sorted(self.visits, key=lambda a: -self.visits[a])[:k]
        return [(a, self.visits[a], self.values.get(a, 0.5)) for a in order]


class MCTS:
    """带 determinization 的蒙特卡洛树搜索。

    流程：每个"世界"（determinization 采样）跑一棵树，最后把各世界根动作的
    估值按访问次数加权平均 —— 这就是 PIMC 的做法。
    """

    def __init__(self, value_fn: ValueFn, filler=None, n_simulations: int = 120,
                 max_depth: int = 12, c_puct: float = 1.4, n_worlds: int = 1,
                 verbose: bool = False, dirichlet_alpha: float = 0.0,
                 policy_fn=None, top_k: int | None = None):
        """Args:
            policy_fn: 可选的策略先验，签名 policy_fn(obs, n_options) -> list[float]。
                给定时，根节点的候选动作会**按先验降序**展开，并可裁剪到 top_k 个。
                这是 AlphaZero 的 P —— 没有它，MCTS 会把 END 和 ATTACH 一视同仁
                地探索，而 V 判不了这种 0.02 量级的差异（实测 search 反而比规则差）。
            top_k: 只探索先验最高的 k 个动作（None=不裁剪）。
        """
        self.value_fn = value_fn
        self.policy_fn = policy_fn
        self.top_k = top_k
        self.filler = filler
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.n_worlds = max(1, n_worlds)
        self.verbose = verbose
        self.dirichlet_alpha = dirichlet_alpha
        self.stats = {"sims": 0, "nodes": 0, "terminal_hits": 0, "worlds": 0}

    # ------------------------------------------------------------ 对外
    def search(self, obs: dict, memory=None, my_deck=None) -> SearchResult:
        """主入口。返回根动作的访问次数与估值。"""
        cur = obs.get("current")
        if not cur:
            return SearchResult({}, {})
        root_player = int(cur.get("yourIndex", 0))
        sel = obs.get("select") or {}
        n_opts = len(sel.get("option") or [])
        if n_opts == 0:
            return SearchResult({}, {})

        agg_v: dict[int, float] = {}
        agg_n: dict[int, int] = {}
        try:
            for wi in range(self.n_worlds):
                world = self._sample_world(obs, memory, my_deck, seed=wi)
                if world is None:
                    continue
                visits, values = self._search_one_world(obs, world, root_player, n_opts)
                self.stats["worlds"] += 1
                for a, n in visits.items():
                    agg_n[a] = agg_n.get(a, 0) + n
                    agg_v[a] = agg_v.get(a, 0.0) + values.get(a, 0.5) * n
        finally:
            try:
                search_end()          # 释放本步搜索的全部状态
            except Exception:
                pass
        # 加权平均
        final_v = {a: (agg_v[a] / agg_n[a] if agg_n.get(a) else 0.5) for a in agg_n}
        if not agg_n:                 # 一个世界都没成功 -> 回退
            return SearchResult({}, {})
        return SearchResult(agg_n, final_v)

    # ------------------------------------------------------------ 内部
    def _sample_world(self, obs, memory, my_deck, seed: int):
        if self.filler is None:
            return None
        try:
            return self.filler.fill(obs, memory, my_deck)
        except Exception as e:
            if self.verbose:
                print(f"    filler 失败: {type(e).__name__}: {e}")
            return None

    def _build_root(self, obs, world, n_opts: int) -> Node | None:
        cur = obs["current"]
        me = int(cur.get("yourIndex", 0))
        try:
            st = search_begin(
                agent_observation=_as_obs_class(obs),
                your_deck=world.get("your_deck") or [],
                your_prize=world.get("your_prize") or [],
                opponent_deck=world.get("opponent_deck") or [],
                opponent_prize=world.get("opponent_prize") or [],
                opponent_hand=world.get("opponent_hand") or [],
                opponent_active=world.get("opponent_active") or [],
            )
        except Exception as e:
            if self.verbose:
                print(f"    search_begin 失败: {type(e).__name__}: {e}")
            return None
        _ = me
        return Node(sid=st.searchId, to_move=int(cur.get("yourIndex", 0)),
                    n_options=n_opts)

    def _search_one_world(self, obs, world, root_player: int, n_opts: int):
        root = self._build_root(obs, world, n_opts)
        if root is None:
            return {}, {}

        order = list(range(n_opts))
        if self.policy_fn is not None:
            try:
                priors = list(self.policy_fn(obs, n_opts))
                if len(priors) == n_opts:
                    order = sorted(order, key=lambda i: -float(priors[i]))
                    self.stats["prior_used"] = self.stats.get("prior_used", 0) + 1
            except Exception as e:
                if self.verbose:
                    print(f"    policy_fn 失败，退化为无先验: {type(e).__name__}: {e}")
        if self.top_k and len(order) > self.top_k:
            order = order[:self.top_k]
            self.stats["pruned"] = self.stats.get("pruned", 0) + 1
        root.untried = order
        for _ in range(self.n_simulations):
            self.stats["sims"] += 1
            path = [root]
            node = root
            depth = 0

            # ---- 1) Selection：沿 UCB 往下走
            while not node.untried and node.children and depth < self.max_depth:
                node = max(node.children.values(),
                           key=lambda c: c.ucb(self.c_puct, node.N))
                path.append(node)
                depth += 1

            # ---- 2) Expansion：展开一个未试过的动作
            child_state = None
            if node.untried and depth < self.max_depth and not node.terminal:
                a = node.untried.pop(0)
                child_state = self._step(node.sid, a)
                if child_state is not None:
                    c_obs = child_state.observation
                    c_cur = getattr(c_obs, "current", None)
                    c_term = bool(c_cur is not None and getattr(c_cur, "result", -1) != -1)
                    c_sel = getattr(c_obs, "select", None)
                    c_n = len(getattr(c_sel, "option", []) or []) if c_sel is not None else 0
                    c_move = int(getattr(c_cur, "yourIndex", node.to_move)) if c_cur else node.to_move
                    child = Node(sid=child_state.searchId, to_move=c_move,
                                 n_options=c_n, terminal=c_term or c_sel is None,
                                 parent=node)
                    node.children[a] = child
                    path.append(child)
                    node = child
                    self.stats["nodes"] += 1
                else:
                    # search_step 失败（多半是该节点已终局）-> 就地评估
                    node.terminal = True

            # ---- 3) Evaluation
            v = self._value_at(node, child_state, root_player, path)

            # ---- 4) Backup：逐层回传，按 to_move 是否变化决定翻转
            self._backup(path, v)

        # 汇总根动作
        visits = {a: c.N for a, c in root.children.items()}
        values = {}
        for a, c in root.children.items():
            # 子节点价值是"子节点 to_move"视角 -> 转到根玩家视角
            values[a] = c.Q if c.to_move == root_player else 1.0 - c.Q
        _ = root_player
        return visits, values

    def _step(self, sid: int, action: int):
        try:
            return search_step(sid, [action])
        except Exception:
            return None

    def _value_at(self, node: Node, child_state, root_player: int, path) -> float:
        """返回【node.to_move】视角的价值。"""
        if child_state is not None:
            c_obs = child_state.observation
            c_cur = getattr(c_obs, "current", None)
            res = getattr(c_cur, "result", -1) if c_cur is not None else -1
            if res != -1:
                self.stats["terminal_hits"] += 1
                if res == 2:
                    return 0.5
                return 1.0 if res == node.to_move else 0.0
            if c_cur is None:
                return 0.5
            v_move = self._eval_obs(c_obs)
            return v_move if node.to_move == int(getattr(c_cur, "yourIndex", node.to_move)) \
                else 1.0 - v_move
        # 没展开成功：用当前节点自身估个值（取反，因为这是"从父节点走过来的"）
        _ = root_player, path
        return 0.5 if node.terminal else 0.5

    def _eval_obs(self, obs_obj) -> float:
        """把 SearchState 的 Observation（dataclass）转回 dict 再交给 value_fn。"""
        d = _observation_to_dict(obs_obj)
        try:
            return float(self.value_fn(d))
        except Exception:
            return 0.5

    def _backup(self, path: list[Node], v: float) -> None:
        """v 是 path[-1].to_move 视角的价值；逐层向上回传。

        关键：不能假设交替行动 —— 同一玩家可能连续决策多次，
        所以只在"父节点 to_move 与子节点不同"时才翻转。
        """
        for i in range(len(path) - 1, -1, -1):
            node = path[i]
            node.N += 1
            node.W += v
            if i > 0:
                parent = path[i - 1]
                if parent.to_move != node.to_move:
                    v = 1.0 - v


# ============================================================ 工具
def _as_obs_class(obs_dict: dict):
    from cg.api import to_observation_class
    return to_observation_class(obs_dict)


def _observation_to_dict(obs_obj) -> dict:
    """SearchState.observation 是 dataclass，转回 dict 以便复用 features。"""
    import dataclasses
    cur = getattr(obs_obj, "current", None)
    sel = getattr(obs_obj, "select", None)

    def conv(o):
        if o is None:
            return None
        # 关键：list 必须递归处理。之前漏了这里，players 仍是 PlayerState
        # 对象列表，features/HeuristicValue 拿到非 dict 直接退化成 0.5，
        # 表现为"搜索里所有动作估值都是 0.500"。
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        if not dataclasses.is_dataclass(o):
            return o
        return {f.name: conv(getattr(o, f.name)) for f in dataclasses.fields(o)}

    def conv_list(xs):
        return [conv(x) for x in (xs or [])]

    out = {"select": None, "logs": conv_list(getattr(obs_obj, "logs", None)),
           "current": None, "search_begin_input": None}
    if sel is not None:
        sd = conv(sel)
        # option 是 list[Option] -> 已是 dict 列表
        out["select"] = sd
    if cur is not None:
        out["current"] = conv(cur)
    return out
