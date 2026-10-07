"""GameMemory —— 在 agent 内累积 obs 给不了的增量信息。

为什么需要它：
    obs["logs"] 是**增量**的（"Events that have occurred since the last selection"），
    只给上次决策以来的事件。不记就永久丢失。
    而对手打出的牌、检索的牌、附的能量，全靠这些日志泄露。

它要解决两类问题：
    1. **对手推断**：累积对手已见卡 → 推断对手卡组构成（filler 用）
    2. **特征供给**：把"历史"压缩成固定长度的特征（V / P 用）

⚠️ 最大的坑：**跨局污染**
    Kaggle 很可能复用进程跨对局运行 agent。如果记忆不重置，
    第二局会带着第一局的对手信息 —— 而且**本地单局测试根本发现不了**。
    所以这里用多重信号检测新对局，并配了回归测试（tests/test_memory.py）。

用法（agent 内）：
    _mem = GameMemory()

    def agent(obs_dict):
        global _mem
        if obs_dict.get("select") is None:   # 开局选牌 = 新对局
            _mem = GameMemory()
        _mem.observe(obs_dict)
        ...
"""
from __future__ import annotations

from collections import Counter

# ---------------------------------------------------------------- 枚举（与 cg.api 对齐）
# 硬编码以避免框架层强依赖引擎导入；数值稳定，见 cg/api.py
LOG_SHUFFLE = 0
LOG_HAS_BASIC = 1
LOG_TURN_START = 2
LOG_TURN_END = 3
LOG_DRAW = 4
LOG_DRAW_REVERSE = 5
LOG_MOVE_CARD = 6
LOG_MOVE_CARD_REVERSE = 7
LOG_SWITCH = 8
LOG_CHANGE = 9
LOG_PLAY = 10
LOG_ATTACH = 11
LOG_EVOLVE = 12
LOG_DEVOLVE = 13
LOG_MOVE_ATTACHED = 14
LOG_ATTACK = 15
LOG_HP_CHANGE = 16
LOG_COIN = 22
LOG_RESULT = 23

AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6

# 只有这些事件携带 cardId（其余是里侧/无卡事件）
LOG_WITH_CARD_ID = {
    LOG_DRAW, LOG_MOVE_CARD, LOG_PLAY, LOG_ATTACH, LOG_EVOLVE,
    LOG_DEVOLVE, LOG_MOVE_ATTACHED, LOG_ATTACK, LOG_HP_CHANGE,
}


class GameMemory:
    """单局内的记忆。每局必须重建。"""

    MAX_HAND_HISTORY = 64

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------ 生命周期
    def reset(self) -> None:
        """清空所有局内状态。新对局必须调用。"""
        self.turns_seen = 0
        self.last_turn = -1
        self.max_turn_seen = -1
        self.logs_processed = 0
        self.game_id = 0

        # 对手（相对"我"的 1-me_idx）已见卡
        self.opp_seen: Counter = Counter()          # cardId -> 次数
        self.my_seen: Counter = Counter()           # 我方已见（用于算未见池）
        self.opp_hand_history: list[int] = []
        self.opp_searches = 0                       # 对手从牌库检索到手牌的次数
        self.opp_energy_attached = 0
        self.opp_plays = 0
        self.opp_discards = 0
        self.my_prizes_taken = 0
        self.opp_prizes_taken = 0
        self._last_my_prize = None
        self._last_opp_prize = None
        self._last_opp_hand = None
        self.finished = False
        self.result = -1

    def is_new_game(self, obs: dict) -> bool:
        """判断这是否是一局新对局的开始。

        多重信号（任一命中即认为新局），按可靠性排序：
          1. **select is None** —— 线上开局的官方信号（agent 被要求返回卡组）。
             线上这是主信号，最可靠。
          2. turn 比上次小 —— 回合数回退，只可能是新局。
          3. turn 回到 0 而之前已见过 >0 的回合 —— 覆盖"上一局停在 turn 0"的边界。
          4. 之前已终局。

        注意：本地 self-play 时 battle_start 直接给卡组，不会出现 select is None，
        所以本地依赖 2/3/4。
        """
        if obs.get("select") is None:
            return True
        cur = obs.get("current")
        if cur is None:
            return True
        turn = int(cur.get("turn", 0))
        if self.last_turn >= 0 and turn < self.last_turn:
            return True
        if turn == 0 and self.max_turn_seen > 0:
            return True
        if self.finished:
            return True
        return False

    def observe(self, obs: dict) -> None:
        """累积一步观测。每次 agent 调用都应调用它。"""
        cur = obs.get("current")

        # 新局检测（必须放在最前）
        if self.is_new_game(obs):
            self.reset()

        if cur is None:
            return

        me_idx = int(cur.get("yourIndex", 0))
        op_idx = 1 - me_idx

        turn = int(cur.get("turn", 0))
        if turn != self.last_turn:
            self.last_turn = turn
            self.turns_seen += 1
        if turn > self.max_turn_seen:
            self.max_turn_seen = turn

        # ---- 累积增量日志
        for lg in (obs.get("logs") or []):
            self._absorb_log(lg, me_idx, op_idx)

        # ---- 快照型信息（全量，直接覆盖，适合做"变化量"）
        players = cur.get("players") or []
        if len(players) > op_idx:
            op = players[op_idx]
            hc = int(op.get("handCount", 0))
            if self._last_opp_hand is not None and hc != self._last_opp_hand:
                self.opp_hand_history.append(hc)
                del self.opp_hand_history[:-self.MAX_HAND_HISTORY]
            self._last_opp_hand = hc

            pc = len(op.get("prize") or [])
            if self._last_opp_prize is not None and pc < self._last_opp_prize:
                self.opp_prizes_taken += (self._last_opp_prize - pc)
            self._last_opp_prize = pc

        if len(players) > me_idx:
            pc = len(players[me_idx].get("prize") or [])
            if self._last_my_prize is not None and pc < self._last_my_prize:
                self.my_prizes_taken += (self._last_my_prize - pc)
            self._last_my_prize = pc

        self.result = int(cur.get("result", -1))
        if self.result != -1:
            self.finished = True

    def _absorb_log(self, lg: dict, me_idx: int, op_idx: int) -> None:
        t = int(lg.get("type", -1))
        who = lg.get("playerIndex")
        cid = lg.get("cardId")
        target = None
        if who == op_idx:
            target = self.opp_seen
        elif who == me_idx:
            target = self.my_seen

        # 只有带 cardId 的事件才泄露信息
        if cid is not None and t in LOG_WITH_CARD_ID and target is not None:
            target[int(cid)] += 1

        if who == op_idx:
            if t == LOG_MOVE_CARD and lg.get("fromArea") == AREA_DECK and lg.get("toArea") == AREA_HAND:
                self.opp_searches += 1
            elif t == LOG_ATTACH:
                self.opp_energy_attached += 1
            elif t == LOG_PLAY:
                self.opp_plays += 1
            elif t == LOG_MOVE_CARD and lg.get("toArea") == AREA_DISCARD:
                self.opp_discards += 1

    # ------------------------------------------------------------ 特征
    def features(self, me_idx: int = 0) -> dict[str, float]:
        """记忆派生的状态特征（全是公开信息，可安全进模型）。"""
        seen_total = sum(self.opp_seen.values())
        seen_unique = len(self.opp_seen)
        hist = self.opp_hand_history
        hand_now = self._last_opp_hand if self._last_opp_hand is not None else 0
        hand_prev = hist[-1] if hist else hand_now
        return {
            "mem_turns_seen": float(self.turns_seen),
            "mem_opp_seen_total": float(seen_total),
            "mem_opp_seen_unique": float(seen_unique),
            "mem_opp_seen_rate": float(seen_total) / max(self.turns_seen, 1),
            "mem_opp_searches": float(self.opp_searches),
            "mem_opp_energy_attached": float(self.opp_energy_attached),
            "mem_opp_plays": float(self.opp_plays),
            "mem_opp_discards": float(self.opp_discards),
            "mem_opp_hand_now": float(hand_now),
            "mem_opp_hand_delta": float(hand_now - hand_prev),
            "mem_opp_hand_max": float(max(hist) if hist else hand_now),
            "mem_opp_hand_samples": float(len(hist)),
            "mem_my_prizes_taken": float(self.my_prizes_taken),
            "mem_opp_prizes_taken": float(self.opp_prizes_taken),
            "mem_prize_race": float(self.my_prizes_taken - self.opp_prizes_taken),
        }

    def option_features(self, obs: dict, idx: int) -> dict[str, float]:
        """记忆派生的选项特征（例如"这张卡对手已经打过"）。"""
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        if idx >= len(opts):
            return {"mem_opt_opp_has_played": 0.0}
        o = opts[idx]
        cid = o.get("cardId")
        if cid is None and o.get("area") == AREA_HAND:
            cur = obs.get("current") or {}
            me_idx = int(cur.get("yourIndex", 0))
            players = cur.get("players") or []
            if len(players) > me_idx:
                hand = players[me_idx].get("hand") or []
                i = o.get("index")
                if i is not None and 0 <= i < len(hand):
                    cid = hand[i].get("id")
        return {"mem_opt_opp_has_played": float(self.opp_seen.get(cid, 0)) if cid else 0.0}

    # ------------------------------------------------------------ 未见池（给 filler 用）
    def unseen_pool(self, my_deck: list[int], obs: dict) -> tuple[list[int], list[int]]:
        """计算双方的"未见卡池"，供 determinization 采样。

        Returns:
            (我方未见池, 对手未见池)
            数量应恰好等于 (牌库+奖赏) 的合计（信息守恒）。
        """
        cur = obs.get("current") or {}
        me_idx = int(cur.get("yourIndex", 0))
        players = cur.get("players") or []
        if len(players) < 2:
            return list(my_deck), []

        me, op = players[me_idx], players[1 - me_idx]

        def visible(p: dict) -> Counter:
            c: Counter = Counter()
            for card in (p.get("discard") or []):
                c[card.get("id")] += 1
            for pk in list(p.get("active") or []) + list(p.get("bench") or []):
                if not pk:
                    continue
                c[pk.get("id")] += 1
                for k in ("energyCards", "tools", "preEvolution"):
                    for e in (pk.get(k) or []):
                        c[e.get("id")] += 1
            for card in (p.get("hand") or []):      # 只有自己的手牌可见
                c[card.get("id")] += 1
            return c

        pool_me = Counter(my_deck)
        for k, v in visible(me).items():
            pool_me[k] -= v

        # 对手卡组未知：用"我方卡组已见的部分"作为对手的先验（自对弈场景），
        # 实际比赛应替换成对手卡组推断模型的输出。
        pool_op = Counter(self.opp_seen)
        for k, v in self.opp_seen.items():
            pool_op[k] += 0
        for k, v in visible(op).items():
            pool_op[k] -= v

        unseen_me = [c for c, n in pool_me.items() for _ in range(max(n, 0))]
        unseen_op = [c for c, n in pool_op.items() for _ in range(max(n, 0))]
        return unseen_me, unseen_op


def memory_wrapper(inner_agent, start_turn: int = 0):
    """把一个普通 agent 包成带记忆的 agent。

    这样现有 agent（main.py / rulebase）不用改动就能享受累積的对手信息。
    """
    mem = GameMemory()

    def agent(obs_dict):
        nonlocal mem
        mem.observe(obs_dict)
        return inner_agent(obs_dict)

    agent.memory = mem          # 便于测试与调试
    agent.inner = inner_agent
    return agent
