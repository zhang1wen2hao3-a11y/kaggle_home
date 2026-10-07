"""determinization 填充器 —— 为 search_begin 提供被抹掉的隐藏信息。

背景（读源码得出的机制）：
    引擎在序列化 search_begin_input 前会调用 erasePlayerData()，
    把隐藏卡清成 cardId = 0（见 State.h:302 / Api.h:105）。
    Search::start 再专门找这些空槽位，用**你传入的预测值**填回去（Search.h:109）。

    被抹掉的：我方牌库+奖赏、对手牌库+手牌+奖赏、里侧 Active。
    保留的  ：弃牌堆、场面、我方手牌、各项数量。

所以：**没有填充器就无法搜索**。这不是可选组件。

三种实现：
    UniformFiller  从"未见池"均匀采样 —— 零成本基线，先跑通闭环
    OracleFiller   用 visualize_data() 的真实隐藏信息 —— **仅本地可用**，
                   用于量化"预测不准的代价"（信息损失成本）
    ModelFiller    用训练出来的对手推断模型 —— 留的口子
"""
from __future__ import annotations

import random
from collections import Counter


class Filler:
    """填充器接口。后续所有实现都遵循它。"""

    name = "base"

    def fill(self, obs: dict, memory, my_deck: list[int]) -> dict | None:
        """产出 search_begin 所需的全部参数。

        Returns:
            dict，键为 your_deck / your_prize / opponent_deck /
            opponent_prize / opponent_hand / opponent_active；
            返回 None 表示无法填充（调用方应回退到不搜索）。
        """
        raise NotImplementedError


def _counts(obs: dict) -> dict | None:
    """从 obs 读出各隐藏区域需要的张数（这些数量是公开信息）。"""
    cur = obs.get("current")
    if not cur:
        return None
    me_idx = int(cur.get("yourIndex", 0))
    players = cur.get("players") or []
    if len(players) < 2:
        return None
    me, op = players[me_idx], players[1 - me_idx]
    active = op.get("active") or []
    return {
        "me_idx": me_idx,
        "my_deck": int(me.get("deckCount", 0)),
        "my_prize": len(me.get("prize") or []),
        "op_deck": int(op.get("deckCount", 0)),
        "op_prize": len(op.get("prize") or []),
        "op_hand": int(op.get("handCount", 0)),
        # 只有对手 Active 是里侧时才需要预测
        "op_active": 1 if (active and active[0] is None) else 0,
    }


def _pad(pool: list[int], need: int, fallback_id: int) -> list[int]:
    """池子不够时补齐（正常情况不该发生；补上宁可让搜索退化也不要崩）。"""
    if len(pool) >= need:
        return pool[:need]
    return pool + [fallback_id] * (need - len(pool))


class UniformFiller(Filler):
    """从"未见池"均匀随机分配 —— 零训练成本的基线。

    理论上不完美（没有利用对手的行为线索），但它给出了 determinstication
    的**下界**：任何模型都必须明显超过它才值得投入。
    """

    name = "uniform"

    def __init__(self, my_deck: list[int], opponent_deck_prior: list[int] | None = None,
                 seed: int | None = None):
        self.my_deck = list(my_deck)
        self.opp_prior = list(opponent_deck_prior) if opponent_deck_prior else list(my_deck)
        self.rng = random.Random(seed)

    def fill(self, obs: dict, memory, my_deck: list[int] | None = None) -> dict | None:
        c = _counts(obs)
        if c is None:
            return None
        deck = list(my_deck or self.my_deck)

        # ---- 我方：牌库+奖赏的合并内容我完全知道，只是不知道怎么分
        pool_me = self._my_unseen(obs, c, deck, memory)
        need_me = c["my_deck"] + c["my_prize"]
        pool_me = _pad(pool_me, need_me, deck[0] if deck else 0)
        self.rng.shuffle(pool_me)
        my_d = pool_me[:c["my_deck"]]
        my_p = pool_me[c["my_deck"]:need_me]

        # ---- 对手：卡组构成**未知**，用先验卡组 − 已见卡 得到候选池
        pool_op = self._op_unseen(obs, c, memory)
        need_op = c["op_hand"] + c["op_deck"] + c["op_prize"]
        fallback = self.opp_prior[0] if self.opp_prior else (deck[0] if deck else 0)
        pool_op = _pad(pool_op, need_op, fallback)
        self.rng.shuffle(pool_op)
        i = 0
        op_h = pool_op[i:i + c["op_hand"]]; i += c["op_hand"]
        op_d = pool_op[i:i + c["op_deck"]]; i += c["op_deck"]
        op_p = pool_op[i:i + c["op_prize"]]

        result = {
            "your_deck": my_d,
            "your_prize": my_p,
            "opponent_deck": op_d,
            "opponent_prize": op_p,
            "opponent_hand": op_h,
            "opponent_active": [],
        }
        if c["op_active"]:
            result["opponent_active"] = [self._guess_op_active(obs, c, memory)]
        return result

    def _my_unseen(self, obs: dict, c: dict, deck: list[int], memory) -> list[int]:
        """我方未见池 = 我的卡组 − 已见（手牌/弃牌/场面）+ 已累积的日志。"""
        pool = Counter(deck)
        cur = obs["current"]
        me = cur["players"][c["me_idx"]]

        def dec(card_id):
            if pool.get(card_id, 0) > 0:
                pool[card_id] -= 1

        for card in (me.get("hand") or []):
            dec(card.get("id"))
        for card in (me.get("discard") or []):
            dec(card.get("id"))
        for pk in list(me.get("active") or []) + list(me.get("bench") or []):
            if not pk:
                continue
            dec(pk.get("id"))
            for k in ("energyCards", "tools", "preEvolution"):
                for e in (pk.get(k) or []):
                    dec(e.get("id"))
        # 场上道具/能量有些不在上面，用日志里我方已见卡再兜一层
        if memory is not None:
            for card_id, n in memory.my_seen.items():
                for _ in range(n):
                    dec(card_id)
        return [cid for cid, n in pool.items() for _ in range(max(n, 0))]

    def _op_unseen(self, obs: dict, c: dict, memory) -> list[int]:
        """对手未见池 = 先验卡组 − 对手已见卡（弃牌/场面/日志）。"""
        pool = Counter(self.opp_prior)
        cur = obs["current"]
        op = cur["players"][1 - c["me_idx"]]

        def dec(card_id):
            if card_id is not None and pool.get(card_id, 0) > 0:
                pool[card_id] -= 1

        for card in (op.get("discard") or []):
            dec(card.get("id"))
        for pk in list(op.get("active") or []) + list(op.get("bench") or []):
            if not pk:
                continue
            dec(pk.get("id"))
            for k in ("energyCards", "tools", "preEvolution"):
                for e in (pk.get(k) or []):
                    dec(e.get("id"))
        if memory is not None:
            for card_id, n in memory.opp_seen.items():
                for _ in range(n):
                    dec(card_id)
        return [cid for cid, n in pool.items() for _ in range(max(n, 0))]

    def _guess_op_active(self, obs: dict, c: dict, memory) -> int:
        """对手 Active 里侧时的猜测：优先用先验卡组里的基础宝可梦。"""
        try:
            from cg.api import all_card_data
            basics = {x.cardId for x in all_card_data() if getattr(x, "basic", False)}
        except Exception:
            basics = set()
        cands = [x for x in self.opp_prior if x in basics]
        return self.rng.choice(cands) if cands else (self.opp_prior[0] if self.opp_prior else 0)


class OracleFiller(Filler):
    """用 visualize_data() 读取**真实**隐藏信息。

    ⚠️ 仅本地可用：它依赖 Battle.battle_ptr，而提交的 main.py 从不调用 battle_start()，
       所以线上跑不了 —— 不会成为作弊通道。
       但**是否允许用它生成训练数据，请自行核对比赛规则**。

    用途：量化"预测不准的代价"（信息损失成本）—— 这是判断对手推断值不值得投入的唯一硬指标。
    """

    name = "oracle"

    def fill(self, obs: dict, memory, my_deck: list[int] | None = None) -> dict | None:
        c = _counts(obs)
        if c is None:
            return None
        try:
            from cg.game import visualize_data
            import json
            replay = json.loads(visualize_data())
        except Exception:
            return None
        if not replay:
            return None

        # 从后往前找一条数量对得上的记录
        me_idx = c["me_idx"]
        for rec in reversed(replay):
            cur = rec.get("current")
            if not cur:
                continue
            players = cur.get("players") or []
            if len(players) < 2:
                continue
            me, op = players[me_idx], players[1 - me_idx]

            def ids(cards):
                return [x.get("id") for x in (cards or []) if x]

            md, mp = ids(me.get("deck")), ids(me.get("prize"))
            od, opp, oh = ids(op.get("deck")), ids(op.get("prize")), ids(op.get("hand"))
            if (len(md) == c["my_deck"] and len(mp) == c["my_prize"] and
                    len(od) == c["op_deck"] and len(opp) == c["op_prize"] and
                    len(oh) == c["op_hand"]):
                active = op.get("active") or []
                return {
                    "your_deck": md,
                    "your_prize": mp,
                    "opponent_deck": od,
                    "opponent_prize": opp,
                    "opponent_hand": oh,
                    "opponent_active": ([active[0].get("id")]
                                        if (active and active[0]) else []),
                }
        return None


class ModelFiller(Filler):
    """用训练出来的对手推断模型填充 —— 给后续工作留的口子。

    约定：model 需实现 `predict_hidden(obs, memory) -> dict`，返回
        {"opponent_deck": [...], "opponent_hand": [...], ...}
    未实现时自动回退到 UniformFiller，保证流水线不会因为模型缺失而断掉。
    """

    name = "model"

    def __init__(self, model=None, fallback: Filler | None = None):
        self.model = model
        self.fallback = fallback or UniformFiller(my_deck=[], seed=0)

    def fill(self, obs: dict, memory, my_deck: list[int] | None = None) -> dict | None:
        if self.model is not None and hasattr(self.model, "predict_hidden"):
            try:
                pred = self.model.predict_hidden(obs, memory)
                if pred:
                    return pred
            except Exception:
                pass
        if my_deck is not None and isinstance(self.fallback, UniformFiller):
            self.fallback.my_deck = list(my_deck)
            self.fallback.opp_prior = list(my_deck)
        return self.fallback.fill(obs, memory, my_deck)


def make_filler(kind: str = "uniform", my_deck=None, **kw) -> Filler:
    """工厂：按名字创建填充器（配置文件里换一行就能切换）。"""
    if kind == "uniform":
        return UniformFiller(my_deck=my_deck or [], **kw)
    if kind == "oracle":
        return OracleFiller()
    if kind == "model":
        return ModelFiller(**kw)
    raise ValueError(f"未知 filler 类型: {kind}")
