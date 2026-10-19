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
    """池子不够时补齐（**已不再用于常规路径**，保留仅为向后兼容）。"""
    if len(pool) >= need:
        return pool[:need]
    return pool + [fallback_id] * (need - len(pool))


def _normalize(pool: list[int], need: int, deck_hint: list[int], rng):
    """把未见池规整到恰好 need 张，并返回实际补齐的张数。

    - **多了**（len > need）：说明有几张卡落在我没建模的区域里
      （实测最常见是 setup 期间多 1 张）。随机剔除多余的，而不是截断尾部 ——
      尾部是按 Counter 插入顺序排的，截断会系统性地偏向某些卡。
    - **少了**（len < need）：说明扣多了。按卡组自身的构成比例补齐，至少保持 archetype。

    ⚠️ 历史 bug（已修）：旧版这里直接 `pool + [fallback_id] * short`，
       而 fallback 取的是 `deck[0]`（基础水能量）。一旦上传 `GameMemory`，
       `my_seen` 这个"日志事件计数"会把同一张卡重复扣很多次，导致池子
       缺 27 张（中位 29），于是 37 张基础能量被塞进牌库 ——
       实测 99.5% 的决策点都发生了这种污染。
    """
    if len(pool) == need:
        return pool, 0
    if len(pool) > need:
        rng.shuffle(pool)
        return pool[:need], 0
    short = need - len(pool)
    src = deck_hint if deck_hint else pool
    if not src:
        return pool + [0] * short, short
    # 按卡组构成循环取样补齐（允许重复：牌库本来就允许同名多张）
    extra = [src[(i * 7919) % len(src)] for i in range(short)]
    return pool + extra, short



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
        # 诊断计数：线上/离线都应该看到 short_me / short_op 接近 0。
        # 若它们很大，说明未见池算错了（历史上曾因误用 GameMemory 而大面积触发）。
        self.stats = {"calls": 0, "short_me": 0, "short_op": 0}

    def fill(self, obs: dict, memory, my_deck: list[int] | None = None) -> dict | None:
        c = _counts(obs)
        if c is None:
            return None
        deck = list(my_deck or self.my_deck)

        # ---- 我方：牌库+奖赏的合并内容我完全知道，只是不知道怎么分
        pool_me = self._my_unseen(obs, c, deck, memory)
        need_me = c["my_deck"] + c["my_prize"]
        pool_me, short_me = _normalize(pool_me, need_me, deck, self.rng)
        self.stats["short_me"] += short_me
        self.stats["calls"] += 1
        self.rng.shuffle(pool_me)
        my_d = pool_me[:c["my_deck"]]
        my_p = pool_me[c["my_deck"]:need_me]

        # ---- 对手：卡组构成**未知**，用先验卡组 − 已见卡 得到候选池
        pool_op = self._op_unseen(obs, c, memory)
        need_op = c["op_hand"] + c["op_deck"] + c["op_prize"]
        pool_op, short_op = _normalize(pool_op, need_op, self.opp_prior or deck, self.rng)
        self.stats["short_op"] += short_op
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
        """我方未见池 = 我的卡组 − 当前可见（手牌/弃牌/场面及附着）。

        ⚠️ **不能**用 memory.my_seen 再扣一次。它统计的是"日志事件次数"（同一张卡
           抽到/打出/附着会被反复记录），拿它当"还剩几张"会把池子扣到只剩十几张。
           实测传入 GameMemory 后 99.5% 的决策点池子不足（中位缺 29 张），
           于是 37 张基础能量被塞进牌库，determinization 完全失真。
        """
        _ = memory          # 保留参数签名，但不参与扣减
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
        return [cid for cid, n in pool.items() for _ in range(max(n, 0))]

    def _op_unseen(self, obs: dict, c: dict, memory) -> list[int]:
        """对手未见池 = 先验卡组 − 对手当前可见卡（弃牌/场面及附着）。

        ⚠️ 与 _my_unseen 同理：**不能**再用 memory.opp_seen 扣一次
           （那是日志事件计数，实测会让 97.4% 的决策点池子不足）。
        """
        _ = memory
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


class CompositionOracleFiller(OracleFiller):
    """用真实隐藏信息，但**打乱区内的具体分配** —— 模拟"对手推断"能拿到的上限。

    为什么 `OracleFiller` 不够：

    | 信息 | 对手推断能不能改善 | `OracleFiller` | 本类 |
    |---|---|---|---|
    | 各区数量（手牌几张、牌库几张） | 不能，本来就是公开的 | 真实 | 真实 |
    | 每区的**多重集合**（构筑猜得准不准） | ✅ 能 | 真实 | **真实** |
    | 区内的**具体分配 / 抽牌顺序** | ❌ 不能（那等于开图） | 真实 | **随机** |

    `OracleFiller` 连顺序都对 ⇒ 10 个"世界"其实是**同一个**世界 ⇒ 方差为 0 ⇒
    `_stats` 里 sd=0、se=0，z 要么 ±inf 要么 0，**配对检验直接失效**。
    实测（4 局 18 个决策）：15 个决策的 z 恰好 0.00，只有 1 个偏离规则 ——
    那不是"完美信息没用"，那是"10 次采样采到了同一个样本"。

    本类保留真实的多重集合与各区张数，只在区内做无放回随机重分配，于是每个 world
    都是**分布正确、顺序随机**的独立样本：既没有开图偏差，又保住了 CRN 需要的方差。
    **这才是对手推断的现实天花板** —— 推断能猜的是构筑，猜不到抽牌顺序。
    """

    name = "composition_oracle"

    def __init__(self, seed: int = 0):
        import random
        self.rng = random.Random(seed)

    def fill(self, obs: dict, memory, my_deck: list[int] | None = None) -> dict | None:
        world = super().fill(obs, memory, my_deck)
        if not world:
            return None
        rng = self.rng

        def redeal(keys):
            """若干区的卡合并后无放回随机重分配，各区张数保持不变。"""
            pool = []
            for k in keys:
                pool.extend(world[k])
            rng.shuffle(pool)
            pos, out = 0, {}
            for k in keys:
                n = len(world[k])
                out[k] = pool[pos:pos + n]
                pos += n
            world.update(out)

        # 我方：牌库 + 奖赏（构筑我知道，只是不知道顺序）
        redeal(["your_deck", "your_prize"])
        # 对方：牌库 + 奖赏 + 手牌（对手推断要猜的就是这个多重集合）
        redeal(["opponent_deck", "opponent_prize", "opponent_hand"])
        # opponent_active 是场上的里侧怪，不动 —— 它是位置，不是抽牌堆
        return world


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
    if kind == "composition_oracle":
        # 对手推断的天花板：构筑正确、顺序随机（详见 CompositionOracleFiller）
        return CompositionOracleFiller(**kw)
    if kind == "model":
        return ModelFiller(**kw)
    raise ValueError(f"未知 filler 类型: {kind}")
