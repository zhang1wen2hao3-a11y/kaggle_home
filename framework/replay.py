#!/usr/bin/env python3
"""回放解析的**唯一正确入口** —— 修的是一个让两份结论作废的静默错位。

## 坑：action 记在**下一步**，不是同一步

Kaggle 回放 `steps[t][j]` 里的 `action` **不是**对 `steps[t][j].observation` 的回答，
而是对 `steps[t-1][j].observation` 的回答（等价：`steps[t+1][j].action` 回答 `steps[t][j].observation`）。

三个独立证据（都在我们自己的回放上实测）：

1. **交卡组那一步**：`steps[0][j]` 是 `select=None`（交卡组）且 `action=[]`；
   `steps[1][j].action` 的长度是 **60** —— 那才是卡表。30/30 局一致。
2. **`IS_FIRST` 不变量**：我们的 agent 恒答"先手"（`OptionType.YES=1`）。
   按"下一步"对齐：395/395 全是 type=1，**没有一次越界**；
   按"同一步"对齐：208 次 type=1，另有 **158 次索引越界**、160 次无 action。
3. **host 契约合法率**（索引在界、数量 minCount..maxCount、无重复）：
   `epsoide_teams2` 同步配对 73.2% → 下一步配对 88.9%。

**代价**：`train_bc.py`（EXP-0089~0098）和 `tools/audit_decisions.py`（EXP-0099/0100）
原先都用同步配对，**标签整体错位一位**，那些结论必须重测。公开 notebook（motono0223 的
tutorial-1）也独立记录了同一现象："同步配对仅 65% 满足 host 契约，下一步配对 99.4%"。

## 附带好处

`steps[1]` 里的 60 长 action 就是**双方完整的 60 张卡表** —— 于是每份回放都能产出两条精确卡表，
这是做卡组 meta 之前拿不到的数据（之前用"可见牌"重建，只有 20~27 张）。
"""
from __future__ import annotations


def our_index(replay: dict, name: str = "zhang1wen2hao3") -> int | None:
    """按 Agents[].Name 找我们的座位（不能假设是 player0：对手常常也是同一份提交）。"""
    names = [a.get("Name") or "" for a in ((replay.get("info") or {}).get("Agents") or [])]
    for i, n in enumerate(names):
        if name in n:
            return i
    return None


def iter_decisions(steps: list):
    """按**正确对齐**产出 (t, j, obs, action)。

    ## 回放格式（实测解码，三条证据互洽）

    同一座位的一个决策，其 observation 会在**连续两步**各出现一次：
    第 t 步 action 为空（决策刚下发），第 t+1 步 obs 相同、action 已填（决策被消费）。
    等待对手期间，该座位的 obs 会被反复投递且 action 一直为空 —— 那些**不是**决策。

    所以正确的配对是：**`obs@(t-1) ↔ action@t`**（同座位、action 非空）。

    验证（317 局我们自己的对局）：
      * host 契约合法率（数量/无重复/索引在界）**100.00%**（17,632 个决策）
      * `IS_FIRST` 不变量：395/395 全选"先手"（我们的 agent 恒答先手）
      * 平均每局 55.6 个决策、其中 5,362 个 MAIN
    反例（旧代码用的"同一步"配对）：合法率仅 86.3%，且 `IS_FIRST` 有 158 次索引越界。
    """
    n = len(steps or [])
    for t in range(1, n):
        cur, prev = steps[t] or [], steps[t - 1] or []
        for j, ent in enumerate(cur):
            if not ent or j >= len(prev):
                continue
            act = ent.get("action")
            if not act:
                continue
            obs = (prev[j] or {}).get("observation") or {}
            if obs.get("select") is None:
                continue
            yield t - 1, j, obs, act


def decks_of(steps: list):
    """从回放里取双方**精确的 60 张卡表**（对方交卡组那一步的 action）。

    返回 {座位: [60 个 cardId]}；缺失的座位不出现。
    """
    out: dict[int, list] = {}
    for t in range(len(steps or []) - 1):
        for j, ent in enumerate(steps[t] or []):
            if not ent or j >= len(steps[t + 1] or []):
                continue
            sel = (ent.get("observation") or {}).get("select")
            if sel is not None:
                continue
            act = (steps[t + 1][j] or {}).get("action")
            if act and len(act) == 60 and all(isinstance(x, int) for x in act):
                out[j] = list(act)
        if len(out) >= 2:
            break
    return out


def legal(sel: dict, act) -> bool | None:
    """host 契约：数量在 [minCount, maxCount]、无重复、索引在界。None = 无法判定。"""
    if not act:
        return None
    opts = sel.get("option") or []
    if not opts:
        return None
    lo, hi = int(sel.get("minCount") or 0), int(sel.get("maxCount") or 0)
    if len(set(act)) != len(act):
        return False
    if not (lo <= len(act) <= hi):
        return False
    return all(isinstance(i, int) and 0 <= i < len(opts) for i in act)
