"""P1 守栏版：在移植包外面套一层**反撤退**覆盖。

规则（只有一条，来自真实对局审计）：如果策略想 **RETREAT**，而我方出战宝可梦此刻
**付得起一个"无特效"的伤害招式**，就把动作改成伤害最高的那个攻击。

依据（`epsoide_port/` 78 局 / 13,116 决策，EXP-0118）：
  * `RETREAT_WHILE_ATTACK` 命中 **177 次，涉及 67/78 局（≈2.6 次/局）**
  * 败局发生率是胜局的 **1.52 倍**（16.5‰ vs 10.8‰）
  * 33 场输局**全部**是"被打完 6 张奖赏"——输在伤害竞速，而撤退会直接放弃本回合的攻击

为什么**不再**要求"无特效招式"：实测这副卡组（Ogerpon）的招式**全部带特效文本**
（`attackId 120 → 伤害 30 / 需 3 能量 / 带文本`），按"只比无特效招式"过滤会让守栏**完全不触发**
（6 局 820 决策 hits=0）。改成"只要存在付得起且伤害>0 的招式就覆盖"。

为什么只覆盖 RETREAT：其它审计项（END_WHILE_ATTACK 0.80×、MISSED_EVOLVE 0.80×、
LOW_DAMAGE_ATTACK 0 次）要么与败局负相关、要么根本不存在。

⚠️ 这是**待验证假设**，不是结论：验收判据是 Bo3 镜像 n=60 得分率 ≥0.55 且 CI 下界 >0.45。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 他们的原始入口（AGENT_DIR 会经 cwd / __file__ / /kaggle_simulations/agent 解析到本目录）
import port_main as _pm  # noqa: E402

_ORIG = _pm.agent

RETREAT = 12
ATTACK = 13

_ATK = {}
try:
    from cg.api import all_attack
    for a in all_attack():
        cost = [int(x) for x in (getattr(a, "energies", None) or [])]
        plain = not (getattr(a, "text", "") or "").strip()
        _ATK[int(a.attackId)] = (int(getattr(a, "damage", 0) or 0), cost, plain)
except Exception:
    _ATK = {}

GUARD = {"hits": 0, "calls": 0}


def _can_pay(have, cost) -> bool:
    """按**属性**判能否支付（无色 0 可由任意能量支付）。"""
    pool = {}
    for t in have or []:
        try:
            t = int(t)
        except Exception:
            continue
        pool[t] = pool.get(t, 0) + 1
    need, colorless = {}, 0
    for t in cost or []:
        t = int(t)
        if t == 0:
            colorless += 1
        else:
            need[t] = need.get(t, 0) + 1
    for t, n in need.items():
        if pool.get(t, 0) < n:
            return False
        pool[t] -= n
    return sum(pool.values()) >= colorless


def _energy_types_of(pk):
    out = []
    for e in (pk.get("energies") or pk.get("energy") or []):
        if isinstance(e, (list, tuple)):
            if len(e) >= 2:
                out += [int(e[0])] * int(e[1])
            elif e:
                out.append(int(e[0]))
        else:
            try:
                out.append(int(e))
            except Exception:
                pass
    return out


def agent(obs, config=None):
    act = _ORIG(obs)
    GUARD["calls"] += 1
    try:
        sel = (obs or {}).get("select")
        if sel is None or not act:
            return act
        opts = sel.get("option") or []
        if not (0 <= act[0] < len(opts)):
            return act
        if int(opts[act[0]].get("type", -1)) != RETREAT:
            return act
        cur = obs.get("current") or {}
        players = cur.get("players") or []
        j = int(cur.get("yourIndex", 0))
        me = players[j] if len(players) > j else {}
        active = next((p for p in (me.get("active") or []) if p), None)
        if not active:
            return act
        have = _energy_types_of(active)
        best_i, best_d = None, 0
        for i, op in enumerate(opts):
            if int(op.get("type", -1)) != ATTACK:
                continue
            info = _ATK.get(int(op.get("attackId", -1)))
            if not info:
                continue
            dmg, cost, plain = info
            # 不要求无特效：本卡组所有招式都带文本；伤害>0 且付得起即可
            if dmg > best_d and _can_pay(have, cost):
                best_i, best_d = i, dmg
        if best_i is None:
            return act
        GUARD["hits"] += 1
        return [best_i]
    except Exception:
        return act
