#!/usr/bin/env python3
"""真实回放定向行为审计：在我们自己的 317 局线上对局里找**可判定型错误**。

## 为什么做这个

历史上最大的两次线上收益都来自这类修复，而不是任何学习型模型：

| 修复 | 本地镜像胜率 | 说明 |
|---|---|---|
| `RETREAT(4)` 排在 `ATTACK(6)` 前面 | 0.2725 → **0.7612** | agent 宁可撤退也不攻击 |
| 多招式时取第一个 ATTACK 选项 | 0.37 → **0.98** | 常常选到 0 伤害的招式 |

所以与其继续折腾学不动的模型（见 `别再试.md` D1~D13），不如回到**能在真实对局里数出来的错**。
本工具的每条审计都必须满足两个条件：

1. **可判定**：不需要搜索、不需要预测对手，只看当时那个 observation 就能判"另一个选项必然不差"
2. **可数**：给出违规次数 / 涉及局数 / 胜局与败局里的分布，才能算杠杆（见 `别再试.md` E2 稀释数学）

## 审计项

| 代号 | 判据 |
|---|---|
| `END_WHILE_ATTACK` | 选了 END，而当场 Active **付得起**的招式伤害 > 0 —— 白送一个回合 |
| `RETREAT_WHILE_ATTACK` | 选了 RETREAT，而当场 Active 付得起的招式伤害 > 0 —— 撤退浪费攻击 |
| `LOW_DAMAGE_ATTACK` | 选了 ATTACK，但存在伤害更高且能量够的 ATTACK —— 打少的 |
| `WASTED_ATTACH` | 选了 ATTACH，但目标宝可梦**所有招式都已启用**（贴了也没用） |
| `MISSED_EVOLVE` | 场上有合法 EVOLVE 却选了别的动作 |
| `END_WHILE_PLAY` | 选了 END，但手上有可打出的牌（PLAY 选项存在） |
| `DEVIATE_FROM_RULE` | 选的动作不是规则会选的（rollout 偏离），再统计其中有多少同时命中上面几项 |

⚠️ 这些是**必要的**错误，不是充分条件：命中 ≠ 一定错，但**未命中也不代表对**。
所以输出既给计数，也给 `--examples` 抽样，人工确认后再决定是否改规则。

用法：
    ./run.sh tools/audit_decisions.py                      # 汇总表
    ./run.sh tools/audit_decisions.py --examples 3         # 每项抽 3 例明细
    ./run.sh tools/audit_decisions.py --dir epsoide --split-loss
"""
import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cg.api import OptionType, all_attack  # noqa: E402
from framework.features import _attack_info, _can_pay  # noqa: E402
from framework.rollout import _get, rule_action  # noqa: E402

OUR_NAME = "zhang1wen2hao3"

_ATTACK = {}          # attackId -> (damage, n_cost, cost_types, 是否无特效)
try:
    for _a in all_attack():
        _cost = list(getattr(_a, "energies", None) or [])
        _plain = not (getattr(_a, "text", "") or "").strip()
        _ATTACK[int(_a.attackId)] = (int(getattr(_a, "damage", 0) or 0),
                                     len(_cost), [int(x) for x in _cost], _plain)
except Exception:
    pass

AUDITS = ("END_WHILE_ATTACK", "RETREAT_WHILE_ATTACK", "LOW_DAMAGE_ATTACK",
          "WASTED_ATTACH", "MISSED_EVOLVE", "END_WHILE_PLAY", "DEVIATE_FROM_RULE")


def _active(player):
    for pk in (player.get("active") or []):
        if pk:
            return pk
    return None


def _energy_total(pk):
    if not pk:
        return 0
    n = 0
    for e in (pk.get("energy") or pk.get("energies") or []):
        n += 1 if not isinstance(e, list) else len(e)
    return n


def _energy_types(pk):
    """这只宝可梦身上的能量类型列表（引擎里 energy 可能是 [[type,n], …] 或 [type, …]）。"""
    out = []
    if not pk:
        return out
    for e in (pk.get("energy") or pk.get("energies") or []):
        if isinstance(e, (list, tuple)):
            if len(e) >= 2:
                out += [int(e[0])] * int(e[1])
            elif e:
                out.append(int(e[0]))
        else:
            out.append(int(e))
    return out


def _target_by_inplay(player, area, idx):
    if area == 4:
        return _active(player)
    if area == 5:
        bench = player.get("bench") or []
        if idx is not None and 0 <= idx < len(bench):
            return bench[idx]
    return None


def audit_one(obs, act_idx):
    """返回该决策命中的审计项集合 + 细节。"""
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    if not (0 <= act_idx < len(opts)):
        return set(), {}
    if len(opts) < 2:
        return set(), {"forced": True}      # 只有 1 个选项 = 被迫决策，没有"错误"可言
    cur = obs.get("current") or {}
    me_idx = int(cur.get("yourIndex", 0))
    players = cur.get("players") or []
    me = players[me_idx] if len(players) > me_idx else {}
    act = _active(me)
    info = _attack_info(act)
    chosen = opts[act_idx]
    ctype = int(chosen.get("type", -1))
    hits, detail = set(), {}

    att_opts = [(i, _ATTACK.get(int(o.get("attackId", -1)), (0, 9, [], False)))
                for i, o in enumerate(opts) if int(o.get("type", -1)) == int(OptionType.ATTACK)]
    have_types = _energy_types(act)
    # ⚠️ 必须用**类型精确**的可付性：只数能量个数会把「有 2 个草能量但招式要 2 水」判成付得起
    # ⚠️ 还必须只比**无特效**招式：0 伤害招式常常「抽 3 张 / 放伤害指示物」，不是劣选
    affordable = [(i, d) for i, (d, _c, ct, plain) in att_opts
                  if plain and d > 0 and _can_pay(have_types, ct)]
    best_aff = max((d for _, d in affordable), default=0)

    if ctype == int(OptionType.END) and info["dmg_now"] > 0:
        hits.add("END_WHILE_ATTACK")
        detail["available_dmg"] = info["dmg_now"]
    if ctype == int(OptionType.END) and any(
            int(o.get("type", -1)) == int(OptionType.PLAY) for o in opts):
        hits.add("END_WHILE_PLAY")
    if ctype == int(OptionType.RETREAT) and info["dmg_now"] > 0:
        hits.add("RETREAT_WHILE_ATTACK")
    if ctype == int(OptionType.ATTACK):
        d, _c, _ct, _plain = _ATTACK.get(int(chosen.get("attackId", -1)), (0, 9, [], True))
        if _plain and best_aff > d:
            hits.add("LOW_DAMAGE_ATTACK")
            detail.update(chosen_dmg=d, best_dmg=best_aff)
    if ctype == int(OptionType.ATTACH):
        tgt = _target_by_inplay(me, chosen.get("inPlayArea"), chosen.get("inPlayIndex"))
        ti = _attack_info(tgt)
        if tgt is not None and ti["min_cost"] > 0 and ti["gap"] == 0.0 \
                and ti["dmg_now"] >= ti["dmg_max"] > 0:
            hits.add("WASTED_ATTACH")           # 全部招式已启用，再贴纯浪费
    if ctype != int(OptionType.EVOLVE) and any(
            int(o.get("type", -1)) == int(OptionType.EVOLVE) for o in opts):
        hits.add("MISSED_EVOLVE")
    try:
        rb = rule_action(sel, obs)
        if rb and rb[0] != act_idx:
            hits.add("DEVIATE_FROM_RULE")
            detail["rule_pick"] = rb[0]
            detail["rule_type"] = int(opts[rb[0]].get("type", -1))
    except Exception:
        pass
    detail["chosen_type"] = ctype
    detail["n_opts"] = len(opts)
    detail["turn"] = int(cur.get("turn", 0) or 0)
    return hits, detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="epsoide")
    ap.add_argument("--examples", type=int, default=0)
    ap.add_argument("--split-loss", action="store_true", help="按胜局/败局分开统计")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(ROOT, args.dir, "*.json")))
    print(f"扫描 {len(files)} 局（{args.dir}）")

    cnt = collections.Counter()          # audit -> 次数
    cnt_rule = collections.Counter()     # audit 且 chosen == 规则选择（真规则 bug）
    cnt_dev = collections.Counter()      # audit 且 rollout 偏离了规则
    games = collections.defaultdict(set)
    cnt_win = collections.Counter(); cnt_loss = collections.Counter()
    dec_win = dec_loss = 0        # 胜局/败局里的决策总数（归一化用）
    total_main = total_dec = 0
    types = collections.Counter()
    examples = collections.defaultdict(list)
    outcomes = collections.Counter()

    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        names = [a.get("Name") or "" for a in ((d.get("info") or {}).get("Agents") or [])]
        ours = [i for i, n in enumerate(names) if OUR_NAME in n]
        if not ours:
            continue
        me = ours[0]
        rew = d.get("rewards") or []
        won = None
        if len(rew) > me:
            won = rew[me] > max([r for j, r in enumerate(rew) if j != me] or [0])
            outcomes["胜" if won else "负"] += 1
        for st in d.get("steps") or []:
            if me >= len(st):
                continue
            ent = st[me]
            obs = ent.get("observation") or {}
            act = ent.get("action")
            sel = obs.get("select")
            if sel is None or not act or not obs.get("current"):
                continue
            total_dec += 1
            if won is True:
                dec_win += 1
            elif won is False:
                dec_loss += 1
            try:
                hits, det = audit_one(obs, int(act[0]))
            except Exception:
                continue
            if det.get("forced"):
                cnt["_forced"] += 1
                continue
            if int(sel.get("type", 0)) == 1:      # MAIN
                total_main += 1
            if not hits:
                continue
            types[int(det.get("chosen_type", -1))] += 1
            dev = "DEVIATE_FROM_RULE" in hits
            for h in hits:
                cnt[h] += 1
                (cnt_dev if dev else cnt_rule)[h] += 1
                games[h].add(f)
                if won is True:
                    cnt_win[h] += 1
                elif won is False:
                    cnt_loss[h] += 1
                if args.examples and len(examples[h]) < args.examples:
                    examples[h].append((os.path.basename(f), det, won))

    print(f"\n我方决策 {total_dec} 个（其中 MAIN {total_main}，"
          f"被迫决策 {cnt['_forced']} 个已排除）；结果分布 {dict(outcomes)}")
    print(f"\n胜局决策 {dec_win} 个 / 败局决策 {dec_loss} 个（归一化分母）")
    print(f"\n{'审计项':<22}{'命中':>7}{'占比':>8}{'涉及局':>7}"
          f"{'规则自己犯':>10}{'rollout偏离':>11}"
          f"{'胜局‰':>7}{'败局‰':>7}{'倍数':>6}")
    for a in AUDITS:
        n = cnt[a]
        if not n:
            print(f"{a:<22}{0:>7}{'-':>8}{0:>7}{'-':>10}{'-':>11}{'-':>7}{'-':>7}")
            continue
        rw = 1000 * cnt_win[a] / max(dec_win, 1)
        rl = 1000 * cnt_loss[a] / max(dec_loss, 1)
        ratio = (rl / rw) if rw > 0 else float("inf")
        print(f"{a:<22}{n:>7}{n / max(total_dec, 1):>7.1%}{len(games[a]):>7}"
              f"{cnt_rule[a]:>10}{cnt_dev[a]:>11}"
              f"{rw:>7.1f}{rl:>7.1f}{ratio:>6.2f}")
    if args.split_loss and outcomes:
        print("\n注：胜/败局内的次数只说明「这个错误在赢的局里也照样犯」，"
              "要判断危害得看它是否集中在败局。")

    if args.examples:
        for a in AUDITS:
            if not examples[a]:
                continue
            print(f"\n=== {a} 样例 ===")
            for fn, det, won in examples[a]:
                print(f"  {fn}  turn={det.get('turn')}  选了 type={det.get('chosen_type')} "
                      f"（共 {det.get('n_opts')} 个选项）"
                      f"  规则想选 type={det.get('rule_type')}"
                      f"  结果={'胜' if won else '负'}")
                extra = {k: v for k, v in det.items()
                         if k not in ("chosen_type", "n_opts", "turn", "rule_type")}
                if extra:
                    print(f"      {extra}")


if __name__ == "__main__":
    main()
