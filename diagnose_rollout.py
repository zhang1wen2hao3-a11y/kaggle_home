#!/usr/bin/env python3
"""离线诊断：一步 rollout 改进到底有没有价值？

## 为什么不直接打对局

端到端对局（rollout_agent vs greedy）的信号被大得多的方差掩盖：
n=200 读 0.5200、对照读 0.5250，两者都在噪声内。要分辨 +0.03 需要约 3400 局。
而"一步改进是否为正"可以用少得多的算力直接测量。

## 做法：world 级训练/测试切分

对每个决策点，采 W 个 determinization（world），然后：

    A 组（前 W/2 个）  ->  用来【选】动作
    B 组（后 W/2 个）  ->  用来【评】这个选择

    B 上比较：选中动作的胜率  -  greedy 动作的胜率  =  一步改进

这排除了"用同一批样本既选又评"的自欺，是策略改进的标准检验。

## 输出

- 不同 margin 阈值下的一步改进（均值 ± 标准误）
- 偏离率
- 若 margin 很大时改进仍显著为正 => 说明信号真实存在，端到端没体现只是被对局方差稀释
- 若各 margin 下改进都≈0 => 这条路的评估器本身没有分辨力，应停止投入

用法：
    ./run.sh diagnose_rollout.py --decisions 300 --worlds 20
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "agents"))

from cg.api import OptionType, SelectType  # noqa: E402
from cg.game import battle_start, battle_select  # noqa: E402
from framework.filler import make_filler  # noqa: E402
from framework.memory import GameMemory  # noqa: E402
from framework.rollout import (  # noqa: E402
    evaluate_actions, has_type, index_of_type, is_main, rule_action,
)
from local_match import load_deck  # noqa: E402


def candidates_of(sel):
    """候选动作下标：greedy 的选择 + ATTACK(若有) + END(若有)。"""
    base = rule_action(sel)[0]
    cands = [base]
    for want in (OptionType.ATTACK, OptionType.END):
        i = index_of_type(sel, want)
        if i is not None and i not in cands:
            cands.append(i)
    return base, cands


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, sd / math.sqrt(n)


def pick(base, cands, halves, margin):
    """用 A 组数据选动作。halves[a] = A 组结果列表。"""
    base_vals = halves[base]
    best_a, best_z = None, -1e9
    for a in cands:
        if a == base:
            continue
        d = [x - y for x, y in zip(halves[a], base_vals)
             if x is not None and y is not None]
        if len(d) < 3:
            continue
        m, se = mean_se(d)
        z = (m / se) if se and se > 0 else (float("inf") if m > 0 else 0.0)
        if z > best_z:
            best_a, best_z = a, z
    if best_a is not None and best_z >= margin:
        return best_a
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", type=int, default=300, help="采样多少个决策点")
    ap.add_argument("--worlds", type=int, default=20, help="（旧参数，等价于 select+eval）")
    ap.add_argument("--select-worlds", default="10,20",
                    help="用于【选动作】的 world 数，逗号分隔可一次测多个设置")
    ap.add_argument("--eval-worlds", type=int, default=40,
                    help="用于【评动作】的 world 数。加大它能显著降低测量噪声")
    ap.add_argument("--deck", default="decks/sample_official.csv")
    ap.add_argument("--deck-opponent", default=None, help="对手卡组（默认同 --deck，即镜像）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margins", default="0,0.5,1.0,1.5,2.0")
    args = ap.parse_args()

    margins = [float(x) for x in args.margins.split(",")]
    dA = load_deck(args.deck)
    dB = load_deck(args.deck_opponent or args.deck)
    filler = make_filler("uniform", my_deck=dA, seed=args.seed)

    # ---- 采集决策点：greedy 自对弈（镜像，和端到端评估一致）
    print(f"采集决策点：{args.deck} vs {args.deck_opponent or args.deck}（镜像）")
    samples = []
    game = 0
    t0 = time.time()
    while len(samples) < args.decisions and game < args.decisions * 4:
        reverse = bool(game % 2)
        obs = battle_start(dA, dB, reverse_player=reverse)[0]
        mem = GameMemory()
        for _ in range(400):
            cur, sel = obs.get("current"), obs.get("select")
            if cur is None or sel is None or cur.get("result", -1) != -1:
                break
            mem.observe(obs)
            if is_main(sel) and has_type(sel, OptionType.ATTACK) and len(sel["option"]) >= 2:
                samples.append((obs, game))
                if len(samples) >= args.decisions:
                    break
            obs = battle_select(rule_action(sel))
        game += 1
    print(f"  采样 {len(samples)} 个决策点，来自 {game} 局，用时 {time.time()-t0:.0f}s\n")

    # ---- 每个决策点做一次 A/B 切分评估
    # 关键：选动作的 world 数与评动作的 world 数分开。
    # 评的 world 越多，一步改进的测量噪声越小（旧版只有 10 个，精度 0.1，估计重尾不稳定）。
    sel_ws = [int(x) for x in str(args.select_worlds).split(",") if x.strip()]
    ev_w = args.eval_worlds
    total_w = max(sel_ws) + ev_w
    print(f"评估：每决策点 {total_w} world = 选动作 {sel_ws} + 评动作 {ev_w}")
    details = []
    t0 = time.time()
    for k, (obs, gid) in enumerate(samples):
        sel = obs["select"]
        base, cands = candidates_of(sel)
        type_map = {a: OptionType(sel["option"][a]["type"]).name for a in cands}
        out, info = evaluate_actions(obs, cands, filler=filler, my_deck=dA,
                                     n_worlds=total_w)
        Aset = {sw: {a: out[a][:sw] for a in cands} for sw in sel_ws}
        B = {a: out[a][max(sel_ws): max(sel_ws) + ev_w] for a in cands}
        details.append((base, cands, Aset, B, type_map))
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(samples)}  ({time.time()-t0:.0f}s)")
    print(f"  完成，用时 {time.time()-t0:.0f}s\n")

    # ---- 按 margin 汇总
    print("=" * 74)
    print(f"{'选worlds':>9} {'margin':>7} {'偏离率':>8} {'一步改进':>10} {'标准误':>8} {'z':>7}")
    print("-" * 74)
    best = None
    for sw in sel_ws:
      for margin in margins:
        diffs = []
        switched = 0
        targets = {}
        for base, cands, Aset, B, type_map in details:
            A = Aset[sw]
            chosen = pick(base, cands, A, margin)
            if chosen != base:
                switched += 1
                tn = type_map.get(chosen, f"idx{chosen}")
                targets[tn] = targets.get(tn, 0) + 1
            vb = [x for x in B[chosen] if x is not None]
            vg = [x for x in B[base] if x is not None]
            if vb and vg:
                diffs.append(sum(vb) / len(vb) - sum(vg) / len(vg))
        m, se = mean_se(diffs)
        z = m / se if se and se > 0 else 0.0
        star = " ★" if (se and abs(z) >= 2) else ""
        print(f"{sw:9d} {margin:7.1f} {switched/max(len(details),1):7.1%} {m:+10.4f} "
              f"{(se if se==se else 0):8.4f} {z:+7.2f}{star}")
        if best is None or m > best[1]:
            best = (margin, m, se, switched, dict(targets), sw)
    print("=" * 74)

    # ---- 偏离目标分析：换到 END 是危险信号，换到 ATTACK 才合理
    if best and best[3] > 0:
        margin, m, se, switched, targets, sw = best
        print(f"\n最优 选worlds={sw} margin={margin} 的偏离去向（共 {switched} 次）：")
        for tn, cnt in sorted(targets.items(), key=lambda kv: -kv[1]):
            print(f"   换到 {tn}: {cnt} 次 ({cnt/switched:.0%})")
        if targets.get("END", 0) / max(switched, 1) > 0.6:
            print("   ⚠ 多数偏离是换到 END —— 需要警惕'提前结束回合'型误判")
        else:
            print("   ✅ 偏离去向不是以 END 为主，行为合理")
    print("\n解读：")
    print("  改进 > 0 且 z >= 2  => 一步改进真实存在，端到端没体现只是被对局方差稀释")
    print("  改进 ≈ 0            => rollout 评估器没有分辨力，这条路应停止投入")
    print("  改进 < 0 且 z <= -2 => 偏离有害，安全阀是必要的")


if __name__ == "__main__":
    main()
