#!/usr/bin/env python3
"""配置扫描：把规则策略的超参跑一遍，用实测胜率选最优。

不靠直觉调参 —— 每个配置跑固定局数，胜率才算数。
"""
from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "agents"))

from local_match import play_match, load_deck  # noqa: E402
from rulebase import make_agent, ORDERS  # noqa: E402


def load(path):
    spec = importlib.util.spec_from_file_location("m_" + os.path.basename(path)[:-3], path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.agent


def evaluate(A, B, deck, games):
    wins = losses = draws = fails = 0
    for g in range(games):
        if g % 2 == 0:
            r, a_idx = play_match(A, B, deck, deck), 0
        else:
            r, a_idx = play_match(B, A, deck, deck), 1
        if r["result"] is None:
            fails += 1
        elif r["result"] == 2:
            draws += 1
        elif r["result"] == a_idx:
            wins += 1
        else:
            losses += 1
    played = wins + losses + draws
    return (wins / played if played else 0.0), wins, losses, draws, fails


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    deck = load_deck()
    R = load("agents/random.py")
    G = load("agents/greedy.py")

    print("对手1 = random, 对手2 = greedy(develop_first)")
    print(f"每个配置 {games} 局（座位交替）\n")
    print(f"{'order':>16} {'go_first':>9} {'bench_full':>11} {'vs random':>10} {'vs greedy':>10}")
    print("-" * 62)
    rows = []
    for order in ORDERS:
        for go_first in (True, False):
            for bench_full in (True, False):
                A = make_agent(order=order, go_first=go_first, bench_full=bench_full)
                wr_r = evaluate(A, R, deck, games)
                wr_g = evaluate(A, G, deck, games)
                rows.append((order, go_first, bench_full, wr_r[0], wr_g[0]))
                print(f"{order:>16} {str(go_first):>9} {str(bench_full):>11} "
                      f"{wr_r[0]:>10.3f} {wr_g[0]:>10.3f}")
    print()
    rows.sort(key=lambda r: -(r[3]))
    print("按 vs random 胜率排序（前 6）：")
    for r in rows[:6]:
        print(f"  {r[0]:>16} go_first={str(r[1]):>5} bench_full={str(r[2]):>5}  "
              f"vs_random={r[3]:.3f}  vs_greedy={r[4]:.3f}")


if __name__ == "__main__":
    main()
