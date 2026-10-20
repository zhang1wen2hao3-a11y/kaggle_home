#!/usr/bin/env python3
"""算力成本测量：单局耗时、时间预算有没有被打穿、还剩多少余量。

为什么需要这个工具：

`main_rollout.py` 有一个单局时间预算 `PTCG_BUDGET_S`（默认 20 s）。超预算后
本局剩余决策**全部退化成纯规则** —— 这是安全阀，但它是静默的，
所以必须能测出来有没有打穿。

实测背景（`sample_official`，2026-10-19）：
    单局 0.7~1.8 s，20 s 的预算**从未触发**
    线上 `actTimeout=0` / `runTimeout=6000`，所以算力余量有 3 个数量级
    —— "加深搜索" 因此从不可能变成可能

用法：
    ./run.sh tools/measure_cost.py agents/ro_attack_w10.py
    ./run.sh tools/measure_cost.py agents/ro_all_w40.py --games 6
    ./run.sh tools/measure_cost.py agents/ro_deck.py --deck decks/water_kyurem.csv
    PTCG_RO_DECK=decks/x.csv ./run.sh tools/measure_cost.py agents/ro_deck.py --deck decks/x.csv

输出里的"投影"一栏是按 runTimeout=6000 s 折算还能把算力放大多少倍。
⚠️ 线上机器比本地慢多少未知，所以投影只是量级参考。
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluate import load_agent                       # noqa: E402
from local_match import play_match, load_deck          # noqa: E402

RUN_TIMEOUT = 6000.0        # 线上 runTimeout（秒），来自 cabt.json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", nargs="?", default="agents/ro_attack_w10.py")
    ap.add_argument("--opponent", default="agents/greedy.py")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--deck", default="decks/sample_official.csv")
    ap.add_argument("--deck-opponent", default=None)
    args = ap.parse_args()

    A = load_agent(args.agent)
    B = load_agent(args.opponent)
    deck = load_deck(args.deck)
    opp_deck = load_deck(args.deck_opponent) if args.deck_opponent else deck

    g = getattr(A, "__globals__", {}) or {}
    budget = g.get("BUDGET_S")
    tripped_any = False

    print("=" * 72)
    print(f"算力成本测量：{args.agent}")
    print("=" * 72)
    print(f"  单局时间预算 PTCG_BUDGET_S : {budget if budget is not None else '（非 rollout agent）'}")
    print(f"  线上 runTimeout            : {RUN_TIMEOUT:.0f} s")
    print(f"  卡组                       : {args.deck}  vs  {args.deck_opponent or args.deck}")
    print("-" * 72)

    walls, trips = [], 0
    for i in range(args.games):
        # agent.reset() 会被 play_match 调用；这里额外确认预算标记的状态
        t0 = time.monotonic()
        r = play_match(A if i % 2 == 0 else B, B if i % 2 == 0 else A,
                       deck, opp_deck, reverse=bool(i % 2))
        dt = time.monotonic() - t0
        walls.append(dt)
        over = (g.get("_OVER_BUDGET") or [False])[0]
        if over:
            trips += 1
            tripped_any = True
        if r.get("error"):
            print(f"  game {i}: 异常 -> {r['error']}")
        print(f"  game {i}: {dt:6.2f}s   预算触发={'是' if over else '否'}")

    n = max(len(walls), 1)
    mean = sum(walls) / n
    worst = max(walls)
    print("-" * 72)
    print(f"  平均 {mean:.2f}s/局   最长 {worst:.2f}s   总 {sum(walls):.1f}s / {len(walls)} 局")
    if budget:
        print(f"  预算占用（按最长局）: {worst / budget:.1%} of {budget:.0f}s")
    print(f"  相对 runTimeout 余量 : {RUN_TIMEOUT / max(worst, 1e-6):,.0f}×")

    print("\n" + "=" * 72)
    if tripped_any:
        print(f"⚠️  有 {trips} 局的时间预算被打穿 —— 那些局的剩余决策已退化成纯规则。")
        print("    对线上提交来说，这意味着**后半局没有搜索**；长局卡组尤其要注意。")
    elif budget:
        print("✅ 预算从未触发。")
        if mean * 10 < budget:
            print(f"    还有 ~{RUN_TIMEOUT / max(mean, 1e-6):,.0f}× 的算力余量可用"
                  f"（例如加深搜索 / 加世界数，虽然实测加世界数是噪声）。")
    else:
        print("（非 rollout agent，没有时间预算概念）")
    print("=" * 72)


if __name__ == "__main__":
    main()
