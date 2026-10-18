#!/usr/bin/env python3
"""B1：用 archetype 卡组池做轮转对战，采集多样化训练数据。

目的：
    V 的学习曲线在单一对手下 225 局就饱和（AUC 0.71），而混合"强弱悬殊"的对手
    反而掉到 0.68。所以要在**可控的多样性**下采数据，看能否突破平台期。

做法：
    读 decks/ 下的全部卡组，做两两轮转（每对都打），座位交替。
    这样同一局面会同时出现在"对强卡组"和"对弱卡组"的数据里，
    模型有机会学到"对手强度"这个隐变量。

用法：
    ./run.sh gen_pool_data.py --games-per-pair 30 --out data/pool_v1.jsonl
    ./run.sh gen_pool_data.py --games-per-pair 20 --oracle --out data/pool_oracle.jsonl
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from framework.dataset import describe, write_records  # noqa: E402
from framework.deck import CardPool, is_legal  # noqa: E402
from gen_data import load_agent, play_and_record  # noqa: E402
from local_match import load_deck  # noqa: E402


def load_decks(pool: CardPool, deck_dir: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for path in sorted(glob.glob(os.path.join(deck_dir, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        deck = pool.deck_from_file(path)
        ok, why = is_legal(pool, deck)
        if not ok:
            print(f"  ⚠️ 跳过非法卡组 {name}: {why}")
            continue
        out[name] = deck
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck-dir", default="decks")
    ap.add_argument("--agent", default="agents/greedy.py", help="双方共用的策略")
    ap.add_argument("--games-per-pair", type=int, default=30,
                    help="每对卡组打多少局（会按座位交替对半分）")
    ap.add_argument("--out", default="data/pool_v1.jsonl")
    ap.add_argument("--no-options", action="store_true",
                    help="不记录选项特征（给 V 用）")
    ap.add_argument("--oracle", action="store_true",
                    help="额外记录真实隐藏信息（给对手推断模型用；仅本地可用）")
    ap.add_argument("--include-sample", action="store_true", default=True,
                    help="把官方样例卡组也纳入（默认纳入）")
    ap.add_argument("--max-games", type=int, default=0, help="总对局上限（0=不限）")
    args = ap.parse_args()

    pool = CardPool()
    deck_dir = args.deck_dir if os.path.isabs(args.deck_dir) else os.path.join(HERE, args.deck_dir)
    decks = load_decks(pool, deck_dir)

    if args.include_sample:
        try:
            decks.setdefault("sample_official", load_deck(os.path.join(HERE, "deck.csv")))
        except Exception:
            pass

    names = list(decks)
    if len(names) < 2:
        raise SystemExit(f"{deck_dir} 下至少要 2 套合法卡组（当前 {len(names)}）")

    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    print(f"卡组 {len(names)} 套，两两组合 {len(pairs)} 对，"
          f"每对 {args.games_per_pair} 局")
    print(f"预计 {len(pairs) * args.games_per_pair} 局 | oracle={'开' if args.oracle else '关'}")
    for n in names:
        print(f"   · {n}")

    agent = load_agent(args.agent)
    all_recs: list[dict] = []
    game_id = 0
    t0 = time.time()

    for pi, (a, b) in enumerate(pairs):
        for g in range(args.games_per_pair):
            recs = play_and_record(agent, agent, decks[a], decks[b], game_id,
                                   bool(g % 2), args.oracle,
                                   record_options=not args.no_options)
            game_id += 1
            if recs:
                for r in recs:
                    r["matchup"] = f"{a}_vs_{b}"
                all_recs.extend(recs)
            if args.max_games and game_id >= args.max_games:
                break
        if args.max_games and game_id >= args.max_games:
            print(f"  达到 max-games={args.max_games}，提前停止")
            break
        dt = time.time() - t0
        print(f"  对 {pi+1}/{len(pairs)} {a} vs {b}: 累计 {game_id} 局 / "
              f"{len(all_recs):,} 条 ({dt:.0f}s)")

    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    n = write_records(out, all_recs)

    print()
    print("=" * 60)
    print(f"写盘 {n:,} 条 -> {out}")
    print(f"对局 {game_id} 局，用时 {time.time()-t0:.0f}s")
    st = describe(all_recs)
    for k, v in st.items():
        print(f"  {k:24s} {v}")
    matchups = len({r.get("matchup") for r in all_recs})
    print(f"  {'matchup 种类':24s} {matchups}")
    if args.oracle and all_recs:
        print(f"  {'含 oracle':24s} {sum(1 for r in all_recs if 'oracle' in r)}/{len(all_recs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
