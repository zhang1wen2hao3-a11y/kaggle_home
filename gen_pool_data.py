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
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from framework.dataset import describe, write_records  # noqa: E402
from framework.deck import CardPool, is_legal  # noqa: E402
from gen_data import load_agent, play_and_record  # noqa: E402


def _agent_epsilon(path: str):
    """读出该策略本轮实际生效的 ε，用于溯源。

    探索器从环境变量 PTCG_EPSILON 取 ε（缺省 0.25），所以这里返回**实际生效值**，
    而不是源码里的默认值表达式 —— 后者曾让 meta 记下一串代码，毫无用处。
    """
    base = os.path.basename(path)
    if "explorer" not in base:
        return 0.0 if ("greedy" in base or "main.py" in base) else None
    try:
        return float(os.environ.get("PTCG_EPSILON", "0.25"))
    except ValueError:
        return os.environ.get("PTCG_EPSILON")
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
    ap.add_argument("--agent", default="agents/greedy.py",
                    help="双方共用的策略（--agent-a/--agent-b 未指定时生效）")
    ap.add_argument("--agent-a", default=None,
                    help="玩家0 的策略（覆盖 --agent）。用于只让一方探索")
    ap.add_argument("--agent-b", default=None,
                    help="玩家1 的策略（覆盖 --agent）")
    ap.add_argument("--games-per-pair", type=int, default=30,
                    help="每对卡组打多少局（按 卡组顺序 x 座位 四路平衡）")
    ap.add_argument("--out", default="data/pool_v1.jsonl")
    ap.add_argument("--no-options", action="store_true",
                    help="不记录选项特征（给 V 用）")
    ap.add_argument("--oracle", action="store_true",
                    help="额外记录真实隐藏信息（给对手推断模型用；仅本地可用）")
    ap.add_argument("--include-sample", action="store_true", default=True,
                    help="把官方样例卡组也纳入（默认纳入）")
    ap.add_argument("--max-games", type=int, default=0, help="总对局上限（0=不限）")
    ap.add_argument("--seed", type=int, default=0,
                    help="探索器随机种子；两侧会自动错开，避免双方同步探索")
    ap.add_argument("--progress-every", type=int, default=20,
                    help="每多少对卡组打印一次进度")
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

    # ---- 加载两侧策略。探索器用【不同种子】，否则双方会在同一批决策上同步偏离 ----
    pa = args.agent_a or args.agent
    pb = args.agent_b or args.agent
    _prev_seed = os.environ.get("PTCG_EPSILON_SEED")
    os.environ["PTCG_EPSILON_SEED"] = str(args.seed * 2 + 1)
    A0 = load_agent(pa)
    os.environ["PTCG_EPSILON_SEED"] = str(args.seed * 2 + 2)
    A1 = load_agent(pb)
    if _prev_seed is None:
        os.environ.pop("PTCG_EPSILON_SEED", None)
    else:
        os.environ["PTCG_EPSILON_SEED"] = _prev_seed
    # 记录本轮实际生效的 ε（而非源码里的默认值表达式）
    eps_a = _agent_epsilon(pa)
    eps_b = _agent_epsilon(pb)

    all_recs: list[dict] = []
    game_id = 0
    t0 = time.time()

    for pi, (a, b) in enumerate(pairs):
        for g in range(args.games_per_pair):
            # 四路平衡：卡组顺序（谁坐玩家0）x 座位（谁有选先后手的权利）
            # 只交替 reverse 的话，(a,b) 永远让 a 拿玩家0，会带进卡组-座位偏置
            swap = bool((g // 2) % 2)
            n0, n1 = (b, a) if swap else (a, b)
            recs = play_and_record(A0, A1, decks[n0], decks[n1], game_id,
                                   bool(g % 2), args.oracle,
                                   record_options=not args.no_options)
            game_id += 1
            if recs:
                for r in recs:
                    r["matchup"] = f"{n0}_vs_{n1}"
                all_recs.extend(recs)
            if args.max_games and game_id >= args.max_games:
                break
        if args.max_games and game_id >= args.max_games:
            print(f"  达到 max-games={args.max_games}，提前停止")
            break
        if (pi + 1) % max(args.progress_every, 1) == 0 or pi + 1 == len(pairs):
            dt = time.time() - t0
            print(f"  对 {pi+1}/{len(pairs)} {a} vs {b}: 累计 {game_id} 局 / "
                  f"{len(all_recs):,} 条 ({dt:.0f}s)")

    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    n = write_records(out, all_recs)

    # ---- 溯源元数据：ε 之前完全没记录，只能事后靠选项特征反算（measure_epsilon.py）。
    #      这里把生成条件直接落盘，杜绝"这批数据是在多大随机概率下跑的"这种黑洞。
    meta = {
        "out": os.path.relpath(out, HERE),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "agent_a": pa,
        "agent_b": pb,
        "epsilon_env": os.environ.get("PTCG_EPSILON", "(未设置，用 agent 文件默认)"),
        "agent_a_epsilon": eps_a,
        "agent_b_epsilon": eps_b,
        "seed": args.seed,
        "deck_count": len(names),
        "decks": names,
        "pair_count": len(pairs),
        "games_per_pair": args.games_per_pair,
        "n_games": game_id,
        "n_records": n,
        "record_options": not args.no_options,
        "oracle": bool(args.oracle),
        "decisions_per_game": round(n / game_id, 1) if game_id else 0,
        "state_feature_dim": len(all_recs[0]["features"]) if all_recs else 0,
        "option_feature_dim": (len(all_recs[0]["options"][0])
                              if all_recs and all_recs[0].get("options") else 0),
    }
    with open(out + ".meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"写盘 {n:,} 条 -> {out}")
    print(f"对局 {game_id} 局，用时 {time.time()-t0:.0f}s")
    print(f"元数据 -> {out}.meta.json")
    st = describe(all_recs)
    for k, v in st.items():
        print(f"  {k:24s} {v}")
    matchups = len({r.get("matchup") for r in all_recs})
    print(f"  {'matchup 种类':24s} {matchups}")
    print(f"  {'K 值':24s} {args.games_per_pair}  (每对卡组局数)")
    print(f"  {'探索':24s} A={meta['agent_a_epsilon']} B={meta['agent_b_epsilon']}")
    if args.oracle and all_recs:
        print(f"  {'含 oracle':24s} {sum(1 for r in all_recs if 'oracle' in r)}/{len(all_recs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
