#!/usr/bin/env python3
"""A 方案：迭代自对弈（AlphaZero 式闭环）。

## 要解决的问题

V 训练在 greedy 局面上、却在 search 局面上使用 —— 分布偏移。
搜索越深越深入外推区，所以"加搜索反而更差"。

AlphaZero 的正统解法：**用当前策略自己产生的对局去训练 V**，循环迭代。

## 与 AlphaZero 的差异（有意为之）

AlphaZero 用纯 self-play。我们这里用 **search vs greedy**：
  · 我们的 search agent 目前弱于 greedy（0.39），纯自对弈两边都弱、局面退化
    （甚至可能互相 END 打成死局）
  · search vs greedy 能保证对局是竞争性的，而且数据分布正好是
    "V 将被使用时的那个分布"（评估也是 vs greedy）

## 每轮做什么

  1. 用当前 V 驱动的 search agent 打 N 局 vs greedy，记录 (局面, 终局)
  2. 用这批数据训新的 V
  3. 评估：V 的 AUC + search(vs greedy) 的实战胜率
  4. 若胜率提升则接受新 V，否则保留旧的（防止震荡）

用法：
    ./run.sh alpha_loop.py --rounds 3 --games 150 --eval-games 100
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "agents"))

from framework.dataset import split_by_game, write_records  # noqa: E402
from framework.models import create  # noqa: E402
from framework.models.lightgbm_model import LGBMModel  # noqa: E402
from gen_data import load_agent, play_and_record  # noqa: E402
from local_match import load_deck, play_match  # noqa: E402
from train_value import auc, brier, log_loss  # noqa: E402


def make_search_agent(model_path: str, deck: list[int], sims: int, depth: int):
    """按给定 V 权重创建全新的 search agent（实例无状态复用风险）。"""
    import search_agent as sa  # noqa: F401  保证模块已导入
    spec = importlib.util.spec_from_file_location(
        "sa_mod", os.path.join(HERE, "agents/search_agent.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = LGBMModel.load(model_path)
    from framework.search import ModelValue
    return mod.SearchAgent(value_fn=ModelValue(model), deck=deck,
                           n_simulations=sims, max_depth=depth)


def gen_data(agent_fn, opponent_fn, deck, opponent_deck, games: int,
             start_id: int) -> list[dict]:
    recs: list[dict] = []
    ok = 0
    for g in range(games):
        # 每局用全新实例，避免跨局状态污染
        a = agent_fn()
        r = play_and_record(a, opponent_fn, deck, opponent_deck,
                            start_id + g, bool(g % 2), oracle=False,
                            record_options=False)
        if r:
            ok += 1
            recs.extend(r)
    return recs


def labels(recs: list[dict]):
    y = []
    for r in recs:
        res = r.get("result")
        if res is None:
            continue
        y.append(0.5 if res == 2 else (1.0 if res == r["player"] else 0.0))
    return y


def train_v(recs: list[dict], out_path: str, val_ratio: float = 0.2):
    tr, va = split_by_game(recs, val_ratio, seed=0)
    Xtr = [r["features"] for r in tr]
    ytr = labels(tr)
    Xva = [r["features"] for r in va]
    yva = labels(va)
    if not Xtr or not Xva:
        return None, {}
    m = create("lightgbm", task="binary")
    m.fit(Xtr, ytr, eval_set=(Xva, yva))
    m.save(out_path)
    p = m.predict(Xva)
    metrics = {"auc": auc(yva, p), "log_loss": log_loss(yva, p),
               "brier": brier(yva, p), "n_train": len(Xtr), "n_val": len(Xva)}
    return m, metrics


def eval_winrate(agent_fn, opponent_fn, deck, opp_deck, games: int) -> float:
    w = l = d = 0
    for g in range(games):
        seat, rev = g % 2, (g // 2) % 2
        a = agent_fn()
        if seat == 0:
            r = play_match(a, opponent_fn, deck, opp_deck, reverse=bool(rev))
        else:
            r = play_match(opponent_fn, a, opp_deck, deck, reverse=bool(rev))
        res = r.get("result")
        if res is None:
            continue
        if res == 2:
            d += 1
        elif res == seat:
            w += 1
        else:
            l += 1
    played = w + l + d
    return (w + 0.5 * d) / played if played else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--games", type=int, default=150, help="每轮生成对局数")
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed-model", default="models/value_pool_v2.json",
                    help="起始 V（用 greedy 数据训出来的那个）")
    ap.add_argument("--out-dir", default="models/alpha")
    args = ap.parse_args()

    deck = load_deck(os.path.join(HERE, "deck.csv"))
    greedy = load_agent(os.path.join(HERE, "agents/greedy.py"))
    out_dir = os.path.join(HERE, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cur_model = os.path.join(HERE, args.seed_model)
    if not os.path.exists(cur_model):
        raise SystemExit(f"起始模型不存在: {cur_model}")

    print("=" * 70)
    print("A 方案：迭代自对弈（search vs greedy）")
    print("=" * 70)
    print(f"每轮 {args.games} 局生成 + {args.eval_games} 局评估 | "
          f"search={args.sims} sims / depth {args.depth}")
    print(f"起始 V: {os.path.basename(cur_model)}\n")

    # 第 0 轮：先用起始 V 测基线胜率
    def mk(path):
        return lambda: make_search_agent(path, deck, args.sims, args.depth)

    t0 = time.time()
    base_wr = eval_winrate(mk(cur_model), greedy, deck, deck, args.eval_games)
    print(f"[基线] 起始 V 的 search vs greedy 胜率 = {base_wr:.3f}")
    best_wr, best_model = base_wr, cur_model

    history = [{"round": 0, "winrate": base_wr,
                "model": os.path.basename(cur_model), "note": "seed"}]

    for rnd in range(1, args.rounds + 1):
        print(f"\n{'-'*70}\n第 {rnd}/{args.rounds} 轮")
        # 1) 用当前 V 生成对局
        t_gen = time.time()
        recs = gen_data(mk(cur_model), greedy, deck, deck,
                        args.games, start_id=rnd * 100000)
        print(f"  生成 {len(recs):,} 条记录 ({time.time()-t_gen:.0f}s)")

        if len(recs) < 2000:
            print("  ⚠️ 数据太少，跳过本轮")
            break

        # 2) 训新 V
        t_tr = time.time()
        new_path = os.path.join(out_dir, f"value_r{rnd}.json")
        m, metrics = train_v(recs, new_path)
        if m is None:
            print("  ⚠️ 训练失败，跳过")
            continue
        print(f"  训练完成 ({time.time()-t_tr:.0f}s): "
              f"AUC={metrics['auc']:.4f}  log_loss={metrics['log_loss']:.4f}  "
              f"(训{metrics['n_train']:,} / 验{metrics['n_val']:,})")

        # 3) 实战评估
        t_ev = time.time()
        wr = eval_winrate(mk(new_path), greedy, deck, deck, args.eval_games)
        print(f"  实战: search vs greedy = {wr:.3f}  ({time.time()-t_ev:.0f}s)  "
              f"(基线 {best_wr:.3f})")

        history.append({"round": rnd, "winrate": wr, "auc": metrics["auc"],
                        "log_loss": metrics["log_loss"],
                        "model": os.path.basename(new_path),
                        "n_records": len(recs)})

        # 4) 接受 or 回滚
        if wr > best_wr:
            print(f"  ✅ 接受新 V（{best_wr:.3f} -> {wr:.3f}）")
            best_wr, best_model = wr, new_path
            cur_model = new_path
        else:
            print(f"  ❌ 未提升（{wr:.3f} <= {best_wr:.3f}），继续用旧 V")
            # 注意：仍然用旧 V 继续下一轮，但也可以选择接受以便探索
            cur_model = new_path if wr >= base_wr else cur_model

    print("\n" + "=" * 70)
    print(f"最佳: {os.path.basename(best_model)}  胜率 {best_wr:.3f}")
    print(f"总耗时 {(time.time()-t0)/60:.1f} 分钟")
    print("=" * 70)
    import json
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=1)
    print(f"历史 -> {os.path.join(out_dir, 'history.json')}")


if __name__ == "__main__":
    main()
