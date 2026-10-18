#!/usr/bin/env python3
"""遗传算法卡组选择器。

思路：**策略固定（贪婪规则），只管卡组**。
    卡组选择是个"有廉价适应度函数"的组合优化问题 —— 一次评估只要几十局
    （~90 局/秒），所以用 GA 搜索比训练模型直接得多。

两种模式：
    vs_baseline  我方卡组 GA 进化，对手固定为样例卡组
    coevolve     "两边各挑一轮"：A 进化 -> B 进化 -> A 进化 -> ...
                 （对手会跟着变强，避免只对一副牌过拟合）

用法：
    # 最简单的：对样例卡组进化 30 代
    ./run.sh ga_deck.py --mode vs_baseline --generations 30 --pop 24 --games 60

    # 双方协同进化，各挑 3 轮
    ./run.sh ga_deck.py --mode coevolve --rounds 3 --generations 15

    # 结果会写到 data/ga_best_deck.csv，可直接当 deck.csv 用
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from framework.deck import (CardPool, crossover, is_legal, mutate_deck,  # noqa: E402
                            random_deck, repair)
from local_match import play_match  # noqa: E402


# ---------------------------------------------------------------- 载入策略
def load_agent(path: str):
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    if d not in sys.path:
        sys.path.append(d)               # append 避免遮蔽标准库
    spec = importlib.util.spec_from_file_location(
        "ga_agent_" + os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


# ---------------------------------------------------------------- 适应度
class Evaluator:
    """用固定的策略跑对局，给卡组打分（胜率）。带缓存，避免重复评估同一副牌。"""

    def __init__(self, agent, games: int = 60, max_seconds: float | None = None):
        self.agent = agent
        self.games = games
        self.max_seconds = max_seconds
        self.t0 = time.time()
        self.cache: dict[tuple, float] = {}
        self.total_games = 0
        self.evals = 0

    @staticmethod
    def _key(deck: list[int]) -> tuple:
        return tuple(sorted(deck))

    def __call__(self, deck: list[int], opponents: list[list[int]]) -> float:
        key = (self._key(deck), tuple(self._key(o) for o in opponents))
        if key in self.cache:
            return self.cache[key]
        if self.max_seconds and time.time() - self.t0 > self.max_seconds:
            return self.cache.get(key, 0.5)

        wins = losses = draws = fails = 0
        per_opp = max(1, self.games // max(len(opponents), 1))
        for opp in opponents:
            for g in range(per_opp):
                seat, rev = g % 2, (g // 2) % 2
                if seat == 0:
                    r = play_match(self.agent, self.agent, deck, opp, reverse=bool(rev))
                else:
                    r = play_match(self.agent, self.agent, opp, deck, reverse=bool(rev))
                self.total_games += 1
                res = r.get("result")
                if res is None:
                    fails += 1
                elif res == 2:
                    draws += 1
                elif res == seat:
                    wins += 1
                else:
                    losses += 1
        played = wins + losses + draws
        score = (wins + 0.5 * draws) / played if played else 0.0
        self.cache[key] = score
        self.evals += 1
        return score

    def elapsed(self) -> float:
        return time.time() - self.t0


# ---------------------------------------------------------------- GA 主体
class GA:
    def __init__(self, pool: CardPool, evaluator: Evaluator, rng: random.Random,
                 pop_size: int = 24, elite: int = 4, tournament: int = 3,
                 mut_rate: float = 0.7, cross_rate: float = 0.4,
                 structural_p: float = 0.25, verbose: bool = True):
        self.pool = pool
        self.ev = evaluator
        self.rng = rng
        self.pop_size = pop_size
        self.elite = elite
        self.tournament = tournament
        self.mut_rate = mut_rate
        self.cross_rate = cross_rate
        self.structural_p = structural_p
        self.verbose = verbose
        self.history: list[dict] = []

    def _tournament(self, scored: list[tuple[float, list[int]]]) -> list[int]:
        cand = self.rng.sample(scored, k=min(self.tournament, len(scored)))
        return max(cand, key=lambda x: x[0])[1]

    def run(self, generations: int, opponents: list[list[int]],
            init_pop: list[list[int]] | None = None) -> tuple[list[int], float]:
        pop = list(init_pop or [])
        # 用对手卡组（通常就是样例卡组）当种子之一 —— 保证 GA 不会比基线更差
        if opponents:
            seed = repair(self.pool, list(opponents[0]), self.rng)
            pop.insert(0, seed)
        while len(pop) < self.pop_size:
            pop.append(random_deck(self.pool, self.rng))
        pop = [repair(self.pool, d, self.rng) for d in pop[:self.pop_size]]

        best_deck, best_score = None, -1.0
        for gen in range(generations):
            g0 = time.time()
            scored = [(self.ev(d, opponents), d) for d in pop]
            scored.sort(key=lambda x: -x[0])
            gen_best, gen_deck = scored[0]
            if gen_best > best_score:
                best_score, best_deck = gen_best, list(gen_deck)

            mean = sum(s for s, _ in scored) / len(scored)
            self.history.append({
                "gen": gen, "best": round(gen_best, 4), "mean": round(mean, 4),
                "deck": list(gen_deck),
                "stats": self.pool.deck_stats(gen_deck),
            })
            if self.verbose:
                dt = time.time() - g0
                print(f"  第{gen+1:>2}代  best={gen_best:.3f}  mean={mean:.3f}  "
                      f"({dt:.1f}s, 累计 {self.ev.total_games} 局)  "
                      f"{self.pool.describe(gen_deck, 4)}")

            # 精英保留 + 繁殖
            nxt = [list(d) for _, d in scored[:self.elite]]
            while len(nxt) < self.pop_size:
                p1 = self._tournament(scored)
                child = p1
                if self.rng.random() < self.cross_rate:
                    p2 = self._tournament(scored)
                    child = crossover(self.pool, p1, p2, self.rng)
                if self.rng.random() < self.mut_rate:
                    child = mutate_deck(self.pool, child, self.rng,
                                        n_swaps=self.rng.randint(1, 3),
                                        structural_p=self.structural_p)
                ok, _ = is_legal(self.pool, child)
                nxt.append(child if ok else repair(self.pool, child, self.rng))
            pop = nxt

        return best_deck, best_score


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["vs_baseline", "coevolve"], default="vs_baseline")
    ap.add_argument("--agent", default="agents/greedy.py", help="固定的策略（不参与进化）")
    ap.add_argument("--baseline-deck", default="deck.csv", help="对手固定卡组")
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--games", type=int, default=60, help="每副卡组的评估局数")
    ap.add_argument("--rounds", type=int, default=3, help="coevolve 模式下交替轮数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--structural-p", type=float, default=0.25,
                    help="结构变异概率（整条进化线替换）；0=纯点变异")
    ap.add_argument("--max-seconds", type=float, default=None, help="总时间上限")
    ap.add_argument("--out", default="data/ga_best_deck.csv")
    ap.add_argument("--history", default="data/ga_history.json")
    args = ap.parse_args()

    pool = CardPool()
    agent = load_agent(args.agent)
    rng = random.Random(args.seed)
    ev = Evaluator(agent, games=args.games, max_seconds=args.max_seconds)

    base_deck = pool.deck_from_file(os.path.join(HERE, args.baseline_deck))
    print(f"卡池: 基础宝可梦 {len(pool.basic_pokemon)} | 训练师 {len(pool.trainers)} | "
          f"基础能量 {len(pool.basic_energy)}")
    print(f"策略: {args.agent}（固定，不进化）")
    print(f"评估: 每副牌 {args.games} 局 | 种群 {args.pop} | "
          f"模式 {args.mode}\n")

    # 样例卡组的基线分（作为参照）
    base_score = ev(base_deck, [base_deck])
    print(f"参照：样例卡组自对弈得分 {base_score:.3f}（应≈0.5）\n")

    ga = GA(pool, ev, rng, pop_size=args.pop, structural_p=args.structural_p)
    t_start = time.time()

    if args.mode == "vs_baseline":
        print("=== vs_baseline：我方进化，对手固定为样例卡组 ===")
        best, score = ga.run(args.generations, [base_deck])
        deck_a, deck_b = best, base_deck
    else:
        print(f"=== coevolve：双方交替进化，共 {args.rounds} 轮 ===")
        deck_a, deck_b = list(base_deck), list(base_deck)
        score = 0.0
        for r in range(args.rounds):
            print(f"\n-- 第 {r+1} 轮 A：我方进化，对手 = B 当前版本 --")
            deck_a, score = ga.run(args.generations, [deck_b], init_pop=None)
            print(f"   A 得分 {score:.3f}")

            print(f"-- 第 {r+1} 轮 B：对手进化，对手 = A 当前版本 --")
            saved = ga.history
            ga.history = []
            deck_b, score_b = ga.run(args.generations, [deck_a], init_pop=None)
            ga.history = saved
            print(f"   B 得分 {score_b:.3f}")

    # ---------------- 最终交叉验证：新卡组 vs 样例卡组
    print("\n=== 最终验证（各 200 局，座位交替）===")

    def head_to_head(da, db, n=200):
        w = l = d = 0
        for g in range(n):
            seat, rev = g % 2, (g // 2) % 2
            r = play_match(agent, agent, da, db, reverse=bool(rev)) if seat == 0 \
                else play_match(agent, agent, db, da, reverse=bool(rev))
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
        return (w + 0.5 * d) / played if played else 0.0, w, l, d

    wr, w, l, d = head_to_head(deck_a, base_deck)
    print(f"  GA卡组 vs 样例卡组 : {wr:.3f}  (W{w} L{l} D{d})")
    if args.mode == "coevolve":
        wr2, w2, l2, d2 = head_to_head(deck_a, deck_b)
        print(f"  GA卡组A vs GA卡组B : {wr2:.3f}  (W{w2} L{l2} D{d2})")

    # ---------------- 保存
    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(str(c) for c in deck_a) + "\n")
    print(f"\n最优卡组已保存 -> {out}")
    print(f"  构成: {pool.describe(deck_a, 10)}")
    print(f"  统计: {pool.deck_stats(deck_a)}")

    hist = args.history if os.path.isabs(args.history) else os.path.join(HERE, args.history)
    with open(hist, "w") as f:
        json.dump({"mode": args.mode, "best_score": score,
                   "total_games": ev.total_games,
                   "elapsed": round(time.time() - t_start, 1),
                   "history": ga.history}, f, ensure_ascii=False, indent=1)
    print(f"进化历史 -> {hist}")
    print(f"\n总计 {ev.total_games} 局对弈，用时 {time.time()-t_start:.1f}s")

    # 记入实验记录
    try:
        spec = importlib.util.spec_from_file_location("el", os.path.join(HERE, "exp_log.py"))
        el = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(el)
        exp_id = el.log_experiment(
            name=f"ga_deck_{args.mode}", agent="ga:deck",
            eval_scheme=f"ga-{args.mode}-g{args.games}-p{args.pop}",
            win_rate=wr, n_games=200,
            changes=f"GA 选卡组（{args.mode}, {args.generations}代, pop{args.pop}, {args.games}局/评估）",
            notes=f"vs样例 {wr:.3f} | 总对局 {ev.total_games} | {pool.describe(deck_a,5)}")
        print(f"已记录 {exp_id}")
    except Exception as e:
        print(f"(未写入实验记录: {e})")


if __name__ == "__main__":
    main()
