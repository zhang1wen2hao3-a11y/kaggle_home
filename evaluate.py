#!/usr/bin/env python3
"""批量评估：agent A 对 agent B 跑 N 局，输出胜率。

座位每局交换，消除先后手偏差。

用法：
    python evaluate.py --agent agents/greedy.py --opponent agents/random.py --games 100
    python evaluate.py --agent agents/greedy.py --opponent agents/random.py --games 200 --log --name v1_greedy
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from local_match import play_match, load_deck  # noqa: E402


def load_agent(path: str):
    """按文件路径加载 agent 函数。"""
    path = os.path.abspath(path)
    # 让 agent 能 import 同目录的兄弟模块（如 rulebase）
    d = os.path.dirname(path)
    if d not in sys.path:
        # 注意：必须 append 而不是 insert(0)，否则 agents/ 下的模块会遮蔽标准库
        sys.path.append(d)
    spec = importlib.util.spec_from_file_location("agent_mod_" + os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "agent"):
        raise SystemExit(f"{path} 里没有 agent 函数")
    return mod.agent


# ============================================================ 并行执行
# 为什么用【进程】而不是线程：
#   引擎是 ctypes 加载的 C++ 单例 —— `Battle.battle_ptr` 是类属性、
#   `GameInitialize()` 只调用一次、搜索状态存在 lib 内部的全局变量里。
#   同一个进程里并发跑两局会互相踩状态，加线程救不了（ctypes 释放 GIL 也没用）。
#
# 为什么用 spawn 而不是 fork：
#   fork 出来的子进程会继承父进程已初始化的 lib 状态；spawn 每个进程全新加载，
#   隔离干净。代价是每个 worker 多花零点几秒初始化。

_W: dict = {}


def _worker_init(agent_path: str, opp_path: str, deck_path: str,
                 opp_deck_path: str | None = None) -> None:
    """每个 worker 进程只执行一次：加载两侧 agent 与卡组。"""
    _W["A"] = load_agent(agent_path)
    _W["B"] = load_agent(opp_path)
    _W["deck"] = load_deck(deck_path)
    _W["opp_deck"] = load_deck(opp_deck_path) if opp_deck_path else _W["deck"]


def _worker_play(g: int) -> dict:
    """打第 g 局。座位与 reverse 由 g 决定，所以结果与执行顺序无关。"""
    A, B = _W["A"], _W["B"]
    dA, dB = _W["deck"], _W["opp_deck"]
    seat = g % 2          # 0: A 坐 0 号位, 1: A 坐 1 号位
    rev = (g // 2) % 2    # 交替"谁能选先后手"
    try:
        if seat == 0:
            r = play_match(A, B, dA, dB, reverse=bool(rev))
        else:
            r = play_match(B, A, dB, dA, reverse=bool(rev))
    except Exception as e:                        # worker 里绝不能让异常炸掉整池
        return {"g": g, "seat": seat, "result": None,
                "error": f"{type(e).__name__}: {e}", "warnings": []}
    return {"g": g, "seat": seat, "result": r["result"],
            "error": r.get("error"), "warnings": r.get("warnings") or []}


AB_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "ab_log.tsv")
AB_HEADER = ["时间", "agent", "对手", "被测卡组", "对手卡组", "方案",
             "有效局", "总局", "W", "L", "平", "失败", "胜率", "耗时s", "s每局"]


def _append_ab_log(args, scheme, played, wins, losses, draws, fails, wr, elapsed) -> None:
    """把这次评估的汇总追加到 logs/ab_log.tsv（TSV，首次自动写表头）。"""
    try:
        os.makedirs(os.path.dirname(AB_LOG), exist_ok=True)
        new = not os.path.exists(AB_LOG)
        with open(AB_LOG, "a", encoding="utf-8") as f:
            if new:
                f.write("\t".join(AB_HEADER) + "\n")
            f.write("\t".join(map(str, [
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                os.path.basename(args.agent), os.path.basename(args.opponent),
                args.deck, args.deck_opponent or args.deck, scheme,
                played, args.games, wins, losses, draws, fails,
                f"{wr:.4f}", f"{elapsed:.1f}", f"{elapsed / max(args.games, 1):.3f}",
            ])) + "\n")
        print(f"已追加日志: logs/ab_log.tsv")
    except Exception as e:            # 记日志失败绝不能影响评估本身
        print(f"（写 ab_log.tsv 失败，已忽略: {e}）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="被测 agent 文件")
    ap.add_argument("--opponent", required=True, help="对手 agent 文件")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1,
                    help="并行进程数（1=串行）。实测加速：3→2.8x, 6→5.5x, 12→6.4x")
    ap.add_argument("--deck", default="deck.csv", help="被测方卡组")
    ap.add_argument("--deck-opponent", default=None,
                    help="对手卡组（默认同 --deck，即镜像对局）")
    ap.add_argument("--eval-scheme", default=None)
    ap.add_argument("--log", action="store_true", help="写入 exp_log")
    ap.add_argument("--name", default=None, help="实验名（--log 时必填）")
    ap.add_argument("--changes", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # 提前校验路径：worker 里才报错的话，Pool 初始化失败的现场会很难看
    for p in (args.agent, args.opponent):
        if not os.path.exists(p):
            raise SystemExit(f"找不到 agent 文件: {p}")
    if args.workers < 1:
        raise SystemExit("--workers 必须 >= 1")

    wins = losses = draws = fails = 0
    all_warnings = []
    errs = []
    t0 = time.time()
    if not args.quiet:
        # 立刻打一行，让人一眼看出"已经在跑了"（第一局可能要好几秒）
        print(f"开始对局：{args.games} 局  |  "
              f"{os.path.basename(args.agent)} vs {os.path.basename(args.opponent)}  |  "
              f"座位×先手 四路交替  |  workers={args.workers}\n"
              f"卡组：{args.deck}  vs  {args.deck_opponent or args.deck}"
              f"{'  （镜像）' if not args.deck_opponent else ''}", flush=True)

    if args.workers > 1:
        import multiprocessing as mp
        # spawn 而不是 fork：引擎是 ctypes 加载的 C++ 单例，fork 出来的子进程
        # 会继承父进程已经初始化的 lib 状态，容易出诡异问题。spawn 每进程全新加载。
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(args.workers, initializer=_worker_init,
                        initargs=(args.agent, args.opponent, args.deck,
                                  args.deck_opponent))
        it = pool.imap_unordered(_worker_play, range(args.games), chunksize=1)
    else:
        pool = None
        _worker_init(args.agent, args.opponent, args.deck,
                     args.deck_opponent)          # 串行时在父进程里加载一次
        it = map(_worker_play, range(args.games))

    try:
        for done, res in enumerate(it, 1):
            g, seat = res["g"], res["seat"]
            if res["result"] is None:
                fails += 1
                errs.append(res.get("error") or "未知")
                if not args.quiet:
                    print(f"  game {g}: 异常 -> {res.get('error')}")
            else:
                if res["result"] == 2:
                    draws += 1
                elif res["result"] == seat:
                    wins += 1
                else:
                    losses += 1
                all_warnings.extend(res["warnings"])

            if not args.quiet and (done == 1 or done % 10 == 0):
                played = wins + losses + draws
                el = time.time() - t0
                eta = el / done * (args.games - done)
                print(f"  {done}/{args.games}  胜率={wins/max(played,1):.3f}  "
                      f"(W{wins} L{losses} D{draws} F{fails})  "
                      f"已用 {el:.0f}s  预计还需 {eta:.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    played = wins + losses + draws
    wr = wins / played if played else 0.0
    dt = time.time() - t0
    scheme = args.eval_scheme or f"vs-{os.path.basename(args.opponent)[:-3]}-n{args.games}"

    print()
    print("=" * 60)
    print(f"被测 agent : {args.agent}")
    print(f"对手       : {args.opponent}")
    print(f"对局       : {played} 有效 / {args.games} 总 (平{draws} 失败{fails})")
    print(f"胜率       : {wr:.4f}   (W{wins} L{losses})")
    if fails:
        print(f"\n\033[31m!!!! {fails}/{args.games} 局因 agent 崩溃而无效 —— 胜率不可信 !!!!\033[0m")
        from collections import Counter
        for msg, c in Counter(errs).most_common(3):
            print(f"    x{c}  {msg}")
        print("    崩溃的对局已排除在胜率之外，请先修掉再解读数字。\n")
    print(f"耗时       : {dt:.1f}s  ({dt/max(args.games,1):.2f}s/局)")
    print(f"评估方案   : {scheme}")
    if all_warnings:
        print(f"⚠ 非法返回 : {len(all_warnings)} 次，例：{all_warnings[:3]}")
    print("=" * 60)

    # ---- 无条件落盘：每次评估都往 logs/ab_log.tsv 追加一行
    # 为什么要有这个：以前结果只靠 `| tee logs/xxx.log` 或者留在终端里，
    # 跑完就丢（后台任务里尤其严重）。实验记录不能依赖调用者的自觉。
    _append_ab_log(args, scheme, played, wins, losses, draws, fails, wr, dt)

    if args.log:
        if not args.name:
            raise SystemExit("--log 需要同时给 --name")
        from exp_log import log_experiment
        exp_id = log_experiment(name=args.name,
                                agent=os.path.basename(args.agent)[:-3],
                                eval_scheme=scheme, win_rate=wr, n_games=played,
                                changes=args.changes,
                                notes=f"对手={os.path.basename(args.opponent)} 平{draws} 失败{fails}")
        print(f"已记录 {exp_id}")


if __name__ == "__main__":
    main()
