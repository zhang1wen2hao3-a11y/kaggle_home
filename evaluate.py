#!/usr/bin/env python3
"""批量评估：agent A 对 agent B 跑 N 局，输出胜率。

座位每局交换，消除先后手偏差。

用法：
    python evaluate.py --agent agents/greedy.py --opponent agents/random.py --games 100
    python evaluate.py --agent agents/greedy.py --opponent agents/random.py --games 200 --log --name v1_greedy
"""
from __future__ import annotations

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="被测 agent 文件")
    ap.add_argument("--opponent", required=True, help="对手 agent 文件")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--eval-scheme", default=None)
    ap.add_argument("--log", action="store_true", help="写入 exp_log")
    ap.add_argument("--name", default=None, help="实验名（--log 时必填）")
    ap.add_argument("--changes", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    A = load_agent(args.agent)
    B = load_agent(args.opponent)
    deck = load_deck()

    wins = losses = draws = fails = 0
    all_warnings = []
    errs = []
    t0 = time.time()
    for g in range(args.games):
        # 座位 + reverse 双重交替：4 局覆盖全部组合，消除 0 号位的结构优势
        seat = g % 2          # 0: A 坐 0 号位, 1: A 坐 1 号位
        rev = (g // 2) % 2    # 交替"谁能选先后手"
        a_idx = seat
        if seat == 0:
            r = play_match(A, B, deck, deck, reverse=bool(rev))
        else:
            r = play_match(B, A, deck, deck, reverse=bool(rev))

        if r["result"] is None:
            fails += 1
            errs.append(r.get("error") or "未知")
            if not args.quiet:
                print(f"  game {g}: 异常 -> {r['error']}")
            continue
        if r["result"] == 2:
            draws += 1
        elif r["result"] == a_idx:
            wins += 1
        else:
            losses += 1
        all_warnings.extend(r["warnings"])

        if not args.quiet and (g + 1) % 10 == 0:
            played = wins + losses + draws
            print(f"  {g+1}/{args.games}  胜率={wins/max(played,1):.3f}  (W{wins} L{losses} D{draws} F{fails})")

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
