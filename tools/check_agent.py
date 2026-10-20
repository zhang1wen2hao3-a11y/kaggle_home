#!/usr/bin/env python3
"""agent 自检：跑几局，确认 rollout 真的在跑，并打印它的实际配置。

用途有两个：

1. **抓 BUG-002 这类"静默退化"**。BUG-002 的表现是：agent 配置看起来正常、
   评分脚本也不报错，但它其实一局都没做搜索 —— 因为时间预算计时器跨局累加，
   累计超预算后该进程里后续所有对局都退化成纯规则。当时的征兆是
   **0.08 s/局（纯规则的速度），而算量差 4 倍的配置耗时完全一样**。
   本工具会把「真 rollout 次数 / 回退次数 / 异常」直接打出来。

2. **确认变体 agent 的配置真的生效**。环境变量在 evaluate 的 worker 里是共享的，
   两个 rollout agent 会互相污染，所以变体都是"烘死"在模块里的。
   这里把烘死的值读回来，避免"以为改了其实没改"。

用法：
    ./run.sh tools/check_agent.py agents/ro_attack_w10.py
    ./run.sh tools/check_agent.py agents/pl_evo.py --games 8 --opponent agents/greedy.py

判读：
    真 rollout > 0            正常
    真 rollout == 0 且有异常   看"首个异常"
    真 rollout == 0 且无异常   高度可疑 —— 配置是 rollout agent 却一次没搜
"""
import argparse
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluate import load_agent                      # noqa: E402
from local_match import play_match, load_deck         # noqa: E402

# agent 背后那个 main_rollout 模块里，值得读回来的全局量
CONFIG_KEYS = ("MODE", "WORLDS", "MARGIN", "MARGIN_K", "CANDS", "BUDGET_S",
               "_ROLLOUT_OK", "_IMPORT_ERR")


def describe_agent(name: str, fn) -> dict:
    g = getattr(fn, "__globals__", {}) or {}
    cfg = {k: g.get(k) for k in CONFIG_KEYS if k in g}
    if "PLAYOUT" in g:
        p = g["PLAYOUT"]
        # 变体把 playout 烘成闭包；能拿到 __name__ 就显示名字
        cfg["PLAYOUT"] = getattr(p, "__name__", "-") if p is not None else None
    print(f"  文件      : {name}")
    if not cfg:
        print("  配置      : （不是 rollout agent，没有 main_rollout 全局量）")
        return cfg
    for k in ("MODE", "WORLDS", "MARGIN", "MARGIN_K", "CANDS", "BUDGET_S"):
        if k in cfg:
            print(f"  {k:<10}: {cfg[k]}")
    if "PLAYOUT" in cfg:
        v = "默认(=兜底规则)" if cfg["PLAYOUT"] is None else cfg["PLAYOUT"]
        print(f"  {'PLAYOUT':<10}: {v}")
    ok = cfg.get("_ROLLOUT_OK")
    print(f"  {'_ROLLOUT_OK':<10}: {ok}" + ("" if ok else f"   ⚠️ {cfg.get('_IMPORT_ERR')!r}"))
    return cfg


def instrument(fn):
    """给 agent 背后的 _choose_action 打计数桩，返回 (清空函数, 统计字典)。"""
    g = getattr(fn, "__globals__", None)
    if not g or "_choose_action" not in g:
        return (lambda: None), {"supported": False}
    stat = {"supported": True, "ok": 0, "err": 0, "first_err": None,
            "worlds": 0, "fail": 0, "deviated": 0}
    orig = g["_choose_action"]

    def wrap(*a, **k):
        try:
            act, info = orig(*a, **k)
            stat["ok"] += 1
            stat["worlds"] += info.get("worlds", 0)
            stat["fail"] += info.get("fail", 0)
            if info.get("chosen") != info.get("base"):
                stat["deviated"] += 1
            return act, info
        except Exception as e:                      # agent() 会吞掉异常并回退规则
            stat["err"] += 1
            if stat["first_err"] is None:
                stat["first_err"] = "".join(
                    traceback.format_exception_only(type(e), e)).strip()
            raise

    g["_choose_action"] = wrap

    def reset():
        for k in ("ok", "err", "worlds", "fail", "deviated"):
            stat[k] = 0
        stat["first_err"] = None
    return reset, stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", help="被测 agent 文件")
    ap.add_argument("--opponent", default="agents/greedy.py")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--deck", default="decks/sample_official.csv")
    ap.add_argument("--deck-opponent", default=None)
    args = ap.parse_args()

    print("=" * 66)
    print("agent 自检")
    print("=" * 66)
    A = load_agent(args.agent)
    B = load_agent(args.opponent)
    cfg = describe_agent(args.agent, A)
    deck = load_deck(args.deck)
    opp_deck = load_deck(args.deck_opponent) if args.deck_opponent else deck
    reset, stat = instrument(A)
    print(f"  对手      : {args.opponent}   卡组: {args.deck}")

    print("\n" + "-" * 66)
    print(f"跑 {args.games} 局（每局都会 reset 预算，与线上每局独立 episode 一致）")
    print("-" * 66)
    walls, wins = [], 0
    agg = {"ok": 0, "err": 0, "worlds": 0, "fail": 0, "deviated": 0}
    for i in range(args.games):
        reset()
        rev = bool(i % 2)
        t0 = time.monotonic()
        r = play_match(A if i % 2 == 0 else B,
                       B if i % 2 == 0 else A,
                       deck, opp_deck, reverse=rev)
        walls.append(time.monotonic() - t0)
        if r.get("result") == (0 if i % 2 == 0 else 1):
            wins += 1
        if stat.get("supported"):
            for k in agg:
                agg[k] += stat.get(k, 0)
        err = r.get("error")
        if err:
            print(f"  game {i}: 异常 -> {err}")
        if stat.get("first_err"):
            print(f"  game {i}: choose_action 首个异常 -> {stat['first_err']}")

    n = max(len(walls), 1)
    mean = sum(walls) / n
    print(f"\n  胜率      : {wins}/{len(walls)}")
    print(f"  耗时      : 平均 {mean:.2f}s/局  最长 {max(walls):.2f}s")
    if stat.get("supported"):
        print(f"  真 rollout: {agg['ok']} 次")
        print(f"  异常回退  : {agg['err']} 次")
        print(f"  世界/失败 : {agg['worlds']} / {agg['fail']}")
        print(f"  偏离规则  : {agg['deviated']} 次"
              f"（占搜索决策 {agg['deviated']/max(agg['ok'],1):.0%}）")

    print("\n" + "=" * 66)
    problems = []
    if cfg and cfg.get("_ROLLOUT_OK") is False:
        problems.append("rollout 未就绪（_ROLLOUT_OK=False），全程走纯规则")
    if stat.get("supported") and agg["ok"] == 0:
        problems.append("一次都没真正搜过 —— 高度可疑（BUG-002 那类静默退化）")
    if stat.get("supported") and agg["err"] > 0:
        problems.append(f"搜索里有 {agg['err']} 次异常被 agent 吞掉了")
    if cfg and cfg.get("MODE") not in (None, "off") and mean < 0.05:
        problems.append(f"平均 {mean:.3f}s/局 低得像纯规则（rollout 通常 ≥0.3s/局）")
    if problems:
        print("⚠️  发现问题：")
        for p in problems:
            print(f"    - {p}")
    else:
        print("✅ 正常：rollout 在跑，没有异常被吞掉")
    print("=" * 66)


if __name__ == "__main__":
    main()
