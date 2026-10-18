"""可切换优先级的规则 agent —— 用环境变量 PTCG_ORDER 选择 ORDERS 里的配置。

用途：**构造对手池**。单个对手当基线会被非传递性骗（实测：rollout 头对头赢
greedy 0.5642，但对随机对手的绝对强度 0.81 < greedy 的 0.86）。
正确做法是让被测 agent 面对同一批对手，比较总胜率。

用法：
    PTCG_ORDER=attack_first ./run.sh evaluate.py \
        --agent agents/greedy.py --opponent agents/rulebase_agent.py --games 100
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rulebase import ORDERS, make_agent  # noqa: E402

ORDER = os.environ.get("PTCG_ORDER", "develop_first")
if ORDER not in ORDERS:
    raise SystemExit(f"PTCG_ORDER={ORDER} 不在 {list(ORDERS)} 里")

GO_FIRST = os.environ.get("PTCG_GO_FIRST", "1") != "0"
BENCH_FULL = os.environ.get("PTCG_BENCH_FULL", "1") != "0"

agent = make_agent(order=ORDER, go_first=GO_FIRST, bench_full=BENCH_FULL)
