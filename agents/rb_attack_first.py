"""规则 agent：order="attack_first"。由 7 配置循环赛自动生成，勿手改。

这样做的原因：evaluate.py 里 agent 和 opponent 在同一个进程，
共用环境变量，无法用 PTCG_ORDER 给两侧不同配置 => 必须按文件区分。
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rulebase import make_agent  # noqa: E402

agent = make_agent(order="attack_first", go_first=True, bench_full=True)
