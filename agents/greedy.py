"""规则基线 agent（提交候选）。

配置由 sweep.py / 两两对决实测选出：develop_first + 先手 + 铺满后备。
见 README「关键发现」。
"""
from rulebase import make_agent

agent = make_agent(order="develop_first", go_first=True, bench_full=True)
