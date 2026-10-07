"""规则基线 agent（提交候选）。

配置由 sweep.py 实测选出，见 README 的实验记录。
"""
from rulebase import make_agent

agent = make_agent(order="develop_first", go_first=True, bench_full=True)
