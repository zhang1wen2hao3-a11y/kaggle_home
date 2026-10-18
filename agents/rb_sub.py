"""子决策 A/B 对照：smart_sub=False"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rulebase import make_agent  # noqa: E402
agent = make_agent(order="safe_v1", go_first=True, bench_full=True,
                   attack_pick="maxdmg", smart_sub=False)
