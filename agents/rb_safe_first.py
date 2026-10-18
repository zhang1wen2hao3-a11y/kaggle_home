"""规则 agent：safe_v1 + attack_pick="first"（现状，作对照）。"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rulebase import make_agent  # noqa: E402
agent = make_agent(order="safe_v1", go_first=True, bench_full=True, attack_pick="first")
