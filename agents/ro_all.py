"""固定配置的 rollout agent：CANDS=all, MARGIN=1.5（用于 A/B 对打，避免环境变量冲突）"""
import os
os.environ["PTCG_ROLLOUT_CANDS"] = "all"
os.environ["PTCG_ROLLOUT_MARGIN"] = "1.5"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "rollout_agent_base", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rollout_agent.py"))
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
agent = _m.agent
