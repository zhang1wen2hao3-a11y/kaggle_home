"""把 busyaprime/Dipam 的移植包接进我们的本地评估框架。

⚠️ 必须补 reset()：他们只在 `obs["select"] is None`（Kaggle 的 episode 开始、
交卡组那一步）把耗时预算 `ROT["est"]` 重置为 600 秒，并清空拒答/重复状态。
我们的 `local_match.play_match` **从不发送 select=None**（这正是当年 BUG-002：
`main_rollout` 的预算计时器跨局累积、20 秒后静默退化成纯规则），
所以这里要把"episode 开始"那一段状态复现出来，否则同一个 worker 里
连打若干局之后他们的 agent 会一直扣预算、跌破 30 秒阈值后退化成随机合法答案
—— 实测会把胜率从 0.80 假降到 0.655。
"""
import importlib.util as _ilu
import os as _os

_D = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("_port_main", _os.path.join(_D, "main.py"))
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)


def reset():
    """复现他们 `select is None` 分支里的按局重置（不含交卡组那一步）。"""
    try:
        _m.REJ["sig"], _m.REJ["tried"] = None, []
        _m.LAST["sig"], _m.LAST["logs"] = None, None
        _m.ROT["est"] = 600.0
        _m.GAME["round"] = None
        # ⚠️ 还要重建 InfoTracker：他们只在"round 变化且 GAME['round'] 非 None"时重建，
        #    而我们上面把 GAME['round'] 清成 None，于是新一局**不会**触发重建，
        #    上一局的隐藏牌追踪会泄漏到下一局（实测会让胜率随局数持续下滑）。
        _m.SA.tracker = _m.InfoTracker(_m.SA.deck)
        _m.SA.forced_run = 0
    except Exception:
        pass


agent = _m.agent
agent.reset = reset
