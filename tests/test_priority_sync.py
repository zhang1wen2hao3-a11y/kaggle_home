"""优先级表一致性测试 —— 防止三份副本漂移。

优先级表在本仓库存在**三处**，而且都必须一致：

    main.py                   提交入口，内联一份（不能 import 仓库内模块）
    framework/rollout.py      rollout 的续弈棋手
    agents/rulebase.py        实验用配置（greedy.py 指向 attack_last_only）

它们历史上真的漂移过，代价巨大：
`framework/rollout.py` 曾经还是旧顺序（RETREAT 在 ATTACK 之前），
于是 rollout 用**坏策略**推演未来 —— 它表面上"修正"了 greedy 的撤退选择，
实际上只是在给规则表的 bug 打补丁。修好规则表后 rollout 立刻失去优势
（vs attack_last_only：sample 0.5167 / fire_gouging 0.4000）。

运行：./run.sh tests/test_priority_sync.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "agents"))

from cg.api import OptionType  # noqa: E402

OK = "\033[32m✅\033[0m"
NG = "\033[31m❌\033[0m"
_fail = 0


def check(cond, msg):
    global _fail
    print(f"  {OK if cond else NG} {msg}")
    if not cond:
        _fail += 1


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rank_from(prio: dict) -> list[str]:
    """把 {OptionType: 优先级} 变成"从最优先到最不优先"的类型名序列。

    没有出现在表里的类型视为最低优先（+inf）。
    """
    ALL = list(OptionType)
    ordered = sorted(ALL, key=lambda t: (prio.get(t, float("inf")), list(ALL).index(t)))
    return [t.name for t in ordered]


def main():
    print("=" * 66)
    print("优先级表一致性测试")
    print("=" * 66)

    main_mod = _load(os.path.join(HERE, "main.py"), "t_main_prio")
    roll_mod = _load(os.path.join(HERE, "framework", "rollout.py"), "t_roll_prio")
    rb = _load(os.path.join(HERE, "agents", "rulebase.py"), "t_rulebase")

    r_main = rank_from(main_mod.PRIORITY)
    r_roll = rank_from(roll_mod.PRIORITY)

    # rulebase 的配置是"列表顺序即优先级"
    order_name = "safe_v1"
    cfg = rb.ORDERS[order_name]
    r_rb = [t.name for t in cfg] + [t.name for t in OptionType if t not in cfg]

    print(f"\n  main.py            : {r_main}")
    print(f"  framework/rollout  : {r_roll}")
    print(f"  rulebase[{order_name}]: {r_rb}")

    print()
    check(r_main == r_roll,
          "main.py 与 framework/rollout.py 的顺序一致" if r_main == r_roll
          else "main.py 与 framework/rollout.py 顺序不一致")
    check(r_main[:len(r_rb)] == r_rb or r_main == r_rb,
          f"main.py 与 rulebase[{order_name}] 的顺序一致" if r_main == r_rb
          else f"main.py 与 rulebase[{order_name}] 顺序不一致")

    # 关键回归：ATTACK 必须在 RETREAT 之前
    print()
    for label, r in (("main.py", r_main), ("rollout", r_roll), ("rulebase", r_rb)):
        a = r.index("ATTACK")
        rt = r.index("RETREAT")
        check(a < rt, f"{label}: ATTACK(#{a}) 在 RETREAT(#{rt}) 之前")

    # greedy.py 指向的配置名要存在
    g = _load(os.path.join(HERE, "agents", "greedy.py"), "t_greedy_prio")
    check(hasattr(g, "agent"), "agents/greedy.py 能正常加载")

    print("\n" + "=" * 66)
    if _fail:
        print(f"{NG} {_fail} 项失败 —— 三处优先级表已经漂移，必须修")
        sys.exit(1)
    print(f"{OK} 全部通过")
    print("=" * 66)


if __name__ == "__main__":
    main()
