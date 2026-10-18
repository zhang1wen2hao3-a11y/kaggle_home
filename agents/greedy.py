"""规则基线 agent（提交候选）。

配置：`safe_v1` + 多招式按伤害择优 + 先手 + 铺满后备 —— 由 8 副卡组的绝对强度实测选出。

## 为什么不是 develop_first（重要）

历史上这里用的是 `develop_first`，它的优先级里 **RETREAT(4) 排在 ATTACK(6) 之前**，
于是只要场上有可撤退的目标，agent 就会**撤退而不攻击**，白白浪费本回合的攻击机会。

这个 bug 长期没被发现，因为所有评估都用 `deck.csv`（sample_official），
而它恰好是唯一不受影响的卡组。换到别的卡组立刻暴露：

    8 副卡组 vs random_baseline 的平均胜率
      develop_first      0.2725      ← 7 副卡组上接近随机
      attack_last_only   0.7612      ← 差 +0.4887

    最极端的 fire_gouging：0.23 -> 0.89

## 也纠正了一个旧的错误结论

旧注释里写过"`retreat_late` 头对头 0.5483 但绝对更差，所以保持 develop_first"。
那个结论的前提是错的 —— `develop_first` 本身就在这些卡组上崩了，
拿它当绝对基准没有意义。正确做法是**跨多副卡组、以 random 为统一参照**比较。

## ⚠️ 与 main.py 必须保持一致

提交用的 `main.py` 里内联了同一张优先级表（它不能 import 本目录的文件）。
两者若不一致，本地测的和线上跑的就是两个策略 —— `tests/test_priority_sync.py` 会检查。
"""
from rulebase import make_agent

agent = make_agent(order="safe_v1", go_first=True, bench_full=True,
                   attack_pick="maxdmg")
