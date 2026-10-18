"""规则基线 agent（提交候选）。

配置由实测选出：develop_first + 先手 + 铺满后备。

⚠️ 注意一个反直觉的结果：`retreat_late`（把 RETREAT 挪到 ATTACK 之后）
   在**头对头**上赢 develop_first（0.5483, n=600），但**绝对强度更差**：
       develop_first vs random = 0.8150
       retreat_late  vs random = 0.7600
   即"能打赢某个对手"不等于"更强"（非传递性）。
   所以基线仍用 develop_first —— 评估必须以统一对手做绝对比较。
"""
from rulebase import make_agent

agent = make_agent(order="develop_first", go_first=True, bench_full=True)
