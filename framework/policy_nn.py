"""策略网络：**逐选项打分 + 作用域 softmax**。

为什么不是常规的"固定输出层 + softmax"：本游戏每一步的合法选项数在 1~122 之间变化，
输出层维度不固定。所以改成「对每个候选算一个分数，只在当前可用选项上归一化」：

    score(s, a) = MLP([state_feat(63), option_feat(40)])
    π(a | s)    = softmax_{a ∈ 当前选项} score(s, a)

这样同一套权重能处理任意动作空间，也不需要 padding / mask 的额外工程。

另带一个 value head（只用状态特征），给 REINFORCE 当基线降方差。
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:                                    # pragma: no cover
    raise SystemExit(f"需要 torch：{e}")


class OptionScorer(nn.Module):
    """给每个 (state, option) 打一个分；再带一个 state 的 value 头。"""

    def __init__(self, n_state: int, n_option: int, hidden: int = 128):
        super().__init__()
        self.n_state = n_state
        self.n_option = n_option
        d = n_state + n_option
        self.body = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(n_state, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    # ---- 前向 ------------------------------------------------------------
    def scores(self, state_feat: torch.Tensor, option_feats: torch.Tensor,
               rule_idx: int | None = None, rule_bonus: float = 0.0) -> torch.Tensor:
        """state_feat: (n_state,)  option_feats: (k, n_option) → (k,) 分数。

        `rule_bonus` 是**结构性规则锚**：给"规则会选的那个选项"加一个固定偏置，
        于是网络学的是**相对规则的残差**（"何时该偏离"），而不是从零学打牌。

        为什么必须有它：随机初始化时 MLP≈0，softmax 在 k 个选项上接近均匀，
        实测**偏离规则 76%、胜率 6%** —— 采到的全是垃圾轨迹，什么都学不到。
        加了偏置之后初始策略 ≈ 规则（胜率立刻回到 0.5 量级），
        探索只发生在"规则不太确定"的地方。这与 rollout 的设计同源：
        **默认跟规则走，只有证据足够才偏离。**
        """
        s = state_feat.unsqueeze(0).expand(option_feats.shape[0], -1)
        x = torch.cat([s, option_feats], dim=1)
        sc = self.body(x).squeeze(-1)
        if rule_bonus and rule_idx is not None and 0 <= rule_idx < sc.shape[0]:
            sc = sc.clone()
            sc[rule_idx] = sc[rule_idx] + rule_bonus
        return sc

    def value_of(self, state_feat: torch.Tensor) -> torch.Tensor:
        return self.value(state_feat).squeeze(-1)

    # ---- 便捷：整条轨迹一起算 --------------------------------------------
    def batch_scores(self, states: torch.Tensor, option_list) -> list:
        """states: (B, n_state)；option_list: 长度 B 的列表，每项 (k_i, n_option)。

        返回长度 B 的列表，每项是 (k_i,) 的分数张量。**不做 padding**，
        因为 k_i 差异很大（1~122），padding 会浪费且容易算错 mask。
        """
        return [self.scores(states[i], option_list[i]) for i in range(states.shape[0])]


def masked_softmax(scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """在一维分数上做 softmax（作用域 softmax）。"""
    if temperature != 1.0:
        scores = scores / temperature
    return torch.softmax(scores, dim=0)


def rule_anchor_probs(k: int, rule_idx: int, eps: float, device=None) -> torch.Tensor:
    """规则锚分布 q = (1-ε)·onehot(规则选择) + ε/K。

    ⚠️ 不能直接用 one-hot 当锚：`KL(π‖onehot)` 在 π 有质量落别处时是 +∞。
    留 ε 的均匀底就有限且数值稳定。
    """
    q = torch.full((k,), eps / k, device=device)
    if 0 <= rule_idx < k:
        q[rule_idx] += (1.0 - eps)
    return q


def kl_to_anchor(probs: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(probs ‖ q)，元素级。"""
    return (probs * (torch.log(probs + 1e-12) - torch.log(q + 1e-12))).sum()


def load_weights(model: OptionScorer, path: str, device="cpu") -> OptionScorer:
    sd = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    return model.to(device)


def save_weights(model: OptionScorer, path: str) -> None:
    torch.save(model.state_dict(), path)
