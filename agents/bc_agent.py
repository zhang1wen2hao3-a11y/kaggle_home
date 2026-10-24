"""把 BC 策略当**策略**用：MAIN 决策交给模型，其余交给规则。

## 为什么现在才值得做

之前（EXP-0089~0098）我们测出"BC 学不动、从没达到规则水平"，据此关闭了这条路。
**那个结论是标签错位造成的假象**：回放的 action 记在下一步同座位（见 `framework/replay.py`）。
修正对齐后同一份数据给出（EXP-0106）：

    BC  holdout top-1 52.4% / 语义 58.6%     ← 模型
    手写规则同批决策   36.9% / 语义 43.9%     ← 我们的规则
    随机基线          21.8%

也就是模型**第一次真的比规则更接近强 agent 的选择**。但"一致率更高"不等于"打得更强"，
所以必须在**真实对局**里验：本 agent 用于镜像 A/B（n=2000，四路座位×先手交替）。

## 判据（跑之前定死）

    对 agents/ro_attack_w10（= 线上配置）镜像 2000 局：
      胜率 ≥ 0.53 且 z ≥ 2   → BC 策略有效，值得进一步做混合/续弈
      0.47~0.53              → 噪声，按项目纪律视为"无效果"
      ≤ 0.47                 → BC 当策略更差，这条路（以当前特征）真正关闭

## 安全设计

* **只接管 MAIN 决策**（占 62.5%）；非 MAIN（弃牌/检索/贴能目标…）全部用 `rule_action`，
  这些子决策此前已证明杠杆极小（0.03pp）。
* 任何异常 → 回退规则；模型文件缺失 → 直接当规则用。
* 与 rollout 不同，本 agent **不做任何搜索**，单次前向，速度远快于预算。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from framework.features import (  # noqa: E402
    OPTION_FEATURE_KEYS, STATE_FEATURE_KEYS, option_features, state_features,
)
from framework.plan_features import (  # noqa: E402
    OPTION_PLAN_KEYS, STATE_PLAN_KEYS, build_plan,
    option_plan_features, state_plan_features,
)
from framework.rollout import is_main, rule_action  # noqa: E402

MODEL = os.environ.get("PTCG_BC_MODEL", os.path.join(_ROOT, "models", "bc_policy.pt"))
_cache = {}


def _load():
    if _cache:
        return _cache
    import torch
    from framework.policy_nn import OptionScorer

    ck = torch.load(MODEL, map_location="cpu", weights_only=False)
    n_state, n_opt = int(ck["n_state"]), int(ck["n_opt"])
    sd = ck["state_dict"]
    # 从 state_dict 反推隐层宽度（OptionScorer 的 body 第一层是 Linear(n_state+n_option, hidden)）
    hidden = int(sd["body.0.weight"].shape[0])
    model = OptionScorer(n_state, n_opt, hidden=hidden)
    model.load_state_dict(sd)
    model.eval()
    _cache.update(model=model, torch=torch, ck=ck, hidden=hidden)
    return _cache


def _card_ids(obs, opts):
    """把每个选项解析成 cardId（PLAY/EVOLVE 的隐含手牌区要走 resolve_cid）。"""
    try:
        from train_bc import resolve_cid
    except Exception:
        return [None] * len(opts)
    out = []
    for o in opts:
        try:
            out.append(resolve_cid(obs, o))
        except Exception:
            out.append(None)
    return out


def agent(obs) -> list:
    sel = obs.get("select")
    if sel is None:                                # 交卡组：读 deck.csv
        for p in (os.path.join(_HERE, "deck.csv"), "deck.csv",
                  "/kaggle_simulations/agent/deck.csv"):
            if os.path.exists(p):
                with open(p) as f:
                    return [int(x) for x in f.read().split() if x.strip()][:60]
        raise SystemExit("找不到 deck.csv")

    base = rule_action(sel, obs)
    if not is_main(sel):
        return base                               # 子决策一律交给规则
    try:
        c = _load()
        torch, model, ck = c["torch"], c["model"], c["ck"]
        opts = sel.get("option") or []
        if len(opts) < 2:
            return base
        plan = build_plan(set())
        sp = state_plan_features(obs, plan)
        sf = np.array([state_features(obs).get(k, 0.0) for k in STATE_FEATURE_KEYS] +
                      [sp.get(k, 0.0) for k in STATE_PLAN_KEYS], np.float32)
        rows = []
        for i in range(len(opts)):
            rows.append([option_features(obs, i).get(k, 0.0) for k in OPTION_FEATURE_KEYS] +
                        [option_plan_features(obs, i, plan).get(k, 0.0) for k in OPTION_PLAN_KEYS])
        ofs = np.array(rows, np.float32)
        # 规则特征（无锚时全 0）；只有当训练时用过锚才需要真实值
        if not ck.get("no_anchor", False):
            try:
                from framework.rule_features import OPTION_RULE_KEYS, option_rule_features, rule_context
                rctx = rule_context(obs, sel)
                ofs = np.concatenate([ofs, np.array(
                    [[option_rule_features(obs, i, rctx).get(k, 0.0) for k in OPTION_RULE_KEYS]
                     for i in range(len(opts))], np.float32)], axis=1)
            except Exception:
                ofs = np.concatenate([ofs, np.zeros((len(opts), 10), np.float32)], axis=1)
        # 卡牌 one-hot
        vocab, other = ck["vocab"], int(ck["other"])
        extra = np.zeros((len(opts), other + 1), np.float32)
        for k, cid in enumerate(_card_ids(obs, opts)):
            extra[k, vocab.get(cid, other) if cid is not None else other] = 1.0
        X = np.concatenate([ofs, extra], axis=1)
        X = (X - ck["mu_o"]) / ck["sd_o"]
        st = torch.from_numpy((sf - ck["mu_s"]) / ck["sd_s"])
        with torch.no_grad():
            sc = model.scores(st, torch.from_numpy(X.astype(np.float32)))
        return [int(torch.argmax(sc).item())]
    except Exception:
        return base
