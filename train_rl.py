#!/usr/bin/env python3
"""REINFORCE + 规则锚：在 MAIN 决策上学一个「比手写规则更好的策略」。

## 判据（写在 `下一步.md` §2.12，跑之前定的）

    vs `greedy`（n=600，SE 2.0pp）  > 0.550  ⇔  z ≥ 2.5
    vs `random_baseline`（n=200）     ≥ 0.850
    止损：3 轮（各 ~10 万局）后 vs greedy 仍 ≤0.52 → 停

## 为什么是这个形态

* **逐选项打分 + 作用域 softmax** —— 动作空间可变（1~122）
* **只在 MAIN 决策上学** —— 子决策杠杆实测只有 0.03pp，学了也是白学
* **规则锚** `β·KL(π‖q_rule)` —— 让策略只能**有限度**地离开规则。
  这条是核心：rollout 之所以能赢规则，靠的是"只有显著更好才偏离"这个保险阀；
  Q 之所以输，是因为蒸馏把这个保险阀丢了、直接 argmax。RL 把保险阀写进目标函数。
* **老师是环境本身**（终局 ±1），不是「rollout 打的代理标签」——
  所以误差表现为**方差**（基线 + 锚可压），而不是**系统性错误**。
* **对手固定 greedy** —— 先隔离非平稳性。

## 用法

    ./run.sh train_rl.py --iters 3 --games-per-iter 20000 --workers 8
    ./run.sh train_rl.py --iters 1 --games-per-iter 400 --workers 4   # 冒烟测试
"""
import argparse
import collections
import importlib.util
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from framework.features import (  # noqa: E402
    OPTION_FEATURE_KEYS, STATE_FEATURE_KEYS, option_features, state_features,
)
from framework.rollout import is_main, rule_action  # noqa: E402

N_S, N_O = len(STATE_FEATURE_KEYS), len(OPTION_FEATURE_KEYS)
CKPT = os.path.join(ROOT, "models", "rl_policy.pt")

_W = {}          # 每个 worker 进程的全局


# ============================================================ worker 侧
def _agent_fn(obs):
    """采样用的 agent：MAIN 上按策略 π 采样，其它决策交给规则。"""
    sel = obs.get("select")
    if sel is None:
        return list(_W["deck"])
    opts = sel.get("option") or []
    base = rule_action(sel)
    if not opts or not is_main(sel) or len(base) != 1 or not _W["train"]:
        return base
    import torch
    with torch.no_grad():
        sf = np.array([state_features(obs).get(k, 0.0) for k in STATE_FEATURE_KEYS],
                      dtype=np.float32)
        ofs = np.stack([
            np.array([option_features(obs, i).get(k, 0.0) for k in OPTION_FEATURE_KEYS],
                     dtype=np.float32) for i in range(len(opts))
        ]) if opts else np.zeros((0, N_O), np.float32)
        st = torch.from_numpy(sf)
        sc = _W["model"].scores(st, torch.from_numpy(ofs),
                                rule_idx=int(base[0]), rule_bonus=_W["bonus"])
        probs = torch.softmax(sc / _W["temp"], dim=0)
        if _W.get("greedy"):
            idx = int(torch.argmax(sc).item())      # 与正式判据（policy_agent TEMP=0）一致
        else:
            idx = int(torch.multinomial(probs, 1).item())
        v = float(_W["model"].value_of(st).item())
    _W["traj"].append({
        "state": sf, "options": ofs, "chosen": idx,
        "rule": int(base[0]), "logp": float(torch.log(probs[idx] + 1e-12).item()),
        "value": v, "temp": _W["temp"],
    })
    return [idx]


def _worker_init(weights_path, deck, temp, bonus=4.0):
    os.environ.setdefault("PTCG_PYTHON", sys.executable)
    import torch
    # ⚠️ 必须限制线程：torch 默认每进程开"核数"个线程，10 worker × 12 = 120 线程，
    #    load average 冲到 59（实测），吞吐反而下降。这个网络 103→128→64→1，单线程够用。
    torch.set_num_threads(1)
    from local_match import load_deck
    from framework.policy_nn import OptionScorer, load_weights
    m = OptionScorer(N_S, N_O)
    if weights_path and os.path.exists(weights_path):
        load_weights(m, weights_path)
    m.eval()
    _W["model"] = m
    _W["deck"] = deck or load_deck("deck.csv")
    _W["temp"] = temp
    _W["train"] = True
    _W["bonus"] = bonus
    _W["greedy"] = False
    from evaluate import load_agent
    _W["opp"] = load_agent(os.path.join(ROOT, "agents", "greedy.py"))
    _W["traj"] = []


def _play_one(_i):
    """打一局（对手固定 greedy），返回该局的轨迹 + 终局奖励。"""
    from local_match import play_match
    opp = _W["opp"]                      # 初始化时加载一次，别每局重载
    _W["traj"] = []
    r = play_match(_agent_fn, opp, _W["deck"], _W["deck"], reverse=bool(_i % 2))
    res = r.get("result")
    if res is None or res == 2:
        reward = 0.0
    else:
        reward = 1.0 if res == 0 else -1.0        # 我们是 player0（play_match 的第一个参数）
    return {"traj": _W["traj"], "reward": reward, "steps": r.get("steps"),
            "crashed": bool(r.get("crashed"))}


# ============================================================ 主进程
def run_iteration(model, args, it):
    import torch
    from framework.policy_nn import save_weights
    save_weights(model, CKPT)                     # worker 从文件加载
    deck = [int(x) for x in open(os.path.join(ROOT, "deck.csv")).read().split()
            if x.strip()][:60]
    ctx = mp.get_context("spawn")
    t0 = time.time()
    games = []
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=(CKPT, deck, args.temp, args.rule_bonus)) as pool:
        for i, out in enumerate(pool.imap_unordered(_play_one,
                                                    range(args.games_per_iter),
                                                    chunksize=4)):
            games.append(out)
            if (i + 1) % max(args.games_per_iter // 10, 1) == 0:
                el = time.time() - t0
                wr = np.mean([g["reward"] > 0 for g in games]) if games else 0
                print(f"    采样 {i + 1}/{args.games_per_iter}  {el:.0f}s  "
                      f"胜率(训练对手 greedy)= {wr:.3f}", flush=True)
    n_traj = sum(len(g["traj"]) for g in games)
    print(f"  采样完成：{len(games)} 局 / {n_traj} 个 MAIN 决策 / "
          f"{time.time() - t0:.0f}s；胜率 = "
          f"{np.mean([g['reward'] > 0 for g in games]):.3f}", flush=True)
    if n_traj == 0:
        raise SystemExit("一个 MAIN 决策都没采到 —— 策略网络没被调用？")

    # ---- REINFORCE 更新
    from framework.policy_nn import kl_to_anchor, rule_anchor_probs
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    stats = collections.Counter()
    for epoch in range(args.epochs):
        order = np.random.permutation(len(games))
        for gi in order:
            g = games[gi]
            if not g["traj"]:
                continue
            G = torch.tensor(g["reward"], dtype=torch.float32)
            opt.zero_grad()
            loss_p = loss_v = loss_a = torch.tensor(0.0)
            for tr in g["traj"]:
                st = torch.from_numpy(tr["state"])
                of = torch.from_numpy(tr["options"])
                sc = model.scores(st, of, rule_idx=tr["rule"],
                                  rule_bonus=args.rule_bonus) / tr["temp"]
                probs = torch.softmax(sc, dim=0)
                v = model.value_of(st)
                adv = (G - v).detach()
                logp = torch.log(probs[tr["chosen"]] + 1e-12)
                q = rule_anchor_probs(len(probs), tr["rule"], args.anchor_eps)
                kl = kl_to_anchor(probs, q)
                loss_p = loss_p - adv * logp
                loss_a = loss_a + kl
                loss_v = loss_v + (v - G) ** 2
                stats["偏离规则"] += int(tr["chosen"] != tr["rule"])
                stats["决策数"] += 1
            n = len(g["traj"])
            loss = (loss_p / n) + args.beta * (loss_a / n) + args.value_coef * (loss_v / n)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
    model.eval()
    print(f"  更新完成（{args.epochs} epoch）；偏离规则比例 = "
          f"{stats['偏离规则'] / max(stats['决策数'], 1):.1%}", flush=True)
    # ⚠️ 两处都要存：itN 是存档，CKPT 是 quick_eval / 下一轮 worker 读的那份。
    # 只存 itN 的话，quick_eval 会读到**本轮更新前**的权重（踩过）。
    save_weights(model, CKPT)
    save_weights(model, os.path.join(ROOT, "models", f"rl_policy_it{it}.pt"))


def quick_eval(args, n=100):
    """训练中的快速评估：当前策略 vs greedy（少量局，只看趋势）。"""
    from framework.policy_nn import OptionScorer, load_weights
    import torch
    m = OptionScorer(N_S, N_O)
    load_weights(m, CKPT)
    m.eval()
    deck = [int(x) for x in open(os.path.join(ROOT, "deck.csv")).read().split()
            if x.strip()][:60]
    ctx = mp.get_context("spawn")
    with ctx.Pool(max(args.workers // 2, 2), initializer=_worker_init,
                  initargs=(CKPT, deck, 0.3, args.rule_bonus)) as pool:   # 评估用低温
        _W["greedy"] = True
        outs = pool.map(_play_one, range(n), chunksize=2)
    wr = np.mean([o["reward"] > 0 for o in outs])
    print(f"  [快速评估 vs greedy] n={n}  胜率 = {wr:.3f}  (SE≈{0.5 / n ** 0.5:.3f})",
          flush=True)
    return wr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--games-per-iter", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1.0, help="规则锚强度（KL 系数）")
    ap.add_argument("--anchor-eps", type=float, default=0.1)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--temp", type=float, default=1.0, help="采样温度（<1 更接近贪心）")
    ap.add_argument("--rule-bonus", type=float, default=4.0,
                    help="结构性规则锚：给规则选项的固定偏置。0=关（不推荐，会退化成乱打）")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--eval-games", type=int, default=100)
    args = ap.parse_args()

    import torch
    from framework.policy_nn import OptionScorer
    torch.set_num_threads(max(os.cpu_count() // 2, 2))
    torch.manual_seed(0)
    np.random.seed(0)
    model = OptionScorer(N_S, N_O)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)

    print("=" * 74)
    print(f"REINFORCE + 规则锚：特征 {N_S}+{N_O}=103 维，"
          f"β={args.beta} ε_anchor={args.anchor_eps} temp={args.temp}")
    print(f"判据（下一步.md §2.12）：vs greedy n=600 > 0.550；"
          f"止损：3 轮后 ≤0.52 则停")
    print("=" * 74)
    for it in range(1, args.iters + 1):
        print(f"\n--- 第 {it}/{args.iters} 轮 ---", flush=True)
        run_iteration(model, args, it)
        if it % args.eval_every == 0:
            quick_eval(args, n=args.eval_games)
    print("\n训练结束。最终模型：models/rl_policy_it*.pt")
    print("下一步：用 evaluate.py 跑 n=600 正式判据 ——")
    print("  PTCG_RL_MODEL=models/rl_policy_it3.pt ./run.sh evaluate.py "
          "--agent agents/policy_agent.py --opponent agents/greedy.py --games 600")


if __name__ == "__main__":
    main()
