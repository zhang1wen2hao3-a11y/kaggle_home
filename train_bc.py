#!/usr/bin/env python3
"""从**强对手的回放**里做行为克隆（BC）——可行性探针。

## 为什么这条路和之前所有学习尝试不同

| 尝试 | 老师是谁 | 强弱 |
|---|---|---|
| Q（失败） | 我们自己的 rollout 打的标签 | 只比规则强 +70 LB |
| RL（失败） | 环境胜负（1 bit / 217 决策） | 无信号 |
| **BC** | **榜首 YumeNeko 1369 / sue124 1277** | **是我们的 3.6 倍** |

这是今天第一次有**真正比我们强的老师**。

## 关键事实（已实测确认）

回放里每个 agent 自己的 `observation` **包含它自己的手牌**，对手的手牌才是 None。
所以每一步都带着"它当时看到了什么 + 它选了什么" —— 这正是 BC 需要的。

## 探针判据（跑之前定）

    留出局的 **top-1 准确率 ≥ 60%**（随机基线 = 每步 1/k，平均约 20%）
      → 通过：策略可学，值得投集成（当 rollout 的续弈策略）
      → 卡在 25~35%：说明这套特征表达不了强策略，不必再谈集成

用法：
    ./run.sh train_bc.py --dir epsoide_field --epochs 8
    ./run.sh train_bc.py --dir epsoide_field --main-only      # 只学 MAIN 决策
"""
import argparse
import collections
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from framework.rollout import _get  # noqa: E402
from framework.features import (  # noqa: E402
    OPTION_FEATURE_KEYS, STATE_FEATURE_KEYS, option_features, state_features,
)
from framework.rollout import is_main, rule_action  # noqa: E402
from framework.plan_features import (  # noqa: E402
    OPTION_PLAN_KEYS, STATE_PLAN_KEYS, build_plan,
    option_plan_features, state_plan_features,
)
from framework.rule_features import (  # noqa: E402
    OPTION_RULE_KEYS, option_rule_features, rule_context,
)

N_PLAN_S, N_PLAN_O = len(STATE_PLAN_KEYS), len(OPTION_PLAN_KEYS)
N_S = len(STATE_FEATURE_KEYS) + N_PLAN_S
N_R = len(OPTION_RULE_KEYS)
N_O = len(OPTION_FEATURE_KEYS) + N_PLAN_O + N_R
CARD_VOCAB_N = 320          # 最高频的 320 张牌做 one-hot，其余进 "other"


def resolve_cid(obs, opt):
    """把 option 的 (area,index) 解析成 cardId。

    ⚠️ 引擎的一个坑：MAIN 里的 `PLAY`(7) / `EVOLVE`(9) 选项**只有 index、没有 area**，
    隐藏的区是**手牌**；`ATTACH`(8) 才带 area=2 + inPlayArea/inPlayIndex（目标是哪只）。
    不兜这一层的话，MAIN 决策的卡牌身份会全部丢失（实测词表只剩 54 张）。
    """
    area, idx = opt.get("area"), opt.get("index")
    if idx is None:
        return None
    if area is None:
        if int(opt.get("type", -1)) in (7, 8, 9):     # PLAY / ATTACH / EVOLVE
            area = 2                                   # 隐含：手牌
        else:
            return None
    cur = obs.get("current") or {}
    pls = cur.get("players") or []
    j = int(cur.get("yourIndex", 0))
    if area == 1:
        arr = (obs.get("select") or {}).get("deck") or []
        return (arr[idx].get("id") if 0 <= idx < len(arr) and arr[idx] else None)
    if len(pls) < 2:
        return None
    arr = {2: pls[j].get("hand"), 3: pls[j].get("discard"), 4: pls[j].get("active"),
           5: pls[j].get("bench"), 6: pls[j].get("prize"),
           12: cur.get("looking")}.get(area) or []
    if 0 <= idx < len(arr) and arr[idx]:
        return arr[idx].get("id")
    return None

OUR_NAME = "zhang1wen2hao3"
CACHE = os.path.join(ROOT, "data",
                     "bc_dataset_noanchor.npz" if os.environ.get("BC_NO_ANCHOR") == "1"
                     else "bc_dataset.npz")


def _extract_file(f, gi, main_only):
    """抽**单个回放**里除我们之外每个 agent 的决策。返回 dict 或 None。

    抽成一个纯函数是为了能用 multiprocessing 并行：单个回放 5 MB，
    特征抽取是纯 Python 计算，单进程 1.6 s/局，3000 局要 80 分钟。
    """
    try:
        d = json.load(open(f))
    except Exception:
        return None
    names = [a.get("Name") or "" for a in ((d.get("info") or {}).get("Agents") or [])]
    if len(names) < 2:
        return None
    # ① 先扫一遍该局每位 agent **见过自己的所有牌**（回放里自己的手牌可见）
    seen = collections.defaultdict(set)
    for st in d.get("steps") or []:
        for j, ent in enumerate(st):
            obs = ent.get("observation") or {}
            cur = obs.get("current") or {}
            pls = cur.get("players") or []
            if j >= len(pls) or not pls[j]:
                continue
            pj = pls[j]
            for zone in ("hand", "discard"):
                for c in (pj.get(zone) or []):
                    if _get(c, "id"):
                        seen[j].add(_get(c, "id"))
            for pk in list(pj.get("active") or []) + list(pj.get("bench") or []):
                if pk and pk.get("id"):
                    seen[j].add(pk.get("id"))
    plans = {j: build_plan(seen[j]) for j in seen}

    out = {"S": [], "O": [], "Y": [], "G": [], "K": [], "AGREE": [],
           "CID": [], "TY": [], "R": [], "stat": collections.Counter()}
    # ⚠️ 必须用 framework.replay.iter_decisions：回放的 action 记在**下一步同座位**，
    #    同步配对会让 34% 的标签越界、其余整体错位一位（EXP-0104 修正）
    from framework.replay import iter_decisions
    for _t, j, obs, act in iter_decisions(d.get("steps") or []):
            sel = obs.get("select")
            if sel is None or not act or not obs.get("current"):
                continue
            if OUR_NAME in (names[j] if j < len(names) else ""):
                continue                      # 跳过我们自己
            plan = plans.get(j) or build_plan([])
            opts = sel.get("option") or []
            if len(opts) < 2 or not (0 <= act[0] < len(opts)):
                continue
            is_m = is_main(sel)
            if main_only and not is_m:
                continue
            try:
                rctx = rule_context(obs, sel)
                sp = state_plan_features(obs, plan)
                sf = np.array([state_features(obs).get(k, 0.0)
                               for k in STATE_FEATURE_KEYS] +
                              [sp.get(k, 0.0) for k in STATE_PLAN_KEYS], np.float32)
                ofs = np.stack([
                    np.array([option_features(obs, i).get(k, 0.0)
                              for k in OPTION_FEATURE_KEYS] +
                             [option_plan_features(obs, i, plan).get(k, 0.0)
                              for k in OPTION_PLAN_KEYS] +
                             [option_rule_features(obs, i, rctx).get(k, 0.0)
                              for k in OPTION_RULE_KEYS], np.float32)
                    for i in range(len(opts))
                ])
            except Exception:
                out["stat"]["特征提取失败"] += 1
                continue
            if not np.isfinite(sf).all() or not np.isfinite(ofs).all():
                out["stat"]["含 NaN"] += 1
                continue
            ids = [resolve_cid(obs, o) for o in opts]
            out["S"].append(sf); out["O"].append(ofs)
            out["Y"].append(int(act[0])); out["CID"].append(ids)
            out["TY"].append([int(o.get("type", -1)) for o in opts])
            out["G"].append(gi); out["K"].append(len(opts))
            # 我们的规则会不会做同样选择（衡量"强 agent 与我们的差异"）
            try:
                rb = rule_action(sel)
                out["AGREE"].append(int(bool(rb) and rb[0] == act[0]))
                # 记下**我们规则**选的是哪个 option：它是判断"模型是否真的学到东西"的基准线。
                # 没有这条线，模型 top-1=32% 看着还行，其实和手写规则一模一样。
                out["R"].append(int(rb[0]) if rb else -1)
            except Exception:
                out["AGREE"].append(0); out["R"].append(-1)
            out["stat"]["MAIN" if is_m else "非MAIN"] += 1
    return out


def _extract_task(args):
    return _extract_file(*args)


def extract(directory, max_games=None, main_only=False, workers=1):
    """directory 支持逗号分隔的多个目录。

    ⚠️ `epsoide/`（我们自己的 316 局）也要抽 —— **我们的对手也是强 agent**，
    每局给出一条完整的对手决策轨迹。不抽它等于白丢 4 倍局数。
    """
    files = []
    for d in str(directory).split(","):
        files += sorted(glob.glob(os.path.join(ROOT, d.strip(), "*.json")))
    files = sorted(set(files))
    if max_games:
        files = files[:max_games]
    print(f"  扫描 {len(files)} 个回放（workers={workers}）…", flush=True)
    tasks = [(f, gi, main_only) for gi, f in enumerate(files)]
    S, O, Y, G, K, AGREE, CID, TY, R = [], [], [], [], [], [], [], [], []
    stats = collections.Counter()
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            it = pool.imap(_extract_task, tasks, chunksize=4)
            for n, r in enumerate(it):
                if r is None:
                    continue
                S += r["S"]; O += r["O"]; Y += r["Y"]; G += r["G"]; K += r["K"]
                AGREE += r["AGREE"]; CID += r["CID"]; TY += r["TY"]; R += r["R"]
                stats.update(r["stat"])
                if (n + 1) % 100 == 0:
                    print(f"    {n + 1}/{len(files)} 局  {time.time() - t0:.0f}s  "
                          f"{len(S)} 决策", flush=True)
    else:
        for t in tasks:
            r = _extract_file(*t)
            if r is None:
                continue
            S += r["S"]; O += r["O"]; Y += r["Y"]; G += r["G"]; K += r["K"]
            AGREE += r["AGREE"]; CID += r["CID"]; TY += r["TY"]; R += r["R"]
            stats.update(r["stat"])
    print(f"  抽到 {len(S)} 个决策：{dict(stats)}  用时 {time.time() - t0:.0f}s")
    print(f"  特征维度：state {N_S}（含 plan {N_PLAN_S}）  "
          f"option {N_O}（含 plan {N_PLAN_O} + rule {N_R}）")
    return S, O, Y, G, K, AGREE, CID, TY, R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="epsoide_field")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--main-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="只用训练集中这一比例的**局**（画学习曲线用）")
    ap.add_argument("--tag", default="", help="日志标记")
    ap.add_argument("--workers", type=int, default=1, help="并行抽取进程数")
    ap.add_argument("--hidden", type=int, default=128, help="MLP 隐层宽度（容量诊断用）")
    ap.add_argument("--label", choices=["teacher", "rule", "deviate"], default="teacher",
                    help="学谁：teacher=强 agent（真 BC）；rule=我们自己的规则（特征充分性诊断）")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(max(os.cpu_count() // 2, 2))
    from framework.policy_nn import OptionScorer
    torch.manual_seed(0); np.random.seed(0)

    if args.reuse_cache and os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        S = list(z["S"]); O = list(z["O"]); Y = list(z["Y"])
        G = list(z["G"]); K = list(z["K"]); A = list(z["AGREE"])
        CID = [list(x) for x in z["CID"]] if "CID" in z.files else [[] for _ in S]
        TY = [list(x) for x in z["TY"]] if "TY" in z.files else None
        R = [int(x) for x in z["R"]] if "R" in z.files else None
        print(f"  复用缓存：{len(S)} 个决策"
              f"{'' if TY is not None else '（老缓存无选项类型，跳过语义指标）'}")
    else:
        print("=" * 74)
        print(f"从强对手回放抽 BC 数据：{args.dir}  main_only={args.main_only}")
        print("=" * 74)
        S, O, Y, G, K, A, CID, TY, R = extract(args.dir, args.max_games,
                                               args.main_only, args.workers)
        if not S:
            raise SystemExit("没抽到数据")
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, S=np.array(S), O=np.array(O, dtype=object),
                            Y=np.array(Y), G=np.array(G), K=np.array(K),
                            AGREE=np.array(A), CID=np.array(CID, dtype=object),
                            TY=np.array(TY, dtype=object), R=np.array(R))
        CID = [list(x) for x in np.array(CID, dtype=object)]
        print(f"  已缓存到 {os.path.relpath(CACHE, ROOT)}")

    G = np.asarray(G)
    games = np.unique(G)
    rng = np.random.RandomState(0); rng.shuffle(games)
    n_ho = max(1, int(len(games) * args.holdout))
    ho = set(games[:n_ho].tolist())
    tr_idx = [i for i in range(len(S)) if G[i] not in ho]
    if args.frac < 1.0:
        # ⚠️ 按**局**采样（不是按决策），否则同一局的决策会跨训练/测试泄漏
        tr_games = sorted({int(G[i]) for i in tr_idx})
        rng2 = np.random.RandomState(1)
        keep = set(rng2.permutation(tr_games)[:max(1, int(len(tr_games) * args.frac))].tolist())
        tr_idx = [i for i in tr_idx if int(G[i]) in keep]
        print(f"  [frac={args.frac}] 只保留 {len(keep)} 局训练数据")
    ho_idx = [i for i in range(len(S)) if G[i] in ho]
    print(f"  按局切分：train {len(tr_idx)} 决策 / {len(games) - n_ho} 局；"
          f"holdout {len(ho_idx)} 决策 / {n_ho} 局")
    print(f"  **强 agent 与我们的规则一致率** = {np.mean(A):.1%}"
          f"（越低说明他们的策略越不同）")
    print(f"  随机基线 top-1 ≈ 1/E[k] = {np.mean([1.0 / k for k in K]):.1%}")

    def _rule_scores(idx):
        """手写规则在同一批决策上的 index 一致率与语义一致率（模型的对照线）。"""
        ok = sem = n = 0
        for i in idx:
            r = R[i] if R is not None else -1
            if r is None or r < 0 or r >= K[i]:
                continue
            n += 1
            ok += int(r == Y[i])
            if TY is not None:
                sem += int((TY[i][r], CID[i][r]) == (TY[i][Y[i]], CID[i][Y[i]]))
        return (ok / max(n, 1), sem / max(n, 1), n)

    if R is not None and TY is not None:
        _ri, _rs, _rn = _rule_scores(ho_idx)
        print(f"  **手写规则基线**（同一 holdout，{_rn} 决策）："
              f"index 一致率 {_ri:.1%}  语义一致率 {_rs:.1%}")

    if args.label == "deviate":
        # 「偏离检测」：把每个决策压成两项 —— 规则的选项 vs 强 agent 的选项，
        # 问特征能不能分辨强 agent 更偏好哪一个。50% = 纯噪声（判据 ≥55%）。
        # ⚠️ 必须配 no-anchor 缓存：锚特征会让模型无脑选规则那项，直接 0%。
        _S, _O, _Y, _K, _G, _CID = [], [], [], [], [], []
        _keep = _ho = _rulewin = 0
        for i in range(len(S)):
            r = R[i] if R is not None else -1
            y = Y[i]
            if r is None or r < 0 or r >= K[i] or r == y:
                continue
            _keep += 1
            pair = [r, y]                       # 固定顺序：0=规则，1=强 agent
            _S.append(S[i]); _O.append(O[i][pair])
            _CID.append([CID[i][r], CID[i][y]])
            _Y.append(1); _K.append(2); _G.append(G[i])
        S, O, Y, K, G, CID = _S, _O, _Y, _K, _G, _CID
        tr_idx = [i for i in range(len(S)) if G[i] not in ho]
        ho_idx = [i for i in range(len(S)) if G[i] in ho]
        print(f"  [label=deviate] {len(S)} 个「规则 vs 强 agent」二选一"
              f"（train {len(tr_idx)} / holdout {len(ho_idx)}）；随机 = 50%")

    if args.label == "rule":
        # 特征充分性诊断：规则是 (obs, sel) 的**确定性函数**，理想情况可学到接近 100%。
        # 如果连自己的规则都学不到，说明缺的是特征，堆数据没用。
        _keep = [i for i in range(len(S)) if R is not None and 0 <= R[i] < K[i]]
        S = [S[i] for i in _keep]; O = [O[i] for i in _keep]
        CID = [CID[i] for i in _keep]; TY = [TY[i] for i in _keep]
        Y = [R[i] for i in _keep]; G = [G[i] for i in _keep]; K = [K[i] for i in _keep]
        A = [A[i] for i in _keep]; R = [R[i] for i in _keep]
        tr_idx = [i for i in range(len(S)) if G[i] not in ho]
        ho_idx = [i for i in range(len(S)) if G[i] in ho]
        print(f"  [label=rule] 改为模仿手写规则：{len(S)} 个决策"
              f"（train {len(tr_idx)} / holdout {len(ho_idx)}）")

    # ---- 卡牌词表（按出现频次取前 CARD_VOCAB_N 张）
    freq = collections.Counter()
    for ids in CID:
        for c in ids:
            if c is not None:
                freq[c] += 1
    vocab = {c: i for i, (c, _) in enumerate(freq.most_common(CARD_VOCAB_N))}
    other = len(vocab)
    n_eff = N_O + other + 1
    print(f"  卡牌词表：{len(vocab)} 张 + other ⇒ 选项特征 {N_O} → {n_eff}")

    def with_cards(i):
        """把第 i 个决策的选项特征拼上卡牌 one-hot。"""
        ofs = O[i]; ids = CID[i]
        extra = np.zeros((ofs.shape[0], other + 1), np.float32)
        for k, c in enumerate(ids):
            extra[k, vocab.get(c, other) if c is not None else other] = 1.0
        return np.concatenate([ofs, extra], axis=1)

    # ---- 特征标准化（用训练集的统计量）
    # 不加这一步会出事：特征里有 HP≈350 这种大数、也有 0/1 布尔，MLP 初始 logit 会很大，
    # 实测 loss 反而 **高于均匀预测**（2.07 > ln(k)≈1.96），也就是"自信地预测错"。
    Xs = np.stack([S[i] for i in tr_idx])
    mu_s, sd_s = Xs.mean(0), Xs.std(0) + 1e-6
    ao = np.concatenate([with_cards(i) for i in tr_idx], axis=0)
    mu_o, sd_o = ao.mean(0), ao.std(0) + 1e-6
    print(f"  特征标准化完成（option 维度 {ao.shape[1]}）")
    _wc_raw = with_cards

    def with_cards(i):                                  # noqa: F811
        ofs = _wc_raw(i)
        return (ofs - mu_o) / sd_o

    def state_of(i):
        return (S[i] - mu_s) / sd_s

    mean_k = float(np.mean([K[i] for i in tr_idx]))
    import math
    print(f"  均匀预测的 loss 参考值 = ln(E[k]) ≈ {math.log(mean_k):.3f}")

    model = OptionScorer(N_S, n_eff, hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    def batches(idx, bs=64):
        order = np.random.permutation(len(idx))
        for i in range(0, len(order), bs):
            yield [idx[j] for j in order[i:i + bs]]

    hc = ht = hc3 = hs = hs3 = 0
    for ep in range(args.epochs):
        model.train()
        tot = corr = 0; loss_sum = 0.0
        for chunk in batches(tr_idx):
            opt.zero_grad()
            losses = []
            for i in chunk:
                st = torch.from_numpy(state_of(i)); of = torch.from_numpy(with_cards(i))
                sc = model.scores(st, of)                     # 不含规则锚
                losses.append(torch.nn.functional.cross_entropy(
                    sc.unsqueeze(0), torch.tensor([Y[i]])))
                corr += int(torch.argmax(sc).item() == Y[i]); tot += 1
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss) * len(chunk)
        # holdout
        # ⚠️ top-1（严格 index 相同）会低估策略质量：手里两张同名卡拉哪张都是对的动作。
        # 所以同时报"语义 top-1"（type+cardId 相同）和"语义 top-3"。
        model.eval()
        hc = ht = hc3 = hs = hs3 = 0
        with torch.no_grad():
            for i in ho_idx:
                st = torch.from_numpy(state_of(i)); of = torch.from_numpy(with_cards(i))
                sc = model.scores(st, of)
                p = int(torch.argmax(sc).item())
                hc += int(p == Y[i]); ht += 1
                top3 = torch.topk(sc, min(3, sc.shape[0])).indices.tolist()
                hc3 += int(Y[i] in top3)
                if TY is not None:
                    sy = (TY[i][Y[i]], CID[i][Y[i]])
                    hs += int((TY[i][p], CID[i][p]) == sy)
                    hs3 += int(any((TY[i][t], CID[i][t]) == sy for t in top3))
        sem = (f"  语义 top-1={hs / max(ht, 1):.1%} 语义 top-3={hs3 / max(ht, 1):.1%}"
               if TY is not None else "")
        print(f"  epoch {ep + 1}/{args.epochs}  loss={loss_sum / max(tot, 1):.4f}  "
              f"train top-1={corr / max(tot, 1):.1%}  **holdout top-1={hc / max(ht, 1):.1%}**"
              f"  top-3={hc3 / max(ht, 1):.1%}{sem}", flush=True)

    # ⚠️ 只存 state_dict 是不够的：推理时必须复现同一套**特征标准化参数**与**卡牌词表**，
    #    否则同一个模型在 agent 里会得到完全不同的分数（这类不一致是静默的）。
    torch.save({
        "state_dict": model.state_dict(),
        "mu_s": mu_s, "sd_s": sd_s, "mu_o": mu_o, "sd_o": sd_o,
        "vocab": vocab, "other": int(other), "n_state": int(N_S), "n_opt": int(n_eff),
        "label": args.label, "no_anchor": os.environ.get("BC_NO_ANCHOR") == "1",
        "cands": "all-options",
    }, os.path.join(ROOT, "models", "bc_policy.pt"))
    if args.epochs > 0:
        print(f"\n最终 holdout top-1 = {hc / max(ht, 1):.1%}  top-3 = {hc3 / max(ht, 1):.1%}"
              + (f"  语义 top-1 = {hs / max(ht, 1):.1%}  语义 top-3 = {hs3 / max(ht, 1):.1%}"
                 if TY is not None and ht else "")
              + (f"  ｜ 手写规则：index {_ri:.1%} 语义 {_rs:.1%}"
                 if R is not None and TY is not None else "")
              + f"（随机 {np.mean([1.0 / k for k in K]):.1%}；判据 ≥60%）")
    else:
        print("\n(--epochs 0：只重建缓存，不训练)")


if __name__ == "__main__":
    main()
