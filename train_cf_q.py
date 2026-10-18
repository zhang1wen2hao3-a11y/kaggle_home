#!/usr/bin/env python3
"""用【反事实】数据训练 Q(s,a) —— 门 2 的判据脚本。

## 和 train_q.py 的区别

`train_q.py` 吃的是旧 rank 数据（每个局面只观测到 1 个动作，只能保留 chosen==1），
那种数据里 P(win|s,a) == P(win|s)，Q 必然退化成 V（实测选项重要性 0.38%~0.59%）。

本脚本吃 `gen_counterfactual.py` 的产物：**同一个局面下有多个候选动作、每个都有
独立的 rollout 胜率标签**。这是唯一能让选项特征获得方差的来源。

## 判据（门 2，决定 PIMC 能不能做）

    选项特征重要性占比    现在 0.59%   →   必须 > 10%
    held-out 最优动作一致率 （随机基线 = 1/k ≈ 33%）→ 应显著更高
    MAE                                    越低越好

用法:
    ./run.sh train_cf_q.py --data data/cf_q_v1.jsonl --cv-folds 5
    ./run.sh train_cf_q.py --data data/cf_q_v1.jsonl --only-spread   # 只用有差异的局面
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_rows(path: str):
    """读 cf 数据，返回 [(game, key, state_feat, [opt_feat...], [label...], [n...])]"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            opts = r.get("options") or []
            labels = r.get("labels") or []
            ns = r.get("ns") or [0] * len(opts)
            if not opts or len(opts) != len(labels):
                continue
            keep = [(o, l, n) for o, l, n in zip(opts, labels, ns) if l is not None]
            if len(keep) < 1:
                continue
            rows.append((
                r["game"],
                (r["game"], r["step"]),
                r["features"],
                [k[0] for k in keep],
                [float(k[1]) for k in keep],
                [int(k[2]) for k in keep],
            ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--num-leaves", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--min-child-samples", type=int, default=100)
    ap.add_argument("--only-spread", action="store_true",
                    help="只用跨动作有差异的局面（更纯的动作信号）")
    ap.add_argument("--drop-prefix", action="append", default=[],
                    help="消融用：丢掉这些特征名/前缀（可重复）。"
                         "例如 --drop-prefix opt_tgt_ 会去掉全部目标状态特征")
    ap.add_argument("--tag", default="", help="输出标签（便于区分消融实验）")
    ap.add_argument("--out", default="models/cf_q_v1.json")
    args = ap.parse_args()

    from framework.models.base import create
    import numpy as np

    path = args.data if os.path.isabs(args.data) else os.path.join(HERE, args.data)
    rows = load_rows(path)
    if not rows:
        raise SystemExit("没有可用数据")

    if args.only_spread:
        rows = [r for r in rows if max(r[4]) - min(r[4]) > 0]
        if not rows:
            raise SystemExit("--only-spread 之后没有数据")

    STATE_KEYS = sorted(rows[0][2].keys())
    OPT_KEYS = sorted(rows[0][3][0].keys())
    if args.drop_prefix:
        def _keep(k):
            return not any(k.startswith(p) for p in args.drop_prefix)
        STATE_KEYS = [k for k in STATE_KEYS if _keep(k)]
        OPT_KEYS = [k for k in OPT_KEYS if _keep(k)]
        print(f"  消融: 丢掉前缀 {args.drop_prefix} -> 剩 {len(STATE_KEYS)}+{len(OPT_KEYS)} 维")
    FEAT_KEYS = STATE_KEYS + OPT_KEYS
    OPT_SET = set(OPT_KEYS)

    games = sorted({r[0] for r in rows})
    print("=" * 72)
    print("反事实 Q(s,a) 训练")
    print("=" * 72)
    print(f"  数据      : {os.path.relpath(path, HERE)}")
    print(f"  局面数    : {len(rows):,}   动作样本: {sum(len(r[3]) for r in rows):,}")
    print(f"  对局数    : {len(games)}")
    print(f"  特征      : {len(STATE_KEYS)} 状态 + {len(OPT_KEYS)} 选项 = {len(FEAT_KEYS)}")
    has_spread = sum(1 for r in rows if max(r[4]) - min(r[4]) > 0)
    print(f"  有差异局面: {has_spread:,} ({has_spread/len(rows):.1%})")
    print("=" * 72, flush=True)

    # ---- 每个 (state, action) 展平成一行
    MAIN_INT = 0        # SelectType.MAIN == 0
    X, y, w, gid, qkey, is_main = [], [], [], [], [], []
    for game, key, sf, ofs, labs, ns in rows:
        svec = [float(sf.get(k, 0.0) or 0.0) for k in STATE_KEYS]
        _m = int(ofs[0].get("opt_select_type", -1)) == MAIN_INT if ofs else False
        for of, lab, n in zip(ofs, labs, ns):
            X.append(svec + [float(of.get(k, 0.0) or 0.0) for k in OPT_KEYS])
            y.append(lab)
            w.append(max(n, 1))
            gid.append(game)
            qkey.append(key)
            is_main.append(_m)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    gid = np.asarray(gid)
    is_main = np.asarray(is_main)
    # qkey 保持 Python list：np.asarray 会把 tuple 拆成二维，取出来的不是可哈希键
    print(f"\n  展平后: {len(y):,} 行")

    # ---- 按局 K 折
    import random
    rng = random.Random(0)
    gs = list(games)
    rng.shuffle(gs)
    folds = [set(gs[i::args.cv_folds]) for i in range(args.cv_folds)]

    mae_list, imp_shares, agree_list, base_agree = [], [], [], []
    grp_agree = {}

    for fi, val_games in enumerate(folds, 1):
        tr = ~np.isin(gid, list(val_games))
        va = ~tr
        if tr.sum() == 0 or va.sum() == 0:
            continue
        m = create("lightgbm", task="regression", num_leaves=args.num_leaves,
                   learning_rate=args.lr, n_estimators=args.n_estimators,
                   min_child_samples=args.min_child_samples, seed=0)
        m.fit_arrays(X[tr], y[tr], FEAT_KEYS, eval_set=(X[va], y[va]))
        pred = m.predict_arrays(X[va])

        mae = float(np.mean(np.abs(pred - y[va])))
        mae_list.append(mae)

        # --- 关键指标：选项特征的重要性占比
        imp = m.feature_importance(k=len(FEAT_KEYS))
        tot = sum(v for _, v in imp) or 1.0
        opt_share = sum(v for k, v in imp if k in OPT_SET) / tot
        imp_shares.append(opt_share)

        # --- held-out 最优动作一致率（只统计有差异的局面）
        va_idx = np.where(va)[0]
        groups = defaultdict(list)
        for j, gi in enumerate(va_idx):
            groups[qkey[gi]].append(j)      # j = 在验证子集里的位置（与 pred 对齐）
        agree = base = n_ok = 0
        grp = {0: [0, 0.0], 1: [0, 0.0]}     # {is_main: [命中数, 总数]}
        for k, idxs in groups.items():
            if len(idxs) < 2:
                continue
            true = [y[va][i] for i in idxs]
            if max(true) - min(true) <= 0:
                continue
            pv = [pred[i] for i in idxs]
            hit = 1 if int(np.argmax(pv)) == int(np.argmax(true)) else 0
            n_ok += 1
            agree += hit
            base += 1.0 / len(idxs)
            g = int(is_main[va_idx[idxs[0]]])
            grp[g][0] += hit
            grp[g][1] += 1
        if n_ok:
            agree_list.append(agree / n_ok)
            base_agree.append(base / n_ok)
            for g in (0, 1):
                if grp[g][1]:
                    grp_agree.setdefault(g, []).append(grp[g][0] / grp[g][1])

        print(f"  折 {fi}: MAE={mae:.4f}  选项重要性={opt_share:.2%}  "
              f"最优动作一致={agree/max(n_ok,1):.1%} (随机 {base/max(n_ok,1):.1%}, n={n_ok})",
              flush=True)

    print("\n" + "=" * 72)
    print("  汇总（按局 K 折）")
    print("-" * 72)
    def line(name, xs, fmt=".4f"):
        if not xs:
            print(f"  {name:22s}  —")
            return
        mu = statistics.mean(xs)
        sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        print(f"  {name:22s}  {mu:{fmt}} ± {sd:{fmt}}   (n={len(xs)} 折)")
    line("MAE", mae_list)
    line("选项特征重要性", imp_shares, ".2%")
    line("最优动作一致率", agree_list, ".2%")
    line("随机基线一致率", base_agree, ".2%")
    if grp_agree.get(1):
        line("  └ MAIN 决策一致率", grp_agree[1], ".2%")
    if grp_agree.get(0):
        line("  └ 非MAIN(子决策)一致率", grp_agree[0], ".2%")

    print("-" * 72)
    if imp_shares:
        share = statistics.mean(imp_shares)
        if share > 0.10:
            print(f"  ✅ 选项重要性 {share:.1%} > 10% —— 评估器能携带动作信息，PIMC 有燃料")
        elif share > 0.03:
            print(f"  ⚠️  选项重要性 {share:.1%}（3%~10%）—— 勉强，建议先做 depth-2 小规模试验")
        else:
            print(f"  ❌ 选项重要性 {share:.1%} < 3% —— 结构上仍不可识别，应停止 ML 方向")
    print("=" * 72)

    # 保存最后一折模型（用于后续 PIMC 试验）
    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    m.save(out)
    print(f"  模型已保存 -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
