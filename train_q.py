#!/usr/bin/env python3
"""训练 Q(s,a) —— 动作价值，而不是局面价值。

## 为什么需要 Q 而不是 V

选动作时我们在做：
    argmax_a V(s_a)          # s_a = 执行动作 a 之后的局面

问题：s_1 和 s_2 往往**几乎一模一样**（同一个宝可梦上场、手牌少一张、
能量多一个，只差挂在谁身上）。V 要从两个几乎相同的局面里读出胜率差，
而真实差异只有 ~0.02，V 的误差却有 ~0.05 —— 信噪比 < 1。

换成 Q：
    argmax_a Q(s, a)         # 同一局面，不同动作，动作本身在输入里

**关键区别**：Q 不需要模型反推"是哪个动作把我带到这里的"，
动作就是输入的一维特征。它消除的是**结构性噪声**，不是"再多训一点"。

P（LTR）已经证明 `(状态 + 选项)` 这个表示信息充分（模仿 greedy top1=0.9994），
现在只需要把目标从"greedy 选了什么"换成"最终是否获胜"。

## ⚠️ 核心限制：只有被选中的动作有标签

    局面 S 有 7 个候选
      -> 只有 greedy 选的那个有观测结果（其他 6 个是反事实，未知）

所以训练**只能用 chosen==1 的行**（71,685 条，而不是 507,997 条）。
对未选动作，Q 靠选项特征外推（"ATTACH 这类动作在类似局面下通常怎样"）。
这比 V 好（V 连动作都没有），但不是完美的。

## 用法

    ./run.sh train_q.py --data data/pool_rank_v1.jsonl --cv-folds 5
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

META_COLS = {"query_id", "game", "label", "n_opts", "chosen", "opt_index"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--min-child-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/q_v1.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import polars as pl

    from framework import cache
    from framework.models.lightgbm_model import LGBMModel
    from train_value import auc, log_loss

    print(f"读取 {args.data}")
    df = cache.load(args.data, kind="rank")
    feat = [c for c in df.columns if c not in META_COLS]

    # ★ 只用被选中的动作（只有它们有观测结果）
    n_all = df.height
    df = df.filter(pl.col("chosen") > 0)
    print(f"  全部候选行 {n_all:,} -> 有标签的 {df.height:,}"
          f"（只保留 chosen==1；未选动作是反事实，无标签）")
    print(f"  局面 {df['query_id'].n_unique():,} / 局数 {df['game'].n_unique()} / 特征 {len(feat)} 维")

    games = df["game"].unique().to_list()
    random.Random(args.seed).shuffle(games)
    K = max(2, args.cv_folds)
    folds = [games[i::K] for i in range(K)]

    kw = dict(num_leaves=args.num_leaves, learning_rate=args.lr,
              n_estimators=args.n_estimators,
              min_child_samples=args.min_child_samples)

    aucs, lls = [], []
    for i in range(K):
        va = df.filter(pl.col("game").is_in(folds[i]))
        tr = df.filter(~pl.col("game").is_in(folds[i]))
        Xtr = tr.select(feat).to_numpy().astype("float32")
        ytr = tr["label"].to_numpy().astype("float32")
        Xva = va.select(feat).to_numpy().astype("float32")
        yva = va["label"].to_numpy().astype("float32")
        m = LGBMModel(task="binary", **kw)
        m.fit_arrays(Xtr, ytr, feat, eval_set=(Xva, yva))
        pva = m.predict_arrays(Xva)
        a, l = auc(yva, pva), log_loss(yva, pva)
        aucs.append(a); lls.append(l)
        if not args.quiet:
            print(f"  折 {i+1}/{K}: 验证 {len(folds[i])} 局/{va.height:>6,} 行  "
                  f"AUC={a:.4f}  log_loss={l:.4f}")

    print("\n" + "=" * 64)
    print(f"  Q(s,a) 验证 AUC      = {statistics.mean(aucs):.4f} ± {statistics.pstdev(aucs):.4f}")
    print(f"  Q(s,a) 验证 log_loss = {statistics.mean(lls):.4f} ± {statistics.pstdev(lls):.4f}")
    print("=" * 64)
    print("  参考：V(s) 在 3 折 CV 下是 0.79 左右。")
    print("  但 AUC 高不代表能选动作 —— 真正的检验是下面的 1-ply 对战。")

    # 全量训练
    X = df.select(feat).to_numpy().astype("float32")
    y = df["label"].to_numpy().astype("float32")
    m = LGBMModel(task="binary", **kw)
    m.fit_arrays(X, y, feat)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    m.save(args.out)
    print(f"\n全量 Q 模型已保存 -> {args.out}")

    imp = m.feature_importance(15)
    if imp:
        tot = sum(v for _, v in m.feature_importance(999))
        print("\n特征重要性 Top15（注意选项特征占多少）：")
        for k, v in imp:
            tag = " ★动作" if k.startswith("opt_") else ""
            print(f"  {k:<30}{v:>10.1f}  {100*v/tot:>5.1f}%{tag}")

    try:
        import importlib.util as iu
        spec = iu.spec_from_file_location("el", os.path.join(HERE, "exp_log.py"))
        el = iu.module_from_spec(spec); spec.loader.exec_module(el)
        ma = statistics.mean(aucs)
        eid = el.log_experiment(
            name="q_lgb_v1", agent="q:lightgbm",
            eval_scheme=f"cv{K}-auc-q", win_rate=ma, n_games=df["game"].n_unique(),
            changes=f"Q(s,a) {len(feat)}维, {K}折按局CV, 仅用 chosen 行",
            notes=f"AUC={ma:.4f}±{statistics.pstdev(aucs):.4f}")
        print(f"已记录 {eid}（win_rate 列存 CV 平均 AUC）")
    except Exception as e:
        print(f"(未写入实验记录: {e})")


if __name__ == "__main__":
    main()
