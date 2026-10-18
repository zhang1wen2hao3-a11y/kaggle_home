#!/usr/bin/env python3
"""训练 P（策略先验）—— 给 MCTS 提供"哪些动作值得探索"。

## 为什么必须有 P

之前的 search agent 是「V + 搜索」，缺了 P。后果：
    MCTS 把所有候选动作一视同仁地用 UCB 探索 —— 它会把 END 和 ATTACH
    平等对待，让 V 去判断。而 V 判不了：两个动作的真实价值差只有 0.02，
    V 的误差却有 0.05。

greedy 的优先级表（PLAY→EVOLVE→ATTACH→…→ATTACK→END）本身编码了大量
领域知识。P 的作用就是**把这份知识注入搜索的先验**，让搜索不去考虑
greedy 认为荒谬的走法。

## 训练目标：Learning to Rank

动作空间是**变长候选列表**，天然对应 LTR：
    一个 query（局面） -> 一组候选文档（候选动作） -> 组内排序

LightGBM 的 LambdaMART 就是为此设计的，`group=每局面的候选数` 直接喂。

## 当前标签的局限（必须知道）

标签是"greedy 实际选了什么"（二值：选中=1，其余=0）。
所以 P 的上限就是 **模仿 greedy**，它不会比 greedy 更聪明。
真正的升级要等搜索标签（MCTS 访问分布）——那是后面的事。

## 评估

    top1    : 排序最高的动作 == greedy 的选择（模仿准确率）
    ndcg@k  : 排序质量
    mrr     : 正确动作的平均倒数排名

用法：
    ./run.sh train_policy.py --data data/pool_rank_v1.jsonl --cv-folds 5
"""
from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

META_COLS = {"query_id", "game", "label", "n_opts", "chosen", "opt_index"}


# ---------------------------------------------------------------- 指标
def ndcg_at_k(y_chosen: list[int], scores: list[float], k: int) -> float:
    """单组 NDCG@k（相关性二值：选中=1）。"""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    dcg = sum(y_chosen[i] / math.log2(r + 2) for r, i in enumerate(order))
    ideal = sorted(y_chosen, reverse=True)[:k]
    idcg = sum(v / math.log2(r + 2) for r, v in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_one(y_chosen: list[int], scores: list[float]) -> float:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    for r, i in enumerate(order):
        if y_chosen[i] > 0:
            return 1.0 / (r + 1)
    return 0.0


def evaluate_groups(groups: list[tuple[list[int], list[float]]]) -> dict:
    """groups = [(y_chosen, scores), ...]"""
    if not groups:
        return {}
    top1 = ndcgs = mrrs = 0.0
    n = len(groups)
    nd3 = nd5 = 0.0
    for ys, sc in groups:
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        if ys[order[0]] > 0:
            top1 += 1
        ndcgs += ndcg_at_k(ys, sc, 1)
        nd3 += ndcg_at_k(ys, sc, 3)
        nd5 += ndcg_at_k(ys, sc, 5)
        mrrs += mrr_one(ys, sc)
    return {"top1": top1 / n, "ndcg@1": ndcgs / n, "ndcg@3": nd3 / n,
            "ndcg@5": nd5 / n, "mrr": mrrs / n, "n_groups": n}


# ---------------------------------------------------------------- 数据
def load_rank(path: str):
    """读 rank 数据，返回 (df, feature_cols)。"""
    import polars as pl
    from framework import cache
    df = cache.load(path, kind="rank")
    feat = [c for c in df.columns if c not in META_COLS]
    return df, feat


def build_fold(df, fa, fb, feat, fold_games):
    import polars as pl
    import numpy as np
    va = df.filter(pl.col("game").is_in(fold_games))
    tr = df.filter(~pl.col("game").is_in(fold_games))
    out = {}
    for tag, d in (("tr", tr), ("va", va)):
        X = d.select(feat).to_numpy().astype("float32")
        y = d["chosen"].to_numpy().astype("float32")
        g = d.group_by("query_id", maintain_order=True).len()["len"].to_numpy()
        out[tag] = (X, y, g, d)
    return out


def predict_groups(df, preds, feat_cols=None):
    """把逐行预测按 query_id 聚成 [(y_chosen, scores), ...]。"""
    groups = []
    qids = df["query_id"].to_numpy()
    chosen = df["chosen"].to_numpy()
    cur_q, ys, sc = None, [], []
    for q, c, p in zip(qids, chosen, preds):
        if q != cur_q:
            if cur_q is not None:
                groups.append((ys, sc))
            cur_q, ys, sc = q, [], []
        ys.append(int(c)); sc.append(float(p))
    if cur_q is not None:
        groups.append((ys, sc))
    return groups


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--min-child-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/policy_v1.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    from framework.models.lightgbm_model import LGBMModel

    print(f"读取 {args.data}")
    df, feat = load_rank(args.data)
    nq = df["query_id"].n_unique()
    ng = df["game"].n_unique()
    print(f"  {df.height:,} 行 / {nq:,} 个局面 / {ng} 局 / 特征 {len(feat)} 维")
    print(f"  平均候选数 {df.height/max(nq,1):.1f}")

    games = df["game"].unique().to_list()
    random.Random(args.seed).shuffle(games)
    K = max(2, args.cv_folds)
    folds = [games[i::K] for i in range(K)]

    kw = dict(num_leaves=args.num_leaves, learning_rate=args.lr,
              n_estimators=args.n_estimators,
              min_child_samples=args.min_child_samples)

    allm = []
    for i in range(K):
        d = build_fold(df, 0, 0, feat, folds[i])
        Xtr, ytr, gtr, _ = d["tr"]
        Xva, yva, gva, dva = d["va"]
        m = LGBMModel(task="rank", **kw)
        m.fit_arrays(Xtr, ytr, feat, eval_set=(Xva, yva),
                     groups=gtr, eval_groups=gva)
        pva = m.predict_arrays(Xva)
        met = evaluate_groups(predict_groups(dva, pva))
        allm.append(met)
        if not args.quiet:
            print(f"  折 {i+1}/{K}: {len(folds[i])} 局/{dva.height:>6,} 行  "
                  f"top1={met['top1']:.3f}  ndcg@3={met['ndcg@3']:.3f}  mrr={met['mrr']:.3f}")

    print("\n" + "=" * 64)
    print(f"  {'指标':<10}{'均值':>10}{'标准差':>10}")
    print("-" * 64)
    for k in ("top1", "ndcg@1", "ndcg@3", "ndcg@5", "mrr"):
        vals = [m[k] for m in allm]
        print(f"  {k:<10}{statistics.mean(vals):>10.4f}{statistics.pstdev(vals):>10.4f}")
    print("=" * 64)
    print("  top1 = 排序最高的动作与 greedy 实际选择一致的比例")
    print("  这是『模仿 greedy』的上限指标 —— P 不会比 greedy 更聪明，")
    print("  它的价值在于给搜索提供先验，把荒谬动作（如 END）的探索概率压下去。")

    # 全量训练并保存
    import polars as pl
    X = df.select(feat).to_numpy().astype("float32")
    y = df["chosen"].to_numpy().astype("float32")
    g = df.group_by("query_id", maintain_order=True).len()["len"].to_numpy()
    m = LGBMModel(task="rank", **kw)
    m.fit_arrays(X, y, feat, groups=g)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    m.save(args.out)
    print(f"\n全量模型已保存 -> {args.out}")

    imp = m.feature_importance(15)
    if imp:
        tot = sum(v for _, v in m.feature_importance(999))
        print("\n特征重要性 Top15：")
        for k, v in imp:
            print(f"  {k:<30}{v:>10.1f}  {100*v/tot:>5.1f}%")

    try:
        import importlib.util as iu
        spec = iu.spec_from_file_location("el", os.path.join(HERE, "exp_log.py"))
        el = iu.module_from_spec(spec); spec.loader.exec_module(el)
        t1 = statistics.mean([x["top1"] for x in allm])
        eid = el.log_experiment(
            name="policy_ltr_v1", agent="policy:lambdarank",
            eval_scheme=f"cv{K}-top1", win_rate=t1, n_games=ng,
            changes=f"LTR 模仿 greedy, {len(feat)}维, {K}折按局CV",
            notes=f"top1={t1:.4f} mrr={statistics.mean([x['mrr'] for x in allm]):.4f} "
                  f"ndcg@3={statistics.mean([x['ndcg@3'] for x in allm]):.4f}")
        print(f"已记录 {eid}（win_rate 列存 top1）")
    except Exception as e:
        print(f"(未写入实验记录: {e})")


if __name__ == "__main__":
    main()
