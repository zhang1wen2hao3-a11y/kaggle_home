#!/usr/bin/env python3
"""训练价值模型 V。

V(s) = 从"当前行动方"视角看的胜率估计。它是搜索的前提：
没有 V，前向推演就只能走到底（成本高 10 倍）；有 V 才能在叶节点截断。

为什么第一个训它：
  * **零依赖数据** —— 自对弈终局就是标签，不需要搜索、不需要对手模型
  * 它是"预测对手"和"搜索"的前置件

用法：
    python train_value.py --data data/selfplay.jsonl
    python train_value.py --data data/selfplay.jsonl --model lightgbm
"""
from __future__ import annotations

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from framework.dataset import read_records, split_by_game, to_value_dataset  # noqa: E402
from framework.models import available, create  # noqa: E402


# ---------------------------------------------------------------- 指标（纯 Python）
def log_loss(y, p) -> float:
    s = 0.0
    for yi, pi in zip(y, p):
        pi = min(max(pi, 1e-9), 1 - 1e-9)
        s += -(yi * math.log(pi) + (1 - yi) * math.log(1 - pi))
    return s / max(len(y), 1)


def accuracy(y, p, thr: float = 0.5) -> float:
    return sum(1 for yi, pi in zip(y, p) if (pi >= thr) == (yi >= 0.5)) / max(len(y), 1)


def auc(y, p) -> float:
    """ROC-AUC（只对二分类标签有效，忽略 0.5 的平局标签）。"""
    pairs = [(pi, yi) for yi, pi in zip(y, p) if yi in (0.0, 1.0)]
    pos = [s for s, lab in pairs if lab == 1.0]
    neg = [s for s, lab in pairs if lab == 0.0]
    if not pos or not neg:
        return float("nan")
    # 用排序法算，避免 O(n^2)
    pairs.sort()
    rank_sum, i = 0.0, 0
    n = len(pairs)
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            if pairs[k][1] == 1.0:
                rank_sum += avg_rank
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def brier(y, p) -> float:
    return sum((yi - pi) ** 2 for yi, pi in zip(y, p)) / max(len(y), 1)


def calibration_table(y, p, bins: int = 5) -> list[tuple[str, int, float, float]]:
    """分桶校准表：(区间, 样本数, 预测均值, 实际胜率)。"""
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, pi in enumerate(p) if (lo <= pi < hi) or (b == bins - 1 and pi >= hi)]
        if not idx:
            continue
        pm = sum(p[i] for i in idx) / len(idx)
        ym = sum(y[i] for i in idx) / len(idx)
        rows.append((f"[{lo:.1f},{hi:.1f})", len(idx), pm, ym))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="linear", help=f"可选: {available()}")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400, help="linear 后端用")
    ap.add_argument("--lr", type=float, default=0.5, help="linear 后端用")
    ap.add_argument("--l2", type=float, default=1e-3, help="linear 后端 L2 正则")
    ap.add_argument("--patience", type=int, default=60,
                    help="linear 后端早停耐心值（验证集损失多少轮无改善即停）")
    ap.add_argument("--out", default="models/value_v1.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    print(f"读取 {args.data}")
    records = read_records(args.data)
    if not records:
        raise SystemExit("数据集为空")
    train, val = split_by_game(records, args.val_ratio, args.seed)
    print(f"  总记录 {len(records)}（{len({r['game'] for r in records})} 局）")
    print(f"  训练   {len(train)}（{len({r['game'] for r in train})} 局）")
    print(f"  验证   {len(val)}（{len({r['game'] for r in val})} 局）  ← 按局切分，无泄漏")

    Xtr, ytr = to_value_dataset(train)
    Xva, yva = to_value_dataset(val)
    print(f"  特征维度 {len(Xtr[0]) if Xtr else 0}")
    base = sum(ytr) / max(len(ytr), 1)
    print(f"  训练集正例率 {base:.3f}（基线预测常数即可得到该准确率）")

    print(f"\n训练模型 = {args.model}")
    kw = {}
    if args.model == "linear":
        kw = dict(task="binary", lr=args.lr, epochs=args.epochs, l2=args.l2,
                  verbose=not args.quiet)
    model = create(args.model, **kw)
    if args.model == "linear":
        model.fit(Xtr, ytr, eval_set=(Xva, yva), patience=args.patience)
    else:
        # LightGBM 等后端同样用验证集做早停（GBDT 不做早停几乎必然过拟合）
        model.fit(Xtr, ytr, eval_set=(Xva, yva))

    # ---------------- 评估
    ptr = model.predict(Xtr)
    pva = model.predict(Xva)
    print("\n" + "=" * 62)
    print(f"{'指标':<22}{'训练集':>14}{'验证集':>14}")
    print("-" * 62)
    print(f"{'log_loss':<22}{log_loss(ytr, ptr):>14.4f}{log_loss(yva, pva):>14.4f}")
    print(f"{'accuracy':<22}{accuracy(ytr, ptr):>14.4f}{accuracy(yva, pva):>14.4f}")
    print(f"{'AUC':<22}{auc(ytr, ptr):>14.4f}{auc(yva, pva):>14.4f}")
    print(f"{'brier':<22}{brier(ytr, ptr):>14.4f}{brier(yva, pva):>14.4f}")
    # 常数基线：永远预测训练集正例率
    const = [base] * len(yva)
    print("-" * 62)
    print(f"{'常数基线 log_loss':<22}{'':>14}{log_loss(yva, const):>14.4f}")
    print(f"{'常数基线 accuracy':<22}{'':>14}{accuracy(yva, const):>14.4f}")
    print("=" * 62)

    print("\n验证集校准（预测值 vs 实际胜率，越接近越好）：")
    print(f"  {'区间':<12}{'样本':>7}{'预测均值':>11}{'实际胜率':>11}")
    for name, n, pm, ym in calibration_table(yva, pva):
        print(f"  {name:<12}{n:>7}{pm:>11.3f}{ym:>11.3f}")

    if hasattr(model, "top_features"):
        print("\n权重绝对值 Top15（标准化后可比）：")
        for k, w in model.top_features(15):
            print(f"  {k:<30}{w:+.4f}")
    if hasattr(model, "feature_importance"):
        imp = model.feature_importance(15)
        if imp:
            print("\n特征重要性 Top15：")
            for k, v in imp:
                print(f"  {k:<30}{v:.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    model.save(args.out)
    print(f"\n模型已保存 -> {args.out}")

    # 记入实验记录
    try:
        import importlib.util as iu
        spec = iu.spec_from_file_location("el", os.path.join(HERE, "exp_log.py"))
        el = iu.module_from_spec(spec); spec.loader.exec_module(el)
        exp_id = el.log_experiment(
            name=f"value_{args.model}_v1", agent=f"value:{args.model}",
            eval_scheme=f"val-auc-{args.model}", win_rate=auc(yva, pva),
            n_games=len({r['game'] for r in val}),
            changes=f"V模型 {args.model}: {len(Xtr[0])}维特征, 按局切分",
            notes=f"log_loss={log_loss(yva, pva):.4f} acc={accuracy(yva, pva):.4f} "
                  f"brier={brier(yva, pva):.4f} (win_rate列存的是AUC)")
        print(f"已记录 {exp_id}（win_rate 列存 AUC）")
    except Exception as e:
        print(f"(未写入实验记录: {e})")


if __name__ == "__main__":
    main()
