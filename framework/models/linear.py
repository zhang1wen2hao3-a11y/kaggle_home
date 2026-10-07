"""纯 Python 逻辑回归 / 线性回归 —— 零依赖基线。

存在的意义：
    框架必须在任何环境都能跑通（本机连 numpy 都没有）。
    同时它是一个**诚实的基线**：如果 LightGBM 打不过它，说明特征有问题，
    而不是模型不够强。

实现要点：
    * 特征标准化（否则量纲差异会让梯度下降极慢）
    * 全批量梯度下降 + L2 正则
    * 二分类用 log-loss，回归用 MSE
"""
from __future__ import annotations

import math

from .base import BaseModel, register


@register
class LinearModel(BaseModel):
    name = "linear"

    def __init__(self, task: str = "binary", lr: float = 0.5, epochs: int = 300,
                 l2: float = 1e-4, verbose: bool = False, **kw):
        super().__init__(**kw)
        self.task = task
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.verbose = verbose
        self.w: list[float] = []
        self.b: float = 0.0
        self.mean: list[float] = []
        self.std: list[float] = []

    # ------------------------------------------------------------ 训练
    def fit(self, rows: list[dict], y: list[float], eval_set=None,
            patience: int = 0, eval_every: int = 10, **kw) -> "LinearModel":
        """训练。

        Args:
            eval_set: 可选的 (X_rows, y) 验证集。给了就启用早停。
            patience: 连续多少轮验证集没有改善就停（0=不早停）。
        """
        if not rows:
            raise ValueError("训练数据为空")
        self.feature_keys = sorted({k for r in rows for k in r})
        X = self._vectorize(rows)
        n, d = len(X), len(self.feature_keys)

        # 标准化（统计量只用训练集，避免验证集泄漏）
        self.mean = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
        self.std = []
        for j in range(d):
            var = sum((X[i][j] - self.mean[j]) ** 2 for i in range(n)) / n
            self.std.append(math.sqrt(var) or 1.0)
        for i in range(n):
            for j in range(d):
                X[i][j] = (X[i][j] - self.mean[j]) / self.std[j]

        Xv = yv = None
        if eval_set is not None:
            Xv = self._vectorize(list(eval_set[0]))
            for i in range(len(Xv)):
                for j in range(d):
                    Xv[i][j] = (Xv[i][j] - self.mean[j]) / self.std[j]
            yv = list(eval_set[1])

        self.w = [0.0] * d
        self.b = 0.0
        best_loss, best_state, bad = float("inf"), None, 0

        for ep in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            loss = 0.0
            for i in range(n):
                z = self.b + sum(self.w[j] * X[i][j] for j in range(d))
                if self.task == "binary":
                    p = _sigmoid(z)
                    loss += -(y[i] * math.log(p + 1e-12) + (1 - y[i]) * math.log(1 - p + 1e-12))
                    err = p - y[i]
                else:
                    loss += (z - y[i]) ** 2
                    err = 2.0 * (z - y[i])
                gb += err
                for j in range(d):
                    gw[j] += err * X[i][j]
            inv = 1.0 / n
            self.b -= self.lr * gb * inv
            for j in range(d):
                self.w[j] -= self.lr * (gw[j] * inv + self.l2 * self.w[j])
            loss *= inv

            if Xv is not None and (ep % eval_every == 0 or ep == self.epochs - 1):
                vl = self._loss_on(Xv, yv)
                if vl < best_loss - 1e-6:
                    best_loss, bad = vl, 0
                    best_state = (list(self.w), self.b)
                else:
                    bad += 1
                if self.verbose:
                    print(f"    epoch {ep:>4}  train={loss:.4f}  val={vl:.4f}"
                          f"{'  *' if bad == 0 else ''}")
                if patience and bad * eval_every >= patience:
                    if self.verbose:
                        print(f"    早停于 epoch {ep}（验证集 {patience} 轮无改善）")
                    break
            elif self.verbose and (ep % 50 == 0 or ep == self.epochs - 1):
                print(f"    epoch {ep:>4}  loss={loss:.4f}")

        if best_state is not None:
            self.w, self.b = best_state
            if self.verbose:
                print(f"    回滚到最佳验证损失 {best_loss:.4f}")
        self.fitted = True
        return self

    def _loss_on(self, X: list[list[float]], y: list[float]) -> float:
        s = 0.0
        d = len(self.feature_keys)
        for i in range(len(X)):
            z = self.b + sum(self.w[j] * X[i][j] for j in range(d))
            if self.task == "binary":
                p = min(max(_sigmoid(z), 1e-9), 1 - 1e-9)
                s += -(y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p))
            else:
                s += (z - y[i]) ** 2
        return s / max(len(X), 1)

    # ------------------------------------------------------------ 推理
    def predict(self, rows: list[dict]) -> list[float]:
        if not self.fitted:
            raise RuntimeError("模型未训练")
        out = []
        d = len(self.feature_keys)
        for r in rows:
            z = self.b
            for j, k in enumerate(self.feature_keys):
                v = r.get(k)
                if v is None:
                    continue
                z += self.w[j] * ((float(v) - self.mean[j]) / self.std[j])
            out.append(_sigmoid(z) if self.task == "binary" else z)
        _ = d
        return out

    # ------------------------------------------------------------ 持久化
    def to_dict(self) -> dict:
        return {"w": self.w, "b": self.b, "mean": self.mean, "std": self.std,
                "feature_keys": self.feature_keys}

    def _load_state(self, s: dict) -> None:
        self.w = s["w"]; self.b = s["b"]
        self.mean = s["mean"]; self.std = s["std"]
        self.feature_keys = s.get("feature_keys", [])

    def top_features(self, k: int = 15) -> list[tuple[str, float]]:
        """按权重绝对值排序（注意：标准化后的权重才可比）。"""
        pairs = sorted(zip(self.feature_keys, self.w), key=lambda kv: -abs(kv[1]))
        return pairs[:k]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)
