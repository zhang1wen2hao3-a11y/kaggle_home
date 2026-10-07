"""LightGBM 适配器 —— 装了就自动可用，没装也不影响零依赖路径。

为什么 LightGBM 特别适合这个任务：
    动作空间是**变长候选列表**，天然对应 Learning-to-Rank 问题。
    LGBMRanker（LambdaMART）就是为"一个 query 多个候选、组内排序"设计的，
    直接传 group=每个决策点的候选数即可 —— 省掉 pointer network 那套。

两种用法：
    LGBMModel(task="binary")   训练 V（状态价值，二分类）
    LGBMModel(task="rank")     训练 P（动作排序，LTR）

安装：pip install lightgbm numpy
"""
from __future__ import annotations

from .base import BaseModel, register

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    np = None
    _HAS_NUMPY = False


@register
class LGBMModel(BaseModel):
    name = "lightgbm"

    @classmethod
    def is_available(cls) -> bool:
        """numpy 和 lightgbm 都装齐才算可用。"""
        if not _HAS_NUMPY:
            return False
        import importlib.util
        return importlib.util.find_spec("lightgbm") is not None

    def __init__(self, task: str = "binary", num_leaves: int = 15,
                 learning_rate: float = 0.03, n_estimators: int = 400,
                 min_child_samples: int = 100, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, reg_lambda: float = 1.0,
                 reg_alpha: float = 0.0, max_depth: int = -1,
                 min_split_gain: float = 0.0, **kw):
        """默认参数偏保守。

        实测教训：默认 num_leaves=31 / n_estimators=300 在这个任务上会
        严重过拟合（训练 AUC 0.99、验证 0.73，且验证 log_loss 反而比线性模型差），
        所以这里特意把 num_leaves 和 learning_rate 调小、min_child_samples 调大。
        """
        super().__init__(**kw)
        self.task = task
        self.params = dict(
            num_leaves=num_leaves, learning_rate=learning_rate,
            n_estimators=n_estimators, min_child_samples=min_child_samples,
            subsample=subsample, subsample_freq=1, colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda, reg_alpha=reg_alpha,
            max_depth=max_depth, min_split_gain=min_split_gain,
            verbose=-1,
        )
        self.booster = None
        self.best_iteration: int | None = None

    # ------------------------------------------------------------ 训练
    def fit(self, rows: list[dict], y: list[float], groups: list[int] | None = None,
            eval_set=None, early_stopping_rounds: int = 50, **kw) -> "LGBMModel":
        self._require()
        if not rows:
            raise ValueError("训练数据为空")
        self.feature_keys = sorted({k for r in rows for k in r})
        X = np.asarray(self._vectorize(rows), dtype=np.float32)
        yv = np.asarray(y)

        # LightGBM 用原生 API（sklearn 包装不便于传 early_stopping）
        import lightgbm as lgb

        if self.task == "rank":
            if groups is None:
                raise ValueError("rank 任务必须提供 groups（每个 query 的候选数）")
            params = dict(objective="lambdarank", **self.params)
            train_set = lgb.Dataset(X, label=yv, group=groups)
        elif self.task == "binary":
            params = dict(objective="binary", metric="binary_logloss", **self.params)
            train_set = lgb.Dataset(X, label=yv.astype(int))
        else:
            params = dict(objective="regression", metric="l2", **self.params)
            train_set = lgb.Dataset(X, label=yv)

        callbacks = []
        valid_sets = None
        if eval_set is not None:
            Xv = np.asarray(self._vectorize(list(eval_set[0])), dtype=np.float32)
            yvv = np.asarray(eval_set[1])
            if self.task == "rank":
                valid_sets = [lgb.Dataset(Xv, label=yvv, reference=train_set)]
            else:
                valid_sets = [lgb.Dataset(
                    Xv, label=(yvv.astype(int) if self.task == "binary" else yvv),
                    reference=train_set)]
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

        self.booster = lgb.train(
            params, train_set, num_boost_round=self.params.get("n_estimators", 400),
            valid_sets=valid_sets, callbacks=callbacks)
        self.best_iteration = getattr(self.booster, "best_iteration", None)
        self.fitted = True
        return self

    def predict(self, rows: list[dict]) -> list[float]:
        self._require()
        if not self.fitted:
            raise RuntimeError("模型未训练")
        X = np.asarray(self._vectorize(rows), dtype=np.float32)
        num_iter = self.best_iteration or 0
        p = self.booster.predict(X, num_iteration=num_iter if num_iter > 0 else None)
        return [float(x) for x in p]

    # ------------------------------------------------------------ 持久化
    def to_dict(self) -> dict:
        self._require()
        if self.booster is None:
            return {"feature_keys": self.feature_keys, "model_str": None,
                    "best_iteration": None}
        return {"feature_keys": self.feature_keys,
                "model_str": self.booster.model_to_string(),
                "best_iteration": self.best_iteration}

    def _load_state(self, s: dict) -> None:
        self._require()
        self.feature_keys = s.get("feature_keys", [])
        self.best_iteration = s.get("best_iteration")
        txt = s.get("model_str")
        if txt:
            import lightgbm as lgb
            self.booster = lgb.Booster(model_str=txt)

    # ------------------------------------------------------------ 工具
    def feature_importance(self, k: int = 20) -> list[tuple[str, float]]:
        """特征重要性 —— 用来检查模型到底在学什么（可解释性是选 GBDT 的理由之一）。"""
        if not self.fitted or self.booster is None:
            return []
        try:
            if self.task == "rank":
                imp = self.booster.feature_importance(importance_type="gain")
            else:
                imp = self.booster.feature_importance(importance_type="gain")
        except Exception:
            return []
        pairs = sorted(zip(self.feature_keys, [float(x) for x in imp]),
                       key=lambda kv: -kv[1])
        return pairs[:k]

    @staticmethod
    def _require():
        if not _HAS_NUMPY:
            raise ImportError("LightGBM 后端需要 numpy：pip install numpy lightgbm")
        try:
            import lightgbm  # noqa: F401
        except ImportError as e:
            raise ImportError("未安装 lightgbm：pip install lightgbm") from e
