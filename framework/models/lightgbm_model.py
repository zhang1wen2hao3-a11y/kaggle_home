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

    def __init__(self, task: str = "binary", num_leaves: int = 31,
                 learning_rate: float = 0.05, n_estimators: int = 300,
                 min_child_samples: int = 20, subsample: float = 0.9,
                 colsample_bytree: float = 0.9, **kw):
        super().__init__(**kw)
        self.task = task
        self.params = dict(
            num_leaves=num_leaves, learning_rate=learning_rate,
            n_estimators=n_estimators, min_child_samples=min_child_samples,
            subsample=subsample, colsample_bytree=colsample_bytree,
            verbose=-1,
        )
        self.booster = None

    # ------------------------------------------------------------ 训练
    def fit(self, rows: list[dict], y: list[float], groups: list[int] | None = None,
            **kw) -> "LGBMModel":
        self._require()
        if not rows:
            raise ValueError("训练数据为空")
        self.feature_keys = sorted({k for r in rows for k in r})
        X = np.asarray(self._vectorize(rows), dtype=np.float32)
        yv = np.asarray(y)

        if self.task == "rank":
            if groups is None:
                raise ValueError("rank 任务必须提供 groups（每个 query 的候选数）")
            import lightgbm as lgb
            self.booster = lgb.LGBMRanker(objective="lambdarank", **self.params)
            self.booster.fit(X, yv, group=groups)
        elif self.task == "binary":
            import lightgbm as lgb
            self.booster = lgb.LGBMClassifier(**self.params)
            self.booster.fit(X, yv.astype(int))
        else:
            import lightgbm as lgb
            self.booster = lgb.LGBMRegressor(**self.params)
            self.booster.fit(X, yv)

        self.fitted = True
        return self

    def predict(self, rows: list[dict]) -> list[float]:
        self._require()
        if not self.fitted:
            raise RuntimeError("模型未训练")
        X = np.asarray(self._vectorize(rows), dtype=np.float32)
        if self.task == "binary":
            return [float(p) for p in self.booster.predict_proba(X)[:, 1]]
        return [float(p) for p in self.booster.predict(X)]

    # ------------------------------------------------------------ 持久化
    def to_dict(self) -> dict:
        self._require()
        if self.booster is None:
            return {"feature_keys": self.feature_keys, "model_str": None}
        return {"feature_keys": self.feature_keys,
                "model_str": self.booster.booster_.model_to_string()
                if hasattr(self.booster, "booster_") else None}

    def _load_state(self, s: dict) -> None:
        self._require()
        self.feature_keys = s.get("feature_keys", [])
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
            imp = self.booster.feature_importances_
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
