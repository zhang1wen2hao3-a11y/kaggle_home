"""模型抽象接口 —— 换模型不动其他代码。

约定（所有模型都必须遵守）：
    fit(rows, y)      rows 是 feature dict 的列表（来自 framework.features）
    predict(rows)     -> list[float]
    save(path)/load(path)

这样一来：
    线性 / LightGBM / XGBoost / sklearn / 神经网络 都能即插即用，
    特征层与训练脚本完全不需要改。
"""
from __future__ import annotations

from typing import Any


class BaseModel:
    """所有模型的基类。子类至少要实现 fit / predict。"""

    name = "base"
    task = "binary"          # binary | regression | rank

    @classmethod
    def is_available(cls) -> bool:
        """该后端在当前环境是否真的可用（依赖是否装齐）。

        注意：类被注册 != 能用。lightgbm 后端在没有 numpy/lightgbm 时
        依然会完成注册（导入时惰性检查），所以必须单独判断可用性，
        否则 available() 会误导使用者。
        """
        return True

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.feature_keys: list[str] = []
        self.fitted = False

    # ---------------------------------------------------------- 训练/推理
    def fit(self, rows: list[dict], y: list[float], **kw) -> "BaseModel":
        raise NotImplementedError

    def predict(self, rows: list[dict]) -> list[float]:
        raise NotImplementedError

    def predict_one(self, row: dict) -> float:
        return self.predict([row])[0]

    # ---------------------------------------------------------- 持久化
    def to_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict) -> "BaseModel":
        raise NotImplementedError

    def save(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name": self.name, "task": self.task,
                       "params": self.params, "state": self.to_dict()}, f)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        import json
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        m = cls(**d.get("params", {}))
        m.feature_keys = d["state"].get("feature_keys", [])
        m._load_state(d["state"])
        m.fitted = True
        return m

    def _load_state(self, state: dict) -> None:
        raise NotImplementedError

    # ---------------------------------------------------------- 工具
    def _vectorize(self, rows: list[dict]) -> list[list[float]]:
        """按 self.feature_keys 把特征字典转成定长向量（缺失补 0）。"""
        keys = self.feature_keys
        idx = {k: i for i, k in enumerate(keys)}
        out = []
        for r in rows:
            v = [0.0] * len(keys)
            for k, val in r.items():
                j = idx.get(k)
                if j is not None:
                    v[j] = float(val)
            out.append(v)
        return out


# ---------------------------------------------------------------- 注册表（口子）
_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """把模型类注册进工厂，便于按名字从配置创建。"""
    _REGISTRY[cls.name] = cls
    return cls


def create(name: str, **params) -> BaseModel:
    """按名字创建模型。

    可用后端取决于装了哪些依赖：
        linear    纯 Python，零依赖，永远可用
        lightgbm  需要 pip install lightgbm numpy
        xgboost   需要 pip install xgboost
        sklearn   需要 pip install scikit-learn

    依赖缺失时**立即报错**（而不是等到 fit 才失败），便于尽早发现问题。
    """
    if name not in _REGISTRY:
        _lazy_import_backends()
    if name not in _REGISTRY:
        raise ValueError(f"未知模型: {name}，已注册: {sorted(_REGISTRY)}")
    cls = _REGISTRY[name]
    if not cls.is_available():
        raise ImportError(
            f"后端 {name} 在当前环境不可用（依赖未装齐）。"
            f"当前可用: {available()}。"
            f"如果需要它，请先 pip install（例如 lightgbm 需要 numpy + lightgbm）")
    return cls(**params)


def available() -> list[str]:
    """当前环境**真正可用**的后端（依赖装齐的）。"""
    _lazy_import_backends()
    return sorted(n for n, cls in _REGISTRY.items() if cls.is_available())


def _lazy_import_backends() -> None:
    """惰性导入可选后端；缺依赖时静默跳过，不影响零依赖路径。"""
    for mod in ("linear", "lightgbm_model", "xgboost_model", "sklearn_model"):
        try:
            __import__(f"framework.models.{mod}")
        except Exception:
            pass
