"""模型层：统一接口 + 各后端实现。

核心是 base.BaseModel 定义的接口，以及 registry 工厂：

    from framework.models import create, available
    print(available())                 # 当前环境可用的后端
    m = create("lightgbm", task="rank")
    m = create("linear", task="binary")

换后端不需要改特征代码和训练脚本 —— 这就是"模型融合的口子"。

后端可用性：
    linear      永远可用（纯 Python，零依赖）
    lightgbm    需 pip install lightgbm numpy
    xgboost     需 pip install xgboost        （未实现，留位）
    sklearn     需 pip install scikit-learn   （未实现，留位）
"""
from .base import BaseModel, create, register, available  # noqa: F401
from . import linear  # noqa: F401  保证零依赖后端总是被注册

__all__ = ["BaseModel", "create", "register", "available"]
