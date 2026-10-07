"""PTCG AI Battle Challenge —— 训练框架。

模块划分：
  features   特征提取（共享层，所有模型都用它）★ 模型融合的口子
  memory     GameMemory：在 agent 内累积增量日志、推断对手信息
  filler     determinization 填充器：给 search_begin 提供隐藏信息
  models     模型抽象接口 + 各后端实现（线性 / LightGBM / …）

设计原则：
  1. 特征层与模型层解耦 —— 特征输出扁平 dict[str, float]，
     任何模型（线性 / GBDT / 神经网络）都能消费同一份特征。
  2. 依赖可选 —— 不装 numpy/lightgbm 也能跑通全流程（纯 Python 基线）。
  3. 只用公开信息做特征 —— 隐藏信息只能用于【训练标签】，不能进特征，
     否则无法迁移到线上。
"""
__all__ = ["features", "memory", "filler", "dataset", "models"]
__version__ = "0.1.0"
