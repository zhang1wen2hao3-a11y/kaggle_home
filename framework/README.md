# 训练框架（framework/）

PTCG AI Battle Challenge 的训练基础设施。目标是**换模型不改其他代码**。

## 模块划分

```
framework/
├── features.py   特征提取（共享层）★ 模型融合的口子
├── memory.py     GameMemory：累积 obs 给不了的增量信息
├── filler.py     determinization 填充器：给 search_begin 提供隐藏信息
├── dataset.py    数据集读写、按局切分
└── models/       模型抽象接口 + 各后端
    ├── base.py            BaseModel + registry 工厂
    ├── linear.py          纯 Python 逻辑回归（零依赖，永远可用）
    └── lightgbm_model.py  LightGBM 适配器（装了才可用）
```

## 数据流

```
                  引擎 obs (公开信息)
                        │
            ┌───────────┼────────────┐
            ↓           ↓            ↓
      GameMemory    features      filler
      (增量日志)   (状态/选项特征)  (隐藏信息)
            │           │            │
            └─────┬─────┘            │
                  ↓                  ↓
            V(s) / P(s,a)      search_begin
                  ↑                  │
                  └──── 标签来自搜索 ─┘
```

## 三条设计纪律

### 1. 特征只用公开信息

`features.py` 里**绝不**使用隐藏信息（对手手牌内容、牌库顺序、里侧奖赏卡）。
线上拿不到，用了模型无法迁移。隐藏信息只能用于**生成训练标签**。

### 2. 按对局切分，不按记录切分

`dataset.split_by_game()` 强制以**对局**为单位划分训练/验证。
同一局的不同步骤高度相关，随机切会让验证集泄漏，把泛化能力估高。

### 3. 视角一致

所有特征和标签都从**当前行动方**的视角计算：
`*_diff` 特征 = 我方 − 对手，V 的标签 = 行动方最终是否获胜。
换个视角，V 的值应该取反 —— 这是一切能自洽的前提。

## 快速上手

```bash
# 1. 生成自对弈数据
python gen_data.py --games 300 --opponent agents/random_baseline.py \
       --out data/sp_vs_random.jsonl --game-offset 0
python gen_data.py --games 300 --opponent agents/greedy.py \
       --out data/sp_vs_greedy.jsonl --game-offset 100000
cat data/sp_vs_*.jsonl > data/selfplay_v1.jsonl

# 2. 训练价值模型 V
python train_value.py --data data/selfplay_v1.jsonl --model linear

# 3. 回归测试
python tests/test_memory.py
```

## 扩展点（"口子"）

### 换模型

`models/base.py` 的 registry 让后端即插即用：

```python
from framework.models import create, available
print(available())                          # ['linear'] 或 ['lightgbm', 'linear']
m = create("lightgbm", task="binary")       # 换一行就换模型
m.fit(X, y)
```

新增后端只需实现 `BaseModel` 的 `fit/predict/to_dict/_load_state`，
再用 `@register` 注册。特征代码和训练脚本完全不用改。

### LTR（动作排序）

`dataset.to_rank_dataset()` 已经产出 `(特征, 相关性标签, group)`，
其中 `group` 是每个决策点的候选数 —— 直接喂给 `LGBMRanker(objective="lambdarank")`。
这是动作空间"变长候选列表"的天然对应形式，省掉 pointer network。

### 对手推断模型

`filler.ModelFiller` 接受任何实现了 `predict_hidden(obs, memory)` 的对象，
未实现时自动回退到 `UniformFiller`，保证流水线不因模型缺失而断。

`gen_data.py --oracle` 会额外记录真实隐藏信息（`visualize_data()`），
可用于**监督学习**训练对手推断模型。

> ⚠️ `visualize_data()` 只在本地可用（它依赖 `Battle.battle_ptr`，
> 而提交的 `main.py` 从不调用 `battle_start`）。是否允许用它生成训练数据，
> 请自行核对比赛规则。

## 关键发现

### 对手强度混合会毁掉 V

实测（600 局自对弈，53 维状态特征，线性模型）：

| 训练数据 | 验证 AUC | 验证 log_loss | 常数基线 log_loss |
|---|---|---|---|
| 只对 `random_baseline` | **0.7202** | 0.6125 | 0.6701 |
| 只对 `greedy` | **0.6968** | 0.6157 | 0.6820 |
| **两者混合** | **0.6336** | **0.6830** | 0.6823 |

混合后 **log_loss 比"永远预测平均值"的常数基线还差**。

原因：同一局面在不同对手下的胜率天差地别（对手是随机策略还是规则策略），
而当前特征**不足以刻画对手强度**，所以混合数据对 V 就是纯噪声。

**这条结论直接约束了对手池的设计**：
训练 V 时对手强度要相对集中，或者必须补上"对手强度/卡组类型"相关特征。

### GameMemory 的跨局污染

Kaggle 很可能复用进程跨对局跑 agent。记忆不重置的话，
第二局会带着第一局的对手信息 —— 而**本地单局测试根本发现不了**。

`memory.py` 用四重信号检测新局（`select is None` / turn 回退 / turn 归零 /
已终局），并配了回归测试 `tests/test_memory.py`（连跑 4 局验证记忆不累加）。
