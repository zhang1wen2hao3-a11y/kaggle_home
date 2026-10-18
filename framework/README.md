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

## 自动 Parquet 缓存（framework/cache.py）

JSONL 有两个硬伤：**体积浪费 6~7 倍**（每个 float 写成文本），
**加载会内存爆炸**（全部变成 Python dict，1.8 GB 文件要 10+ GB 内存 —— 踩过 OOM）。

所以训练前会自动把 JSONL 转成列式 parquet：

```python
from framework import cache
df = cache.load("data/pool_v4_v.jsonl", kind="value")   # 自动建缓存
lf = cache.scan("data/selfplay_v1.jsonl", kind="rank")  # 惰性，可超内存数据
```

**实测收益**（pool_v2_v，11.7 万行）：

| | JSONL | Parquet |
|---|---|---|
| 加载 | 1.41 s | **0.04 s（36×）** |
| 体积 | 172 MB | **2.2 MB** |
| 端到端训练 | 13.1 s | **4.6 s（2.9×）** |
| 峰值内存 | 1.31 GB | **424 MB** |

设计纪律：
1. **按源文件 size + mtime + 版本号判新鲜度** —— 源数据重新生成会自动重建
2. **分块写多个 part**（默认 5 万行/块），绝不一次性把全部行读进内存
3. `kind="value"` 给 V 用（每行一个决策点）；`kind="rank"` 给 P 用
   （**拍平**成每行一个候选动作，`query_id` 即 LTR 的 group）

## ⚠️ 非传递性：头对头赢 ≠ 更强

一个真实踩到的坑：

```
retreat_late（把 RETREAT 挪到 ATTACK 之后）
  vs develop_first : 0.5483  (n=600, 2.4σ 显著)  ← 头对头赢
  vs random_baseline: 0.7600                     ← 绝对强度更差
develop_first
  vs random_baseline: 0.8150 ± 0.0194            ← 绝对强度更好
```

**A 能打赢 B，但 B 对第三方更强。** 卡牌/石头剪刀布式的非传递性是常态。

**结论：任何"改进了 X"的结论，必须同时用统一对手做绝对比较，
不能只看头对头。** 这就是为什么要有固定的基准对手。

## ⚠️ 已证实的负面结论：当前 V 不能用于选动作

加了 P（策略先验）之后做了系统测量（400 局，SE≈0.024）：

| 配置 | vs greedy |
|---|---|
| greedy 自对照 | 0.50 |
| **top_k=1**（只保留 P 的 top-1，等价 greedy） | **0.53** ✅ 管线正确 |
| **top_k=3**（V 在 greedy 认可的 3 个里选） | **0.3725** ❌ 低 5.2σ |

**V 在"接近最优的动作之间"做选择时是有害的**——它系统性地选了更差的。
诊断显示搜索倾向于跳过 `PLAY`（铺场）而选择 `ATTACH`/`ATTACK`（即时动作）。

根因：V 预测的是"这个局面的胜率"，但选动作需要的是
**"这个动作造成的胜率变化"**。而后者（~0.02）远小于 V 的误差（~0.05）。

> **AUC 0.83 的价值函数，在选动作上可能完全没用。** 这是两回事。

修法（都不便宜）：
1. 迭代自对弈 —— 但前提是 search 已经 ≥ greedy（鸡生蛋）
2. 训练 **Q(s,a)** 而不是 V(s) —— 直接学"这动作多好"，不是"这局面多好"
3. 只在特定战术问题（如"能否 KO"）上用搜索，不做通用策略

**当前结论：提交应直接用 greedy（我们最好的 agent）。搜索链路保留作为后续工作。**

## ⚠️ 评估纪律：必须用按局 K 折 CV

**这是踩过的最贵的坑。** 实测：同一数据集、同样特征，**只改验证集切分种子**：

```
seed=0 ACC=0.7454   seed=3 ACC=0.7980
seed=1 AUC=0.8294   seed=4 AUC=0.8434
seed=2 AUC=0.7944   seed=5 AUC=0.7305
标准差 0.041，极差 0.113
```

**噪声 ±0.04，而特征改进通常只有 ±0.005 量级 —— 单次切分根本测不出来。**

根因：只有 550 局，且**同一局内几十~几百条记录高度相关**，
有效样本量是【局数】而不是【记录数】。110 局的验证集太小。

所以：

```bash
# ❌ 单次切分，结果不可信，不要用它下结论
./run.sh train_value.py --data ... --model lightgbm

# ✅ 按局 K 折，看 mean ± std
./run.sh train_value.py --data ... --model lightgbm --cv-folds 5
```

**判据：只差 < 2×std 的差异，一律视为不显著。**
（据此，KO 威胁特征 53→63 维的 +0.0015 属于噪声。）

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

> ⚠️ 命令一律用 `./run.sh` 前缀。它会自动挑到依赖齐全的解释器
> （本机 `base` 是 3.14 且零依赖，`py310` 才有 numpy/lightgbm）。
> 先跑 `./run.sh --check` 看环境诊断。

```bash
# 1. 生成自对弈数据
./run.sh gen_data.py --games 300 --opponent agents/random_baseline.py \
        --out data/sp_vs_random.jsonl --game-offset 0
./run.sh gen_data.py --games 300 --opponent agents/greedy.py \
        --out data/sp_vs_greedy.jsonl --game-offset 100000
cat data/sp_vs_*.jsonl > data/selfplay_v1.jsonl

# 2. 训练价值模型 V
./run.sh train_value.py --data data/selfplay_v1.jsonl --model lightgbm

# 3. 回归测试
./run.sh tests/test_memory.py
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
