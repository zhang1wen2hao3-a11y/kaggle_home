# PTCG AI Battle Challenge 实验工程

Kaggle 比赛 [The Pokémon Company - PTCG AI Battle Challenge Playground](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground)
的智能体开发工程。

- **任务**：写一个 agent，让宝可梦集换式卡牌对战（Pokémon TCG）的三局两胜（Bo3）对局胜率最大化
- **提交形式**：`.tar.gz` 包（`main.py` + `deck.csv` + `cg/`，`-f <file>` 上传，必须带 `-m` 留言）
  —— 提交的是**代码**，不是预测结果
- **指标**：`cabt_bo3`（三局两胜）
- **截止**：2027-01-08
- **限制**：每天 5 次提交，**总共 2 次计分提交**；队伍上限 5 人

## ⚠️ 数据授权

`data/` 下是比赛官方数据（卡牌数据库 + C++ 引擎），**已在 `.gitignore` 中排除，不会进入仓库**。
首次使用请先执行：

```bash
bash fetch_data.sh
```

## 当前状态（TL;DR）

### 提交物：**前向搜索 rollout + 官方样例卡组**（LB 562.9，榜 276/439）

| 项 | 值 |
|---|---|
| **在窗口里的 2 份** | `dist/ptcg_rollout_sample.tar.gz`（562.9）+ `dist/ptcg_greedy.tar.gz`（357.6） |
| **LB** | **562.9** · 榜中位 664.3 · **排名 276/439** · 榜首 1368 |
| 规则基线（`main.py`） | **0.815 ± 0.019 vs random_baseline**（n=400），本地最强且零依赖 |
| 等价性 | `main.py` vs `agents/greedy.py` = **0.510**（n=300）→ 同一策略 |

> **卡组 × 策略有强交互，不能当成两个可加的杠杆。**
> 2×2 同快照实测（见 [`LB记录.md`](LB记录.md)）：
>
> | | 规则 greedy | rollout | 加搜索的收益 |
> |---|---:|---:|---:|
> | `sample_official` | 357.6 | **562.9** | **+205.3** |
> | `grass_heracross` | 395.5 | 388.9 | **−6.6** |
>
> 加搜索在官方样例卡组上值 +205，在我们自造卡组上是 0。
> 所以**必须一起调**，而且本地固定策略下的卡组排名对线上没有预测力。

> `main.py` 因提交要求必须自包含（不能 import `agents/`），与 `agents/greedy.py`
> 是同一策略的两份实现 —— **改动时两边都要改**，并用
> `./run.sh evaluate.py --agent main.py --opponent agents/greedy.py` 回归（应≈0.5）。

> ★ **窗口里现在有一个免费实验槽**：下一份提交会顶掉最老的 `greedy`（357.6），
> 而队伍分取 max 已经是 562.9 —— **新提交再差也不会让分数掉**。用它去试最贵的那个假设。

### 已建成的组件

| 组件 | 状态 | 结论 |
|---|---|---|
| 规则策略（greedy） | ✅ | 本地最强，零依赖，留在窗口里兜底 |
| 对局/评估框架 | ✅ | 座位+reverse 双交替、崩溃检测、按局 CV |
| 特征体系 | ✅ | 局面 **63 维**（48 静态 + 15 记忆）+ 选项 **40 维** |
| 卡组池（25 套 archetype） | ✅ | 25 套、600 种 matchup 全覆盖 |
| **确定性填充器（determinization）** | ✅ 修好 BUG-001 | 剩余张数 93.3% / 99.3% 精确重建 |
| **前向搜索 rollout（规则策略续弈）** | ✅ **真的赢** | `sample_official` 上 LB **+205.3**；但 `grass_heracross` 上 **−6.6** |
| V（价值模型 LightGBM） | ✅ 训出来了 | AUC **0.78 ± 0.04**，❌ 但不能选动作 |
| P（策略先验 LTR） | ✅ 训出来了 | 模仿 greedy **top1 = 0.9994**（天花板 = greedy） |
| **Q(s,a)（反事实 GBDT）** | ✅ 训出来了 | 选项重要性 **34.79%**，❌ 但选动作 **0.465**（Gate 3 失败） |
| MCTS 搜索闭环（V + P） | ✅ 工程跑通 | ❌ **打不过 greedy** |
| GA 卡组选择器 | ⚠️ 可跑 | ❌ 产出的卡组**更差** |
| 对手推断模型 | ❌ 未开始 | 缺 oracle 数据 |

### 有效 vs 无效（全部经过统计检验）

**有效：**
- ATTACK 排第一位是灾难（0.17 vs 0.85），必须"先铺场后攻击"
- 开局铺满后备（否则主力倒下空场判负）
- **前向搜索 rollout 是真收益，但只在"规则打法次优"的卡组上**：
  `sample_official` 上 LB **+205.3**（Gate 1，本地配对 0.5817，n=600，z=+4.06）；
  `grass_heracross` 上 **−6.6**（打平）
- **卡组与策略强交互**：不能把"卡组收益"和"策略收益"相加外推 —— 实测 562.9，不是预测的 678
- **卡组多样性显著提升 V**：单对手 AUC 0.722 → 多样卡组池 0.766（控制记录数后仍 +0.044）
- 修复 `st_turn` 伪特征（回合奇偶编码了"谁在行动"）——它曾导致搜索 96% 选 END
- 修复 BUG-001（填充器把 GameMemory 的日志计数当剩余张数）——99.5% 的局面 determinization 失真

**无效 / 已证伪：**
| 尝试 | 结果 |
|---|---|
| MCTS 搜索（V + 搜索） | **0.39 vs greedy 0.5** ❌ |
| + P 先验裁剪 top_k=3 | **0.3725**（400 局，低 5.2σ）❌ |
| 1-ply（纯 V 选动作） | **0.20** ❌ |
| 启发式 V | **0.25** ❌ |
| 加深搜索（150 sims / depth 10） | **更差**（0.225）❌ |
| **Q(s,a) 反事实 GBDT 选动作** | **0.4650 vs 规则**（n=600，z=**−1.72**）❌ **比规则还差** |
| Q 换 margin 再试 | 0.35 / 0.445 / 0.415 / 0.425（margin 0/0.05/0.1/0.2）❌ 全部 <0.47 |
| GA 优化卡组 | 产出**构造更差**的卡组（无用能量 + 单张填充）❌ |
| `retreat_late` 配置 | 头对头赢但**绝对强度更差**（0.760 vs 0.815）❌ |

> **为什么 V 不能选动作**：V 预测的是「这个局面的胜率」，但选动作需要
> 「这个动作造成的胜率变化」。后者量级 ~0.02，远小于 V 的误差 ~0.05。
> **AUC 0.83 的价值函数，在选动作上可以完全没用。** 详见 `framework/README.md`。

> **那直接学 Q(s,a) 呢？也不行，这条路我们走到了尽头。**
> 用 2,400 局反事实 rollout 标了 69,042 个动作标签（`data/cf_q_final.jsonl`），
> 把选项特征重要性从 **0.59% 拉到 34.79%**（Gate 2 通过）——
> 但**选动作胜率只有 0.4650**，比纯规则还低，换 margin 也救不回来。
> 特征提取能力已经到位，差的是**36% 的 argmax 错误率**（63.60% vs 随机 42.19%）：
> 决策点上一个错误就直接抵消掉全部收益，而终局 playout 近似无偏，
> 所以 95 维的 GBDT 打不过一次真实的 rollout。
> **结论：PIMC / 学习型 Q 到此为止，不要再投入。** 详见 `framework/README.md`。

### 三条评估纪律（都踩过坑才总结出来）

1. **必须用按局 K 折 CV** —— 单次切分的 AUC 标准差达 0.041，而特征改进只有 0.005 量级
2. **头对头赢 ≠ 更强** —— 非传递性是常态，任何改动都要同时看对统一基准的绝对胜率
3. **样本量要够** —— 检测 0.05 的差异需要 ~1600 局；80 局的结论全是噪声

### 下一步（按价值排序）

| 优先 | 做什么 | 理由 |
|---|---|---|
| 1 | **用免费实验槽试"卡组 × 策略"组合** | 窗口里 `greedy`(357.6) 顶掉了也不亏，队伍分守住 562.9。这是**零风险**的，见 `LB记录.md` |
| 2 | 扩大 rollout 覆盖面 | `sample_official` 上 rollout 值 +205，说明**规则打法次优的地方还有空间**；成本 O(决策数²)，可以从候选集/世界数往上推（目前 margin 1.5 全候选 / 8 世界） |
| 3 | 换更好的卡组 | rollout 在 `sample_official` 上已经是 562.9，同策略换卡组仍值 **+174**（vs `grass_heracross`）。但本地排名没有预测力，**只能靠线上免费槽试** |
| 4 | 生成 oracle 数据 → 对手推断 | 让 determinization 从"按牌库比例瞎猜"变成"按对手实际构筑推断"，直接提升 rollout 的信号质量 |
| ~~—~~ | ~~卡组与策略分开调、收益可加~~ | ❌ 已证伪：加搜索在 sample 上 +205、在 heracross 上 0 |
| ~~—~~ | ~~Q(s,a) / PIMC / 迭代自对弈~~ | ❌ **已四条独立证据证伪**，不再投入（见上） |

## 历史基线记录

| 实验 | Agent | 本地胜率 | 对局数 | 对手 |
|---|---|---|---|---|
| exp01_random_selfplay | random_baseline | 0.4475 | 400 | random_baseline（框架自检） |
| exp02_greedy_vs_random | greedy | 0.850 | 200 | random_baseline |
| exp03_main_vs_random | main.py | 0.805 | 200 | random_baseline |
| exp04_main_vs_greedy | main.py | 0.5067 | 300 | greedy（等价性验证） |

> ⚠️ EXP-0002 与 EXP-0003 是**同一策略**的两次测量，0.850 与 0.805 的差异属抽样噪声。
>
> ⚠️ `random_baseline` 作为基准**已无区分度**：所有合理配置都能赢它 80%+。
> 需要更强的基准对手才能区分策略优劣。

## 关键发现

以下数字均为**修正后**的实测结果，指标统一为 `vs random_baseline`，且框架已通过
`random vs random ≈ 0.5` 自检（见 `experiments.csv` / `实验记录.md`）。

### 1. 决策顺序是决定性的：ATTACK 绝不能排第一

PTCG 的**攻击会结束回合**。把 ATTACK 排在优先级首位，agent 就退化成"每回合只平A、
从不铺场/进化/充能"。n=200：

| 配置（go_first=True, bench_full=True） | vs random_baseline |
|---|---|
| `attack_first`（错误示范） | **0.170** |
| `attack_last_only` | 0.780 |
| `attach_first` | 0.790 |
| `no_retreat` | 0.830 |
| **`develop_first`（定稿）** | **0.845** |

正确顺序：`PLAY → EVOLVE → ABILITY → ATTACH → RETREAT → DISCARD → ATTACK → END`

### 2. 开局必须铺满后备

`SETUP_BENCH_POKEMON` 取满 `maxCount`。只上 1 只后备时，主力被击倒后 Active 空场
直接判负（败因 3）。对照实验（n=60，vs random_baseline）：

| 配置 | bench_full=False | bench_full=True |
|---|---|---|
| `attack_first` | 0.100 | **0.167** |
| `develop_first` | 0.733 | **0.867** |

### 3. 修好前两条之后，剩下的顺序差异基本是噪声

两两对决 n=200，四个"合理"配置互有胜负、全部落在 0.4～0.55：

| 对局 | A 胜率 |
|---|---|
| `develop_first` vs `attach_first` | 0.500 |
| `no_retreat` vs `attack_last_only` | 0.540 |
| `develop_first` vs `no_retreat` | 0.440 |
| `develop_first` vs `attack_last_only` | 0.480 |

**结论：不要在这些顺序上继续调参** —— 收益已在噪声内。要提升必须换更强的机制
（见下方"内置前向搜索"）。

### 4. 评估框架本身踩过的坑（比策略更重要）

- **`agents/random.py` 遮蔽了标准库 `random`**。加了 `sys.path.insert(0, agents/)` 后，
  该文件里的 `import random` 导入了它自己 → 对手开局即崩。而 `play_match` 当时把崩溃
  **静默记成"输"**，导致所有配置都被记成 100% 胜率。
  修复：改名 `random_baseline.py` + `sys.path` 改用 `append` + 崩溃显式标记 `crashed` 并单独统计。
- **0 号位有结构优势**：`battle_start(reverse_player=False)` 时"选先后手"
  （`SelectContext.IS_FIRST`）只发给 0 号位。只交替座位时 random 自对弈会跑出 0.62 而非 0.5。
  修复：座位与 `reverse` 双重交替，`play_match(..., reverse=...)`。
- **教训**：胜率数字在框架自检通过之前一律不可信。先跑 `random vs random ≈ 0.5`，
  再解读任何策略对比。`evaluate.py` 现在会在出现崩溃时红字警告并把它排除在胜率之外。

### 5. 当前基线

`agents/greedy.py` 与提交入口 `main.py` 同为 `develop_first + 先手 + 铺满后备`。
直接评估 `main.py`（n=200）：**0.840**，与 `greedy.py` 一致 —— 提交入口可用。

## 内置前向搜索

`cg.api` 提供 `search_begin` / `search_step`，可把对手手牌/牌库/奖赏卡做 **determinization**
后向前推演 —— 这是打不完美信息卡牌的关键工具，也是下一步的主要方向。

⚠️ **读源码得出的机制**（`Api.h:105` / `State.h:302` / `Search.h:109`）：
引擎在序列化 `search_begin_input` 前会调用 `erasePlayerData()`，
把隐藏卡清成 `cardId = 0`；`Search::start` 再专门找这些空槽位，
用**你传入的预测值**填回去。

> 被抹掉的：我方牌库+奖赏、对手牌库+手牌+奖赏、里侧 Active
> 保留的　：弃牌堆、场面、我方手牌、各项数量

**所以没有填充器就无法搜索** —— 这不是可选组件。见 `framework/filler.py`。

## 训练框架

见 [`framework/README.md`](framework/README.md)。核心是**模型无关**。

### ⚠️ 一律用 `./run.sh`，别直接 `python`

本机有两个解释器，训练必须用依赖齐全的那个（装了 numpy/lightgbm/torch）：

| 环境 | 依赖 | 能做什么 |
|---|---|---|
| `base` (3.14.6) | 零依赖 | 只能跑评估/对局 |
| **`py310` (3.10.21)** | 齐全 | ✅ 训练、评估、对局全都行 |

`./run.sh` 会自动挑到合适的解释器，所以所有命令统一写成：

```bash
./run.sh --check                              # 环境诊断（选了哪个解释器、有哪些依赖）
./run.sh evaluate.py --agent main.py --opponent agents/random_baseline.py --games 200

# 生成自对弈数据（--oracle 可额外记录真实隐藏信息）
./run.sh gen_data.py --games 300 --opponent agents/random_baseline.py --out data/sp_vs_random.jsonl

# 训练价值模型 V（--model 可换 lightgbm / linear）
# 训练前会自动把 JSONL 转成 parquet 缓存（加载快 36x、省内存 3x）
./run.sh train_value.py --data data/pool_v4_v.jsonl --model lightgbm

# 手动预建/查看缓存（可选，训练时也会自动建）
./run.sh -m framework.cache data/pool_v4_v.jsonl --kind value --show

# GameMemory 跨局污染回归测试
./run.sh tests/test_memory.py
```

需要指定别的解释器时：`PTCG_PYTHON=/path/to/python ./run.sh ...`

（`make` 目标也都已经走 `./run.sh`：`make check / eval / gen / train / test`）

### ⚠️ 训练框架的第一个实测结论：对手强度混合会毁掉 V

600 局自对弈、53 维状态特征、线性模型：

| 训练数据 | 验证 AUC | 验证 log_loss | 常数基线 |
|---|---|---|---|
| 只对 `random_baseline` | **0.7202** | 0.6125 | 0.6701 |
| 只对 `greedy` | **0.6968** | 0.6157 | 0.6820 |
| **两者混合** | **0.6336** | **0.6830** | 0.6823 |

混合后 **log_loss 比"永远预测平均值"还差**。同一局面在不同对手下胜率天差地别，
而特征不足以刻画对手强度 → 混合数据对 V 就是纯噪声。

**这直接约束对手池设计**：训练 V 时对手强度要相对集中，或必须补上对手强度特征。

## 目录结构

```
# ── 提交物（这个 + deck.csv + cg/ 就是全部；打成 tar.gz 上传）──
├── main.py                # ★ 提交入口（自包含，规则 greedy）
├── main_rollout.py        # ★ 提交入口（前向搜索版，另需 framework/）
├── deck.csv               # ★ 提交卡组（gitignore：比赛方数据）
├── cg/                    # ★ 官方引擎（gitignore，比赛方授权）
├── make_submission.py     # 打包 + 静态/动态自检（exec 模拟，Kaggle 没有 __file__）
│
├── run.sh                 # ★ 统一解释器入口（务必用它跑脚本）
│
# ── 对局与评估 ──
├── local_match.py         # 本地对局框架（座位+reverse 双交替、崩溃检测）
├── evaluate.py            # 批量胜率评估（--workers 走 spawn）
├── sweep.py               # 策略配置扫描
├── exp_log.py             # 实验记录工具
├── experiments.csv        # 实验记录表（唯一事实来源）
├── 实验记录.md            # 实验记录的 markdown 版（exp_log.py md 生成）
├── LB记录.md              # ★ 排行榜读数与"计分窗口"机制（只有 2 次计分提交）
│
# ── 数据生成 ──
├── gen_data.py            # 单对手自对弈数据生成
├── gen_pool_data.py       # 跨卡组池轮转采数据（25 套 / 600 matchup）
├── gen_counterfactual.py  # ★ 反事实 Q 数据：同一局面分叉多个动作 + rollout 打标签
├── make_decks.py          # 按 archetype 构造卡组
│
# ── 训练 ──
├── train_value.py         # 训练 V（价值模型），支持 --cv-folds 按局 CV
├── train_policy.py        # 训练 P（策略先验，LTR / LambdaMART）
├── train_cf_q.py          # 训练 Q(s,a)（反事实），报告选项重要性 / argmax 一致率
├── alpha_loop.py          # 迭代自对弈循环（AlphaZero 式）
├── ga_deck.py             # GA 卡组选择器（实测效果不好，保留作记录）
│
# ── 框架 ──
├── framework/             # 见 framework/README.md
│   ├── features.py        #   特征提取（局面 63 维 + 选项 40 维）
│   ├── memory.py          #   GameMemory（增量日志累积、防跨局污染）
│   ├── filler.py          #   determinization 填充器（BUG-001 已修）
│   ├── rollout.py         # ★ 前向搜索 rollout（规则续弈 + CRN + 离散 margin）
│   ├── search.py          #   MCTS（UCB / 回传 / 先验裁剪）—— 已证伪
│   ├── deck.py            #   卡组表示 / 合法性 / 遗传算子
│   ├── cache.py           #   自动 parquet 缓存
│   ├── dataset.py         #   数据集读写与按局切分
│   └── models/            #   模型抽象接口 + linear / lightgbm 后端
│
├── agents/                # agent 实现
│   ├── rulebase.py        #   参数化规则策略（8 种 order 配置）
│   ├── greedy.py          #   ★ 当前基线
│   ├── rollout_agent.py   #   ★ 前向搜索 agent（env 可调：mode/worlds/margin/cands）
│   ├── q_agent.py         #   Q 推理 agent（实测打不过规则，保留作记录）
│   ├── random_baseline.py #   基准对手（已无区分度）
│   └── search_agent.py    #   搜索 agent（MCTS + P + V，实测打不过 greedy）
│
├── decks/                 # 25 套 archetype 卡组
├── tests/                 # 回归测试（优先级表同步、rollout 差分测试、内存污染）
│
├── data/                  # 比赛数据 + 数据集 + parquet 缓存（gitignore）
└── models/                # 训练产物（gitignore，可用脚本重建）
```

## 使用

```bash
python exp_log.py list                      # 按胜率降序列出实验
python exp_log.py add --name v1 --agent random --win 0.42 --games 200
python exp_log.py lb --exp EXP-0001 --score 350.0
python exp_log.py md                        # 生成 实验记录.md
```

在评估脚本里自动记录：

```python
from exp_log import log_experiment
log_experiment(name="v1_rulebase", agent="rulebase", win_rate=0.42, n_games=200,
               eval_scheme="vs-random-bo3", changes="最小可跑 agent")
```

## 本地自对弈

```bash
pip install kaggle-environments
python -c "from kaggle_environments import make; e=make('cabt'); e.run(['agents/random.py','agents/random.py'])"
```
