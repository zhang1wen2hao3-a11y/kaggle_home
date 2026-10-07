# PTCG AI Battle Challenge 实验工程

Kaggle 比赛 [The Pokémon Company - PTCG AI Battle Challenge Playground](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground)
的智能体开发工程。

- **任务**：写一个 agent，让宝可梦集换式卡牌对战（Pokémon TCG）的三局两胜（Bo3）对局胜率最大化
- **提交形式**：`submission.py`（提交的是代码，不是预测结果）
- **指标**：`cabt_bo3`（三局两胜）
- **截止**：2027-01-08
- **限制**：每天 5 次提交，**总共 2 次计分提交**；队伍上限 5 人

## ⚠️ 数据授权

`data/` 下是比赛官方数据（卡牌数据库 + C++ 引擎），**已在 `.gitignore` 中排除，不会进入仓库**。
首次使用请先执行：

```bash
bash fetch_data.sh
```

## 当前基线

| 项 | 值 |
|---|---|
| **提交入口**（真正上传的） | `main.py`（自包含；+ `cg/` + `deck.csv` 即构成提交） |
| 等价实验版 | `agents/greedy.py` |
| 策略配置 | `develop_first` + `go_first=True` + `bench_full=True` |
| 基准对手 | `agents/random_baseline.py` |
| 等价性验证 | `main.py` vs `agents/greedy.py` ≈ **0.5**（EXP-0004, n=300）→ 同一策略 |

> `main.py` 因提交要求必须自包含（不能 import `agents/`），与 `agents/greedy.py` 是同一策略的
> 两份实现 —— **改动时两边都要改**，并用 `evaluate.py main.py vs agents/greedy.py` 回归验证。

| 实验 | Agent | 本地胜率 | 对局数 | 对手 |
|---|---|---|---|---|
| exp01_random_selfplay | random_baseline | 0.4475 | 400 | random_baseline（框架自检） |
| exp02_greedy_vs_random | greedy（基线） | 0.850 | 200 | random_baseline |
| exp03_main_vs_random | main.py（基线） | 0.805 | 200 | random_baseline |
| **exp04_main_vs_greedy** | main.py | **0.5067** | 300 | greedy（等价性验证） |

> ⚠️ EXP-0002 与 EXP-0003 是**同一策略**的两次测量，0.850 与 0.805 的差异属抽样噪声
> （n=200 时标准误≈0.025），不是版本差异。

> ⚠️ `random_baseline` 作为基准**已无区分度**：所有合理配置都能赢它 80%~88%，
> 继续用它无法区分策略优劣。下一步需换成搜索型对手。

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

## 目录结构

```
.
├── fetch_data.sh          # 重新拉取比赛数据到 data/
├── exp_log.py             # 实验记录工具：追加 / 查看 / 生成 markdown / 写入 LB
├── experiments.csv        # 实验记录表（唯一事实来源，utf-8-sig 可直接用 Excel 打开）
├── 实验记录.md            # 实验记录的 markdown 版（由 exp_log.py md 生成）
├── data/                  # 比赛数据（gitignore，不入库）
└── agents/                # agent 实现
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
