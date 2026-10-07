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

## 当前成绩

| 实验 | Agent | 本地胜率 | 对局数 | 对手 |
|---|---|---|---|---|
| exp_random_selfplay | random | 0.500 | 100 | random（自检基准） |
| **exp_greedy_v1** | rulebase | **1.000** | 100 | random |
| exp_greedy_selfplay | rulebase | 0.500 | 100 | rulebase（自对弈） |

> 本地对局框架 `local_match.py`，座位每局交换，详见 `实验记录.md`。

## 关键发现

规则策略的**决策顺序**是决定性的，实测（每配置 60 局）：

| 配置 | vs random | vs develop_first |
|---|---|---|
| `attack_first`（攻击优先） | 1.000 | **0.033** |
| `develop_first`（铺场优先） | 1.000 | **0.667** |
| `attach_first` | 1.000 | 0.600 |
| `no_retreat` | 1.000 | 0.467 |

两条结论：

1. **ATTACK 绝不能排第一**。PTCG 的攻击会结束回合，攻击优先等于每回合只平A、从不铺场，
   对 `develop_first` 胜率仅 0.033 —— 比随机还差。必须"先铺场/进化/充能，最后攻击"。
2. **开局必须铺满后备**（`SETUP_BENCH_POKEMON` 取满 `maxCount`）。只上 1 只后备时，
   主力被击倒后 Active 空场直接判负（败因 3）。这一条把 `attack_first` 从 0.10 拉到 1.000。

⚠️ **random 已经不够用作基准了** —— 所有合理配置都能 100% 打赢它，失去区分度。
下一步需要更强的对手（更细的规则策略，或用 `search_begin` 做前向搜索的 agent）。

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
