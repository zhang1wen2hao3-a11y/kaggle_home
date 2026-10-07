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

本地对局框架 `local_match.py`；座位与"选先后手权"双重交替，详见 `实验记录.md`。

| 实验 | Agent | 本地胜率 | 对局数 | 对手 |
|---|---|---|---|---|
| exp01_random_selfplay | random_baseline | 0.455 | 400 | random_baseline（框架自检） |
| **exp02_greedy_vs_random** | rulebase | **0.855** | 200 | random_baseline |
| exp03_main_vs_random | main.py（提交版） | 见 `实验记录.md` | 200 | random_baseline |

## 关键发现

### 1. 决策顺序是决定性的：ATTACK 绝不能排第一

PTCG 的**攻击会结束回合**。把 ATTACK 排在优先级首位，agent 就变成"每回合只平A、
从不铺场/进化/充能"，实测 n=200：

| 配置 | vs random_baseline |
|---|---|
| `attack_first`（错误示范） | **0.170** |
| `develop_first`（先铺场后攻击） | **0.855** |
| `attach_first` | 0.790 |
| `no_retreat` | 0.833 |

正确顺序：`PLAY → EVOLVE → ABILITY → ATTACH → RETREAT → DISCARD → ATTACK → END`

### 2. 开局必须铺满后备

`SETUP_BENCH_POKEMON` 要取满 `maxCount`。只上 1 只后备时主力被击倒后 Active 空场
直接判负（败因 3）。这一条把 `attack_first` 从 0.10 拉到 ~0.17，也把
`develop_first` 从 0.467 拉到 0.667（对旧版 greedy 的对照实验）。

### 3. 评估框架本身踩过的坑（值得记住）

- **`agents/random.py` 遮蔽了标准库 `random`**。加了 `sys.path.insert(0, agents/)` 后，
  该文件里的 `import random` 导入了它自己 → 对手开局即崩。而 `play_match` 当时把崩溃
  静默记成"输"，导致**所有配置都被记成 100% 胜率**。修复：文件改名 `random_baseline.py`
  + `sys.path` 改用 `append` + 崩溃显式标记 `crashed` 并单独统计。
- **0 号位有结构优势**：`battle_start(reverse_player=False)` 时，"选先后手"
  （`SelectContext.IS_FIRST`）只发给 0 号位。只交替座位的话，random 自对弈会跑出
  0.62 而非 0.5。修复：座位与 `reverse` 双重交替（`play_match(..., reverse=...)`）。
- 教训：**胜率数字在框架自检通过之前一律不可信**。先用 `random vs random ≈ 0.5`
  验证框架，再解读任何策略对比。

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
