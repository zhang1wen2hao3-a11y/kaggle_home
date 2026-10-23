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

### 提交物：**前向搜索 rollout + 官方样例卡组**（LB ~460，榜 276/439）

| 项 | 值 |
|---|---|
| **在窗口里的 2 份** | `dist/ptcg_rollout_sample.tar.gz`（**460.5，仍在下降**）+ `dist/ptcg_greedy.tar.gz`（346.4） |
| **LB** | **460.5** · 榜中位 664.3 · **排名 276/439** · 榜首 1368 |

> ⚠️ **LB 读数是会向下漂的。** 同一份提交 2 小时前读到 562.9，现在 460.5。
> **只有同一时刻、同一状态（都活跃 / 都冻结）的读数才能相减。**
> 详见 [`LB记录.md`](LB记录.md) —— 这份文档已经因为违反这条被修正过三次。
| 规则基线（`main.py`） | **0.815 ± 0.019 vs random_baseline**（n=400），本地最强且零依赖 |
| 等价性 | `main.py` vs `agents/greedy.py` = **0.510**（n=300）→ 同一策略 |

> **卡组 × 策略有强交互，不能当成两个可加的杠杆。**
> 2×2 同快照实测（见 [`LB记录.md`](LB记录.md)）：
>
> | | 规则 greedy | rollout | 加搜索的收益 |
> |---|---:|---:|---:|
> | `sample_official` | 346.4 | **460.5** | **+114.1** |
> | `grass_heracross` | 395.5（冻结） | 388.9（冻结） | **−6.6** |
>
> 加搜索在官方样例卡组上值 **+114**（不是之前说的 +205 —— 那个数是拿还在下降的
> 早期读数算的）。`grass_heracross` 那两行都是**冻结值**，可比；但它和
> `sample_official` 的**活跃值**不能比，所以"换卡组值多少"这个问题**目前还没有答案**。

> `main.py` 因提交要求必须自包含（不能 import `agents/`），与 `agents/greedy.py`
> 是同一策略的两份实现 —— **改动时两边都要改**，并用
> `./run.sh evaluate.py --agent main.py --opponent agents/greedy.py` 回归（应≈0.5）。

> ★ **窗口里现在有一个免费实验槽**：下一份提交会顶掉最老的 `greedy`（357.6），
> 而队伍分取 max 已经是 460.5 —— **新提交再差也不会让分数掉**。用它去试最贵的那个假设。

### 已建成的组件

| 组件 | 状态 | 结论 |
|---|---|---|
| 规则策略（greedy） | ✅ | 本地最强，零依赖，留在窗口里兜底 |
| 对局/评估框架 | ✅ | 座位+reverse 双交替、崩溃检测、按局 CV |
| 特征体系 | ✅ | 局面 **63 维**（48 静态 + 15 记忆）+ 选项 **40 维** |
| 卡组池（25 套 archetype） | ✅ | 25 套、600 种 matchup 全覆盖 |
| **确定性填充器（determinization）** | ✅ 修好 BUG-001 | 剩余张数 93.3% / 99.3% 精确重建 |
| **前向搜索 rollout（规则策略续弈）** | ✅ **真的赢** | 同卡组同快照 LB **+114.1**（`sample_official`，还在收敛） |
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
  `sample_official` 上同快照 LB **+114.1**（Gate 1，本地配对 0.5817，n=600，z=+4.06）
- **LB 读数是会向下漂的 Elo**：同一份提交 2 小时内从 562.9 掉到 460.5。跨时刻的差值一律作废
- **不能把"卡组收益"和"策略收益"相加外推** —— 那次预测 678，实测 460 且仍在下降
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
| **rollout worlds 10 → 40** | **0.4833**（n=300，SE 0.029，z=−0.58）❌ 纯噪声 |
| **rollout MODE attack → all** | **0.4867**（n=300，SE 0.029，z=−0.46）❌ 纯噪声 |
| **determinization 用真实构筑** | **0.5033**（n=300，z=+0.11）❌ 隐藏信息猜对了也**一分不赚** |
| **margin 扫描 0.5 → 2.5** | 两端显著更差（−2.42σ / −2.66σ），中间 0.75~2.0 是**平台** ❌ 当前值已在平台上 |
| **子决策目标选择（`smart_sub`）** | **0.5100**（n=300，z=+0.35）❌ 噪声 |
| **续弈（playout）策略换 develop_first** | **0.4833**（n=300，z=−0.58）❌ 噪声 |
| **续弈策略换 evolve_first** | 初筛 0.5533（z=+1.85）→ **复测 n=2000 = 0.5035**（z=+0.31）❌ 赢家诅咒 |
| **续弈策略换 attack_first（正对照）** | 0.4567（z=−1.50）方向对但不显著 |
| **depth-2（对手回应取最坏）** | **0.5133**（n=600，z=+0.65）❌ 噪声；成本 0.61→1.20 s/局 |
| **换卡组（24 套挑战 rollout+sample）** | 22 套明显更弱；**前 2 名复测后落回 0.5** ❌ 见下 |

> ⚠️ 最后两行是**修复 BUG-002 之后**重测的。修复前的同一组 A/B 是 0.08 s/局、
> 两个算量差 4 倍的配置耗时完全相同 —— 那是 bug 把 rollout 静默降级成规则了。
> **rollout 的配置旋钮（世界数 / 覆盖面 / determinization 精度）已经全部饱和**，
> 不要再调。连"把隐藏卡猜对"都是 0 —— 详见下面的 `CompositionOracleFiller` 实验。

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

### 日志：`evaluate.py` 每次跑完都会落盘

`logs/ab_log.tsv` 是一行一次的汇总（时间 / agent / 对手 / 卡组 / 方案 / 正负 / 胜率 / 耗时），
**无条件追加**，不需要 `| tee`，也不怕后台任务里丢掉。

> 为什么要这样：以前结果只留在终端，后台跑的扫描跑完就没了（`logs/` 还是 gitignore 的）。
> 实验记录不能依赖调用者的自觉。

两个地方的分工：

| 位置 | 内容 | 进仓库 |
|---|---|---|
| `logs/ab_log.tsv` | **所有**评估的原始汇总，防丢 | ❌ gitignore |
| `experiments.csv` | 事后挑出**有意义**的那些，结构化记录 | ✅ 唯一事实来源 |

```bash
# 跑完把它提升成正式记录（否则只是个临时数字）
./run.sh -c "from exp_log import log_experiment; log_experiment(
    name='xxx', agent='agents/a', eval_scheme='vs-b-n300', win_rate=0.51, n_games=300)"
```

### 三条评估纪律（都踩过坑才总结出来）

1. **必须用按局 K 折 CV** —— 单次切分的 AUC 标准差达 0.041，而特征改进只有 0.005 量级
2. **头对头赢 ≠ 更强** —— 非传递性是常态，任何改动都要同时看对统一基准的绝对胜率
3. **样本量要够** —— 检测 0.05 的差异需要 ~1600 局；80 局的结论全是噪声

### ⚠️ 卡组挑战赛：24 套自造卡组 vs `sample_official`（rollout 两侧镜像外战）

每套 160 局（SE≈0.040），**全部 24 套完成**：

| 排名 | 卡组 | n=160 | 复测 n=600 | 结论 |
|---:|---|---:|---:|---|
| 1 | `grass_heracross` | 0.5875 (z=+2.21) | **0.5050 (z=+0.24)** | 无差别 |
| 2 | `lightning_zeraora` | 0.5563 (z=+1.42) | **0.4867 (z=−0.65)** | 无差别 |
| 3 | `metal_mawile` | 0.4562 (z=−1.11) | — | 无差别 |
| 4~6 | `dragon_latias` / `metal_zacian` / `water_abomasnow` | 0.3250~0.3937 | — | 明显更弱 |
| 7~24 | 其余 18 套 | 0.1250~0.3063 | — | 明显更弱（z 至 −9.5） |

**结论：`sample_official` 就是我们卡组池里最好的卡组，卡组这条线在本池内已经榨干。**

### ⚠️ 教训：挑最大值 = 挑中了噪声（今天犯了三次）

同一个错误今天出现了三次，形状完全一样 —— **小样本里跑出一个 +2σ 的「赢家」，换成正确样本量后落回 0.5**：

| 实验 | 初筛 | 复测 |
|---|---|---|
| 卡组挑战赛 · `grass_heracross` | n=160 → **0.5875**（z=+2.21） | n=600 → **0.5050**（z=+0.24） |
| 卡组挑战赛 · `lightning_zeraora` | n=160 → **0.5563**（z=+1.42） | n=600 → **0.4867**（z=−0.65） |
| 续弈策略 · `playout_evolve` | n=300 → **0.5533**（z=+1.85） | n=2000 → **0.5035**（z=+0.31） |

**规律：在一组候选里取最大值，最大值天然偏高。** k 个候选、每个 n 局、真值都是 0.5 时，
最大值的期望约为 `0.5 + z_k × SE`，k=4 时 z≈1.6，k=24 时 z≈1.9。

⇒ **任何"从 N 个配置里挑最好"的扫描，第一名都必须用独立的大样本复测才能当真。**
扫描只配用来**淘汰**（底部那些离 0.5 有 5~9σ 的是真的差），不配用来**选拔**。

### ⚠️ 卡组挑战赛：从 24 个候选里挑最大值 = 挑中了噪声

24 个候选、每个 n=160、**真值都是 0.5** 时，最大值的期望约为
`0.5 + 1.9 × 0.040 = 0.576` —— 而观测到的是 **0.5875**。
**挑战赛的第一名完全可以由纯噪声解释**，复测后立刻落回 0.5050。

同样是 24 选 1，但**底部是真的**：0.30 / 0.20 / 0.125 这些离 0.5 有 5~9.5 个标准误，
不可能是噪声。所以正确的读法是——

> **"这批卡组里有一大部分明显更差"是可证的；"其中某几套更好"是不可证的。**

用 n=160 的排名去选卡组，只能用来**淘汰**，不能用来**选拔**。要选拔，就得对被选中的
那 1~2 个候选重新用 n=600 复测（成本 24 分钟），而且**必须**这么做 —— 不做就等于把
噪声当结论。

### 下一步（按价值排序）

| 优先 | 做什么 | 理由 |
|---|---|---|
| 1 | **子决策策略**：那 18% 从来没被决策过的选择 | `ATTACH_TO`（平均 **6.55** 个选项、93.5% 需要抉择）等 context 现在由**引擎枚举顺序**决定，`_POKEMON_TARGET_CTX` 甚至没覆盖 `ATTACH_TO`。子决策的对手是"任意"，门槛极低 |
| 2 | **margin 自适应 + 重扫** | 代码注释自己说候选 k 从 3 增到 12 时阈值应提到 ~2.0+，但实现里固定 1.5，而我们在平均 11.8 个候选上用它；且**所有历史调参都在 BUG-002 修复前做的**，已作废 |
| 3 | playout 策略（评估器偏差） | 判断"哪个动作更好"靠 playout，而 playout 用的是和兜底相同的 `safe_v1` —— 同一个策略既当选手又当裁判 |
| ~~—~~ | ~~在我们的 25 套卡组里换一套更好的~~ | ❌ 24 套挑战赛：22 套明显更弱，前 2 名复测落回 0.5。`sample_official` 是池内最优 |
| ~~—~~ | ~~调 rollout 的世界数 / 覆盖面 / determinization~~ | ❌ 全部证伪：worlds 10→40、MODE attack→all、连"用真实构筑造世界"都是噪声 |
| ~~—~~ | ~~卡组与策略分开调、收益可加~~ | ❌ 已证伪：那次外推的 678 实测只有 460 |
| ~~—~~ | ~~Q(s,a) / PIMC / AlphaZero 式自弈~~ | ❌ **改进算子断了**：Q 选动作 0.4650（比规则还差）。详见 `framework/README.md` |

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

### 3. 顺序里只有一件事重要：**ATTACK 不能被 RETREAT/DISCARD 挤到后面**

2026-10-20 用 n=600（SE 2.0pp）把优先级表重测了一遍——之前那批是 n=200、且
在 BUG-002 修复前，测不出 2~4pp 的差异：

| 优先级表 | 胜率 | z | 结论 |
|---|---:|---:|---|
| `safe_v1`（当前）：ATTACH→EVOLVE→PLAY→ABILITY→**ATTACK**→END→DISCARD→RETREAT | 基线 | 0 | — |
| `evolve_first`：EVOLVE→PLAY→ABILITY→ATTACH→**ATTACK**→RETREAT→DISCARD→END | 0.4983 | −0.08 | 噪声 |
| `play_first`：PLAY→EVOLVE→ABILITY→ATTACH→**ATTACK**→END→DISCARD→RETREAT | 0.4883 | −0.57 | 噪声 |
| `play_before_evolve`：ATTACH→PLAY→EVOLVE→ABILITY→**ATTACK**→END→DISCARD→RETREAT | 0.4733 | −1.31 | 噪声 |
| `develop_first`：PLAY→EVOLVE→ABILITY→ATTACH→**RETREAT→DISCARD→ATTACK**→END | **0.4167** | **−4.08** | ❌ **显著更差** |

关键在读表：前四行**ATTACK 都在 RETREAT/DISCARD 之前**，彼此全是噪声；
唯一崩掉的那行恰恰是 **RETREAT→DISCARD 插在 ATTACK 前面**。

⇒ **"先抽牌还是先贴能量"完全不重要**（这是直觉上最想问的那个问题，答案是不重要）。
真正致命的只有一条：**别让撤退/弃牌挡在攻击前面**——那正是 2026-10-07 那个让
8 套卡组平均 0.2725 的 bug。当前 `safe_v1` 在这条上是正确的，也不需要再调。

### 3b.（历史记录）修好前两条之后，剩下的顺序差异基本是噪声

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

> 📌 **接下来做什么看 [`下一步.md`](下一步.md)**；**已证伪、别再试的看 [`别再试.md`](别再试.md)**；**引擎字段（含隐含语义）看 [`引擎字段清单.md`](引擎字段清单.md)**
>
> ；**已经证伪、别再试的看 [`别再试.md`](别再试.md)**
> （后者汇总了 A~D 共 30 项关闭的方向、7 个方法论陷阱、7 条已验证正确、4 个未试方向）
>
>  —— 状态快照、已证伪清单（别再试）、
> 明天的具体任务与判据、以及免费提交槽的使用规则。

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
├── tools/                 # 诊断工具（做实验前先过一遍）
│   ├── check_agent.py     #   agent 自检：真 rollout 次数 / 被吞掉的异常 / 配置是否真生效
│   ├── inspect_subdecisions.py  # 子决策盘点：占比、选择空间、是否退化、选项解析成具体牌
│   └── measure_cost.py    #   单局耗时 / 预算是否被打穿 / 还剩多少算力余量
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
