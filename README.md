# PTCG AI Battle Challenge 实验工程

Kaggle 比赛 [The Pokémon Company - PTCG AI Battle Challenge Playground](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground)
的智能体开发工程。队伍 `zhang1wen2hao3`。

| 项 | 值 |
|---|---|
| **任务** | 写一个 agent 打宝可梦集换式卡牌，最大化三局两胜（Bo3）胜率 |
| **提交形式** | `.tar.gz`（`main.py` + `deck.csv` + `cg/` + 权重），`-f` 上传、必须带 `-m` |
| **指标** | `cabt_bo3`（Elo 式，**会漂，见下文**） |
| **截止** | 2027-01-08 |
| **限制** | 每天 5 次提交，**总共 2 个计分位**（最近两份在池子里被拉去配对） |
| **队分** | **1082.1**（`57871935` `ptcg_sub_g6_db04` = db04 卡组 + PPO 训练权重）；另一臂 port@db04 1028.2 |

> **提交策略（用户定，2026-11-19）：两个计分位全部当探针，不留锚。**
> 原文："不需要守的那格，一开始分数高没什么用，就怎么能方便我们训练就怎么提交就行。"
> 依据：同一份提交读数一天内摆动 ±250 分（`LB记录.md` 被这条规律修正过三次）。
> 因此**任何跨时刻的 LB 差值一律作废**，只在"同一快照、同一状态"下读。

## ⚠️ 数据授权

`data/`、`cg/`、`deck.csv` 是比赛方数据（引擎 + 卡牌库），**已在 `.gitignore` 中排除**。首次使用：

```bash
bash fetch_data.sh
```

## 当前状态（TL;DR）

**agent = 神经策略（4.55M 参数 PPO）+ 卡组。规则 greedy 只作为对照与兜底。**

| 项 | 值 |
|---|---|
| 队分 | **1082.1**（G6 = db04 卡组 + 训练权重） |
| 面板读数 | 全格最高 `cand9D6ER4` **72.2%**（靠 7e39 格 88% 拉高）；剔除 7e39 后 `cand0904R1` **61.8%** 最高 |
| 已确认的真收益 | ① 训练配方本身（G4-B 0.5337 [0.5093,0.5581] vs port，首次显著）② **每个训练块做 SWA** ③ db04 换牌（+53.9 队分） |
| 已归零的方向 | 对位特化（specialist）4 条线全零 · 奖励塑形 S0–S4 · 开局 BC · 蒸馏 D1 · 决策重排 · LR/批次/调度 A2–A4 · 跨 run 权重平均 |
| 唯一未闭环线索 | **KL 锚 0.005 + 块内 SWA**（单种子单格领先，摆幅 13.3→4.7pp，待 3 种子×17 格） |

### v2 面板读数（17 格 × 150 局 / 种子，`logs/meta/<cand>_v2_s950000/`）

| 驾驶员 | 卡组 | 全格加权 | 剔除 7e39 | 格子数 |
|---|---|---:|---:|---:|
| `cand9D6ER4` | dunsparce_abra | **72.2%** | 58.6% | 17 |
| `cand0904R1` | dreepy_drakloak | 53.2% | **61.8%** | 17 |
| `cand3EAAR1` | dreepy_drakloak | 45.5% | 50.1% | 17 |
| `candE1D7R4` | z_ogerpon_toolbox_miwa | 42.1% | 37.9% | 17 |
| `lrsweep_a2_*_SWA` | e1d7 | 43.3% | — | 17 |
| `cand9D6E_SWA4` | dunsparce_abra | 78.1% | 56.7% | **2（不完整，不可比）** |

> ⚠️ **7e39 格单独读**：那是卡组结构问题（同驾驶员的端口版本也只有 35~41%，
> 9d6e/2617/f5f8 打它 60~89%），而且**我们的 7e39 比真人强**（真人天梯胜率 33%），
> 面板在这一格会过度惩罚我们 ⇒ 排强度看"剔除 7e39"列（与 LB 的 Spearman = +1.00）。

## 我们的 agent 现在是什么

三层，全部在 `port_repo/`：

| 层 | 位置 | 说明 |
|---|---|---|
| 引擎 | `cg/`（比赛方 `libcg.so`）+ `tools/` ctypes 封装 | 官方 C++ 引擎，`libcg.so` 用 `std::random_device`，不可播种 |
| 环境 | `port_repo/native/ptcg/`（C） | 观测/编码/追踪/卡组池全部重写成 C，与 Python 版逐位一致（`make -C native parity`） |
| 模型 + 训练 | `port_repo/native/model/`（CUDA）+ `src/`（PufferLib `c5d3c63` + 本地补丁） | **训练循环里没有 PyTorch** |

### 模型规格

| 项 | 值 |
|---|---|
| 参数量 | 4.55 M |
| 结构 | `PT_D=256`、`PT_H=8`、`PT_FFN=512`、`PT_LAYERS=4`、horizon 64 |
| 权重格式 | `.bin` = `PT_PARAMS` 顺序的扁平 fp32，**18,205,408 B**（长度不对就是坏文件） |
| 转换 | `PYTHONPATH=build/port_plain:. ./tools/py tools/native_weights.py import\|export` |
| 训练驱动 | `port_repo/native/train_native.py`，常用入口 `tools/db04_line.sh` |

### 训练循环的关键机制（都是踩坑换来的）

- **league / 冻结 bank**：`BANKS` 里的 `.bin` 作为冻结对手，`frozen_bank_pct = LEAGUE_PCT`，
  每个 bank 拿 `floor(agents×pct)` 个 env；约束 `league_pct × n_banks ≤ 1`，`PT_MAX_BANKS=16`。
- **加权靠重复**：想让某个对手占 40% 权重，就是把它在 `BANKS`/`SLOTS` 里重复写 4 次。
  没有"权重"参数，只有位置。
- **bank 的卡组**由 `BANKDECKS` 钉死；未标注的行是自博弈（`MIX_SELF`），其对手卡组从池里按
  `--deck-weights-file` 抽。
- **SWA**：每个训练块结束存快照 `experiments/native-<RUN>/<step:016d>.bin`，
  `tools/swa_avg.py` 做等权平均。这是目前**唯一可复现**的正收益旋钮，见下。

## 评估体系：判据先于实验

### 面板 v2（唯一的强度口径）

`decks/testset_meta_real_v2.tsv`：**17 格 × 150 局**，每格 = 一个"对手卡组 + 最强驾驶员"，
种子基 950000（逐局种子 950000/950002/…，所以**同种子可逐局配对**）。

```bash
# 本地跑一个候选的整张面板
bash tools/local_panel_v2.sh cand9D6ER4 6

# 配对检验（新候选 vs 对照，同种子逐局）
PYTHONPATH=build/port_plain:. ./tools/py tools/panel_paired.py --a cand9D6ER4 --b candE1D7R4 --tag v2

# 排名（自动剔除 7e39）
PYTHONPATH=build/port_plain:. ./tools/py tools/panel_rank.py --cands cand9D6ER4 cand0904R1 --tag v2
```

### 四条评估纪律（每条都付过代价）

1. **≥3 个种子（450 局配对）才能当真。** 单种子 150 局的"赢家"平均虚高 ~1.5σ。
   本项目的失败模式统一叫**胜者诅咒**：4 块里挑最好的一块，好块的好是噪声。
2. **"从 N 个候选里取最大"不是选拔，只是淘汰。** k=4 时最大值期望 `+1.6σ`，k=24 时 `+1.9σ`。
   扫描只能用来扔掉底部的，第一名必须独立大样本复测。
3. **头对头赢 ≠ 更强。** 非传递性是常态，任何改动都要看对统一基准的绝对胜率。
4. **面板有偏，LB 有漂**：面板过惩罚 7e39；LB 单次读数 ±250 分。
   任何"LB 涨了 X"必须同快照双读数，或改用回放统计。

## 分数地图（按卡组）

| 我方 | 最强对手卡组 | 缺口类型 | 证据 |
|---|---|---|---|
| 0904（dreepy） | 7e39 | 34.9% × 权重 56.7% = 优先级 #1 | 真人 n=11 |
| e1d7（ogerpon） | 7e39 | 34.9% × 46.5% = #2 | n=22 |
| 3eaa（dreepy） | 7e39 | 34.9% × 20.0% = #3 | n=6 |
| 9d6e（dunsparce） | — | 全格 72.2% 最高，但 7e39 格 88% 是主要来源；剔除后 58.6% | 面板 17 格 |

逐格缺口明细在 `docs/我方vs高分选手_逐格缺口_*.csv`、`docs/格子差距诊断_*.csv`。

## ✅ 有效（经过统计检验）

| 手段 | 效果 |
|---|---|
| **训练配方本身**（PPO + league + 真实野区卡组分布） | G4-B **0.5337 [0.5093,0.5581]** vs 端口权重（首次显著）；实盘队分 G6 1082.1 vs port@db04 1028.2（+53.9） |
| **每个训练块做 SWA** | 块间摆幅 **13.3pp → 4.7pp**；LR 扫描里 A2 配方 **36.4% → 43.3%**（CI [+1.3,+12.2]）；17th place 同法 +4.3pp |
| **换牌：db04** | 真实胜率 0.642 vs 7e39 0.568；队分 +53.9（同配方同 seed，只换牌） |
| 规则 greedy（`agents/greedy.py`） | 仍是零依赖兜底与对照基线，`main.py` 与其等价（0.510, n=300） |

## ❌ 已证伪（不要再试）

| 方向 | 结果 |
|---|---|
| **对位特化 p1/p2/p3/p4** | 权重位移仅 **0.0007%~0.0016%**（正常一轮 PPO 是 0.014%~0.05%），17 格 × 3 种子后**全部归零**；详见下节 |
| 奖励塑形 S0/S1/S2/S3B/S3C | λ=1 无效(+0.5pp)；奖品稠密奖励 **−8.6pp**（reward hacking）；PBRS 无优势(−4.2pp)；可攻击性进 Φ 无效 |
| 开局 BC（B1/B1b） | 前 3 回合冻结骨干微调，**改不动行为** |
| 蒸馏 D1（含 4M 版） | +1.4pp 在噪声内，目标格没保住 |
| 熵系数 S4 | 无收益 |
| 决策重排（branch probe） | **−6.8pp，z=−6.0** |
| LR / 批次 / 调度 A2–A4 | 14th 的配方（1e-5 + cosine + 4×batch）在本项目**一分不赚**；A2_SWA 3 种子 = **+0.0pp** |
| 跨 run 权重平均 | 无效（`tools/swa_avg.py` 仍在，但只用于块内） |
| 旧规则线的 MCTS / Q(s,a) / PIMC / 24 套卡组挑战 / rollout 旋钮 | 全部封死，见 `别再试.md`（30 项） |

## 对位特化（specialist）：为什么归零，以及唯一还活着的线索

**规格（用户定）**：先用 Bayesian 后验预测对手卡组，若 `P(最高) − P(当前) > 10%` 则切到该卡组
的对位最佳 agent；多个 agent 合并要求特攻胜率差 ≥5pp，否则用非特攻；主 agent 取
`覆盖率 × 胜率差` 最高的综合 agent。

**实测**（`docs/对位特化记录.csv`、`专项训练诊断_2026-11-27.md`）：

| 线 | 单种子/单格 | 3 种子 / 17 格 | 终判 |
|---|---|---|---|
| p1 0904→7e39 | +5.3pp (n=150) | +1.6pp（n.s.） | 零 |
| p2 e1d7→7e39 | +3.9pp，CI [−0.8,+8.6] | 3 种子 +3.1 / +0.6 / −2.8 → **+0.3pp** | 零 |
| p3 3eaa→7e39 | — | −1.6pp | 零 |
| p4 0904→7e39 + KL 锚 | raw 50.0/46.7/36.7/44.7（极差 13.3pp） | 未做 3 种子 | **未闭环** |

**根因（决定性证据）**：专项臂与起点的全局相对 L2 位移只有 **0.0007%~0.0016%**，
而历史上"能测出东西"的一轮是 0.014%~0.05% —— 位移小 10~70 倍。
`LR=2e-5` + 锚到起点 + 每块 500k 步 = **优化压力不足**，块间差异是采样噪声，不是策略震荡。
外部同赛道真实量级也只有 **+3~5pp/格**（Petit Canard 12 个 matchup specialist 平均 +3.5pp）；
拿 6th 的 +12~14pp 当目标是量级搞错（那是"切专家 vs 自己的泛用策略"，不是 specialist 训练收益）。

**唯一线索**：KL 锚 `KL_COEF=0.005` + 块内 SWA，四块 `48.7 / 44.0 / 47.3 / 48.0%`（baseline 43.3%），
摆幅压到 4.7pp、四块全在基线之上 —— **单种子、单格**，需要 3 种子 × 17 格才能判。
实现：`LOSS_ANCHOR_KL=7`（`port_repo/native/src/pufferlib.cu`），
`--kl-anchor-bin/--kl-anchor-coef`（`train_native.py`），`KL_ANCHOR=` 环境变量（`tools/db04_line.sh`）。

> **训练器层面的实现**：`LOSS_ANCHOR_KL=7` 用**采样 KL** `E[logπ_cur − logπ_ref]`（不是精确 KL），
> 锚 = 传入的 `--kl-anchor-bin`。**锚到哪由上层决定**：旧的对位特化脚本锚的是固定起点，
> evo loop 已改成**每块追最新自己（last-best 棘轮，`ANCHOR_SLACK_PP` 容差）**。
> 尚未做的是"精确 KL"（需要把锚的 logits 搬进 loss kernel）。

## 训练对手强度体检（`lgw` 口径，2026-11-27 用户规则）

> "如果 A 对战 bank B 的胜率高于 0.8 或者低于 0.1，都需要关注对手强度是否在线，
> 不符合的需要筛选掉，不能作为训练对手。"

| 胜率 | 含义 | 处理 |
|---|---|---|
| **> 0.8** | bank 是沙包，梯度信号"白捡" | 踢出训练对手，换更强的 bank |
| **0.1 ~ 0.8** | 健康 | 保留（`lgw` 落在 **0.4–0.7** 最好） |
| **< 0.1** | bank 太强，learner 拿不到可用梯度 | 踢出，或先降低对手强度再训 |

**离线体检**（训练前，独立工具）：

```bash
./tools/py tools/bank_audit.py \
    --agents cand9D6ER4 cand9D6E_SWA4 ab9d6e_r2A \
    --banks /root/league_g0003/bank_0.bin /root/league_g0003/bank_1.bin \
    --gpu root@connect.nmb2.seetacloud.com --n 450 --emit-ok
# 输出每个候选可用的 BANKS= 列表；有任何不合格 → 退出码 1
```

**在线体检**（训练中）：训练日志里 `lgw 0.456 [0:0.58/0.07 1:0.27/0.06 …]` 的方括号是**逐 bank
在线胜率**（`league_w_<b>/league_n_<b>`），evo loop 每轮训练完会把最后一段解析出来，
任何 bank 落出 (0.1, 0.8) 就告警，并在下一轮体检时剔除。

**在 evo loop 里的落地**：每轮训练前对「族首 × 每个 bank」各跑 450 局，
**逐族**生成可用 bank 列表（某个 bank 对某族太强/太弱就只对该族剔除），
再拿这份列表去跑该族的训练；如果一个都不剩，该族退化为自博弈 + 加权抽样并告警。

## 卡组资产

| 路径 | 内容 |
|---|---|
| `decks/pool_field/` | **196 套真实野区卡组**（60 个 archetype 名；**36 个名字只有 1 套**） |
| `decks/pool_final/` | **211 套**训练池 = 每 archetype ≤10 变种 + Kaggle 近邻补充（`tools/build_pool_final.py`） |
| `decks/weights_pool_final.txt` | 抽样权重（v3 格式）：一族内 champion **0.70** / 其余共 0.30；单套一族给 **1.0**；未列出的默认 1e-4 |
| `decks/testset_meta_real_v2.tsv` | 面板 v2 的 17 格 + 权重（强度口径的**唯一**定义） |
| `data/deck_corpus_local.csv` | 6258 局本地回放抽出的 342 套真实卡组（外部语料） |

> 卡组 id = `decks/pool_final` 下排序后的位置（`native/tools/export_tables.py <out> --pool decks/pool_final`）。
> 池子里 champion 是"字母序第一"，不是我们实际用的那套 —— 需要 `main=` 覆盖。

## 机器与运维纪律

- **1 台 GPU**（RTX 4090 24G，`/root/ptcg_train_kit/port_repo`）+ **CPU 评估机**（`/root/eval_cpu_kit`，32 核 / 14 worker）。
- **任务一跑完立刻关机**（GPU 按小时计费），并且**永远再挂一个固定时点的兜底关机**
  （`tools/force_finish_deadline.sh`）—— 主链的 watcher 死过两次。
- 关机前**必须先拉回产物并校验尺寸**：`.bin` 必须是 **18,205,408 B**；
  `vs_*.json` 每候选必须 **17 个**。校验不过不关机。
- 产物不许放 `/tmp`（会消失，一次丢过 4 个权重）；`/tmp` 也不许当 askpass 唯一来源。
- 后台任务用 `setsid nohup` + `pgrep -c -f` 自检；**杀进程用字面 PID，不要 `pkill -f` 脚本名**（会杀掉自己的 shell）。
- 详见 `docs/机器与环境搭建清单.md`、`AGENTS.md`。

## 目录结构

```
├── main.py                  # 提交入口（自包含；出包用 make_submission.py）
├── deck.csv  cg/            # 比赛方数据（gitignore）
│
├── port_repo/               # ★ 主线：神经策略
│   ├── native/
│   │   ├── ptcg/            #   C 版引擎环境（与 Python 逐位一致）
│   │   ├── model/           #   CUDA 模型（4.55M）
│   │   ├── src/             #   PufferLib 补丁版（含 KL 锚 loss）
│   │   ├── train_native.py  #   训练驱动
│   │   └── tools/           #   export_tables / native_weights / model_parity
│   └── puffer_ptcg/         #   编译产物 _C*.so
│
├── agents/                  # 规则基线（greedy / rollout_agent / random_baseline…）
├── framework/               # 旧规则线框架（搜索 / 填充器 / 特征），已封存
├── build/agents/            # ★ 候选权重（<name>/model.pt + .bin）
├── decks/                   # 卡组资产（见上表）
├── logs/meta/               # 面板结果 logs/meta/<cand>_v2_s<seed>/vs_<cell>.json（gitignore）
│
├── tools/                   # ★ 主力工具（100+）
│   ├── db04_line.sh         #   训练入口（env 透传 INIT/T/LR/LEAGUE_PCT/KL_ANCHOR…）
│   ├── local_panel_v2.sh    #   本地跑整张 v2 面板
│   ├── panel_paired.py      #   同种子逐局配对检验
│   ├── panel_rank.py        #   候选排名（剔除 7e39）
│   ├── bo3_eval.py          #   单格 Bo3 评估（生产 ABI）
│   ├── swa_avg.py           #   权重平均（tool）
│   ├── build_pool_final.py  #   训练卡池构建
│   ├── driver_registry.py   #   驾驶员注册表（check = 0 error，禁止手改表）
│   ├── evo_loop.py          #   ★ 进化式选育循环（20 候选/轮，bank 闸门 +1.5pp）
│   └── force_finish_deadline.sh  # 兜底关机
│
└── docs/                    # 结论与记录（见下）
```

## 文档索引（读这几份就够）

| 文档 | 内容 |
|---|---|
| `docs/训练路线结论_2026-11-19.md` | **主线结论全集**（§1–§28）：每轮实验的判据、读数、口径修正 |
| `专项训练诊断_2026-11-27.md` | specialist 为什么归零 + 外部 9 份方案对照 + 补足顺序 |
| `AGENTS.md` | 常用流程索引、命名规范、统计纪律、机器纪律（**开工先读**） |
| `docs/机器与环境搭建清单.md` | GPU/CPU 环境搭建、10 类坑 |
| `docs/驾驶员与覆盖规则.md` | 驾驶员表的覆盖规则与固定评测流程（禁止手改表） |
| `LB记录.md` | LB 读数机制（会漂、会冻结）与三次翻车复盘 |
| `别再试.md` | 30 项已关闭方向 + 方法论陷阱（**动手前先看**） |
| `训练手册.md` | 训练/评估的完整操作手册 |
| `高分段架构清单.md` | 公开榜高分方案的可复现架构清单 |

## 使用

```bash
bash fetch_data.sh                    # 拉比赛方数据（首次）
make -C port_repo/native test         # 环境不变量测试（不需要 GPU）
bash port_repo/native/build_native.sh # 编译 CUDA 扩展（需要 nvcc，1~2 分钟）

# 训练（GPU 机器上）
bash tools/db04_line.sh               # 默认配方 400k 步
INIT=/root/cand0904R1.bin T=2000000 KL_ANCHOR=/root/cand0904R1.bin KL_COEF=0.005 \
  bash tools/db04_line.sh             # KL 锚实验

# 评估（CPU 机器上，或本地）
bash tools/local_panel_v2.sh <候选> 6
PYTHONPATH=build/port_plain:. ./tools/py tools/panel_paired.py --a <新> --b <对照> --tag v2
```

## 下一个动作：进化式选育循环（2026-11-27 定稿）

对位特化单线训练已经证伪（位移 0.0007%~0.0016%），改成**多族并行 + 选育**：

**一轮 = 4 个族 × 4M 步**，每族拆 **4 块 × 1M 步**，产出 **4 个块输出 + 1 个族内 SWA** = 5 个候选，
4 族共 **20 个**。训练配方统一带 **KL 锚 0.005 + cosine 退火**。

> **KL 锚锚的是「最新的自己」（last-best 棘轮），不是族首/genesis。**
> 每块训完立刻跑 450 局 vs 冻结 bank：只要这块不比当前锚差超过 `ANCHOR_SLACK_PP`（默认 1pp），
> 就把下一块的 `KL_ANCHOR` 切到这块；否则保持上一个最好的。
> 块间这 450 局的胜率同时用于选育（省一次重复评测），SWA 也在这 4 块上做。

**晋级闸门（对冻结 bank 的 450 局平均胜率）**：

| 候选类型 | 晋级线（用户原话） |
|---|---|
| 块输出（族 A1 的第 k 块） | `w(块) ≥ w(族首 A1) + 1.5pp` —— 原话："A1 对 bank B 平均胜率 50，那你至少得有 51.5" |
| 族内 SWA | `w(swa) ≥ max_k w(bp_k) + 1.5pp` —— 原话："A1 50, A2 51，那 r1Aswa 至少需要 52.5" |

过线者按 w 排序取前 4 个作为下一轮族首（每族最多 2 个，防 4 个 SWA 全来自一族）；
**一个都不过就停止**；最多 5 轮 = 80M 步。

**起始池**：不限于最强 agent —— 把**同牌表**的 15 个 agent（`cand9D6ER4`/`SWA3`/`SWA4`/
`ab9d6e_r2A`… 含特化产物、strong 矩阵上的驾驶员）各跑 450 局对 bank，取前 4 个当族首，
所以"特化后没涨但也没掉"的模型天然有位置。

```bash
./tools/py tools/evo_loop.py --dry-run        # 只打印选育逻辑，不占机器
GPU=root@<gpu> GPU_PORT=38237 EVAL_HOST=root@113.44.139.218 \
  ./tools/py tools/evo_loop.py                # 真跑（断点续跑：logs/evo_state.json）
```

成本估算（SPS≈1600）：80M 步 ≈ **14 h GPU**；每轮 bank 评测 20 候选 × 450 局 ≈ 15 min，
面板复核 3 种子 × 17 格 ≈ 30 min/model。⇐ 开机前先挂 `tools/force_finish_deadline.sh` 兜底。

后续（按价值排序）：

| 优先 | 做什么 | 判据 |
|---|---|---|
| 1 | **跑通上面这条 evo loop（9d6e 起手）** | 至少一轮有候选过 +1.5pp 闸门；否则说明配方/步数还不够 |
| 2 | **卡组后验路由**（Bayesian 对手卡组预测 + 10% 切换阈值） | 先离线验证后验 top-1 准确率（6th 报 99% by turn 2） |
| 3 | 每轮报 `rel_L2` 位移 | 位移 <0.01% 就别指望能测出胜率差 |
| 4 | last-best ratchet + 精确 KL（仅在 1 有正信号后） | 同 1 |
