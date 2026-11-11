# 参考工程对比：为什么我们训练一直不涨点

参考工程：`samples/pkmn-kaggle-main/pkmn-kaggle-main`（12th/6807，最终分 1186.4，Alakazam + Dragapult 双卡组）。

## 一句话结论

**不是架构问题，是训练量级差 3 个数量级 + 缺自适应对手机制。** 我们自己的 kit **已经实现了**
自对弈（同一张网打双方、negamax GAE）、KL 门、frozen bank——但这些机制我们只喂了 400k 步，
参考工程喂了 **3B 步**（deck finetune）/ **11B 步**（generalist）。

## 关键参数对比

| 维度 | 我们（阶段三） | 参考工程 | 比值 |
|---|---|---|---|
| 训练步数 | **400k**（~7 min） | generalist **11B** + finetune **3B** | ~**7,500×**（finetune 口径） |
| 吞吐 | 957 SPS（1×2080Ti） | 16.5k SPS（4×H200） | 我们单卡慢 ~70×（折算后） |
| 模型 | **5M**（端口权重） | **20M** transformer | 4× |
| 对手池 | **187** 副 + 10% 冻结银行 | **4,941** 副（3517 Kaggle + 1424 Limitless） | 26× |
| 自对弈 | ✅ 有（双方同网，negamax） | ✅ 有 | 相同 |
| teacher 蒸馏 | ❌ 无（只有后置 kl_blend 拉回） | ✅ 有（--teacher-kl-coef 0.005） | 缺 |
| gated promotion | ❌ 无（银行是静态的） | ✅ 赢过旧 self ≥0.53 才晋升 | 缺 |
| two-net 自适应 finetune | ❌ 无 | ✅ A(舰)↔B(对手) 轮训 6-8 轮 | 缺 |
| 反拖延 terminal | ❌ 无 | ✅ 单回合 200 步判负 | 缺 |
| 评估 | 200 场/格 | **≥1600 场/格 + 复种子** | 8×+ |
| 奖励 | +1/−1 | +1/−1（+ 反拖延判负） | 基本同 |

## 三个根因（按重要性）

### 1. 训练量级差 3 个数量级 —— 主因

参考工程的 **deck finetune 就要 3B 步**，且是"+2pp 就冻结 A、训 B 追回、重复 6-8 轮"的节奏，
每一轮都是一次 2pp 的增量。我们的阶段三 400k 步 = 它的 **0.013%**，还不到它一轮的量。

证据闭环（我们自己的数据）：换 INIT 后权重位移只有中位 **0.26%**，面板 −1.4pp（噪声带内）；
参考工程每轮 finetune 是 +2pp 的可测增量。**"没练够"和"没测到"同时成立。**

折算到我们的硬件：3B 步在 4×H200 上是 ~50 小时墙钟；在 1×2080Ti（957 SPS）上等价
**~170M 步 / 50 小时**。所以"复现它的 finetune 量级"我们负担得起（50 小时 GPU），
但我们一直在跑 7 分钟的版本。

### 2. 缺"会追回"的对手 —— 机制差异

我们的对手是「90% 自对弈随机场地牌 + 10% 冻结银行 shup_a0.5」。参考工程的 finetune 里，
对手网 B 会**专门训练来追回 A 打出的分数**，A、B 交替冻结晋升。frozen bank 是死的，
two-net 是活的——这是它能持续 +2pp、我们停在 0 的关键差别。

### 3. 评估太粗 —— 看不见小增益

他们 ≥1600 场/格 + 复种子；我们 200 场/格（§59：单格 ±6.5pp，加权 ±2.5pp）。
一个 2pp 的 finetune 增益在我们的评估下**不可见**。所以他们能靠 +2pp 的节奏滚雪球，
我们连第一次 +2pp 都测不出来。

## 结论：两个问题分开答

**要不要用它的工程训练我们的模型？** —— **不**。理由：
1. 它说 engine binaries 是专有的，我们得换成自己的 `cg/` 引擎，等价于重写；
2. 20M 模型 + 11B 步的算力（4×H200）我们根本没有；
3. 我们的 kit 已经实现了它的核心机制（自对弈 + negamax + KL 门 + frozen bank），
   缺的是**量级**和**晋升循环**，不是从头换工程。

**能不能看出区别优化我们的工程？** —— **能，而且便宜**，按性价比排序：

| # | 动作 | 成本 | 预期 |
|---|---|---|---|
| 1 | **跑一个 20-40M 步的 finetune**（先验证"只是没练够"） | GPU 6-12 h | 决定性：若权重位移随步数线性增长，实锤量级问题 |
| 2 | **加 gated promotion**（银行从"静态"改成"赢过旧 self 就换"） | 改造 trainer | 让对手活起来，复刻它 finetune 的核心 |
| 3 | **对手池 187 → ~1900**（deck_corpus + ladder 档案里现成的牌表） | 数据整理 | 26× 多样性，接近它的 4,941 |
| 4 | **加反拖延 terminal**（单回合 200 步判负） | env.c 小补丁 | 防策略学成拖回合 |
| 5 | 评估提到 ≥800 场/格 + 复种子 | CPU 节点 | 让 2pp 增益可见 |

## 附：它的自对弈与我们的对照（确认不是缺自对弈）

参考 `rl/train_selfplay.py` 开头：live net 打双方（pure self-play）、negamax GAE、
teacher-student 蒸馏、gated promotion（winrate ≥ promote-winrate 才换 teacher）。

我们的 `port_repo/native/README.md` + `PATCHES.md`：同一张网打双方、`ptcg_puff_advantage`
negamax（座位切换翻号）、KL/value-loss 门、frozen-bank loader。

⇒ 两边**都是自对弈**，我们缺的不是自对弈本身，是它配套的 **teacher 蒸馏 + 晋升循环 + 步数预算**。

## 预算与杠杆（2026-11-06 追加）

参考工程 ≈ 22.7B 步（generalist 11B + finetune 5.7B + 早期/失败 ~6B），
在 4×H200 @16.5k SPS 上 ≈ 16 天墙钟，按 H200 云价 ≈ **$5k–10k**（按需上沿）。

我们：1×2080Ti @957 SPS。光 generalist 的 11B 步放到我们单卡 = **133 天**；
单次 finetune 400k 步 = 7 分钟。**算力预算差两个数量级，这不是调参能补的。**

结论：**别复现它的量级**。我们有 1060 分靠"端口权重 + 选卡组"，不是靠 RL。
RL 要不要继续，取决于一个便宜的分水岭实验（见下）。

## 分水岭实验（`tools/ph3_watershed.sh`，T=30M）

**目的**：区分"没练够"（量级问题）和"RL 信号没动"（机制问题）。

| 项 | 配置 |
|---|---|
| 与 v2 唯一差别 | `T`：400k → **30M**（75×） |
| 成本 | 957 SPS → **8.7 小时**（1×2080Ti，¥20–40） |
| 判据 | 跑完量**中位张量相对 L2**（`tools/weight_dist.py`，剔除 5 张静态表） |

**权重位移标定**（中位张量相对 L2）：

| 模型 | 中位位移 | 含义 |
|---|---|---|
| 端口自身 | 0.000% | 基准 |
| `ph3_2617_v2`（400k） | 0.316% | 我们现在的 finetune |
| `ph3_9d6e_v2`（400k） | 0.341% | 同上 |
| `g6_db04`（天梯 1060，多轮） | 0.502% | 我们最好的 RL 模型 |
| `g8_bc_parent`（BC 全重训） | **1.921%** | BC 全重训量级 |
| `bc_26c6`（BC 全重训） | **2.046%** | 同上 |

**判读**：30M 步后
- 中位位移 **>1%**（逼近 BC 量级）→ **只是没练够**，瓶颈是预算，下一步决定投更大 GPU 还是转非 RL 杠杆；
- 仍 **≈0.3%** → **RL 信号没动**（对手太弱 / 奖励太稀 / KL 门卡死），先修机制再谈预算。

配套测量工具：`tools/weight_dist.py`（`.bin` 自动转 `.pt`）。跑完的第二指标是 12 格
`meta_eval` 面板（与 v2 同机同种子），看 meta_w 是否越过 ±2pp 噪声带。

## 我们早试过 A/B 轮训（"双头对弈"），差别在哪（2026-11-06）

我们的 `build/dual_kl_loop.sh`（A=26c6、B=3eaa）和它的 two-net finetune **形状相同**：
A/B 交替、对手用对面的产物、双向换边。但我们 8 轮 69 分钟就"小增益后见顶"（§48.3），
它 3B 步 6-8 轮持续 +2pp。差别不是思路，是四个量级上的东西：

| 维度 | 我们的双头对弈 | 参考工程 two-net finetune | 后果 |
|---|---|---|---|
| 起点 | BC 克隆（26c6 侧 0.14，端口打 BC 0.8~0.92） | 11B 步 generalist（会打 ~5000 副牌） | 我们 A/B 一开始就是弱鸡，没有"够得着的强对手" |
| 对手范围 | **1v1**：A 打 26c6，B 打 3eaa | A 打舰牌，B 打**整个 meta + 克制池** | 1v1 两三轮互相摸透就顶了；池子能持续给压力 |
| 每轮步数 | **100k 步/轮** | **~500M 步/轮**（3B ÷ 6-8 轮） | 差 5000 倍 |
| 换边机制 | 固定每轮换边 + α=0.25 权重回拉 | **领先 2pp 才冻结换边** + 训练内 teacher-KL | 固定换边让一方领先就打崩另一方（它说的 collapse） |

我们自己的记录早就点破了根因（§21.10）：

> "联赛的瓶颈不是训练配置，是对手强度。除了端口自己，池子里全是 BC 克隆（端口打它们 0.8–0.92）。
> 从这个起点做对抗学习，无论打多少步都缺少够得着的强对手。"

⇒ 它对"梯云纵"为什么能成、我们为什么归零，答案是**用 11B 步 generalist 先造出一个够强的对手**，
再在几千副牌的池子上、以 500M 步/轮的预算做门控换边。这三条我们一条都不满足。

## 对分水岭实验的修正

据此，30M 步的分水岭判据要补一层：如果 30M 步后权重仍 ≈0.3%，
**最可能的解释就是上面这条（对手太弱，梯度≈0），而不是"奖励没写对"**——
这是我们已有 5-6 次证据支持的头号嫌疑。真正要动的变量是**对手强度**，步数是次要的。

## 落地：teacher_loop（冻结 teacher + 门控晋升 + 两网轮训）

按"一边冻结一边训，A 打舰牌、B 打所有牌"升级了旧 `dual_kl_loop.sh`，对齐参考工程的 two-net 结构。

**新脚本**：
- `tools/teacher_loop.sh` —— GPU 机上的编排：A/B 轮训 + KL 回拉 + 门控晋升
- `tools/teacher_gate.sh` —— 晋升 h2h（镜像，同卡组），返回胜率
- `tools/db04_line.sh` —— 新增 `P_ANCHOR_LEARNER/SELF_OPP/LEAGUE_OPP` 三个旋钮（B 阶段 learner 打所有牌）

**与参考工程 still 有的三个偏差（诚实标注）**：
1. **KL 是事后回拉**（`kl_blend` α≈0.10），不是训练内逐 step 的 teacher-KL 蒸馏。
2. **teacher 只作用在 A**（舰牌侧）；B 侧没有独立 teacher。
3. **每轮步数**：默认 2M（v1 先验证循环能转、晋升能触发），不是参考工程的 ~500M/轮。

**晋升判据（预注册）**：
- 先量镜像基线 `B0`（起点 A vs teacher 的同文件镜像，处理座位偏置——实测镜像≠0.5，如 2617 是 0.567）。
- 晋升条件：`A vs teacher 镜像胜率 ≥ B0 + 3pp`（`PROMOTE_MARGIN=0.03`，`N_GATE=400`）。
- 保守（宁可不晋升也不晋升噪声）：镜像 n=400 的 CI ≈ ±4.9pp，3pp 余量接近噪声下沿。

**上机前必须冒烟的三处**（GPU 机上验证，本机无法复现）：
1. `BANKDECKS=` 空串时冻结 bank 是否真的打池子所有牌（还是退化成锚牌）。
2. `P_ANCHOR_LEARNER=0.0` 时 learner 是否真的抽 pool、ANCHOR 只作占位不报错。
3. `teacher_gate.sh` 在 GPU 机本地能否跑（依赖 eval_pair + 引擎；GPU 机 96 核）。

**判据（跑完循环后）**：
- 若 `teacher` 在 R 轮内至少晋升 1 次，且最终 A 的 12 格 `meta_w` ≥ 端口 +3pp ⇒ 机制有效，值得加预算/加轮数；
- 若 R 轮 0 晋升、A 面板不涨 ⇒ 与旧 dual 同样的"对手不够强 / 步数不够"问题，回来看权重位移。

## 训练侧接入：对手驾驶员按 choice 矩阵选（2026-11-06）

### 已有什么（原 teacher_loop 就有）

| 需求 | 实现 | 位置 |
|---|---|---|
| A/B 换边（A 训时 B 冻结、B 训时 A 冻结） | ✅ | `teacher_loop.sh` ① `BANKS=$B`；④ `BANKS=$A` |
| A 永远用指定卡组 | ✅ | `ANCHOR=$SHIP` + `P_ANCHOR_LEARNER=1.0` |
| B 的卡组按覆盖率 | ✅ | `DWF=decks/weights_field_b05.txt` |

### 这次补的

**① A 阶段：对手驾驶员查矩阵**（新 `tools/choice_to_banks.py`）

把 `the-pockmon-battle-choice.csv` 里 A 那一行（对手牌 → 最强驾驶员）翻译成训练器的
`BANKS/BANKDECKS/LEAGUE_PCT`：**按实测覆盖率分配 bank 槽位**，每个槽位钉一副牌、带该牌的驾驶员权重。

```
$ ./tools/py tools/choice_to_banks.py --a-deck db04
对手卡组   槽位  覆盖率  驾驶员
7e39        4   34.7%  init
2617        1   11.7%  init
e8ee        1    6.2%  g6_db04     ← 矩阵说这格用 g6_db04
f5f8        1    5.7%  g6_db04
0d86        1    5.5%  g6_db04
9d6e        1    4.8%  init
```

**② B 阶段：对手 A 用 init**（`BANKS=$INIT_A BANKDECKS=$SHIP`），不再用 A 训后的权重。

### 引擎硬限制（绕不过去）

```python
assert a.league_pct * len(a.league_bins) < 0.5, "league banks would own >50% of rows"
```

⇒ **指定驾驶员最多覆盖 49% 的对局**，其余 51% 仍是自对弈（learner 自己打覆盖率抽的牌）。
`train_native.py` 的这条断言是有意的安全阀，不是可以调的参数。

另外：bank 的卡组是**钉死**的（`bank-deck` override 直接覆盖覆盖率抽样），
所以「对手卡组分布」实际由 **bank 槽位分配**决定 —— 本工具用最大余数法按覆盖率分配 16 个槽位来近似。

### 目前生效的实际效果

`choice` 表里 **A=26c6/slowking 这一行全是 `init`**（slowking 的驾驶员数据几乎没有），
所以这个功能现在跑出来等价于"端口权重当对手"。矩阵里任何一个格子从 `init` 变成训练件，
重新跑 `teacher_loop.sh` 就自动生效，**不需要改代码**。

### 待办

- `/root/kl/` 需要补 `ph3_2617_v1.bin / ph3_2617_v2.bin / ph3_9d6e_v2.bin`
  （`DEFAULT_PILOT_MAP` 已写死这些路径，缺文件会回退 init 并告警）。
- `matrix_full.csv` 的 7 副牌之外（`e8ee`/`db04`/`0904`/`26c6`/`e1d7`）的驾驶员对比仍缺数据，
  补齐后矩阵会自动出现更多非 init 格。

## A/B 切换改成动态（2026-11-06）

### 之前为什么是固定步数

`teacher_loop.sh` 最初按固定 `STEPS` 跑每个阶段——这是我图省事。参考工程是

> "Train the nets in turns, starting with Net A until it **improves its WR by 2pp** over baseline,
> then freeze Net A and start to train Net B to recover the lost points, repeat."

固定步数的代价：A 可能 20M 就到顶、后 30M 白跑；B 也可能早该切却没切。

### 现在的实现（`tools/ab_watch.py` + `teacher_loop.sh DYN=1`）

**原料是免费的**：训练器每个 epoch 都打印 per-bank 胜率

```
epoch 50 step 1638400 ... lgw 0.612 [1:0.58/0.05][2:0.71/0.05]...
                          ↑ bank 胜率/出现率 —— bank 钉死了卡组，所以这就是"对某副牌的胜率"
```

**触发逻辑**：
1. 覆盖率加权对位胜率 `wr_bank = Σ coverage_b × wr_b / Σ coverage_b`；
2. `baseline` = 前 `--warm` 个 epoch 的均值（默认 30）；`ema` = 最近 `--ema` 个 epoch（默认 20）；
3. **A 阶段**：`ema - baseline ≥ THRESH_PP`（默认 **+2pp**）→ `pkill` 训练，取最新快照当 A；
4. **B 阶段**：`ema - baseline ≥ THRESH_PP`（B 追回失分）→ 停，本轮结束。

**快照密度**：DYN 模式默认 `CHECKPOINT_INTERVAL=20`（= 655k 步），
杀训练时最多丢 0.7M 步（固定模式是 100 epoch = 3.3M）。

### 噪声提醒（预注册）

league 占 49% 时，每 epoch 约 **100 局** league 对局；`warm=30` + `ema=20` 的窗口约 2000–3000 局
⇒ 加权胜率的 CI 约 **±2pp**。也就是说 **+2pp 的触发阈值贴着噪声底**，会偶发误触发。

可调的三个旋钮：
- `THRESH_PP=3.0` —— 更保守（推荐先用这个）；
- `LEAGUE_PCT` 提高（上限 0.49）→ 每 epoch 样本更多；
- `--warm/--ema` 放大 → 曲线更平滑但响应更慢。

## 多轮交替：每轮必须带上"增强后的 A/B"（2026-11-06）

参考工程是 **6–8 轮**交替（`--ft-balance-mode turns`：*"promotion-driven strict alternation; phase ends
when the trainee earns its cross-gate promotion, `--ft-phase-timebox` 120M backstop"*），
门槛 `--cross-gate-delta 0.03` = **3pp**，且是相对"冻结 vs 冻结"基线（因为对位 EV ≠ 0.5）。

我们原来的 `teacher_loop.sh` 有个致命缺陷：**多轮时 A 的对手仍来自静态 choice 表，B 的增强权重从不回流**，
第 2 轮等于把第 1 轮重跑一遍。已修：

| 阶段 | 第 1 轮 | 第 i≥2 轮 |
|---|---|---|
| A 训 | 对手 = 矩阵驾驶员（9 槽） | 对手 = 矩阵驾驶员 **+ 当前 B 的 2 个槽**（`B_SLOTS=2`，不钉牌 → 按覆盖率抽牌） |
| B 训 | 对手 A = 刚训完的 A | 对手 A = 上一轮的 A（`B_OPP=A` 默认；可设 `init` 回退） |

槽位分配：`league_pct = 0.49 / (矩阵槽 + B_SLOTS)`，仍然卡在引擎 50% 上限内。
watcher 的加权里，B 槽按覆盖率均值 1.0 计（它打全池）。

**默认值同步**：`ROUNDS=3`（先用 3 轮验证，参考工程是 6–8）、`THRESH_PP=3.0`（对齐 cross-gate-delta 0.03）。

⚠️ 语义提醒：第 1 轮 B 阶段的门控晋升用的是"镜像 h2h A vs teacher"，
而 B 阶段对手是 A —— 两者测的不是同一个量，**不要用 B 阶段的训练胜率当晋升依据**。

## 切换到纯交替自博弈（MODE=ab，2026-11-06）

用户拍板：**与参考工程对齐**——A/B 攻防交替，一侧稳定提升 **2pp** 就换边，跑满指定轮数后两侧停止。
不再需要"49% 冻结 + 51% 自对弈"的混合方案。

### 机制差异（为什么这次可以不要 50% 上限）

| | 混合模式（MODE=matrix） | 纯交替（MODE=ab） |
|---|---|---|
| 行构成 | 49% 冻结对手 + 51% 自对弈 | **100% 冻结对手**（单智能体收集） |
| learner 策略梯度 | 自对弈半边给双份（negamax） | 只有自己那半边 |
| `league_pct × n_banks` | 必须 <0.5 | **=1.0** |
| 对手 | choice 矩阵驾驶员（静态清单） | **当前 B / 当前 A**（活对手，每轮更新） |

参考工程的 finetune 就是后者：*"the live net always pilots --finetune-deck; the other seat is a
FROZEN net (--opponent-ckpt)... **Collection becomes single-agent**"*。

### 改动

1. **`train_native.py`**：`assert league_pct*n < 0.5` → `assert 0 < league_pct*n <= 1.0`，
   并在一侧 >50% 时打警告（提醒"纯交替模式：learner 没有 live 对手信号，靠 teacher KL 锚 + 离线审计"）。
2. **`teacher_loop.sh`**：新增 `MODE=ab`（默认）。A 阶段 `BANKS=$B`（不钉牌 → B 按覆盖率抽全池）、
   `LEAGUE_PCT=1.0`；B 阶段 `BANKS=$A`（绑舰牌）、`LEAGUE_PCT=1.0`。
3. **修 bug**：`A_NEW/B_NEW` 跨轮未重置，导致第 2 轮 B 阶段的对手停在第 1 轮 A 上（已修，dry-run 验证）。
4. **`launch_ab.sh`** 默认：`MODE=ab`、`THRESH_PP=2.0`、`ROUNDS=3`。

### 交替链（dry-run 实测）

```
r1A: INIT=init        BANKS=B(init)      league=100%
r1B: INIT=init        BANKS=A(r1A)       league=100%
r2A: INIT=A(r1A)      BANKS=B(r1B)       league=100%   ← 增强的 B 回流成对手
r2B: INIT=B(r1B)      BANKS=A(r2A)       league=100%   ← 增强的 A 回流成对手
... 满 ROUNDS 轮后停止
```

### 附带收益与风险

- **收益**：league 行从 49% 提到 100% ⇒ 每 epoch 样本翻倍，加权胜率噪声 ±2pp → **±1.4pp**，2pp 阈值更可信。
- **风险**：learner 完全没有 live 对手信号，对手质量决定一切（Goodhart）。
  缓解：KL 回拉 α=0.10 + 门控晋升（镜像 h2h）+ **只用离线审计判断**，不看训练胜率。
