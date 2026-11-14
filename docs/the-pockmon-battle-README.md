# 三张 battle 表的填充说明

生成器：`tools/build_battle_tables.py`（可重跑，幂等覆盖）

```bash
./tools/py tools/build_battle_tables.py
```

## 编码与格式

- **编码 GB18030**（模板原本是 GBK；选手名含越南语/土耳其语字符，GBK 编不下，故用其超集）。
  Excel 直接双击可正常显示。
- 表头与模板**逐字一致**，未增删列。

## 卡组名聚合规则（2026-11-06 更新：真复合用 A-B，微调仍用主体名）

原始卡组名是 `<archetype>__<hash>`，太长且同 archetype 有多个 decklist。统一映射到简洁名，规则：

| 情形 | 命名 | 例子 |
|---|---|---|
| **真·复合**：两条线各 ≥4 张（各是一个完整体系） | **A-B**（按卡数降序，最多 3 段） | `dunsparce_dudunsparce` → **dunsparce-froslass**；`abra_kadabra` → **alakazam-dunsparce** |
| **微调**：第二条线只有 2~3 张 | **单主体名** | `dreepy_drakloak`（munkidori 仅 1.5 张）→ **dragapult**；`dwebble_crustle`（kangaskhan 1.7 张）→ **crustle** |

**判据是实测卡数**（不是猜）：对每个 archetype，取 `data/deck_corpus.csv` 里它的所有 60 张实卡，
按卡表进化链归到各条线，算平均卡数；**≥4 张 = 真体系，2~3 张 = 微调**。

46 个 archetype → **36 个名字**，其中 **23 个复合名**。完整表在
`tools/build_battle_tables.py` 的 `ARCH2SHORT`（改一行就能覆盖生成结果）。

典型例子：

| archetype | 卡数构成 | 名字 |
|---|---|---|
| `teal_mask_ogerpon_ex_applin`（我们的 7e39） | applin 6.0 + meganium 5.5 + ogerpon 4.0 | **applin-meganium-ogerpon** |
| `dunsparce_abra`（我们的 0bf5/9d6e/db04） | alakazam 11.9 + dunsparce 7.0 | **alakazam-dunsparce** |
| `grookey_thwackey`（我们的 2617） | rillaboom 8.0 + applin 7.7 | **rillaboom-applin** |
| `shuppet_dhelmise`（我们的 0d86） | shuppet 7.0 + dunsparce 5.0 + dhelmise 4.0 | **shuppet-dunsparce-dhelmise** |
| `dreepy_drakloak` | dragapult 10.9 + munkidori 1.5 | **dragapult**（微调） |
| `slowpoke_slowking` | slowking 7.0 + kangaskhan 3.0 | **slowking**（微调） |

⚠️ **判断说明**（可覆盖）：
- 顺序按**卡数降序**，不是按"招牌卡"。所以 7e39 排成 `applin-meganium-ogerpon`（applin 卡最多），
  若想按招牌名排成 `ogerpon-applin-meganium`，改 `ARCH2SHORT` 一行即可。
- 3 段的（`shuppet-dunsparce-dhelmise`、`applin-meganium-ogerpon`、`dwebble-cinderace-kangaskhan`、
  `great_tusk-koraidon-stonjourner`、`grimmsnarl-froslass-munkidori`、`dwebble-froslass-munkidori`）
  都是三条线各 ≥4 张，压缩到 2 段会丢信息。
- 两个 1~4 样本的怪牌手动修正：`hop_s_snorlax_hop_s_cramorant` → `cramorant-kangaskhan`；
  `girafarig_enamorus` → `girafarig`（实卡里没有 girafarig 线，属数据噪声）。

## cards.csv —— 卡组宇宙（36 行，按主体/复合聚合）

| 列 | 口径 |
|---|---|
| `DeckID` | 该主体下**样本最多的那个 decklist** 哈希（代表，不是穷举） |
| `卡组` | 主体/复合名（上节映射） |
| `Meta 占比` | 该主体在 1756 局真人天梯档案里的出现占比（同一主体多个 decklist 相加） |
| `样本局数` | 座位出现次数（一局 2 个座位） |

⚠️ 同样有**采样偏差**：档案里 slowking 28.5% / shuppet 28.5% / alakazam 22.6% 是因为两位选手的局数占大头。
我们自己的实测场地（582 局）里是 ogerpon 34.7%、rillaboom 11.7%……见 `docs/opponent_pool_weights.md`。

## match.csv —— 对位记录（491 行，按主体聚合）

**三条整理规则**：

1. **同一对只留一条**：以「(主体, 玩家) 实体对」为唯一键（同一主体的不同 decklist 会塌成一行），同一对的两个方向**相加合并**
   （例：`f5f8/YumeNeko` vs `26c6/ミワハルキ` 原本两个方向 110-113 与 107-115，
   合并后 225-220 共 445 局，只占一行）。
2. **胜率高的一侧放左**（`玩家卡组/玩家` 列），另一侧放右（`对手卡组/对手`）⇒ 每行左边胜率恒 ≥ 50%。
3. **排序**：按 左卡组名 → 左玩家 → 右卡组名 → 右玩家。卡组名是 ASCII（archetype__hash），
   即**字母序**；中文选手名因未装 `pypinyin`，按 Unicode 码位排序（安装后会自动改走真拼音）。

用 `玩家` 列区分三块来源：

| 块 | `玩家` / `对手` 列 | 来源 |
|---|---|---|
| ① 真人 vs 真人 | 选手名（如 `YumeNeko`） | `data/ladder_games.tsv`，同组合 **≥3 局**才收录 |
| ② 我们 vs 对手 | 我们的提交件名（`g6_db04`/`ala`/`g0002`…）/ 对手选手名 | `data/our_games.tsv`，640 局 |
| ③ 端口 vs 端口 | 两边都是 `端口权重` | `data/scheduler/matrix_full.csv`，7×7 矩阵 |

- `胜局/败局` 由原始记录聚合；`平局` **全部为 0**（数据源没有平局记录，不是漏填）。
- **对手卡组未识别（`?`）的 58 局不进表**——归不到具体卡组，计入只会污染统计。
- 原始 574 行（真人 92 + 我们 433 + 端口 49）→ 同对合并 + 按主体聚合后 491 行。

⚠️ **踩过的坑**：`our_games.tsv` 的卡组列存的是**全名**（`dunsparce_abra__9d6e`），
早期版本直接 `[:4]` 取 ID，得到 `duns`/`teal`/`abra` 这种假 ID，把表污染了 47 行。
现在统一走 `hid()`：有 `__` 就取后缀，`?` 直接判为未识别。

## players.csv —— 选手与克隆体（104 行）

| 列 | 口径 |
|---|---|
| `DeckID` / `卡组` | 同 cards.csv |
| `玩家` | 真人选手名，或我们本地的 agent 目录名 |
| `类型` | `真人` / `端口基线` / `BC克隆` / `PPO训练` |
| `父玩家` | 克隆体的启发来源：BC 克隆填其语料贡献最大的真人教师；端口/PPO 留空 |
| `Checkpoint` | 本地权重路径 `build/agents/<name>/model.pt`；真人为空 |
| `数据来源` | 真人是 `Kaggle 回放（N 局）`；本地 agent 是 `端口仓库权重` / `我们训练` |

**类型是按权重身份判定的，不是按目录名前缀**：

| 权重 md5（前 12） | 判定 | 例子 |
|---|---|---|
| `8b5fcda672ac` | 端口基线 | `port_7e39`、`c_grookey`、`port_0904`… 共 12 个 |
| `6007d6f083` | BC克隆 | `g8_bc_parent`、`opp_shuppet_human` |
| 其它 | PPO训练 | `g6_db04`、`g8_shuppet`、`ph3_2617_v2`… 共 8 个 |

（这条判定规则是踩坑后加的：早期按目录名前缀分类，把 `opp_shuppet_human` 误判成端口权重，
它其实是 shuppet 的 BC 父权重。）

## 已知缺口

1. **冠军卡组的真人样本极少**：`0904`(Petit Canard 1510)、`e1d7`(ミワハルキ 1580) 在我们抓的档案里
   几乎不出现，cards.csv 里占比为 0 —— 不是没有玩家打，是没抓到回放。
2. **平局数据缺失**（见上）。
3. `DeckID` 只保留 4 位哈希，理论上可能撞车（目前 119 副无冲突）。

## the-pockmon-battle-choice.csv —— 每格该用哪个驾驶员

**结构**：行 = 挑战者卡组，列 = 被挑战卡组，格 = 打这一格胜率最高的**驾驶员**（agent 名）。

**生成器**：`tools/build_choice_table.py`（只做汇总，不新增对局）

**选择规则（带噪声阈值，不裸取 max）**：

1. 某个**训练过的驾驶员**比 `init`（端口权重）高 **≥5pp** ⇒ 用它；
2. 否则用 `init`；
3. 只有 `init` 没测过、其他驾驶员有数据时，才退回"取最高"并在明细里标注。

**为什么加 5pp 阈值**：单格 n=200 的 95% CI ≈ ±6.5pp，裸取 max 等于**系统性挑中幸运值**（赢家诅咒）。
实测 78 个非 `init` 格子里只有 **7 个**真的领先 ≥5pp，其余 71 个都在噪声内——所以表里绝大多数格子是 `init`。

**当前结果**（12×12，78 格有数据）：

| 驾驶员 | 格数 | 说明 |
|---|---|---|
| `init` | 68 | 端口权重（默认） |
| `g6_db04` | 8 | db04 专精（天梯 1060），其中 5 格是"只有它测过" |
| `shuppet_specialist` | 1 | `g8_bc_parent` 打 shuppet，`0d86→3eaa` +12.2pp |
| **`r1A`** | **2** | 10M 步 slowking 专才：`26c6→26c6`（镜像）+12.0pp、`26c6→0d86` +7.0pp |
| `ph3_2617_v1` | 1 | `2617→26c6` +8.0pp |

**唯一的大信号是 shuppet**：`db04 → 0d86`，`g6_db04` 0.980 vs `init` 0.020（+96pp）——
因为端口权重根本不会打 shuppet（全场地 0.01 胜率），这正是之前就该换 BC 驾驶员的那一格。

**默认填充（66 格）**：没测过的格子**先填 `init`**（假定未知就用端口权重），
后续训出更强的驾驶员再覆盖。明细表用 `状态` 列区分：
- `实测`（78 格）—— 有真实对局数据，按 5pp 阈值选；
- `默认`（66 格）—— 没有数据，先按 `init` 填。
缺数据的主要是 `26c6`/`e1d7`/`e8ee` 从未作为挑战者评测，以及 `init` 对 `0904`/`26c6`/`e1d7`/`e8ee` 没测过。

**配套**：
- `the-pockmon-battle-choice-detail.csv` —— 全部 144 格 + 胜率/n/**状态（实测/默认）**/依据
- `the-pockmon-battle-choice-all.csv` —— 全部原始记录（含落选驾驶员），供审计

### 2026-11-12 更新：r1A（slowking 10M 步专才）进表

- 数据源：`logs/meta/meta_r1A/`（12 格 × 200 场，seed 950000）+ 同协议基线 `logs/meta/meta_port_26c6/`
- 判据照旧（≥5pp）：r1A 只在 **2 个对位**上达标
  - `26c6 → 26c6`（镜像）**0.562 vs 0.443，+12.0pp**
  - `26c6 → 0d86` **0.920 vs 0.850，+7.0pp**
- 其余格子不达标（7e39 +1.2、2617 +2.0、db04 +3.0、e1d7 +2.7；0bf5 −5.5、9d6e −6.8）
- **加权净效果 −0.6pp**（0.2592 → 0.2527）⇒ 进表是"局部最优"，不代表整体变强
- 26c6 行的其余 10 格仍是 `init`（没有训练件领先 ≥5pp）

⚠️ 注意：slowking(26c6) 在我们实测场地的占比是 **0%**，所以这一行目前**不影响天梯**；
它的价值是"为一个未来的 slowking 提交/对手提供最强驾驶员"。

## cards.csv 新增第 5 列「最高LB」（2026-11-12）

口径：**每副牌最高的可查天梯分**。

| 来源 | 用途 |
|---|---|
| `data/deck_corpus.csv` + `data/lb_scores.json` | 外部强玩家用这副牌打出的最高分 |
| `data/lb_ours.json` | **我们自己提交过**的牌，用我们的天梯分补 |
| 两者都没有 | **NA** |

### 为什么有「（xx·我们）」这样的行

`cards` 按**主体聚合**（如 `alakazam-dunsparce` 把 0bf5/9d6e/f5f8/e8ee/db04 合成一行），
聚合后取的是**组内最高分**（1428，Ondrej 的 f5f8/0bf5）。
这会**盖掉我们自己 decklist 的分数**（db04 只有 1060.3）。所以当我们的分被组内更高分盖住时，
单独补一行 `xxx（hash·我们）`，让两个数都能看到。

### 关键读数

| 主体 | 组内最高 LB | 我们自己的 |
|---|---|---|
| `shuppet-dunsparce-dhelmise` | **1594.4**（ミワハルキ） | 0d86·我们 779.6 |
| `slowking` | **1594.4**（ミワハルキ） | — |
| `applin-meganium-ogerpon`(7e39) | **1446.2**（Dipam） | 7e39·我们 880.9 |
| `alakazam-dunsparce`(0bf5/9d6e/f5f8/db04) | **1428.0**（Ondrej） | 9d6e·我们 992.1、db04·我们 **1060.3** |
| `rillaboom-applin`(2617) | **1312.9**（Darren） | — |
| `ogerpon`(e1d7) | 736.2 | e1d7·我们 875.5 |

⇒ **我们所有牌的天花板都远高于自己的成绩**：最强主体能到 1428~1594，我们最高 1060.3。
NA 只剩 2 行（`girafarig`、`voltorb`，各 1~2 局、无分数记录）。

## the-pockmon-battle-choices-win-rate.csv —— a→b 胜率矩阵（2026-11-12 新增）

**结构**：行 = 我方（挑战者）卡组 a，列 = 对手（被挑战者）卡组 b，格 = **我方 a 打对手 b 的胜率**（0~1，4 位小数）；
**对角 `a→a`（自己打自己）填 `/`**，按约定不做镜像对局统计。

**口径**（与 choice / choice-detail 两张表一致）：

| 侧 | 驾驶员 | 依据 |
|---|---|---|
| 我方 a | 该卡组**最强驾驶员** | 取自 `choice.csv` 选出来的那个：`db04→g6_db04`、`2617→ph3_2617_v1`、`26c6→r1A`、`0d86→shuppet_specialist`，其余 = `init`（端口权重） |
| 对手 b | **端口权重 + 该卡组** | 与 testset_meta / panel 的口号"只差卡组"一致，保证格子之间只差牌 |

**数据来源（两段，都能追到文件）**：

1. **老 eval**（90 格）：`the-pockmon-battle-choice-detail.csv` 里 `状态=实测` 的行，n=178~300，同一格不同来源协议略有差异；
2. **新补**（本轮 51 格）：`tools/fill_winrate.sh` —— **200 场 Bo3、座位交替、seed-base 970000、单协议**，
   结果落在 `logs/winrate/<a>_<b>.json`。老 eval 覆盖的 90 格里，凡"反向有数据但驾驶员/协议不匹配"的 25 格也一并重跑，
   避免用 `1−反向` 推算出错配的对位（例：`db04→0904` 的反向是 `g6_db04` 打端口，不是端口打端口）。

**生成器**：`./tools/py tools/build_winrate_table.py`（`--dry` 只看覆盖率；新跑结果优先于老 eval）

**读数要点 / 已知陷阱**：

- **`(a,b) + (b,a) ≠ 1`**。两侧驾驶员不同（如 `db04` 行用 `g6_db04`，别行用 `init`），且老数据 n 不一致。
  最典型：`2617→0d86 = 1.0000` 而 `0d86→2617 = 0.2900`（和 = 1.29）——这一格老数据用的是 `opp_shuppet_human`
  这个强驾驶员，不是端口。**不能拿对角外侧相加当"自洽性检查"**，只能逐格看。
- **极端值是真的，不是 bug**：`0d86→db04 = 0.0000`、`0d86→e1d7 = 0.0150`、`0d86→e8ee = 0.0000`
  （端口权重根本不会打 shuppet）；反向 `db04→0d86 = 0.9800`。这类格子**不能插值成 0.5**。
- 覆盖率：老 eval 90 + 新补 51 = 141 格，去重后 132 个非对角格全覆盖（对角 12 个填 `/`）。

**为什么要补这 51 格**：`(a,b)` 和 `(b,a)` 是两次独立的 `eval_pair` 任务，老 eval 只围着天梯占比 98% 的 7 副牌做，
`e1d7`/`e8ee` 整行整列零数据（它们天梯占比 0%，从没当过对手也没当过"我方卡组"）。

**在哪跑**：`tools/pack_eval_cpu.sh` 打 24 MB 纯 CPU 包（引擎 `libcg.so` 走 ctypes，不绑 Python ABI），
32 核机上 16 面板并行，单格 **84~135 秒**（本机 12 核/6 面板要 320 秒）。

### 2026-11-13 更新：`9d6e` 行换成多种子数据，并引入"驾驶员 sidecar"

**新增 sidecar**：`the-pockmon-battle-choices-win-rate-pilot.csv`（我方卡组 → 驾驶员 → 面板 tag → 每格 n）。
纯数字矩阵看不出"这一行是谁跑的"，sidecar 把可追溯性补上；`build_winrate_table.py` 会按它覆盖对应行，
所以重建不会回落成老的 init 数据。

**`9d6e` 行现在用 3 个种子合并（n≈600/格）的端口驾驶员**：

```
./tools/py tools/set_winrate_pilot.py --ship 9d6e --pilot init \
    --tag meta_port_9d6e --tag meta_port_9d6e_s950001 --tag meta_port_9d6e_s950002
./tools/py tools/build_winrate_table.py
```

变化（旧 → 新）：`0904 0.455→0.472`、`2617 0.420→0.452`、`26c6 0.733→0.718`、`3eaa 0.680→0.630`、
`7e39 0.875→0.898`、`e1d7 0.925→0.933`、`e8ee 0.530→0.513`、`f5f8 0.570→0.513`。
单格 n 从 178~300 提到 600，CI 从 ±6.5pp 收到 ±3.7pp。

**为什么这一行的驾驶员不是本轮新训的 `ab9d6e_r1A`**：3 种子合并后 `ab9d6e_r1A` 的加权
Δ = **+0.19pp**（95% CI `[−1.74, +2.11]`）——统计上与端口不可区分。单种子时的 "+2.4pp" 其实是
**基线在 seed 950000 上偏低**造成的（基线自己从 0.6824 跳到 0.7044）。

但它在**两个具体对位上真的有优势**（n=600、Bonferroni 校正后仍显著）：
`abra_e8ee +12.8pp`、`abra_f5f8 +11.8pp`；代价是 `镜像 dunsparce_9d6e −8.8pp`。
所以处理方式是：**matrix 行整体仍用端口**，把这两个格子写进 `choice.csv`（见下）。

### 2026-11-13 更新：`choice.csv` / `choice-detail.csv` 纳入 `ab9d6e_*` 四个驾驶员

| 驾驶员 | 来源 | 舰牌 | 步数 |
|---|---|---|---|
| `ab9d6e_r1A` | A/B 第 1 轮 A | 9d6e | 4,849,664（+ 另 2 个种子面板 n≈600） |
| `ab9d6e_r1B` | 第 1 轮 B（通才） | 9d6e（评估时） | 4,030,464 |
| `ab9d6e_r2A` | 第 2 轮 A（对手换成 r1B） | 9d6e | 4,030,464 |
| `ab9d6e_r2B` | 第 2 轮 B | 9d6e（评估时） | 3,670,016 |

> ⚠️ 名字带 `ab9d6e_` 前缀是必须的：**上一轮 slowking 专才也叫 `r1A`**（`data/ab_run1/r1A.bin`，
> choice 表里 `26c6` 那两格用的就是它）。不加前缀会在表里撞车。

本轮只在 `9d6e` 行改了 **2 格**：

| 格 | 原 | 现 | 依据 |
|---|---|---|---|
| `9d6e → e8ee` | init 0.513 | **ab9d6e_r2A 0.670** | +15.7pp，3 种子 n=600，CI[+10.1,+21.0] |
| `9d6e → f5f8` | init 0.513 | **ab9d6e_r2A 0.653** | +14.0pp，3 种子 n=600，CI[+8.4,+19.4] |

> 这两格先被 `ab9d6e_r1A` 占过（+12.8/+11.8pp，3 种子），补了 `ab9d6e_r2A` 的 3 个种子后它的领先更大
> （+15.7/+14.0pp），按"取最高且已验证"换成 `r2A`。`ab9d6e_r1B` / `ab9d6e_r2B` 仍只有单种子数据，
> 留在 UNVERIFIED 名单里、只入明细表。

**选表规则加了一条硬门槛**：只有**多种子验证过**的驾驶员才有资格改表
（单种子 n=200 的 ±6.5pp 噪声会让"取最高"系统性挑中幸运值）。因此 `ab9d6e_r1B/r2A/r2B`
只进明细表、不参与选表——`r2A` 在单种子上对 `e8ee/f5f8` 分别有 +15.0/+14.0pp，
但那是没验证过的数，先不用。

`依据` 列现在会标 `多种子验证 n=600` 或 `单种子 n=200`；**标"单种子"的格子是启发式，未经证实**
（例如本轮 `9d6e → 9d6e` 镜像格因基线刷新变成 `ph3_9d6e_v2` 0.540 vs init 0.488，就是单种子 +5.2pp）。

## the-pockmon-battle-choice-in-table.csv —— 自家"最佳驾驶员"之间的内部对局（2026-11-13 新增）

**用途**：挑"最泛用的 agent"。`choices-win-rate.csv` 每行 = 一副我方卡组 + 它的最佳驾驶员；
本表记录**这些最佳驾驶员两两之间**打过什么、谁赢。

**规则**：

1. 12 副卡组 → C(12,2) = **66 个无序对**，每个对局**只记一条**（不是两个方向各一条）；
2. **以获胜多的一方作为挑战者**：行里 `挑战者` 是胜率高的一方，`胜率` 恒 ≥ 0.5；
3. 按 `挑战者` 名字字典序排序（同挑战者再看被挑战者）；
4. `驾驶员` 列 = **挑战者**用的驾驶员，被挑战者用的写在 `来源` 列里。

**驾驶员口径**（与 win-rate 矩阵每行一致）：`db04→g6_db04`、`2617→ph3_2617_v1`、`26c6→r1A`（slowking 10M 专才）、
`0d86→shuppet_specialist`（= `g8_bc_parent`），其余 = `init`（端口权重）。

**数据来源**：

- **两侧都是 `init`**（28 对）：由 win-rate 矩阵双向推出 —— 矩阵 `(a,b)` 是 `init@a vs 端口@b`、
  `(b,a)` 是 `init@b vs 端口@a`，同一个对局的两次独立测量，合并 `cell(a,b)` 与 `1 − cell(b,a)`
  （胜场与场次相加，n=400）。`来源` 列写 `derived a→b/b→a`。
- **任一侧不是 `init`**（38 对）：必须实跑，`tools/run_in_table.sh` 产出 `logs/in_table/<a>_<b>[__s2].json`，
  每对 **2 个种子 × 200 场 Bo3**（总场次与一次 400 场相同，但种子级方差减半 —— 见 `docs/ab_run2.md` §14.1）。

**生成器**：`./tools/py tools/build_in_table.py`（`--dry` 只打印还缺哪些，并写 `logs/in_table_missing.txt`）。

**当前状态**：28/66 已填；缺的 38 对全部涉及 4 个非 `init` 驾驶员
（`shuppet_specialist` 12 对、`ph3_2617_v1` 11 对、`r1A` 11 对、`g6_db04` 12 对，含互相之间的 6 对），
评估机在跑到一半时失联，待机器恢复后 `bash tools/run_in_table.sh` 幂等补齐。

### 2026-11-13：CPU 机自动接手（`tools/watch_cpu_and_run.sh` + `tools/run_cpu_all.sh`）

评估机会随时被停，所以把"等机器 → 部署 → 跑完 → 拉结果 → 刷新表"做成一个可挂着的守护脚本：

```bash
bash tools/watch_cpu_and_run.sh root@113.44.139.218    # 每 60 秒探一次，最多 12 小时
```

机器上线后它会（全部幂等，断线重连后继续）：

1. **远端没 kit** ⇒ 本地 `pack_eval_cpu.sh`（162 MB，含 12 个面板对手 + 4 个候选 + 4 个历史驾驶员）→ 推 → `setup.sh`；
2. **有 kit 但缺驾驶员** ⇒ 只补推 4 个 `model.pt` 并铺 agent 目录；
3. 起 `tools/run_cpu_all.sh`，串行跑三件事：

| # | 任务 | 规模 |
|---|---|---|
| ① | 补 `choice-in-table` 缺的自家对局 | 38 对 × 2 种子 × 200 场 |
| ② | 给只有单种子数据的驾驶员补种子（`ab9d6e_r1B` / `ab9d6e_r2B`） | 4 个面板 |
| ③ | `9d6e→2617`（权重 9.6% 的最大漏分格）做到 n=1000 | 2 个候选 |

4. 跑完自动拉回 `logs/{in_table,meta,hiN_*}` 并执行 `build_in_table.py` 刷新表。

## the-pockmon-battle-choice-self.csv —— 每副牌"自己的最强驾驶员"（2026-11-13 新增）

**为什么需要**：`choice.csv` 的格子是「**我方**打这格该用谁」，回答不了"这**副牌**交给谁驾驶最强"。
而训练时的对手池需要后者 —— 每副对手牌都应该由它自己的最强驾驶员来开。

**生成器**：`./tools/py tools/build_pilot_table.py`
（读 `choice-detail.csv` 推导 + `choice-all.csv` 取同格 init 基线）

**规则**：对该牌自己那一行里所有**非 init** 驾驶员记录，取胜率最高者；该行全是 init 则记 `init`。

**当前结果（12 副牌，5 副有非 init 驾驶员）**：

| 卡组 | 最强驾驶员 | 胜率 | 样本 | 相对同格 init |
|---|---|---|---|---|
| `db04` | `g6_db04` | 0.9800 | 200 | **+96.0pp**（init 0.02） |
| `9d6e` | `ab9d6e_r2A` | 0.6700 | 600 | **+15.7pp**（init 0.5133） |
| `0d86` | `shuppet_specialist` | 0.5618 | 178 | **+12.2pp**（init 0.4400） |
| `2617` | `ph3_2617_v1` | 0.3300 | 200 | +8.0pp（init 0.2500） |
| `26c6` | `r1A`（slowking 专才） | 0.9200 | 200 | +7.0pp（init 0.8500） |
| `0904` / `0bf5` / `3eaa` / `7e39` / `e1d7` / `e8ee` / `f5f8` | `init` | — | — | 该行无非 init 驾驶员 |

**消费方**：`tools/choice_to_banks.py --strong` 默认读这张表（`--pilot-table` 可覆盖），
把"对手驾驶员"逐槽填进 `BANKS`，按场地权重分配槽位（`--cov-src weights`）、
`--quota 1.0` 时全部行都是冻结对手。

```bash
./tools/py tools/build_pilot_table.py                       # 先刷新表
./tools/py tools/choice_to_banks.py --a-deck 7e39 --strong --max-banks 8 --quota 1.0 --cov-src weights --emit-env
```

## 数据母表与重建流程（2026-11-14）

**母表**：`the-pockmon-battle-choice-in-table.csv`（列：`挑战者, 被挑战者, 驾驶员, 胜率, 样本场次, 来源`）
= **唯一证据源**，两条命令即可从它重建出两张交付表：

```bash
./tools/py tools/export_master.py       # ① 从所有证据源导出母表
./tools/py tools/build_from_master.py   # ② 由母表重建 choice / choice-detail / win-rate / pilot
```

### 证据源（按优先级）

| 优先级 | 来源 | 说明 |
|---|---|---|
| 3 | `logs/meta/<tag>/vs_*.json` | 12 格面板；多种子（`_s95000x`）先按格池化 |
| 2 | `data/opp_matrix/panel_*.json` | 7 格历史面板（cand = 我方 agent） |
| 1 | `data/scheduler/matrix_full.csv` | 49 格远程矩阵（双方端口） |

### 由母表重建的四张表

| 表 | 规则 |
|---|---|
| `choice.csv` / `choice-detail.csv` | **格级**：该格非 init 驾驶员需比同格 init 高 **≥5pp 且池化 n≥400**（多种子验证），否则用 `init` |
| `choices-win-rate.csv` | **行级**：每副牌按 12 格**场地加权**选一个驾驶员（比 init 高 ≥1pp 才替换），整行用该驾驶员的胜率；对角线填 `/` |
| `choices-win-rate-pilot.csv` | 每行的驾驶员 + 场地加权 + 样本（可追溯） |

> 两张表的判据不同是有意的：`choice.csv` 回答"这一格我方派谁"，可以逐格换人；
> `choices-win-rate.csv` 回答"这副牌整体交给谁"，必须整行一致，否则矩阵不可比。

**当前母表**：219 条记录 / 95 格 / 14 个驾驶员。
