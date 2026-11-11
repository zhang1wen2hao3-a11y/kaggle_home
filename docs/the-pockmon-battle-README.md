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
