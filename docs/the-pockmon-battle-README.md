# 三张 battle 表的填充说明

生成器：`tools/build_battle_tables.py`（可重跑，幂等覆盖）

```bash
./tools/py tools/build_battle_tables.py
```

## 编码与格式

- **编码 GB18030**（模板原本是 GBK；选手名含越南语/土耳其语字符，GBK 编不下，故用其超集）。
  Excel 直接双击可正常显示。
- 表头与模板**逐字一致**，未增删列。

## 卡组名聚合规则（2026-11-06 新增）

原始卡组名是 `<archetype>__<hash>`（如 `slowpoke_slowking__26c6`），太长且同一 archetype 有多个 decklist。
现在统一映射到**简洁主体名**，规则是「取牌组主体（主攻手/最终进化），主体基本一致就归同一个名字」：

| 例子 | 简洁名 | 判定依据 |
|---|---|---|
| `slowpoke_slowking__*` | **slowking** | 进化线慢波→呆壳兽，最终进化 |
| `teal_mask_ogerpon_ex_applin__*` | **ogerpon** | 主攻手是厄鬼椪 |
| `dunsparce_abra__*` / `abra_kadabra__*` | **alakazam** | 60 张里 Abra+Kadabra+Alakazam 有 12 张、Dunsparce+Dudunsparce 只有 7 张 ⇒ 主体是胡地，土龙节节是抽牌引擎 |
| `dreepy_drakloak__*` / `z_dragapult_dreepy_petit__*` | **dragapult** | 进化线最终形态 |
| `shuppet_dhelmise__*` / `shuppet_banette__*` | **shuppet** | 约定俗成 + 本项目一贯叫法（进化终点其实是 Banette） |

**46 个 archetype → 26 个主体名**。完整映射表在 `tools/build_battle_tables.py` 的 `ARCH2SHORT`，
改一行就能重新归属（表是可审计、可覆盖的，不是硬编码猜的）。

几个判断说明（可覆盖）：
- `dunsparce_dudunsparce` / `dudunsparce_dunsparce` 保留为 **dudunsparce**（没有胡地线，纯土龙节节引擎）。
- `raging_bolt_ex_teal_mask_ogerpon_ex` 归到 **ogerpon**（虽然主攻是猛雷鼓，但只有 1 个样本，影响极小）。
- `mega_kangaskhan_ex_dwebble` / `cinderace_mega_kangaskhan_ex` / `slowpoke_mega_kangaskhan_ex` 都归 **kangaskhan**。

## cards.csv —— 卡组宇宙（26 行，按主体聚合）

| 列 | 口径 |
|---|---|
| `DeckID` | 该主体下**样本最多的那个 decklist** 哈希（代表，不是穷举） |
| `卡组` | 主体名（上节映射） |
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
