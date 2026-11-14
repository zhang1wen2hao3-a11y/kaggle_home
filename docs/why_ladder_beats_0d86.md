# 天梯为什么能赢 0d86（shuppet），我们本地差在哪一步

数据口径：
- 天梯：`data/ladder_games.tsv`（1756 局真人档案，双座位），卡组身份 = `sha256(sorted ids)[:8]` 前 4 位；
- 本地：`docs/the-pockmon-battle-choice-in-table.csv`（母表）、`logs/meta/*`（面板）；
- 卡表：`decks/pool_field/*.csv` + `data/EN_Card_Data_R2_full.csv`（卡面效果原文）。

---

## 1. 天梯侧：谁在赢 shuppet，靠什么赢

按「对手卡组 + 对手玩家」聚合（只用 n ≥ 10 的格）：

| 对手卡组 | 对手玩家 | 胜率 | n |
|---|---|---|---|
| **f5f8** | **YumeNeko** | **89.6%** | 67 |
| **e8ee** | **YumeNeko** | **90.9%** | 11 |
| 26c6 | ミワハルキ | 61.4% | 101 |
| ea39 | Raihan Ramadistra | 59.6% | 47 |
| e1f9 | RtoABC2 | 56.8% | 44 |
| 2617 | ANMOL GARG | 36.4% | 11 |
| **9d6e（我们的牌）** | Charles Backman | **0.0%** | 20 |
| c21e | バーベナヘレナ… | 0.0% | 22 |
| 9080 | Tys TANA | 0.0% | 10 |

按**卡**聚合（同一批对局，按卡组是否带某张牌分组）：

| 分组 | vs shuppet 胜率 | n | 卡组数 |
|---|---|---|---|
| 带 **Battle Cage** | **60.7%** | 168 | 14 |
| 不带 Battle Cage | 36.2% | 550 | 65 |
| 带 **Enhanced Hammer** | 57.1% | 175 | 12 |
| 带 **Toucannon 系** | **83.3%** | 120 | 3 |
| 都没有 | 37.3% | 533 | 59 |

**+24.5pp（Battle Cage）/ +46pp（Toucannon 系）**，不是噪声级别。

### 机制（卡面原文，不是推测）

- **Battle Cage（竞技场）**：“Prevent all damage counters from being placed on Benched
  Pokémon (both yours and your opponent's) by effects of attacks and Abilities …”
  → 直接关掉 shuppet 的 Banette / Gwynn 铺替补席伤害的引擎；竞技场场上只能有一张，
  下它同时把 shuppet 的 **Prism Tower** 顶掉（双杀）。
- **Enhanced Hammer（物品）**：“Discard a Special Energy from 1 of your opponent's Pokémon.”
  → shuppet 的能量底是 **4× Telepath Psychic Energy（特殊能量）**，这是精准拆引擎。
- **Toucannon（Feather Rondo 60 + 双方替补席每只 +20）**：不吃铺伤，靠双方铺场数量反向放大伤害。

⇒ f5f8 / e8ee 是**为打 shuppet 设计的卡组**：YumeNeko 用它吃 ミワハルキ 的 shuppet，
局部对位 84~90%。

---

## 2. 我们本地差在哪一步（三层，逐层可验证）

### (a) 卡组层：我们选的牌里没有解

`9d6e` / `db04`（我们历史上最看好的两副）：**Battle Cage 0 张、Enhanced Hammer 0 张。**
天梯同款 `9d6e` 由 Charles Backman 本人打 shuppet 是 **0/20**。
我们的 choice 表之所以把 9d6e/db04 排前面，是因为 field 权重来自**我们自己 582 局的历史占比**
（见第 3 节）。

### (b) 驾驶员层：同一副牌，本地 6.7%、天梯 89.6%

| 场景 | 数值 |
|---|---|
| 本地：f5f8(init 端口) vs 0d86(**g8_bc_parent**) | **6.7%**（= 1 − 0.9325，n=400）|
| 天梯：f5f8(YumeNeko) vs 0d86 | **89.6%**（n=67）|
| 本地：f5f8(init) vs 0d86(init) | 99.2%（0d86 侧 0.83%，n=600）|

反证「有卡 ≠ 会打」：`c21e` 带 4 张 Battle Cage + 1 张 Enhanced Hammer，
天梯对 shuppet **0/22** —— 卡在手上不会用等于没有。

更狠的自证：本地同一对局的两个方向，都被「有专精的一方」打成 63~92%。

| 本地格 | 数值 |
|---|---|
| 0d86@`g8_bc_parent` vs 26c6(init) | 63.5% |
| 26c6@init vs 0d86(init) | 85.0% |
| 26c6@r1A vs 0d86(init) | 92.0% |

⇒ **本地表测的是「哪边有专精」，不是牌面强度。** 差的“那一步”是**时机决策**
（何时下竞技场、何时弃特能、何时换人），不是价值函数——这一点从
`0d86@g8_bc_parent` 与 `0d86@init` 的 0.9325 / 0.0083 反差就能定性。

### (c) 度量层：母表 219 条记录的对手驾驶员全是 `init`

顶分段对手是**专精驾驶**（ミワハルキ / YumeNeko 这类），我们所有格子却拿自己的弱端口当对手
⇒ 系统性高估 20~30pp。方案 A（每格用该卡组当前最强驾驶员当对手）方向正确，
30 个 A 面板还排在 wishlist 后面，到位后 choice 表才有意义。

---

## 3. 顶分段其实是「三个牌」的石头剪刀布（我们权重错了）

LB ≥ 1350 的座位构成（n=1833）：

| 卡组 | 顶分段占比 | 全梯占比 |
|---|---|---|
| 26c6（slowking） | **54.6%** | 28.5% |
| 0d86（shuppet） | **43.5%** | 28.5% |
| f5f8（abra+Toucannon） | 0.2% | 15.1% |
| 7e39 | 0.4% | 1.2% |
| 其它 | 1.3% | 26.7% |

而我们的 `decks/weights_field_b05.txt`：7e39 **28.6%**、shuppet 9.7%、slowking 5.1%、f5f8 4.7%。
**权重几乎完全反了**（这是我们自己 582 局的历史占比，带镜像自放大）。

但 YumeNeko 的 f5f8 的 532 局里：450 局打 26c6（**50.0%**）、82 局打 0d86（**84.1%**）
⇒ 他确实在顶分段（只是 LB 快照 1145 过期）。三个牌的关系：

| 对位 | 天梯实测 |
|---|---|
| f5f8 → 0d86 | **84.1%**（n=82） |
| 26c6 → 0d86 | **61.4%**（n=101） |
| f5f8 → 26c6 | 50.0%（n=450） |

按顶分段份额算期望（假设我们打得出对应水平）：

| 我们玩 | 顶分段期望胜率 |
|---|---|
| f5f8（真会打） | 0.546×0.50 + 0.435×0.841 ≈ **63.9%** |
| 26c6（真会打） | ≈ 55%（61.4% 打 shuppet、50% 打自己） |
| 0d86（真会打） | 0.546×0.386 + 0.435×0.50 ≈ **42.8%** |

⇒ 我们现在的 0d86 提交方向，在顶分段是**期望最差**的一角；
`f5f8/e8ee` 这条线才是天花板（它同时刻 shuppet 84% 且与 slowking 五五开）。

---

## 4. 因此行动

1. **（已开跑）** YumeNeko 的 f5f8（512 局）+ e8ee（136 局）BC 数据物化 + kit
   （`data/bc_yumeko_abra*.npz` → `dist/bc_kit_yumeko_abra*.tar.gz`）。GPU 开机后训练 `bc_yumeko`，
   按标准 B 筛（vs `g6_db04` + 同牌镜像 vs 端口 + vs `g8_bc_parent`@0d86 + vs port@26c6）。
2. **训练优先级重排**：`shuppet-specialist` 降级；改为
   ① `bc_yumeko@f5f8` 的 RL 专精（对手 = `g8_bc_parent`@0d86 + port/专精 26c6）；
   ② slowking（26c6）专精——顶分段 54.6%，我们所有驾驶员只有 ~35%。
3. **权重重建**：`decks/weights_top.txt`（顶分段份额）另行保存，choice 表用双权重出两版，
   提交决策按「目标区间」选。
4. **提交前置条件**：我们的 0d86 面对 BC deck 天梯只有 39%；在对手库补上 BC deck 之前，
   任何“0d86 很稳”的结论都不可信。
