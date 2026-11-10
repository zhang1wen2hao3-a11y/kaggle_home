# 对手池卡组比例：构成、依据与待定项

> 生成脚本：`tools/build_deck_weights.py`　数据源：`data/our_games.tsv`（我们自己 582 局真实天梯）
> 采用版本：**`models/weights/conf/weights_field_b05.txt`**（冠军保底 b=0.05）
> 最后更新：2026-11-06

## 1. 一句话

对手分布不再手工拍数字，改成 **实测场地比例 + 冠军卡组固定保底**：

```
w'(d) = (1 − K·b) · f(d)  +  b · 1[d ∈ 冠军卡组]
```

`f(d)` = 我们实测遇到的场地分布；`K` = 冠军卡组数（4）；`b` = 每副冠军牌的保底比例（0.05）。
先给每副冠军牌固定 b 的份额，剩下的 `1 − K·b` 按实测比例分 —— 冠军比例被抬到至少 b，
其余卡组按比例压缩，总强度上升但场地形状不变。

## 2. 为什么这么定

1. **场地会漂，手工数字会过期。** 旧 `weights_2617.txt` 给 0bf5 0.12，我们 582 局实测只有 3.8%；
   实测 6.2% 的 `abra_kadabra__e8ee` 完全没有权重；实测 1.9% 的 `db04` 走 `default 0.0001`。
2. **纯场地分布会把多样性压死。** 7e39 一个占 34.7%，ESS 只有 6.1（≈ 训练时真正"轮得开"的对手数）。
3. **冠军牌值得额外练。** 它们教师是 1500+ 分、语料 13~21 万帧；多练几 pp 是廉价的鲁棒性投资。
4. **加保底反而提升多样性**：ESS 6.1（b=0）→ **8.5（b=0.05）** → 9.5（b=0.10）。
   直觉上"给固定几副牌权重会收窄分布"，实际因为纯场地里 7e39 太大，压缩它反而把空间放出来。
   ⇒ 保底同时改善强度和覆盖。

## 3. 冠军卡组的判定（数据驱动）

判据：**语料帧数 ≥ 100k 且教师天梯 ≥ 1500**（见 `下一步.md` §56.4 的语料表）。

| 卡组 | 教师 | 教师 LB | 帧数 | 我们实测遇到 |
|---|---|---|---|---|
| `z_dragapult_dreepy_petit__0904` | Petit Canard | 1510 | 209,735 | 0.0% |
| `slowpoke_slowking__26c6` | ミワハルキ | **1580** | 167,053 | 0.0% |
| `z_ogerpon_toolbox_miwa__e1d7` | ミワハルキ | **1580** | 137,602 | 0.0% |
| `shuppet_dhelmise__0d86` | ミワハルキ | **1580** | 130,236 | 5.5% |

**`7e39` 不算冠军牌**：它场次最多（34.7%），但最好教师只有 Dipam 1478 / 仅 1,768 帧
（其余 21,562 帧来自 875 分的 d_tree_suki）。按"帧数 + 教师分"这条判据它够不上。

⚠️ 注意：三副冠军牌（0904/26c6/e1d7）在我们 582 局里**一次都没遇到**。
给它们 15% 权重是"请陪练"的投资，不是场地复刻 —— 这也是 b 必须小的原因（b=0.10 时三者占 30%，就过头了）。

## 4. 关键指标

| 文件 | 冠军保底 | 显式副数 | 权重合计 | ESS |
|---|---|---|---|---|
| `weights_2617.txt` | 旧（手拍） | 9 | 1.0000 | 5.6 |
| `weights_field_b0.txt` | b=0 | 36 | 1.0000 | 6.1 |
| `weights_field_b05.txt` | **b=0.05** | 36 | 1.0000 | 8.5 |
| `weights_field_b10.txt` | b=0.10 | 36 | 1.0000 | 9.5 |

**镜像率**（学习者自己那副牌在对手座位的比例 = 自对弈比例）：

| 学习者卡组 | 旧文件 | b=0 | b=0.05 |
|---|---|---|---|
| `2617` | 12.8% | 11.7% | **9.6%** |
| `9d6e` | 8.0% | 4.8% | **4.0%** |
| 训练日志里的实测 | 12.8% ✓ | — | — |

镜像率会随 b 一起变小（因为"剩下的 1−K·b"被压缩）。若要把镜像率固定成独立旋钮，
需要再加 `--mirror-share` 参数，但那会变成第二个变量，先不动。

## 5. 实测场地分布（582 局，按出现次数排序）

| # | 卡组 | 局数 | 实测占比 | 冠军 |
|---|---|---|---|---|
| 1 | `teal_mask_ogerpon_ex_applin__7e39` | 202 | 34.7% |  |
| 2 | `grookey_thwackey__2617` | 68 | 11.7% |  |
| 3 | `abra_kadabra__e8ee` | 36 | 6.2% |  |
| 4 | `abra_kadabra__f5f8` | 33 | 5.7% |  |
| 5 | `shuppet_dhelmise__0d86` | 32 | 5.5% | ★ |
| 6 | `dunsparce_abra__9d6e` | 28 | 4.8% |  |
| 7 | `dunsparce_abra__0bf5` | 22 | 3.8% |  |
| 8 | `dreepy_drakloak__3eaa` | 21 | 3.6% |  |
| 9 | `dunsparce_abra__db04` | 11 | 1.9% |  |
| 10 | `abra_kadabra__3f45` | 11 | 1.9% |  |
| 11 | `dreepy_drakloak__94ec` | 10 | 1.7% |  |
| 12 | `mega_lucario_ex_solrock__77a5` | 10 | 1.7% |  |
| 13 | `dreepy_drakloak__ea39` | 9 | 1.5% |  |
| 14 | `snover_mega_abomasnow_ex__b702` | 8 | 1.4% |  |
| 15 | `dreepy_drakloak__b381` | 8 | 1.4% |  |
| 16 | `grookey_thwackey__7d48` | 7 | 1.2% |  |
| 17 | `abra_kadabra__f06b` | 5 | 0.9% |  |
| 18 | `dreepy_drakloak__a413` | 4 | 0.7% |  |
| 19 | `mega_kangaskhan_ex_dwebble__7e79` | 4 | 0.7% |  |
| 20 | `beldum_metang__f91d` | 3 | 0.5% |  |
| 21 | `teal_mask_ogerpon_ex_applin__e938` | 3 | 0.5% |  |
| 22 | `dunsparce_abra__7c22` | 3 | 0.5% |  |
| 23 | `iono_s_voltorb_iono_s_tadbulb__7dc2` | 3 | 0.5% |  |
| 24 | `mega_kangaskhan_ex_dwebble__0382` | 3 | 0.5% |  |
| 25 | `teal_mask_ogerpon_ex_applin__5c8e` | 2 | 0.3% |  |
| 26 | `dunsparce_abra__9080` | 2 | 0.3% |  |
| 27 | `beldum_metang__a47a` | 2 | 0.3% |  |
| 28 | `mega_lucario_ex_solrock__b757` | 2 | 0.3% |  |
| 29 | `riolu_lunatone__d974` | 2 | 0.3% |  |
| 30 | `dreepy_drakloak__07be` | 2 | 0.3% |  |
| 31 | `riolu_mega_lucario_ex__c6ca` | 2 | 0.3% |  |
| 32 | `mega_kangaskhan_ex_dwebble__4e3f` | 2 | 0.3% |  |
| 33 | `dwebble_crustle__1079` | 2 | 0.3% |  |
| 34 | `abra_kadabra__c21e` | 1 | 0.2% |  |
| 35 | `iron_thorns_ex_pikachu_ex__6db9` | 1 | 0.2% |  |
| 36 | `abra_kadabra__41dd` | 1 | 0.2% |  |
| 37 | `dreepy_drakloak__6083` | 1 | 0.2% |  |
| 38 | `snover_mega_abomasnow_ex__38dc` | 1 | 0.2% |  |
| 39 | `dreepy_drakloak__5244` | 1 | 0.2% |  |
| 40 | `mega_lucario_ex_makuhita__b383` | 1 | 0.2% |  |
| 41 | `shuppet_banette__f2af` | 1 | 0.2% |  |
| 42 | `dwebble_crustle__ca60` | 1 | 0.2% |  |
| 43 | `riolu_mega_lucario_ex__08e2` | 1 | 0.2% |  |
| 44 | `mega_lucario_ex_solrock__b395` | 1 | 0.2% |  |
| 45 | `munkidori_marnie_s_impidimp__c20a` | 1 | 0.2% |  |
| 46 | `mega_kangaskhan_ex_dwebble__0fd0` | 1 | 0.2% |  |
| 47 | `mega_kangaskhan_ex_dwebble__e700` | 1 | 0.2% |  |
| 48 | `snover_mega_abomasnow_ex__39d1` | 1 | 0.2% |  |
| 49 | `duraludon_archaludon_ex__da1d` | 1 | 0.2% |  |
| 50 | `dudunsparce_abra__121f` | 1 | 0.2% |  |
| 51 | `dunsparce_dudunsparce__caea` | 1 | 0.2% |  |
| 52 | `dwebble_crustle__4bf5` | 1 | 0.2% |  |
| 53 | `duraludon_archaludon_ex__5ac3` | 1 | 0.2% |  |

合计 582 局 / 53 副（另有 58 条记录对手未识别，已排除）

## 6. 三档权重对照（只列显式写进文件的 36 副）

| 卡组 | 实测 | 旧(手拍) | b=0 | b=0.05 | b=0.10 | 冠军 |
|---|---|---|---|---|---|---|
| `teal_mask_ogerpon_ex_applin__7e39` | 34.7% | 34.0% | 35.9% | 28.6% | 21.3% |  |
| `shuppet_dhelmise__0d86` | 5.5% | — | 5.7% | 9.7% | 13.6% | ★ |
| `grookey_thwackey__2617` | 11.7% | 13.0% | 12.1% | 9.6% | 7.2% |  |
| `z_dragapult_dreepy_petit__0904` | 0.0% | 10.0% | 0.0% | 5.1% | 10.2% | ★ |
| `slowpoke_slowking__26c6` | 0.0% | 4.0% | 0.0% | 5.1% | 10.2% | ★ |
| `z_ogerpon_toolbox_miwa__e1d7` | 0.0% | 5.0% | 0.0% | 5.1% | 10.2% | ★ |
| `abra_kadabra__e8ee` | 6.2% | — | 6.4% | 5.1% | 3.8% |  |
| `abra_kadabra__f5f8` | 5.7% | 5.0% | 5.9% | 4.7% | 3.5% |  |
| `dunsparce_abra__9d6e` | 4.8% | 8.0% | 5.0% | 4.0% | 2.9% |  |
| `dunsparce_abra__0bf5` | 3.8% | 12.0% | 3.9% | 3.1% | 2.3% |  |
| `dreepy_drakloak__3eaa` | 3.6% | 9.0% | 3.7% | 3.0% | 2.2% |  |
| `dunsparce_abra__db04` | 1.9% | — | 2.0% | 1.6% | 1.2% |  |
| `abra_kadabra__3f45` | 1.9% | — | 2.0% | 1.6% | 1.2% |  |
| `dreepy_drakloak__94ec` | 1.7% | — | 1.8% | 1.4% | 1.1% |  |
| `mega_lucario_ex_solrock__77a5` | 1.7% | — | 1.8% | 1.4% | 1.1% |  |
| `dreepy_drakloak__ea39` | 1.5% | — | 1.6% | 1.3% | 0.9% |  |
| `snover_mega_abomasnow_ex__b702` | 1.4% | — | 1.4% | 1.1% | 0.8% |  |
| `dreepy_drakloak__b381` | 1.4% | — | 1.4% | 1.1% | 0.8% |  |
| `grookey_thwackey__7d48` | 1.2% | — | 1.2% | 1.0% | 0.7% |  |
| `abra_kadabra__f06b` | 0.9% | — | 0.9% | 0.7% | 0.5% |  |
| `dreepy_drakloak__a413` | 0.7% | — | 0.7% | 0.6% | 0.4% |  |
| `mega_kangaskhan_ex_dwebble__7e79` | 0.7% | — | 0.7% | 0.6% | 0.4% |  |
| `beldum_metang__f91d` | 0.5% | — | 0.5% | 0.4% | 0.3% |  |
| `teal_mask_ogerpon_ex_applin__e938` | 0.5% | — | 0.5% | 0.4% | 0.3% |  |
| `dunsparce_abra__7c22` | 0.5% | — | 0.5% | 0.4% | 0.3% |  |
| `iono_s_voltorb_iono_s_tadbulb__7dc2` | 0.5% | — | 0.5% | 0.4% | 0.3% |  |
| `mega_kangaskhan_ex_dwebble__0382` | 0.5% | — | 0.5% | 0.4% | 0.3% |  |
| `teal_mask_ogerpon_ex_applin__5c8e` | 0.3% | — | 0.4% | 0.3% | 0.2% |  |
| `dunsparce_abra__9080` | 0.3% | — | 0.4% | 0.3% | 0.2% |  |
| `beldum_metang__a47a` | 0.3% | — | 0.4% | 0.3% | 0.2% |  |
| `mega_lucario_ex_solrock__b757` | 0.3% | — | 0.4% | 0.3% | 0.2% |  |
| `riolu_lunatone__d974` | 0.3% | — | 0.4% | 0.3% | 0.2% |  |
| `dreepy_drakloak__07be` | 0.3% | — | 0.4% | 0.3% | 0.2% |  |
| `riolu_mega_lucario_ex__c6ca` | 0.3% | — | 0.4% | 0.3% | 0.2% |  |
| `mega_kangaskhan_ex_dwebble__4e3f` | 0.3% | — | 0.4% | 0.3% | 0.2% |  |
| `dwebble_crustle__1079` | 0.3% | — | 0.4% | 0.3% | 0.2% |  |


## 7. 已知缺陷（改之前先看这里）

1. **评估面板只覆盖 70% 真实对手。** 7 格面板（`decks/panel_real.tsv`）覆盖 582 局中的 406 局；
   剩下 30% 完全没测，其中最大的是 `abra_kadabra__e8ee`（6.2%）、`db04`（1.9%）、`3f45`（1.9%）。
   ⇒ 面板分与天梯分对不上的主要原因之一。要扩到 ~12 格。
2. **矩阵里没有 db04。** 49 格矩阵是 7 副牌的世界，而 db04 正是我们天梯 1060 分用的牌，
   选卡组框架因此无法给 db04 打分。（`port_db04` 面板已在补测）
3. **`field_real.json` 已过期**：它是更早样本（7e39 45.6% / 0bf5 14.5%），与现在 582 局
   （34.7% / 3.8%）差得远。加权面板目前仍用它，需重算。
4. **冠军保底不是救命稻草。** 旧文件其实≈ b=0.05（0904 0.10 + e1d7 0.05 + 26c6 0.04 = 19%），
   而那个配方下三次训练仍是 0 ~ −4.7pp。它能修"分布错配"，不一定能修"训练不涨"。
5. **默认尾部 0.0001**：187 副池子里未显式列出的约 150 副共享极小权重，等价于忽略。
   如果某天遇到罕见牌被打崩，要回来把这副提权。

## 8. 下一步实验（等 GPU，26 分钟，单变量）

| 臂 | 对手分布 | 与上一臂唯一差别 |
|---|---|---|
| A | `weights_field_b0.txt` | 基线（纯实测场地） |
| B | `weights_field_b05.txt` | 只加冠军保底 5% |

其余固定：`INIT=/root/kl/port_w.bin`、`ANCHOR=grookey_thwackey__2617`、`BANKS=/root/kl/shup_a0.5.bin`
（`LEAGUE_PCT=0.10`、`BANKDECKS=shuppet_dhelmise__0d86`）、`T=400000`、`SEED=424242`、`THREADS=8`。

## 9. 复现

```bash
# 重新生成（b 可调）
./tools/py tools/build_deck_weights.py --b 0.05 --out models/weights/conf/weights_field_b05.txt
./tools/py tools/build_deck_weights.py --b 0.05 --dry-run      # 只打印表格

# 推到训练机（仓库 decks/ 目录下读）
scp models/weights/conf/weights_field_b05.txt \
    root@<gpu>:/root/ptcg_train_kit/port_repo/decks/
# 训练时用 DWF=decks/weights_field_b05.txt 覆盖默认值
```
