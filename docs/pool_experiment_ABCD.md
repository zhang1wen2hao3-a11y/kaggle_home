# 对手池 A/B/C/D 对照实验（预注册）

> 目的：检验"给训练对手池加**冠军卡组**和**其他强策略**"能否在**固定 Meta 测试集**上提升强度。
> 关键约定：**训练时的胜率不是评价标准**。若加压后训练胜率下降、而固定测试集胜率上升，这可能是正向结果。
> 最后更新：2026-11-06

## 1. 四组设计

| 组 | 本地对手 | 冠军卡组 | 其他强策略 | 权重文件 | ESS |
|---|---|---|---|---|---|
| A（基线） | 100% | 0% | 0% | `weights_exp_A.txt` | 6.1 |
| B（轻量） | 85% | 10% | 5% | `weights_exp_B.txt` | 8.2 |
| C（中等） | 60% | 25% | 15% | `weights_exp_C.txt` | 12.3 |
| D（强压力） | 40% | 40% | 20% | `weights_exp_D.txt` | 12.7 |

三块混合公式（`tools/build_deck_weights.py --mix`）：

```
w(d) = a_local·f(d) + a_champ/|C|·1[d∈冠军] + a_other/|O|·1[d∈其他强]
```

块内：本地块按我们 582 局实测比例；冠军块、其他强块按**均匀**（它们大多在我们场地占比为 0，没有实测比例可分）。
四个文件都已用训练器同款解析规则自检（版本 3、权重合计 = 1.0、名字全部存在于 `pool_fields` 187 副）。

## 2. 三组怎么划出来的（数据驱动）

**本地对手** = `data/our_games.tsv` 里我们实际遇到的对手，按 582 局实测占比。

**冠军卡组（4 副）** = 教师天梯 ≥ 1500：

| 卡组 | 教师 | LB | 我们遇到 |
|---|---|---|---|
| `z_dragapult_dreepy_petit__0904` | Petit Canard | 1510 | 0.0% |
| `slowpoke_slowking__26c6` | ミワハルキ | 1594 | 0.0% |
| `z_ogerpon_toolbox_miwa__e1d7` | ミワハルキ | 1580 | 0.0% |
| `shuppet_dhelmise__0d86` | ミワハルキ | 1594 | 5.5% |

**其他强策略（7 副）** = 教师 LB ∈ [1300, 1500) 且我们遇到 < 1%（`data/deck_corpus.csv` + `data/lb_scores.json` 按 deck_hash 取拥有者最高分）：

`dreepy_drakloak__a413`(1428)、`dreepy_drakloak__9aec`(1428)、`dreepy_drakloak__2228`(1428)、
`cinderace_mega_kangaskhan_ex__1eb0`(1428)、`dreepy_drakloak__c73f`(1313)、
`mega_kangaskhan_ex_dwebble__7e79`(1313)、`beldum_metang__e1f9`(1311)

**`7e39` 不算冠军牌**：场次最多（34.7%）但最好教师只有 Dipam 1478 / 1,768 帧。

## 3. 固定测试集（四组共用，与各组训练分布解耦）

`decks/testset_meta.tsv`，12 格，**所有对手统一用「端口权重 + 该卡组」驾驶**（格子之间只差卡组）：

| 组 | 格子 |
|---|---|
| local（8） | 7e39、2617、e8ee、f5f8、9d6e、0bf5、3eaa、db04 |
| champ（4） | 0904、26c6、e1d7、0d86 |

权重取我们 582 局的实测占比（0904/26c6/e1d7 为 0，0d86 5.5%）。

## 4. 五项指标（`tools/meta_eval.sh` 直接输出）

| # | 指标 | 定义 | 用途 |
|---|---|---|---|
| 1 | **meta_w** | 12 格按实测占比加权的胜率 | 衡量整体真实对局强度（主指标） |
| 2 | **champ** | 4 副冠军牌的平均胜率 | 强度压力是否转成学习收益 |
| 3 | **local** | 8 副本地牌的平均胜率 | 是否牺牲原有能力 |
| 4 | **worst** | 12 格里最低的胜率 | 防止严重退化 |
| 5 | **seed_var** | 同组不同训练种子的指标方差 | 训练稳定性 |

## 5. 预注册判据（先定规则，防事后解释）

对每组相对 A 的差值（Δ），按 §59 的噪声带：**单格 n=200 的 CI ≈ ±6.5pp，12 格加权 ≈ ±2pp**。

| 判定 | 条件 |
|---|---|
| **B/C/D 中任一组"成功"** | `Δmeta_w ≥ +3pp` **且** `Δlocal ≥ −2pp` **且** `Δworst ≥ −5pp` |
| 值得扩种子的候选 | `Δmeta_w ≥ +1.5pp` 且 `Δchamp > 0` |
| 判为无效 | `Δmeta_w < +1.5pp`，或 `Δlocal < −3pp`（牺牲本地能力） |
| 判为有害 | `Δmeta_w ≤ −3pp` 或 `Δworst ≤ −8pp` |

**训练侧指标（train win rate / fwr / ep_len）一律不参与判定**，只作诊断。
理由：已复现 5 次以上——`fwr` 与面板/天梯强度无单调关系（§56.2）。

## 6. 成本

| 阶段 | 内容 | 机器 | 时长 |
|---|---|---|---|
| 训练 | 4 组 × 1 种子（T=400k） | GPU 2080Ti | ≈13 min/组 → **52 min** |
| 评估 | 4 组 × 12 格 × 200 场 | CPU 32 核/16 面板 | ≈21 min/组 → **84 min** |
| 复评 | 胜出组 × 3 种子 | GPU + CPU | ≈2 h |

本地（12 vCPU/6 面板）评估约 66 min/组 ≈ 4.4 h，慢 3 倍，不适合。

固定超参（四组完全相同，只动对手分布一个变量）：
`INIT=/root/kl/port_w.bin`、`ANCHOR=grookey_thwackey__2617`、`BANKS=/root/kl/shup_a0.5.bin`
`LEAGUE_PCT=0.10`、`BANKDECKS=shuppet_dhelmise__0d86`、`T=400000`、`SEED=424242`、`THREADS=8`

## 7. ⚠️ 已知的仪器问题（2026-11-06 修复）

`build/agents/` 下多个 port agent 的 `model.pt` 原是**符号链接**，统一指向
`build/port_plain/model.pt`。用 `cp` 覆盖模型时会**顺着链接写穿**，把共享的端口权重文件改坏，
于是那些 agent 实际加载的是被覆盖后的权重（且互相污染）。

- **已修复**：全部 agent 的 `model.pt` 已实体化；`build/port_plain/model.pt` 已用干净副本恢复
  （md5 `8b5fcda672ac`，与 `opp_3eaa`/`opp_dunsparce_0bf5` 一致）。
- **受影响的臂**（结果作废，需重跑）：`ph3_2617_v2`、`ph3_9d6e_v2`、`port_9d6e`、
  `g6_db04_on_2617`、`ph3_2617_v1`、`port_db04`，以及 `c_grookey_port` 的 `3eaa` 单格。
- **未受影响的臂**：`bc_grookey3`（本地，2026-11-05 22:30–23:15 跑）、
  `bc_grookey3_node` / `port_grookey_node`（评估机，独立 kit）。
- **教训**：以后给 agent 目录换权重一律 `rm -f dir/model.pt && cp ...`，并校验 md5；
  或直接在 `decks/testset_meta.tsv` 里用实体目录路径。
