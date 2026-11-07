# 下一批 PPO 训练包 + 判定规则（准备好，等机器）

## 前提：为什么先不训

面板（7 个真实对手、1400 局、k 次重复）能分辨的最小差异约 **2~4pp**（基线 4 次极差 4.86pp）。
而我们所有 PPO 产出的差异都在 ±2pp 内 ⇒ **面板测不出来**。因此这轮改用真实天梯选牌：

| 我们的牌 | 真实天梯胜率 | n |
|---|---|---|
| `dunsparce_abra__db04` | 0.439 | 387 |
| `teal_mask_ogerpon_ex_applin__7e39` | 0.452 | 279 |
| **`dunsparce_abra__9d6e`** | **0.595** | 42 |
| `grookey_thwackey__2617` | 0.462 | 26 |

提交位正在把它们各自攒到 120+ 局（不花 GPU）：`58074897 = g6_9d6e`、`58070565 = g6_7e39`。

## 决策规则（预注册，避免事后找理由）

拿到 9d6e / 7e39 各 ≥120 局后：

| 条件 | 结论 | 行动 |
|---|---|---|
| `9d6e` ≥ 0.52 且 CI 下界 > 0.44 | 换牌有效 | PPO 打在 **anchor=9d6e**（下面 T1/T2） |
| `9d6e` 落在 0.44~0.52 | 与 db04 同水平 | 不换牌，PPO 继续 db04，起点用 `r1A2`（T3） |
| `9d6e` < 0.44 | 42 局是噪声 | 不换牌，同上；并把"换牌"这条线归档 |
| `7e39` 的 ogerpon 镜像格 ≥0.45 且总胜率 ≥0.48 | 7e39 可作为主牌 | 追加 PPO：anchor=7e39（T2 提级） |

## 三个候选训练包（每包 13 分钟，T=400k，配方与 G6 完全一致）

| 包 | anchor | INIT | BANKS | BANKDECKS |
|---|---|---|---|---|
| `t1_9d6e` | `dunsparce_abra__9d6e` | `g6_db04.bin` | `g6_db04.bin` | `teal_mask_ogerpon_ex_applin__7e39` |
| `t2_7e39` | `teal_mask_ogerpon_ex_applin__7e39` | `g6_db04.bin` | `g6_db04.bin` | `dunsparce_abra__0bf5` |
| `t3_db04_r1A2` | `dunsparce_abra__db04` | `r1A2_db04_g6.bin` | `g6_db04.bin` | `teal_mask_ogerpon_ex_applin__7e39` |

（`DWF=decks/weights_real.txt`、`SEED=424242`、`THREADS=8`、`LEAGUE_PCT=0.2`，其余同 G6）

## 机器起来后的执行清单

```bash
# ① 训练机（GPU）：开机后确认仓库与权重都在
ssh -p <port> root@<host> 'ls /root/ptcg_train_kit/port_repo/init/; nvidia-smi -L'

# ② 跑三包（远端脚本，链式、失败即停、产物校验 18,205,408 B）
#    见 build/chain_t.sh（本文件同目录下一节），scp 上去 nohup 执行

# ③ 拉回（本地）
./tools/py tools/remote_train.py find --name t1_9d6e
scp -P <port> root@<host>:/root/autodl-tmp/models/t*_*.bin models/ppo/

# ④ 评估机（CPU 32 核）：传新模型 → 面板 1400 局 ×k 次
./tools/py tools/two_node.py eval --names db04_vs_grookey --name t1_9d6e --deck dunsparce_abra__9d6e --push
```

## 已归档的分析数据（本机）

- `docs/strength_table.md`：真实天梯 vs 面板对照（两格反向 ±70~90pp）
- `data/replay_matchups.tsv`：3827 条真实对局（team/deck/胜负）
- `ptcg_league/logs/realpanel/*.json`：面板结果（k 次重复，直接可算均值与极差）
- `decks/panel_real.tsv` + `decks/field_real.json`：真实对手面板与权重（覆盖 64.9% 场次）
