# models/weights —— 训练机权重与配置的本地备份

无卡实例上用 rsync 拉下来的（2026-11-05），因为 GPU 机器随时可能关机，而这些 `.bin` 本地一份都没有。
每个 `.bin` 18,205,408 B = 127 张量 / 5,001,164 参数。

## 目录

| 路径 | 内容 | 说明 |
|---|---|---|
| `kl/*.bin` | `/root/kl/` 全部 7 个权重 | 训练与评测的主权重（`port_w`/`g6_db04`/`g8_bc_parent`/`bc_3eaa`/`bc_26c6`/`mix50`/`shup_a0.5`）+ 若干 `.pt` 中间产物 |
| `ph3/*.bin` | `/root/autodl-tmp/ph3/` | 阶段三三个学习者：`ph3_db04_v1` / `ph3_2617_v1` / `ph3_9d6e_v1`，含各自 `run*.log` |
| `logs/ph3_*.log` | 训练日志副本 | 记录 config、deck 权重生效情况、逐 epoch 曲线 |
| `conf/` | `decks/pool_field`(187 副) + `decks/weights_*.txt`(11 个分布) | 复现训练配置所必需 |

## 关键验证：`port_w.bin` 就是端口权重

`.bin` 是老版 pickle/C 格式，torch 2.6 直接 `torch.load` 会报 `invalid load key`。
本地用 `port_repo/native/tools/native_weights.py import` 转 pt 后逐张量比较：

| 对比 | 张量哈希 |
|---|---|
| `kl/port_w.bin`（转 pt） | `14f2890f` |
| `build/agents/port_7e39/model.pt` | `7b78e3fc` |
| `build/agents/port_on_9d6e/model.pt` | `7b78e3fc` |
| `build/agents/c_grookey/model.pt` | `7b78e3fc` |
| `build/agents/g6_db04/model.pt` | `de47cb60` |

哈希不一致，但**逐参数差值只有 `card_static` 一张表**（静态卡表 buffer），
可学参数相对 L2 = **0.0033%**、max|Δ| = 0（除 `card_static`）。
⇒ `port_w.bin` ≡ 本地 7 个 port agent 的权重，差别仅在静态卡表导出方式。

## 用途

1. **重新训练 2617 / 9d6e 的 INIT**：`INIT=/root/kl/port_w.bin`（而不是 `g6_db04.bin`），
   起点即"这副牌会打"的端口策略。
2. **难度阶梯对手**：`shup_a0.5.bin` 是最优档（opp-win 0.3425、信号 0.90）；
   本地评测若要复现该对手，需先用 `native_weights.py import` 转 pt 再包成 agent 目录。
3. **训练日志**：`ph3_2617_v1` 与 `ph3_9d6e_v1` 的 `INIT` 都是 `g6_db04.bin`，
   这是"初始化选错"的直接书面证据（不是推测）。
