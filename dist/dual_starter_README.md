# 双训 + 规则路由：起步包

两个工具，都不需要 GPU；重活（PPO 训练）仍在 GPU 机上跑。

## 1. `tools/dual_train.py` —— 双模型互训编排器

补齐了 `ptcg_league/` 缺的两块：**角色互换的跨代 driver** 和 **停止判据**。

```bash
# ① 初始化（选两边起始模型）
./tools/py tools/dual_train.py init --names db04_vs_x --a g6_db04 --b <另一侧模型名>

# ② 打包本轮（纯 CPU，秒级；打印传机命令）
./tools/py tools/dual_train.py pack --names db04_vs_x \
    --anchor dunsparce_abra__db04 --dwf decks/weights_real.txt --full-pool

# ③ 在 GPU 机上
#    tar xzf <run>_kit.tar.gz && cd <run>_kit
#    PTCG_REPO=<训练仓库> bash apply.sh --run

# ④ 回传 .bin 后，本机收集 + 评估 + 判定
./tools/py tools/dual_train.py collect --names db04_vs_x --bin ~/xxx.bin

# ⑤ 随时看轮次表
./tools/py tools/dual_train.py status --names db04_vs_x
```

**停止判据**（写在 state 里，可 init 时改）：
- `min_delta=0.02`：本轮 field-weighted EV 提升 <2pp 记为“没提升”
- `patience=2`：**两侧各连续 2 轮没提升 ⇒ 停止**，输出双方最佳模型
- `max_rounds=8`

**主判据是 field-weighted EV，不是头对头。** 头对头会奖励“专门克制对方”——G13 就是这么练的，
`lgw` 健康但顶部加权 +0.0pp。

## 2. `tools/router_core.py` —— 规则路由

在线决定“这一手用哪个模型”，可被 `main.py` 直接 import。

- **冷启动**：信息为 0 → 一律泛用模型
- **只在回合边界切换**，最多 `MAX_SWITCHES`（默认 2）次
- **上下文交接**：`shared_tracker=True`（零成本，推荐）或独立 tracker + 历史重放（1~4 ms）
- **兜底**：任何异常都退回泛用模型，绝不让路由层导致 forfeit

离线自检（不需要 GPU）：

```bash
./tools/py tools/router_core.py                      # 打印原型族索引
./tools/py tools/router_core.py --replay <回放.json> --trace   # 看第几手能判定对手
```

**已实测**（40 局真实回放）：
- 首次判定中位 **第 16 个决策**（≈第 2~3 回合），范围 9~59
- 判定的那批 **15/16 正确**（唯一的错是 e1d7↔7e39，都是 ogerpon 族内混淆）
- 判不出来就什么都不做（回退泛用）

## 提交资源上限（官方）

| 项 | 上限 | 实测占用 |
|---|---|---|
| 提交体积 | 197.7 MiB | 基底 20.6 + **每模型 17.1** ⇒ 可放约 10 个模型 |
| RAM | 12.2 GiB | 627 MB + 每模型 22 MB |
| vCPU | 2 | 单次决策 28 ms |
