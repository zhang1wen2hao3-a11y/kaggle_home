# 双训 + 规则路由：起步包

两个工具，都不需要 GPU；重活（PPO 训练）仍在 GPU 机上跑。

## 1. `tools/dual_train.py` —— 双模型互训编排器

补齐 `ptcg_league/` 缺的两块：**角色互换的跨代 driver** 和 **停止判据**。

```bash
./tools/py tools/dual_train.py init --names db04_vs_grookey --a g6_db04 --b grookey --primary A
./tools/py tools/dual_train.py pack --names db04_vs_grookey \
    --anchor dunsparce_abra__db04 --dwf decks/weights_real.txt --full-pool
#   → 传到 GPU 机： tar xzf <run>_kit.tar.gz && cd <run>_kit
#                  PTCG_REPO=<训练仓库> bash apply.sh --run
./tools/py tools/dual_train.py collect --names db04_vs_grookey --bin <拉回的.bin>
./tools/py tools/dual_train.py status  --names db04_vs_grookey
```

### 判据（两层，别只用一层）

| 层 | 规则 | 为什么 |
|---|---|---|
| 统计显著 | 本轮 vs **同一侧上一轮**的同种子配对 McNemar，`p < alpha(0.05)` | 点估计相减会把抽样波动当信号 |
| 实际够大 | `delta ≥ min_delta(0.02)` | **8 个种子全赢/0 输就有 p=0.008，但实际只 +0.6pp**——纯显著性会让循环为噪声级提升一直跑 |

### 换向与停止

```
本轮「双层判定」通过 ⇒ 下一轮继续练同一侧（还在涨，换走浪费一轮）
没通过            ⇒ 换到另一侧
两侧各自连续 patience(2) 轮没通过 ⇒ 停
```

> `patience` / `max_rounds` 是**预算护栏**，不是“预计第几轮收敛”。
> 真收敛只能看 Δ 序列：**逐轮收窄并趋近 0 = 收敛；来回跳 = 还在噪声里，规则不可信。**
> 单轮成本 ≠ 收敛时间：一轮训练是固定 400k 步 ⇒ 实测 **wall 488~490s ≈ 8 分钟**（常数）。

## 2. `tools/router_core.py` —— 规则路由

在线决定“这一手用哪个模型”，可被 `main.py` 直接 import。

- **冷启动**：信息为 0 → 一律泛用模型
- **只在回合边界切换**，最多 `MAX_SWITCHES`（默认 2）次
- **上下文交接**：`shared_tracker=True`（零成本，推荐）或独立 tracker + 历史重放（1~4 ms）
- **兜底**：任何异常退回泛用模型，绝不让路由层导致 forfeit
- **分类器**：覆盖率 ≥0.80 且领先 ≥0.15 才认，否则判“未判定”→ 不切

离线自检（不需要 GPU）：

```bash
./tools/py tools/router_core.py                      # 打印原型族索引
./tools/py tools/router_core.py --replay <回放.json> --trace   # 第几手能判定对手
```

**已实测**（40 局真实回放）：首次判定中位 **第 16 个决策**（≈第 2~3 回合）；
判定批 **15/16 正确**（唯一错是 e1d7↔7e39，同属 ogerpon 族内混淆）；判不出来就回退泛用。

## 3. `tools/make_agent_dir.sh`

铺标准 agent 目录（骨架软链 + model.pt + deck.csv），评测/联赛都认这个结构。

## 提交资源上限（官方）+ 实测占用

| 项 | 上限 | 实测 |
|---|---|---|
| 提交体积 | 197.7 MiB | 基底 20.6 + **每模型 17.1** ⇒ 可放约 10 个模型 |
| RAM | 12.2 GiB | 627 MB + 每模型 22 MB |
| HDD | 11.8 GiB | 解包约 60 MB |
| vCPU | 2 | 单次决策 28 ms，一局约 9 s |
