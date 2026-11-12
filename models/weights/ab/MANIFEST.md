# A/B 交替自博弈 · 第 1 轮 A 阶段权重（9d6e 专精）

本目录是 **MODE=ab（纯交替）长跑第 1 轮的 A 侧产物**，`.bin` 为引擎原生权重格式，
可直接当训练起点（`INIT=`）或转成 `model.pt` 做评估。

## 文件

| 文件 | 大小 | md5 | 说明 |
|---|---|---|---|
| `r1A.bin` | 18,205,408 | `28d27691c9aa548c4f22f76a23039645` | A 阶段训练产物（4,849,664 步，ep148） |
| `r1A_blend.bin` | 18,205,408 | `babd27ac173291b006ad07d604dc9787` | `r1A` 按 α=0.10 往端口权重做了 **KL 回拉**（混合 125 个张量，沿用 2 个缓冲） |
| `r1B.bin` | 18,205,408 | `56ab89c926e3296351522b49c4e1578e` | B 阶段训练产物（通才，4,030,464 步，ep123，timebox 停） |

## 怎么来的

| 项 | 值 |
|---|---|
| 机器 | `connect.westb.seetacloud.com:38325`（RTX 4080 SUPER 32 GB） |
| 启动命令 | `SHIP=dunsparce_abra__9d6e MODE=ab ROUNDS=2 THRESH_PP=3.0 AB_MAX_STEPS≈4M bash tools/launch_ab.sh` |
| 起点权重 | `port_w.bin`（md5 `2bbba9587f7a1a6747fd80cc23f4c141`，即 port 策略） |
| 训练步数 | 4,849,664（ep148，SPS 1705，墙钟 2868 s） |
| 学习者座位 | **A 钉死 9d6e**（`--p-anchor-learner 1.0`），每局都打这副 |
| 对手座位 | 冻结 `port_w` 打覆盖率抽牌（`--league-pct 1.0`，100% 行都是它；`deck-weights-file decks/weights_field_b05.txt`，ESS 8.7） |
| 其他 | `--total-agents 512 --minibatch 512 --num-threads 8 --lr 2e-4 --seed 424243 --no-anneal-lr`，快照间隔 20 epoch（655k 步） |
| 结束方式 | **timebox**（4M 步到点换边），不是胜率触发 |

日志里可核对的分布证据：`lgfrac 1.000`（全 league 行）、`tl0:1.00/to0:0.05`
（学习者 100% 打 9d6e，对手只有 ~5% 抽到 9d6e = 池内占比）。

## 训练期表现（别当成质量证明）

```
基线（skip 5 + warm 30）  0.355
末段 EMA20                0.352     ← Δ ≈ 0pp
`--skip` 后全程 Δ 范围    −0.4pp … +2.1pp
```
即：**对冻结端口的训练期胜率没有可辨识的增益**（与之前 10M 步 slowking 的"位移大、收益零"一致）。
是否变强只能看离线 12 格面板（`tools/meta_eval.sh`，种子 950000，n=200），
对照基线 `端口@9d6e = 0.6723`。

## 权重位移（`tools/weight_dist.py`）

```
全局相对 L2        1.7062%
中位张量相对 L2    1.791%（最大 17.95%）
参数量             5,001,164
标定：400k RL≈0.32% / g6_db04≈0.50% / BC 全重训≈1.9~2.0% / slowking 10M≈3.81%
```
→ 位移已达"BC 全重训"量级，但胜率没跟上。

## 用法

```bash
# ① 直接当训练起点（引擎原生格式）
INIT=models/weights/ab/r1A_blend.bin bash tools/db04_line.sh

# ② 转 model.pt 做评估
PYTHONPATH=build/port_plain:. ./tools/py port_repo/native/tools/native_weights.py import \
  models/weights/ab/r1A.bin models/ppo/r1A.pt
bash tools/make_agent_dir.sh build/agents/r1A_9d6e models/ppo/r1A.pt decks/pool_field/dunsparce_abra__9d6e.csv
bash tools/meta_eval.sh --cand build/agents/r1A_9d6e --tag r1A_9d6e

# ③ 重做 KL 回拉
PYTHONPATH=. PTCG_NW=port_repo/native/tools/native_weights.py PTCG_BUNDLE=. \
  ./tools/py tools/kl_blend.py --new-bin models/weights/ab/r1A.bin \
  --init-bin models/weights/kl/port_w.bin --alpha 0.10 --out-bin /tmp/blend.bin
```

> ⚠️ 快照目录里的 `state_*.bin`（36 MB）是**含优化器/计数的 resume 状态**，当权重 import 会
> `AssertionError`。纯权重只有 `<step>.bin`（18 MB）这一种。

---

# 第 1 轮 B 阶段（通才）

| 项 | 值 |
|---|---|
| 文件 | `r1B.bin`（md5 `56ab89c926e3296351522b49c4e1578e`） |
| 起点权重 | `port_w.bin`（`experiments/native-r1B/init.bin` 的 md5 与 port_w 完全一致，已核对） |
| 训练步数 | 4,030,464（ep123，SPS 1704，墙钟 2394 s） |
| 学习者座位 | **B 不钉牌**（`--p-anchor-learner 0.0`）→ 每局按覆盖率从 187 副池里抽牌 |
| 对手座位 | 冻结 `r1A_blend`（md5 `babd27ac…`）**钉死打 9d6e**（`--bank-decks dunsparce_abra__9d6e`） |
| `--league-pct` | 1.0（100% 行都是这个冻结对手） |
| 结束方式 | **timebox**（4M 步到点，`★触发(timebox 4030464>=4000000)`） |

日志分布证据：`lgfrac 1.000`、`tl0:0.02~0.07 / to0:1.00`
（学习者只有 ~4% 的对局抽到 9d6e = 池内占比，对手 100% 打 9d6e）。

## B 的训练期表现（同样不能当质量证明）

```
基线（skip 5 + warm 30）  0.351
全程 Δ 范围               −1.8pp … +1.2pp，均值 −0.59pp
前半段 lgw 0.343 → 后半段 0.344     ← 无提升
```
注意 **B 的 lgw 不能和 A 的比**：A 是固定用自己最强的一副牌打池子，B 是被迫用池里
它不熟的弱牌去打一副专精牌，起点胜率天然低。

## 第 2 轮 A 阶段的对手（已生效）

```
--league-ckpts /root/autodl-tmp/teacher_loop_ab/r1B.bin   ← 增强后的 B
--bank-decks   （缺省 = 不钉牌）                          ← B 按覆盖率抽牌打 A
--p-anchor-learner 1.0                                    ← A 仍钉 9d6e
```
这才是"对手逐步变强"的通路：第 1 轮 B 只被钉在 9d6e 当陪练，第 2 轮起 B 反过来用一副
任意牌来考验 A。
