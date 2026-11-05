# run kit 的生命周期（**不要预存，按需现打**）

## 关键修正（2026-11-05）

**反向包不需要单独打** —— `dual_train.py pack` 会自动换向：

```
collect 判定"本轮没显著提升"  ⇒ state 里 turn 翻到另一侧
pack    读 state              ⇒ 自动用「另一侧的模型当 init、本侧模型当 bank」
```

所以任何时候都只有一条命令：`./tools/py tools/dual_train.py pack --names <state> --dwf ...`

### 但这里修掉了一个真 bug

老版 `pack` 把 `--anchor` 和 `--bank-decks` **都写死成同一个 CLI 参数**，
于是**交叉卡组对训会错**：换向到 grookey 侧时，会把 grookey 专家钉到 db04 上——
正是「专家打生牌 → 陪练强度虚高 → 学不到东西」那个坑。

现在每侧记住自己的卡组：

| 训哪侧 | anchor（learner 100% 打） | bank 钉在 |
|---|---|---|
| A | `A.deck` | `B.deck` |
| B | `B.deck` | `A.deck` |

`init` 时必须显式给 `--a-deck` / `--b-deck`。

## 盘上现有的包（都有明确用途）

| 包 | 用途 |
|---|---|
| `ptcg_train_kit_v3.tar.gz` | 首次部署（两个 C 补丁 + 5 个工具 + 187 池，自洽） |
| `r1_db04_vs_grookey_kit.tar.gz` | **当前该跑的**：主侧第 1 轮 |
| `r1_db04_vs_abra_kit.tar.gz` | 主侧备用：陪练换 abra（已有 G11-A +9.5pp 实证） |
| `dual_starter.tar.gz` | 本机编排器 + 路由框架（不传 GPU 机） |

## 标准循环（每一轮只有这两步）

```bash
# ① 本机：出包（秒级）
./tools/py tools/dual_train.py pack --names db04_vs_grookey \
    --dwf decks/weights_real.txt --full-pool

# ② GPU 机：解包 → 预检 → 开训（约 8 分钟）
tar xzf <run>_kit.tar.gz && cd <run>_kit
PTCG_REPO=~/ptcg_train_kit/port_repo bash apply.sh --run

# ③ 本机：收模型、评估、判定、自动决定下一轮训谁
./tools/py tools/dual_train.py collect --names db04_vs_grookey --bin <拉回的.bin>
./tools/py tools/dual_train.py status  --names db04_vs_grookey
```

## 规矩

**只有当"这一轮现在就要跑"时才打包。** 判据：
1. `status` 里 `turn` 指向谁，就训谁（它已按"显著提升才继续同侧"自动决定）
2. 主侧 `flat` 涨到 `patience` 才说明这条线走不通，那时才考虑换成另一对（如 abra）
3. **不在测过之前为多种可能各打一个包**
