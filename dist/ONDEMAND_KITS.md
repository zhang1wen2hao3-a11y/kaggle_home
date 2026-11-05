# 按需生成 run kit（现打现用，不预存）

**原则：只为"现在要跑的那一轮"打包。** run kit 全部由现成件合成，`make_line_kit.py` 几秒出包，
预存一堆"可能用得上"的包只会让"该跑哪个"变模糊（2026-11-05 用户指出后已删掉一个这样的包）。

## 已在盘上的（都有明确用途）

| 包 | 用途 |
|---|---|
| `ptcg_train_kit_v3.tar.gz` | 首次部署（补丁+工具+187 池，自洽） |
| `r1_db04_vs_grookey_kit.tar.gz` | **当前该跑的**：主侧第 1 轮，陪练=BC grookey |
| `r1_db04_vs_abra_kit.tar.gz` | 主侧备用：陪练换成 abra（已验证方向，加固用） |
| `dual_starter.tar.gz` | 本机编排器 + 路由框架（不传 GPU 机） |

## 按需生成（需要时现打）

**反向：把 grookey 陪练练强**（只有当主侧第 1 轮毫无提升、怀疑陪练太弱时才需要）

```bash
./tools/py tools/make_line_kit.py \
  --init bc_pool/grookey.bin --run r1_grookey_vs_db04 \
  --anchor grookey_thwackey__2617 --dwf decks/weights_real.txt \
  --banks models/ppo/g6_db04.bin --bank-decks dunsparce_abra__db04 \
  --league-pct 0.20 --track "dunsparce_abra__db04,grookey_thwackey__2617" \
  --stage-full-pool --out /root/r1_grookey_vs_db04
```

**后续轮次**：把 `--init` 换成上一轮的产出、`--banks` 换成新的陪练、
`--run` 换个名字即可。用 `dual_train.py pack` 可以自动算出这些参数：

```bash
./tools/py tools/dual_train.py pack --names <state名> \
    --anchor <本轮 learner 的卡组> --dwf decks/weights_real.txt --full-pool
```

## 判断"该打哪个包"的依据

1. 主侧（要提交的那副牌）**有没有提升** → 没提升才去练陪练
2. `dual_train.py status` 里的 `flat` 计数：主侧 flat 涨到 `patience` 就换方向
3. 不要在没测过之前就为多种可能各打一个包
