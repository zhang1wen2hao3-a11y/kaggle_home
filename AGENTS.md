# AGENTS.md —— 接手这个仓库前先读（给 AI 助手 / 新会话用）

工作目录：`/home/zhang/code/kaggle/the-pokemon-company`
（每次 `bash` 都要显式 `cd`，默认 cwd 不是这里；`./tools/py` 是仓库内 python 包装器。）

## 0. 有现成方法就别重新推导（用户 2026-11-18 明确要求）

| 要做的事 | 照这份做 | 别做 |
|---|---|---|
| **抓排行榜高手的回放** | **`docs/抓对手回放.md`**（四步链路 + 8 个坑 + 抓完怎么分析）；一键：`./tools/py tools/fetch_team_replays.py --top 20 --per-team 20 --out epsoide_top20 --sleep 1.5` | 别再从 CLI 试探"榜单怎么下、回放怎么取" |
| 提交 Kaggle | `python3 tools/kaggle_submit.py <tar.gz> "<说明>"`（上传域名 IP 被黑洞时会自动钉可达 IP） | 别用裸 `kaggle competitions submit`（会卡在 0%） |
| 提交包 | `bash tools/make_sub.sh <agent名> [deck.csv]`（文件名 ≤30 字符，否则 Validation 失败） | — |
| 训练收尾 | `GPU_HOST=… FINAL=… RUN=… NAME=… DECK=… bash tools/finish_run_generic.sh`（md5 校验→转 pt→关 GPU→推 CPU→面板→聚合→关机） | 别手搓 rsync/转换 |
| 多候选派发 | `bash tools/serial_dispatch.sh <host> "cand:cell1 cell2" ...` | 别同时起两个 `dispatch_panel.py`（会互相抢空闲机判定，两个队列都不推进） |
| 评估机连不上 | 直接改本地跑：`bash tools/eval_candidate_local.sh <cand> "<cells>"`（不用问） | 别干等 |

## 1. 卡组/驾驶员的命名与标注（用户要求）

- 提到一副牌**不能只给 4-hex id**，要给：**卡组名 + 最高分玩家 + 该玩家分数 + 牌子在天梯的胜率/局数**。
  用 `./tools/py tools/deck_label.py <4hex>` 生成，例：`abra_kadabra__f5f8（最高分 YumeNeko 1145，该牌 55.3%/532 局）`。
- 卡组身份 = `sha256(','.join(sorted(60 个卡号)))[:8]`；池子文件名 `<archetype>__<8hex>`。
- 一个实体一个名字；驾驶员名不要等于卡组 4-hex id。

## 2. 统计纪律（写死，不许事后放宽）

- 逐格结论：**Δ≥5pp 且合并 n≥400**；整行结论：Δ≥1pp。
- 面板 n=200/格 ⇒ ±6.9pp；n=150/格 ⇒ ±8pp。差 1–2pp 一律视为噪声，别包装成"提升"。
- 改动前先写下判据（预注册），跑完只报"过了/没过"。

## 3. 机器纪律

- 训练：GPU 跑完 → **md5 校验 → 转 pt → 立刻关机**；评估：结果 **拉回并校验数量后** → 关机。
- **双重保险（2026-11-19 用户要求）**：在任何机器上开工前，先挂一个 systemd 兜底定时关机——
  `bash tools/arm_deadline_shutdown.sh <host> 02:00`（或派发时带 `DEADLINE=02:00`）。
  原因：看门狗靠"日志标记 + 结果数量"，一旦调度器被中断或 ssh 因主机指纹失败，
  脚本会**退出但不关机**，机器空转（已经发生过两次）。定时关机由机器自己执行，不受我这边的进程/网络影响。
- **新物化卡组后要把池子同步到评估机**，否则远端 `make_agent_dir` 找不到 deck.csv（2026-11-19 踩过）：
  `rsync -az -e ssh --include '*.csv' --exclude '*' decks/pool_field/ <host>:/root/eval_cpu_kit/decks/pool_field/`
  再把 `decks/testset_meta_real.tsv`、`tools/driver_registry.py` 一起推过去。
- **所有跨机 ssh/scp/rsync 都要带 `-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null`**：
  新机器第一次连接时没这条会 `Host key verification failed`，而收尾脚本会把"连不上"当成机器已释放
  ⇒ 2026-11-19 就因此把面板跑到了开发机本地（用户明确禁止），并且让 123 关不掉机。
- 中途不要用 `pgrep -f "<字面量>"`：会匹配到自己那条 ssh/bash 命令行（`pkill` 把自己 shell 杀了 3 次）。用中括号形式 `[e]val_pair`。
- `ssh` 端口是小写 `-p`（大写 `-P` 是 tag 选项，会白等 4 小时）。
- 不要长 sleep 等任务；起任务后 12–20 s 内做一次探活，之后靠日志轮询。

## 4. 每轮都要维护的表（别让表变旧）

`bash tools/refresh_tables.sh` 会重算（**任何一步失败都非零退出，不许跳过**）：
① 母表 → ②③ 交付表 → **④ 必须卡组清单 → ⑤ 补必须卡组的驾驶员（BC/init）** → ⑥ 名字校验
→ **⑦ 投资分 → ⑧ 覆盖校验（必须卡组一条不漏）** → ⑨ 方案A 测试集。

- **训练资源按投资分排**：`docs/卡组天花板与投资分.csv`。
  公式 `投资分 = 100 × (缺口G × 覆盖C × 高峰H)^(1/3)`，
  口径见 **`docs/投资分口径.md`**（用户 2026-11-19 要求：向**高分卡组 + 高覆盖率**倾斜，
  缺口 = 远端高分 − 我们在这副牌上的分）。
- **必须卡组**（17 格 ∪ 母表 ∪ 榜前 N 队≥10 局用过的牌，现在 28 副）见 `docs/必须卡组清单.csv`；
  缺驾驶员会被自动补成 `bc_<4hex>`（该原型有 BC 权重）或 `init_<4hex>`（端口权重），并登记注册表。
- **卡组淘汰机制**：`tools/prune_decks.py`（口径见 **`docs/淘汰机制.md`**）——
  弱（强度全场最低）+ 无格压力（<55%）+ 环境无覆盖（<2%）三条同时成立才淘汰，冠军级（天花板≥1500）保护；
  产出 `docs/卡组淘汰与新增建议.csv`。刷新第 ⑨ 步是 `--check` 硬闸门：该淘汰却还留在表里就非零退出。
- 旧脚本 `tools/build_investment_table.py` 已降级为**库**（提供真实份额权重/母表/天花板规则/17 格面板分），
  被 `tools/build_investment_score.py` import；不要再直接跑它生成投资分。

## 5. 当前状态入口

- 计划与结论：`docs/下一步计划_2026-11-18.md`（§7 e1d7 专项、§8 面板与 LB 口径失准）
- LB 读数历史：`LB记录.md`
- 提交复盘：`docs/提交复盘-输在哪里.md`
