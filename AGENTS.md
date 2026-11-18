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
- 中途不要用 `pgrep -f "<字面量>"`：会匹配到自己那条 ssh/bash 命令行（`pkill` 把自己 shell 杀了 3 次）。用中括号形式 `[e]val_pair`。
- `ssh` 端口是小写 `-p`（大写 `-P` 是 tag 选项，会白等 4 小时）。
- 不要长 sleep 等任务；起任务后 12–20 s 内做一次探活，之后靠日志轮询。

## 4. 每轮都要维护的表（别让表变旧）

`bash tools/refresh_tables.sh` 会重算：
`docs/the-pockmon-battle-choice-in-table.csv`（母表）、`docs/卡组天花板与投资分.csv`（天花板/投资分）、
`decks/testset_meta_real.tsv`（17 格真实份额测试集）、`decks/testset_meta_A.tsv`。

## 5. 当前状态入口

- 计划与结论：`docs/下一步计划_2026-11-18.md`（§7 e1d7 专项、§8 面板与 LB 口径失准）
- LB 读数历史：`LB记录.md`
- 提交复盘：`docs/提交复盘-输在哪里.md`
