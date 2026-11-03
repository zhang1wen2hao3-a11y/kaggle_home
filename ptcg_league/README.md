# PTCG 训练联赛（Training League）

目的：**训练可以无限自动跑，晋级只由 `rules.json` 里写死的规则决定**。
你只做三件事：启动、看报告、必要时回滚。

```
./tools/py ptcg_league/run_league.py status     # 看池子 / PFSP 权重 / 断点
./tools/py ptcg_league/run_league.py next       # 跑一代（自己会停在需要 GPU 的那一步）
看 ptcg_league/reports/gen_NNNN.md              # 每代一份报告
```

---

## 1. 三个池子

| 池 | 现在有什么 | 谁往里写 | 能被覆盖吗 |
|---|---|---|---|
| `init` 初始策略池 | 6 个 BC brain + 出厂端口 | `run_league.py init` | **永不**（`Registry.add` 直接抛错） |
| `evo` 进化训练池 | 每代一条 PPO 分支的检查点 | 训练产出 | 只追加 |
| `league` 正式联赛池 | 通过评测的模型（`best`/`opponent`/`specialist`） | `decide` 步骤 | 只追加，`best` 指针可移动/回滚 |

初始池里每个模型都冻了 `frozen.json`（源 ckpt 的 sha256、`model.pt` 的 sha、卡组的 sha）。

**当前初始池**（`./tools/py tools/bc_pool_registry.py` 可重建这张表）

| agent | 卡组 | 帧数 | 端口基线一致率 | BC 后一致率 |
|---|---|---|---|---|
| `bc:deck_0d86` | shuppet_dhelmise | 170,997 | 44.7% | **73.1%** |
| `bc:deck_26c6` | slowpoke_slowking | 135,184 | 62.8% | **81.0%** |
| `bc:deck_f5f8` | abra_kadabra | 76,862 | 52.3% | **69.6%** |
| `bc:agent_RtoABC2` | beldum_metang | 16,864 | 34.5% | **75.9%** |
| `bc:agent_Raihan` | dreepy_drakloak | 22,046 | 51.1% | **62.8%** |
| `bc:agent_sue124` | shuppet_dhelmise | 19,830 | 42.1% | **66.8%** |
| `init:port_plain` | teal_mask_ogerpon | — | — | 出厂权重 |

> ⚠️ 实测：`build/port_plain/model.pt` 导出成 `.bin` 后与 `port_repo/native/pt_model_field.bin`
> **sha256 完全相同**（`3e50b694…`）。所以"出厂端口"和"通用 init"是**同一份权重**，
> 不再单独登记成两个对手（同权重互打只浪费评测预算）。

---

## 2. 一代 = 固定 8 步（每步落盘，可断点续跑）

| # | 步骤 | 产出 | 失败/中断后 |
|---|---|---|---|
| 1 | `plan` 定分支、卡组、父模型、抽 bank | `gen/NNNN/plan.json` | 重跑即重建 |
| 2 | `train` PPO 分支 | `pools/evo/<branch>/final.bin` | **本机没 GPU → 写 `pending/gen_NNNN_train.sh` 并停在 `awaiting=train`** |
| 3 | `stage` 把 ckpt 变成可评测 agent 目录 | `stage.json`（含权重 sha） | 幂等 |
| 4 | `screen` 筛查 | `screen.json` | 已有结果直接复用（协议固定 ⇒ 可复现） |
| 5 | `confirm` 确认 | `confirm.json` | 同上；筛查挂了就跳过 |
| 6 | `anchor` 固定评测集 | `anchor.json` | 同上 |
| 7 | `decide` 判定 + 晋级/特色/淘汰 + 池上限 | `decide.json`、`registry.jsonl` | 已判定就不重判 |
| 8 | `pfsp` 更新采样权重 + 出报告 | `pfsp.json`、`reports/gen_NNNN.md` | 幂等 |

断点在 `state/league_state.json` 的 `awaiting` 里。`next --resume` 从断点续，不会重跑整代。

---

## 3. 预注册规则（`rules.json`，每代记 `rules_sha`）

计量单位 = **一场 Bo3**（胜 1 / 平 0.5 / 负 0）。

**晋级（三关全过）**

1. 筛查 400 场对当前 best：得分 ≥ 0.52 **且** 异常 = 0；
2. 确认 1600 场对当前 best：配对胜率 **95% CI 下界 > 0.50**；
3. 固定评测集（7 个冻结对手 × 50 场）：相对 best 的回落 ≤ 0.02。

**保留为特色对手**：确认场次 `CI 上界 < 0.50`（总体更差），但对某个 archetype
（≥200 场）得分 ≥ 0.58。这类模型进正式池但**不**当 best。

**淘汰**：其余全部，包括 `CI 跨 0.50` —— 那是"**不能宣称提升**"，不是"变差了"。

**为什么必须"固定评测集"**：只跟当前 best 比会出现"接力式漂移"（每代都赢一点，
十代后可能整体崩）。固定 roster 是**不动的尺**。

---

## 4. 对手采样：PFSP `w_i = p_i(1-p_i)`

- `p_i` = 训练策略对对手 i 的最近一次胜率；没量过的给 0.5（不确定性最大 ⇒ 该多练）。
- **地板配额 0.05**：胜率只有 5% 的纯克制对手，`p(1-p)` 只有 0.0475，16 个对手里会被饿死 ——
  而它恰好是最该练的。所以每个对手保底 5%。
- **单头上限 0.35**：不许一个对手吃掉全部预算。
- **两阶段**（`groups`）：先抽 archetype 再抽个体；类别权重取组内 `p(1-p)` 的最大值而不是求和，
  否则"卡表多的大类"会因为人头多而自动吃满预算。抽过的类别先撤下来，不让同一原型占两个名额。
- **强制配额**（`rules.banks.pinned_archetypes`）：某些原型每代必须有一个代表在场。
  现在钉的是 `shuppet_dhelmise` —— 不是为了"练强敌"，而是因为**不钉住就永远学不到**：
  我们这副牌的 `Powerful Hand` 是招式效果，被 `Hide 'n' Sneak` 全额免疫，真实天梯对 shuppet 0/16。

---

## 5. 硬件现实逼出来的两条设计

**① 联赛池 8–16 个，但每代只上 2–4 个 bank。**
训练器在 `create_pufferl` 里把 `league-ckpts` 烘成 cudagraph 指针，跑起来**不能加**；
每 bank ≈ 1.0 GiB（512 agents 实测 15.8 GB → +2 bank 18.2 GB），
且有断言 `league_pct × n_banks < 0.5`。所以本机生成的 `league_pct = min(0.4/n, 0.12)`。
池子靠**每代重抽**覆盖，不需要原生训练器支持权重渐变。

抽 bank 时**先把候选池过滤干净再抽**（不能"抽完再丢掉与 init 同权重的那个"，
否则 bootstrap 时 parent 就在池里，有 ~35% 概率白丢一个名额）。

**② 本机（RTX 3050 / 4 GB）跑不了原生 PPO。** 训练在 4090 上做，
本机只做 plan / 评测 / 判定，靠 `tools/league_kit.py` 打一个 63 MB 的包传过去。

---

## 6. 训练健康度的早期信号（预注册）

日志里只看两个数：

| 指标 | 健康 | 红旗 |
|---|---|---|
| `lgw`（对 bank 的胜率） | **0.4–0.7** | `> 0.9` 且 `ep_len` 下降 |
| `ep_len`（每局决策数） | 稳定/上升 | 明显下降 |

历史上 `lgw 0.96–0.98` + `ep_len 176→123` = 训练退化成"自己跟自己下"，
20M 步后 `real 0.7150`，与基线完全相同。所以只要出现这个组合，**这一代不用等评测就能判废**。

顺带：现在抽 bank 时会**剔掉与 init 同权重（sha 相同）的候选** —— 冻结自快照本身不算错，
但它占掉一个 bank 名额却不提供新信息，正是上面那个退化模式的燃料。

---

## 7. 今晚的流程（4 条命令）

```bash
# ① 本机：生成计划（已做过，重跑无副作用）
./tools/py ptcg_league/run_league.py next --gens 1 --banks 3
# ② 本机：打包（63 MB，含 init + 3 个 bank + run.sh + MANIFEST）
./tools/py tools/league_kit.py --gen 1
# ③ 4090：解压就跑
scp dist/league_g0001.tar.gz <4090>:~/work/
#    ssh <4090>: cd ~/work && tar xzf league_g0001.tar.gz && cd league_g0001 && bash run.sh
# ④ 本机：把 final.bin 拉回来，续跑
scp <4090>:~/work/league_g0001/final.bin ptcg_league/pools/evo/g0001_7e39_port_plain/final.bin
./tools/py ptcg_league/run_league.py next --resume
```

`run.sh` 会先校验包的 sha（传坏了当场停），再查显存有没有残留进程占着，
训练完打印 `lgw`/`ep_len` 两行健康指标。

---

## 8. 出问题怎么办

| 现象 | 处理 |
|---|---|
| 某代卡在 `awaiting=train` | 把 ckpt 放回 `plan.ckpt_out`，`next --resume` |
| 晋级后发现不对 | `run_league.py rollback`（默认回上一个 best）或 `--to <agent_id>` |
| 怀疑规则被改过 | `run_league.py verify`（比对最近 5 代 `rules_sha`） |
| 池子被近似模型塞满 | `prune_league()` 按（相似度, anchor 分）贪心淘汰，`decide` 每代自动执行 |
| 评测出现 `异常 > 0` | **该数字一律不算数**（异常 = 抛异常或随机兜底），先查内存/超时 |
| 想小规模验证链路 | `--rules ptcg_league/rules_pilot.json`（场次缩到 1/25，报告里 `rules_sha` 会不同） |

## 9. 目录

```
ptcg_league/
├── league.py            核心逻辑：注册表 / 状态 / PFSP / decide()（纯函数，可单测）
├── run_league.py        调度器：init / next / status / verify / rollback
├── eval_pair.py         单项任务：固定种子配对评测（复用 tools/bo3_eval.py 的引擎面板）
├── report.py            报告渲染（只读 json，随时可重渲）
├── rules.json           预注册规则（v1）
├── rules_pilot.json     小规模试跑规则（禁止用它晋级正式 best）
├── state/registry.jsonl 追加式登记表
├── state/league_state.json  代计数 / best 指针 / PFSP / 断点 / 历史
├── pools/{init,evo,league}/
├── gen/NNNN/*.json      每代的 8 个产出
├── reports/             每代报告 + overview.md
├── pending/gen_NNNN_train.sh   给 GPU 机器看的启动脚本
└── logs/
```
