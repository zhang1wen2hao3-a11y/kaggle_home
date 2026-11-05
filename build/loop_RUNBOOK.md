# 无人值守循环整包（ptcg_loop）

一个包，机器自己跑完：**训练 → 评估 → 判定 → 继续/换边 → 直到两侧都停摆**。

## 首次(只做一次)

```bash
tar xzf ptcg_loop_kit.tar.gz && cd ptcg_loop
bash port_repo/native/build_native.sh      # ★ 不要设 PTCG_SM（脚本按 nvidia-smi 自动探测）
ls port_repo/native/puffer_ptcg/_C*.so     # 确认编出 .so
python3 -c "import sys;sys.path.insert(0,'.');import tools.bo3_eval" && echo "评估器可导入"
```

## 建一对要互训的模型

```bash
python3 tools/dual_train.py init --names db04_vs_grookey \
    --a g6_db04 --a-deck dunsparce_abra__db04 \
    --b grookey --b-deck grookey_thwackey__2617 \
    --primary A
# --a/--b 可给 models/ppo/xxx.bin 或裸名字（自动在 models/ppo、bc_pool 里找）
# --primary：我们要**提交**的那一侧，判据用 field_ev；另一侧用"打主侧的头对头"
```

## 开跑（无人值守）

```bash
bash loop.sh db04_vs_grookey 8
```

每轮它会：

1. `train` —— 用 state 里的 learner/bank 直接调 `port_repo/tools/db04_line.sh`（约 8 分钟）
2. `collect` —— 首次会先量 **round 0 基线**（该侧起始模型），再把本轮模型与它做
   **同种子配对 McNemar + Δ≥2pp** 判定
3. 判定结果决定：继续训同一侧 / 换另一侧 / 记录 flat

**停止条件**：两侧各自连续 2 轮未显著提升，或到轮数上限。停止后打印两侧最佳模型。

## 自测（不训练，只验链路）

```bash
bash loop.sh db04_vs_grookey 2 --dry
```
`--dry` 会把 `train` 换成"复制 init 当产物"，因此链路必然判"无提升"→ 第二轮换边 →
第三轮进入停摆计数。**用来确认评估/判定/换向/停止都通，不烧 GPU。**

> 想更快：`init` 时加 `--matches-per-cell 5`，面板缩到 25 局/轮，几十秒跑完一整轮。

## 目录布局（为什么长这样）

`eval_pair.py` / `bo3_eval.py` 按 `ROOT/tools/bo3_eval.py` + `build/agents/<名>/` 这套
相对路径找东西，所以整包刻意做成**仓库的瘦身副本**：

```
ptcg_loop/
├── loop.sh                 ← 入口
├── tools/                  dual_train / bo3_eval / db04_line / make_agent_dir / patch_bank_decks / py
├── ptcg_league/eval_pair.py
├── framework/replay.py
├── build/port_plain/       cg/ ptcg/ data/ main.py ...（agent 目录的软链目标）
├── build/agents/           15 个面板对手（按我们 640 局实测频率排，覆盖 87.3%）
├── decks/pool_field/       187 副 + weights_real.txt
├── models/ppo/             起始模型
├── port_repo/              训练仓库（两个 C 补丁已在）
├── ptcg_league/state/       dual_<名>.json 是循环的全部记忆，**别删**
└── RUNBOOK.md
```

## 常见问题

| 现象 | 原因/处理 |
|---|---|
| `no kernel image is available` | 编译时设了 PTCG_SM；删掉环境变量重编 |
| `✗ 已有循环在跑 <名>` | 有另一个 `loop.sh` 持有 `logs/loop_<名>.lock`。先确认它是不是还活着，死了再 `rm -f logs/loop_<名>.lock`。**别两个循环同时跑同一个 state**：各自推进 round 会互相覆盖 |
| `✗ 评估回收不全（0/400 场）` | worker 被 OOM 杀了（同目录 `*.log` 是 0 字节）。脚本已自动降到 4/2/1 个 worker 重试，全降完还不行就说明机器内存真不够：先别跑别的评估，或把 `--matches-per-cell` 调小 |
| 评估很慢 | 面板 15→5 个对手、或 `--matches-per-cell` 调小；14 vCPU 下 1250 局约 10~20 分钟 |
| 想从某个中间模型重开 | 改 `state/dual_<名>.json` 里对应侧的 `model`，再 `loop.sh` |
| 两侧都 flat 但你觉得还能练 | 加 `--patience 3` 或换 bank（重新 `init` 一个新 state） |
