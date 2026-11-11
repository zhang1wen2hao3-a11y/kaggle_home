#!/usr/bin/env bash
# **分水岭实验**：30M 步 finetune，验证"训练不涨点是不是因为没练够"。
#
# 与 ph3_2617_v2 只差一个变量：T（400k → 30M，75×）。其余逐字相同：
#   INIT=/root/kl/port_w.bin  ANCHOR=2617  DWF=weights_2617.txt
#   BANKS=shup_a0.5@10%  AG=512 MB=512 LR=2e-4 SEED=424242 THREADS=8
#
# 判据（跑完后按顺序执行）：
#   1. 权重位移：./tools/py tools/weight_dist.py --bin <final.bin>
#      → 中位张量相对 L2 >1%（接近 BC 量级）= 只是没练够；仍 ≈0.3% = RL 信号没动
#   2. 面板：拉到本地、转 pt、建 agent、跑 12 格 meta_eval（tools/eval_queue.sh 追加）
#
# 成本：957 SPS → 30M 步 ≈ 8.7 小时（1×2080Ti）。训练器有 state_latest.bin 断点，
# 中途断了可改 T 为剩余步数从断点续（db04_line.sh 支持）。
set -uo pipefail
cd "$(dirname "$0")/.."
echo "========== ph3_2617_watershed =========="
echo "T = 30,000,000（= v2 的 75×）  其余与 ph3_2617_v2 完全相同"
INIT=/root/kl/port_w.bin \
ANCHOR=grookey_thwackey__2617 \
TRACK=grookey_thwackey__2617,shuppet_dhelmise__0d86,teal_mask_ogerpon_ex_applin__7e39 \
DWF=decks/weights_2617.txt \
BANKS=/root/kl/shup_a0.5.bin \
BANKDECKS=shuppet_dhelmise__0d86 \
LEAGUE_PCT=0.10 \
T=30000000 AG=512 MB=512 LR=2e-4 SEED=424242 THREADS=8 \
OUT=/root/autodl-tmp/ph3 \
RUN=ph3_2617_watershed TAG=2617 \
bash tools/db04_line.sh 2>&1 | tee /root/autodl-tmp/ph3/run_2617_watershed.log
