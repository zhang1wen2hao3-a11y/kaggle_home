#!/usr/bin/env bash
# **阶段三 v2**：换卡组训练，修掉 v1 的初始化错误。
#
# v1 的错误（`models/weights/ph3/run_2617.log` 第 8/21 行有据）：
#   INIT=/root/kl/g6_db04.bin ——db04 专家权重 + 2617 卡组 ⇒ 先"忘掉"再"学新的"，
#   RL 预算被遗忘吃掉，成品比"直接用端口权重打这副牌"还差 12~15pp。
#
# v2 改一条：**INIT 换成已经在打这副牌的端口权重**（起点就是该卡组的端口水平）。
# 其余（卡组、对手分布、bank、难度、超参、种子）全部与 v1 逐字相同 —— 只动一个变量。
#
# 用法（GPU 机上，仓库根目录）：
#   bash tools/ph3_v2.sh 2617      # → /root/autodl-tmp/ph3/ph3_2617_v2.bin
#   bash tools/ph3_v2.sh 9d6e      # → /root/autodl-tmp/ph3/ph3_9d6e_v2.bin
#   bash tools/ph3_v2.sh both      # 两个顺序跑（约 20 分钟，含 pool 加载）
set -uo pipefail
cd "$(dirname "$0")/.."

DECK_ANCHOR_7E39="teal_mask_ogerpon_ex_applin__7e39"

run_one() {
  local what=$1 anchor track tag
  case "$what" in
    2617) anchor="grookey_thwackey__2617"
          track="grookey_thwackey__2617,shuppet_dhelmise__0d86,$DECK_ANCHOR_7E39" ;;
    9d6e) anchor="dunsparce_abra__9d6e"
          track="dunsparce_abra__9d6e,shuppet_dhelmise__0d86,$DECK_ANCHOR_7E39" ;;
    *) echo "✗ 只支持 2617 / 9d6e"; return 2;;
  esac
  tag="ph3_${what}_v2"
  echo "========== $tag =========="
  echo "anchor   = $anchor"
  echo "INIT     = /root/kl/port_w.bin   ← 与 v1 的唯一差别（v1 用的是 g6_db04.bin）"
  INIT=/root/kl/port_w.bin \
  ANCHOR="$anchor" \
  TRACK="$track" \
  DWF=decks/weights_2617.txt \
  BANKS=/root/kl/shup_a0.5.bin \
  BANKDECKS=shuppet_dhelmise__0d86 \
  LEAGUE_PCT=0.10 \
  T=400000 AG=512 MB=512 LR=2e-4 SEED=424242 THREADS=8 \
  OUT="/root/autodl-tmp/ph3" \
  RUN="$tag" TAG="$what" \
  bash tools/db04_line.sh 2>&1 | tee "/root/autodl-tmp/ph3/run_${what}_v2.log"
}

case "${1:-}" in
  both) run_one 2617; run_one 9d6e ;;
  ""|*) run_one "${1:-}" ;;
esac
