#!/usr/bin/env bash
# **在 GPU 机上启动 A/B 长跑**（含 50M 步方案）。
#
#   bash tools/launch_ab.sh                      # 默认：26c6/slowking，25M/阶段（总计 50M）
#   STEPS=50000000 bash tools/launch_ab.sh       # 每阶段 50M（总计 100M，约 29h）
#   SHIP=dunsparce_abra__db04 bash tools/launch_ab.sh
#
# 结构（与参考工程 two-net 一致）：
#   A 阶段：A=舰牌专精（指定卡组），对手驾驶员**按 choice 矩阵选**（覆盖率分配 bank 槽位）
#   B 阶段：B=通才（覆盖率先验抽所有牌），对手 A 用 **init** 权重打舰牌
#   每轮结束做 KL 回拉 + 门控晋升（镜像 h2h ≥ B0+3pp 才把 teacher 换成 A）
#
# 权重快照：DYN 模式默认 20 epoch（= 655k 步）——杀训练最多丢 655k 步；
# 每个都能用 `./tools/py tools/weight_dist.py --bin <快照>` 量位移 → 判 scaling。
set -uo pipefail
cd /root/ptcg_train_kit/port_repo

SHIP="${SHIP:-slowpoke_slowking__26c6}"
STEPS="${STEPS:-25000000}"        # 每阶段步数；总计 = 2 × STEPS × ROUNDS
ROUNDS="${ROUNDS:-3}"          # 对齐参考工程的 6–8 轮交替（我们先用 3 轮）
INIT="${INIT:-/root/kl/port_w.bin}"
OUT="${OUT:-/root/autodl-tmp/teacher_loop_ab}"
ALPHA="${ALPHA:-0.10}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20}"   # DYN 模式：杀训练最多丢 655k 步
THRESH_PP="${THRESH_PP:-3.0}"          # 对齐参考工程 cross-gate-delta 0.03，也更抗我们的 ±2pp 噪声
B_SLOTS="${B_SLOTS:-2}"
PROMOTE_MARGIN="${PROMOTE_MARGIN:-0.03}"
N_GATE="${N_GATE:-400}"
mkdir -p "$OUT"

echo "===== 预检 ====="
fail=0
need() { [ -e "$1" ] && echo "  ✓ $1" || { echo "  ✗ 缺 $1"; fail=1; }; }
need "$INIT"
need /root/kl/g6_db04.bin
need /root/kl/g8_bc_parent.bin
need decks/weights_field_b05.txt
need decks/pool_field
need tools/choice_to_banks.py
need docs/the-pockmon-battle-choice.csv
need tools/teacher_loop.sh
grep -q bank_deck native/ptcg/env.c 2>/dev/null && echo "  ✓ bank-decks 补丁" || { echo "  ✗ 缺 bank-decks 补丁"; fail=1; }
[ "$fail" = 0 ] || { echo "✗ 预检失败，先补齐缺件（tools/push_ab_kit.sh 从本地推）"; exit 1; }

TOTAL=$(( STEPS * 2 * ROUNDS ))
HRS=$(awk -v s="$TOTAL" 'BEGIN{printf "%.1f", s/957/3600}')
echo
echo "A/B 长跑：SHIP=$SHIP  每阶段 $((STEPS/1000000))M 步 × 2 × $ROUNDS 轮 = $((TOTAL/1000000))M 步"
echo "预计 ${HRS} 小时（957 SPS）  快照间隔 $((CHECKPOINT_INTERVAL*32768/1000000)).2M 步"

nohup env STEPS="$STEPS" ROUNDS="$ROUNDS" OUT="$OUT" ALPHA="$ALPHA" \
  CHECKPOINT_INTERVAL="$CHECKPOINT_INTERVAL" PROMOTE_MARGIN="$PROMOTE_MARGIN" N_GATE="$N_GATE" \
  bash tools/teacher_loop.sh "$SHIP" "$INIT" "$INIT" "$ROUNDS" "$STEPS" "$ALPHA" \
  > "$OUT/run.log" 2>&1 &
echo "✓ 已启动 pid=$!   日志 $OUT/run.log"
echo
echo "监控："
echo "  tail -5 $OUT/run.log"
echo "  ls experiments/native-r*A/*.bin | tail -3      # 权重快照"
echo "  grep -E 'epoch (5|10) ' experiments/native-r*A/logs 2>/dev/null | tail -2"
