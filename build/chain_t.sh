#!/usr/bin/env bash
# 下一批 PPO（t1/t2/t3）远端链式脚本：放到训练机上 nohup 执行即可。
# 每包 400k 步 ≈ 13 分钟；产物必须是 18,205,408 B，否则停链。
set -u
R=/root/ptcg_train_kit/port_repo
INIT_G6=/root/ptcg_train_kit/port_repo/init/g6_db04.bin
INIT_R1A2=/root/autodl-tmp/models/r1A2_db04_g6.bin
OUT=/root/autodl-tmp/models
LOG=/root/autodl-tmp/runlogs
mkdir -p "$OUT" "$LOG"
run_one() {   # run_one <run> <anchor> <init> <bankdeck>
  local run=$1 anchor=$2 init=$3 bd=$4
  [ -f "$OUT/$run.bin" ] && { echo "===== $run 已存在，跳过"; return 0; }
  echo "===== $run start $(date +%H:%M:%S) anchor=$anchor init=$(basename "$init") bankdeck=$bd"
  ( cd "$R" && env PYTHON=/root/miniconda3/bin/python3 INIT="$init" \
      OUT="$OUT" RUN="$run" T=400000 THREADS=8 LEAGUE_PCT=0.2 SEED=424242 \
      ANCHOR="$anchor" TRACK="$bd,$anchor" DWF=decks/weights_real.txt \
      BANKS="$init" BANKDECKS="$bd" \
      bash tools/db04_line.sh > "$LOG/$run.log" 2>&1 )
  local sz=$(stat -c%s "$OUT/$run.bin" 2>/dev/null || echo 0)
  echo "===== $run done $(date +%H:%M:%S) size=$sz"
  [ "$sz" = "18205408" ] || { echo "✗ $run 产物异常，停链"; return 1; }
}
run_one t1_9d6e       dunsparce_abra__9d6e                   "$INIT_G6"   teal_mask_ogerpon_ex_applin__7e39 || exit 1
run_one t2_7e39       teal_mask_ogerpon_ex_applin__7e39      "$INIT_G6"   dunsparce_abra__0bf5                || exit 1
run_one t3_db04_r1A2  dunsparce_abra__db04                   "$INIT_R1A2" teal_mask_ogerpon_ex_applin__7e39 || exit 1
echo "CHAIN_T_DONE $(date +%H:%M:%S)"
