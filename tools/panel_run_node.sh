#!/usr/bin/env bash
# 在评估机上跑 7 格面板（由 tools/evnode_panel.sh 推送执行，不手工调用）
set -uo pipefail
cd /root/ptcg_eval
TAG=$1; CAND=$2; N=${N:-200}; SEED=${SEED:-940000}
OUT=/root/panel/$TAG; mkdir -p "$OUT"
echo "[$(date +%T)] start $TAG cand=$CAND n=$N seed=$SEED" >> "$OUT/run.log"
while IFS=$'\t' read -r name dir _deck; do
  case "$name" in \#*|"") continue;; esac
  f="$OUT/vs_$name.json"
  [ -f "$f" ] && { echo "[$(date +%T)] skip $name" >> "$OUT/run.log"; continue; }
  echo "[$(date +%T)] cell $name ..." >> "$OUT/run.log"
  python3 ptcg_league/eval_pair.py --cand "$CAND" --opp "$name=$dir" \
      --matches "$N" --workers 0 --seed-base "$SEED" --out "$f" --logdir "$OUT/$name" >> "$OUT/run.log" 2>&1
done < /root/panel/panel_real_node.tsv
echo "[$(date +%T)] DONE $TAG" >> "$OUT/run.log"
