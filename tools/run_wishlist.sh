#!/usr/bin/env bash
# **按愿望单跑评估**（两台机器各跑一个 shard，互不重复；幂等）
#   bash tools/run_wishlist.sh 0        # 跑 shard 0
#   SHARD=1 bash tools/run_wishlist.sh  # 或这样
set -uo pipefail
cd "$(dirname "$0")/.."
SHARD="${1:-${SHARD:-0}}"
WL="${WL:-logs/wishlist.tsv}"
[ -f "$WL" ] || { echo "✗ 缺 $WL"; exit 1; }
if [ "$SHARD" = "all" ]; then
  TOT=$(wc -l < "$WL")
else
  TOT=$(awk -F'\t' -v s="$SHARD" '$5==s' "$WL" | wc -l)
fi
mkdir -p logs; echo "$SHARD" > logs/.runner_mode
echo "[$(date +%H:%M)] shard $SHARD：$TOT 个面板"
i=0
while IFS=$'\t' read -r ag tag sd proto shard; do
  [ "$SHARD" = "all" ] || [ "$shard" = "$SHARD" ] || continue
  i=$((i+1))
  if [ "$proto" = "A" ]; then t="${tag}_A_s${sd}"; TS=decks/testset_meta_A.tsv
  else t="${tag}_s${sd}"; TS=; fi
  if [ -f "logs/meta/$t/summary.json" ]; then echo "[$i/$TOT] skip $t"; continue; fi
  [ -d "build/agents/$ag" ] || { echo "[$i/$TOT] ✗ 缺 agent $ag"; continue; }
  echo "[$i/$TOT] --- $t（$proto）"
  TS="$TS" SEED=$sd bash tools/meta_eval.sh --cand "build/agents/$ag" --tag "$t" 2>&1 | tail -1
done < "$WL"
rm -f logs/.runner_mode
echo "[$(date +%H:%M)] shard $SHARD 完成（跳过已存在的）"
