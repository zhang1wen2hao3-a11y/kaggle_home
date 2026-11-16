#!/usr/bin/env bash
# 在评测机上跑**真实口径面板**的指定格子（新格才需要跑，老 12 格直接复用 A 面板结果）
#   bash tools/real_panel_run.sh <候选agent> <seed> <cell1> [cell2 ...]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
CAND="$1"; SEED="$2"; shift 2
TS=decks/testset_meta_real.tsv
TAG="$(basename "$CAND")_real_s${SEED}"
OUT="logs/meta/$TAG"; mkdir -p "$OUT"
for cell in "$@"; do
  dir=$(awk -F'\t' -v c="$cell" '$1==c{print $2}' "$TS")
  [ -n "${dir:-}" ] || { echo "✗ $cell 不在 $TS"; continue; }
  [ -d "$dir" ] || { echo "✗ $cell 的 agent 不存在：$dir"; continue; }
  f="$OUT/vs_$cell.json"
  [ -s "$f" ] && { echo "[skip] $cell"; continue; }
  ./tools/py ptcg_league/eval_pair.py --cand "build/agents/$CAND" --opp "$cell=$dir" \
      --matches 150 --workers 24 --seed-base "$SEED" --out "$f" 2>&1 | tail -1
done
echo REAL_PANEL_DONE
