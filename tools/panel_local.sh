#!/usr/bin/env bash
# **本地跑某候选的 17 格面板**（不占云机；用来补云端没覆盖的候选）
#   bash tools/panel_local.sh <候选> [TESTSET] [TAG] [WORKERS] [EXTRA_CELLS...]
#   例：bash tools/panel_local.sh cand9D6ER3 decks/testset_meta_real_v2.tsv v2 8
# 说明：本机 12 核，workers 别开太大（默认 8）；每格 150 局串行跑，整面板约 60~80 分钟。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
NAME="${1:?候选名}"; TS="${2:-decks/testset_meta_real_v2.tsv}"; TAG="${3:-v2}"; W="${4:-8}"
OUT="logs/meta/${NAME}_${TAG}_s950000"; mkdir -p "$OUT"
[ -s "build/agents/$NAME/model.pt" ] || { echo "✗ 缺 build/agents/$NAME/model.pt"; exit 1; }
echo "[$(date +%H:%M:%S)] 本地面板 $NAME 口径=$TS 标签=$TAG workers=$W"
awk -F'\t' '!/^#/ && NF>=3 && $1!="tail_mix"{print $1"\t"$2}' "$TS" | while IFS=$'\t' read -r cell dirp; do
  f="$OUT/vs_$cell.json"
  [ -s "$f" ] && { echo "  skip $cell（已有）"; continue; }
  [ -d "$dirp" ] || { echo "  ✗ $cell 缺 $dirp"; continue; }
  echo "[$(date +%H:%M:%S)] 跑 $cell ← $dirp"
  ./tools/py ptcg_league/eval_pair.py --cand "build/agents/$NAME" --opp "$cell=$dirp" \
    --matches 150 --workers "$W" --seed-base 950000 --out "$f" >> "logs/local_${NAME}_${cell}.log" 2>&1 \
    || echo "  ✗ $cell 失败（见 logs/local_${NAME}_${cell}.log）"
done
n=$(ls "$OUT"/vs_*.json 2>/dev/null | wc -l)
echo "[$(date +%H:%M:%S)] $NAME 完成 $n/17 格"
./tools/py tools/panel_rank.py --cands "$NAME" --tag "$TAG" 2>/dev/null | tail -2
