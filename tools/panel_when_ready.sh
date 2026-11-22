#!/usr/bin/env bash
# **等训练产物落地后自动跑面板**（训练/评估解耦：GPU 在训，CPU 机等着接活）
#   bash tools/panel_when_ready.sh <host> <标记文件> "candCB25R1:beldum-metang__cb25 cand049CR1:dreepy-drakloak__049c" [TESTSET] [TAG] [WORKERS]
#   每个 spec = <候选名>:<牌表文件名(不含 .csv)>；候选权重必须已由收尾脚本拉到 models/ppo/<候选>.pt
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
H="${1:?host}"; MK="${2:?标记文件}"; SPECS="${3:?spec}"; TS="${4:-decks/testset_meta_real_v2.tsv}"
TAG="${5:-v2}"; W="${6:-24}"
rm -f "$MK"
for spec in $SPECS; do
  NAME="${spec%%:*}"; DECK="${spec#*:}"
  echo "[$(date +%H:%M:%S)] 等 models/ppo/$NAME.pt（最多 120 分钟）"
  for i in $(seq 1 240); do [ -s "models/ppo/$NAME.pt" ] && { echo "[$(date +%H:%M:%S)] ✓ $NAME 到位"; break; }; sleep 30; done
  [ -s "models/ppo/$NAME.pt" ] || { echo "✗ $NAME 权重没等到，跳过"; continue; }
  bash tools/make_agent_dir.sh "build/agents/$NAME" "models/ppo/$NAME.pt" "decks/pool_field/$DECK.csv" | tail -1
  bash tools/panel_on_host.sh "$NAME" "$H" "$TS" "$TAG" "$W"
done
echo "PANEL_WHEN_READY_DONE" > "$MK"
echo "[$(date +%H:%M:%S)] 全部完成 → $MK"
