#!/usr/bin/env bash
# 铺一个标准 agent 目录（评测/联赛都认这个结构）：骨架软链 + model.pt + deck.csv
#   bash tools/make_agent_dir.sh build/agents/ag_xxx models/ppo/xxx.pt decks/pool_field/xxx.csv
set -euo pipefail
DST="${1:?目标目录}"; PT="${2:?model.pt 路径}"; DECK="${3:?deck.csv 路径}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SK="$ROOT/build/port_plain"
rm -rf "$DST"; mkdir -p "$DST"
for l in EN_Card_Data.csv cg data main.py ptcg r2e_manifest.json; do
  ln -s "$SK/$l" "$DST/$l"
done
cp -f "$PT" "$DST/model.pt"
cp -f "$DECK" "$DST/deck.csv"
echo "✓ $DST  (model=$(basename "$PT") deck=$(basename "$DECK") $(wc -l < "$DST/deck.csv") 张)"
