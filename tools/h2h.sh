#!/usr/bin/env bash
# **两个 agent 头对头**（同机同种子，座位由 eval_pair 均衡）。
#   bash tools/h2h.sh <A目录> <B目录> [场数] [tag]
# 例：bash tools/h2h.sh build/agents/r2A_9d6e build/agents/r1A_9d6e 400 r2A_vs_r1A
set -uo pipefail
cd "$(dirname "$0")/.."
A="${1:?A 目录}"; B="${2:?B 目录}"; N="${3:-400}"; TAG="${4:-$(basename "$A")_vs_$(basename "$B")}"
SEED="${SEED:-980000}"; OUT="${OUT:-logs/h2h}"
mkdir -p "$OUT"
./tools/py ptcg_league/eval_pair.py --cand "$A" --opp "enemy=$B" \
  --matches "$N" --workers 0 --seed-base "$SEED" --out "$OUT/$TAG.json" --logdir "$OUT/$TAG" 2>&1 | tail -2
