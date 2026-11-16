#!/usr/bin/env bash
# **本地兜底评测**：远程评估机都连不上时，在开发机直接跑真实口径面板。
#   bash tools/eval_candidate_local.sh <候选> "<空格分隔的格>" ["<第二种子格>"]
# 与远端跑法同一口径（同一 eval_pair、同一 Bo3、座位交替），只是 worker 少一些、串行按格跑。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
NAME="${1:?}"; CELLS="${2:-}"; SEED2_CELLS="${3:-}"
WORKERS="${WORKERS:-8}"
[ -d "build/agents/$NAME" ] || { echo "✗ 本地缺 build/agents/$NAME（先 make_agent_dir）"; exit 1; }
run() {  # run <seed> <cells...>
  local seed="$1"; shift
  for cell in "$@"; do
    local tag="${NAME}_real_s${seed}" out="logs/meta/${NAME}_real_s${seed}/vs_${cell}.json"
    mkdir -p "logs/meta/${tag}"
    [ -s "$out" ] && { echo "[skip] $cell s$seed"; continue; }
    local dir; dir=$(awk -F'\t' -v c="$cell" '$1==c{print $2}' decks/testset_meta_real.tsv)
    [ -n "${dir:-}" ] || { echo "✗ $cell 不在真实口径测试集"; continue; }
    echo "→ $cell s$seed ($dir)"
    ./tools/py ptcg_league/eval_pair.py --cand "build/agents/$NAME" --opp "$cell=$dir" \
        --matches 150 --workers "$WORKERS" --seed-base "$seed" --out "$out" 2>&1 | tail -1
  done
}
run 950000 $CELLS
[ -n "$SEED2_CELLS" ] && run 950001 $SEED2_CELLS
echo LOCAL_EVAL_DONE
