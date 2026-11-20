#!/usr/bin/env bash
# **对称补测：双方都用强驾驶员，逐对实测**（2026-11-20 用户要求，本地跑）
#
# 为什么：矩阵 cell 只记"我方派谁"，对手侧长期是端口权重 ⇒ 所有"打某对手 89%"的结论都是拿软对手量的。
# 这里把每一对的**两边都换成非 init 驾驶员**，各打 150 局，得到"强 vs 强"的真实读数。
#
# 坐标对应 `docs/the-pockmon-battle-choice.csv`（15 行 × 18 列，牌序：数字在字母前）。
#
#   bash tools/sym_pair_eval.sh [每对局数=150] [workers=8]
#
# 输出：logs/meta/sym/<A>_vs_<B>.json，并汇总打印。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
MATCHES="${1:-150}"
WORKERS="${2:-8}"
OUTDIR=logs/meta/sym
mkdir -p "$OUTDIR"

# "A的驾驶员 A牌 B的驾驶员 B牌 坐标"
PAIRS=(
  "cand7e39S   7e39   ab9d6e_r1A 9d6e  11行13列"
  "bc_3eaa     3eaa   ab9d6e_r2A 9d6e   8行13列"
  "candE1D7R2  e1d7   ab9d6e_r2A 9d6e  14行13列"
  "ab9d6e_r2A  9d6e   ph3_2617_v2 2617 12行7列"
  "ab9d6e_r2A  9d6e   cand0bf5S  0bf5  12行4列"
  "cand7e39S   7e39   candf5f8E   f5f8  11行18列"
  "cand0bf5S   0bf5   candf5f8E   f5f8  （无行·单向）"
  "ph3_2617_v2 2617   cand7e39S  7e39   6行12列"
  "cand7e39S   7e39   ph3_2617_v2 2617 11行7列"
  "candE1D7R2  e1d7   cand7e39S  7e39  14行12列"
  "cand7e39S   7e39   candE1D7R2 e1d7  11行16列"
)

echo "共 ${#PAIRS[@]} 对，每对 $MATCHES 局，workers=$WORKERS"
for spec in "${PAIRS[@]}"; do
  read -r A A_d B B_d coord <<< "$spec"
  out="$OUTDIR/${A}__vs__${B}.json"
  if [ -s "$out" ]; then echo "[skip] $A vs $B"; continue; fi
  echo "→ $A($A_d) vs $B($B_d)   $coord"
  ./tools/py ptcg_league/eval_pair.py --cand "build/agents/$A" --opp "$B_d=build/agents/$B" \
     --matches "$MATCHES" --workers "$WORKERS" --seed-base 950200 --out "$out" 2>&1 | tail -1
done
echo "SYM_DONE"
