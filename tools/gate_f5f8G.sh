#!/usr/bin/env bash
# 在评测机上跑 **candf5f8G 的门禁**（预注册判据见 docs/驾驶员与覆盖规则.md §1 与 run_f5f8_gap.sh 头）
#   G1 vs c_grookey(端口@2617)   G2 vs ph3_2617_v1   G3 vs opp_3eaa(端口@3eaa)
#   G5 vs g8_bc_parent(守 0d86 格)   G4 方案A 12 格面板
#   bash tools/gate_f5f8G.sh            # 全部
#   bash tools/gate_f5f8G.sh cells      # 只跑三格（快）
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
WHAT="${1:-all}"
cell() { local out="data/opp_matrix/$2.json"; [ -s "$out" ] && { echo "skip $2"; return; }
  ./tools/py ptcg_league/eval_pair.py --cand build/agents/$1 --opp "$3" --matches 200 --workers 24 \
    --seed-base "$4" --out "$out" 2>&1 | tail -1; }
if [ "$WHAT" = all ] || [ "$WHAT" = cells ]; then
  for sd in 950000 950001 950002; do
    cell candf5f8G f5f8G_vs_c_grookey_s$sd   "g=build/agents/c_grookey"   $sd
    cell candf5f8G f5f8G_vs_ph3_2617v1_s$sd  "p=build/agents/ph3_2617_v1" $sd
    cell candf5f8G f5f8G_vs_opp_3eaa_s$sd    "e=build/agents/opp_3eaa"    $sd
  done
  cell candf5f8G f5f8G_vs_shbc_s950000 "s=build/agents/g8_bc_parent" 950000
fi
if [ "$WHAT" = all ] || [ "$WHAT" = panel ]; then
  TS=decks/testset_meta_A.tsv N=150 SEED=950000 bash tools/meta_eval.sh \
    --cand build/agents/candf5f8G --tag candf5f8G_A_s950000 2>&1 | tail -2
fi
echo GATE_F5F8G_DONE
