#!/usr/bin/env bash
# **7 格加权面板评测**：按真实天梯分布（decks/field_real.json）给一个 agent 打加权分。
#
#   bash tools/panel_eval.sh --cand build/agents/g8_bc_parent --tag g8bc
#   N=400 SEED=913000 bash tools/panel_eval.sh --cand models/archive/bc_grookey3 --tag bcg3
#
# 为什么要固定 SEED：跨机噪声 ±4~6pp，只有**同种子**的两个 agent 才可配对比较。
# 输出：logs/panel/<tag>/vs_<cell>.json（逐格）+ weighted.json（加权结果），可断点续跑。
set -uo pipefail
cd "$(dirname "$0")/.."
CAND=""; TAG=""
while [ $# -gt 0 ]; do
  case "$1" in --cand) CAND="$2"; shift 2;; --tag) TAG="$2"; shift 2;; *) shift;; esac
done
[ -n "$CAND" ] || { echo "✗ 缺 --cand"; exit 2; }
TAG="${TAG:-$(basename "$CAND")}"
N="${N:-200}"; SEED="${SEED:-940000}"
OUT="logs/panel/$TAG"
mkdir -p "$OUT"
echo "候选 $CAND → $OUT（每格 $N 场，种子基准 $SEED）"
while IFS=$'\t' read -r name dir _deck; do
  case "$name" in \#*|"") continue;; esac
  [ -n "${dir:-}" ] || continue
  f="$OUT/vs_$name.json"
  [ -f "$f" ] && { echo "[skip] $name"; continue; }
  echo "########## $TAG vs $name ##########"
  ./tools/py ptcg_league/eval_pair.py --cand "$CAND" --opp "$name=$dir" \
      --matches "$N" --workers 0 --seed-base "$SEED" --out "$f" --logdir "$OUT/$name" 2>&1 | tail -2
done < <(grep -v '^#' decks/panel_real.tsv)

./tools/py - "$OUT" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
w = json.load(open("decks/field_real.json"))
print(f"\n{'格':<18}{'胜率':>7}{'95% CI':>18}{'n':>5}{'权重':>7}")
num = den = 0.0
used = 0
for f in sorted(glob.glob(d + "/vs_*.json")):
    cell = os.path.basename(f)[3:-5]
    o = json.load(open(f))["overall"]
    ww = w.get(cell)
    wtxt = f"{ww:.3f}" if ww is not None else "—"
    print(f"{cell:<18}{o['score']:>7.3f}{'[%.3f,%.3f]' % (o['ci_low'], o['ci_high']):>18}{o['n']:>5}{wtxt:>7}")
    if ww is not None:
        num += ww * o["score"]; den += ww; used += 1
if den:
    res = {"tag": os.path.basename(d), "weighted": round(num / den, 4),
           "cells": used, "n_per_cell": None}
    print(f"\n加权面板（{used}/7 格，覆盖权重 {den:.3f}）= **{num/den:.4f}**")
    json.dump(res, open(os.path.join(d, "weighted.json"), "w"), ensure_ascii=False, indent=1)
else:
    print("\n⚠️ 没有任何格子能匹配 field_real.json 的键名")
PY
