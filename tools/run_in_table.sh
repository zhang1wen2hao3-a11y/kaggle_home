#!/usr/bin/env bash
# **补 the-pockmon-battle-choice-in-table.csv 缺的自家对局**（在装好 eval kit 的 CPU 机上跑）
#
#   bash tools/build_in_table.py --dry     # 先看缺哪些 → logs/in_table_missing.txt
#   bash tools/run_in_table.sh             # 在目标机上补（幂等）
#
# 每对跑 **2 个种子 × 200 场 Bo3**（不是一次 400 场）：总场次一样，但种子级方差减半
# —— 本轮实测同一格单种子能差 5pp 以上，见 docs/ab_run2.md §14.1。
set -uo pipefail
cd "$(dirname "$0")/.."
N="${N:-200}"
LIST="${LIST:-logs/in_table_missing.txt}"
OUT="${OUT:-logs/in_table}"
mkdir -p "$OUT"
[ -f "$LIST" ] || { echo "✗ 缺 $LIST：先跑 ./tools/py tools/build_in_table.py --dry"; exit 1; }
TOT=$(grep -c . "$LIST")
echo "待补 $TOT 对 × 2 种子 × $N 场"
i=0
while read -r x y px py ax ay; do
  [ -n "${x:-}" ] || continue
  i=$((i+1))
  for spec in "1200000:" "1300000:__s2"; do
    sb="${spec%%:*}"; suf="${spec##*:}"
    f="$OUT/${x}_${y}${suf}.json"
    [ -f "$f" ] && { echo "[$i/$TOT] skip $x vs $y$suf（已有）"; continue; }
    t0=$(date +%s)
    ./tools/py ptcg_league/eval_pair.py --cand "build/agents/$ax" --opp "enemy=build/agents/$ay" \
      --matches "$N" --workers 0 --seed-base "$sb" --out "$f" --logdir "$OUT/${x}_${y}${suf}" \
      > "$OUT/${x}_${y}${suf}.log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [ "$rc" = 0 ] && [ -f "$f" ]; then
      s=$(./tools/py -c "import json;o=json.load(open('$f'))['overall'];print(f\"{o['score']:.4f} n={o['n']}\")")
      echo "[$i/$TOT] ✓ $x($px) vs $y($py)$suf  $s (${dt}s)"
    else
      echo "[$i/$TOT] ✗ $x vs $y$suf 失败 rc=$rc（看日志）"
    fi
  done
done <<< "$(grep . "$LIST")"
echo "=== 全部结束，回到本地跑 ./tools/py tools/build_in_table.py 生成表 ==="
