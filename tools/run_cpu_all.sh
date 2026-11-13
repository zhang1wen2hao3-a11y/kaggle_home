#!/usr/bin/env bash
# **CPU 机的一次性总队列**（幂等，机器随时断都能续）：
#   ① 补 choice-in-table 缺的自家对局（38 对 × 2 种子）
#   ② 给还只有单种子数据的驾驶员补种子（ab9d6e_r1B / r2B）
#   ③ 把 9d6e 行漏分最大的格子（2617，权重 9.6%）做到 n=1000，给矩阵定个准数
set -uo pipefail
cd "$(dirname "$0")/.."
echo "===== [$(date +%H:%M)] ① 内部对局 ====="
bash tools/run_in_table.sh 2>&1 | tail -3

echo "===== [$(date +%H:%M)] ② 单种子驾驶员补种子 ====="
for spec in "ab9d6e_r1B:build/agents/ab9d6e_r1B" "ab9d6e_r2B:build/agents/ab9d6e_r2B"; do
  tag="${spec%%:*}"; ag="${spec##*:}"
  [ -d "$ag" ] || { echo "  ✗ 缺 $ag，跳过"; continue; }
  for sd in 950001 950002; do
    t="${tag}_s${sd}"
    [ -f "logs/meta/$t/summary.json" ] && { echo "  [skip] $t"; continue; }
    echo "  --- $t"
    SEED=$sd bash tools/meta_eval.sh --cand "$ag" --tag "$t" 2>&1 | tail -1
  done
done

echo "===== [$(date +%H:%M)] ③ 9d6e→2617 高样本（n=1000）====="
for spec in "port:build/agents/port_on_9d6e" "r2A:build/agents/ab9d6e_r2A"; do
  c="${spec%%:*}"; ag="${spec##*:}"
  [ -d "$ag" ] || { echo "  ✗ 缺 $ag，跳过"; continue; }
  f="logs/hiN_2617_${c}.json"
  [ -f "$f" ] && { echo "  [skip] $c（已有）"; continue; }
  ./tools/py ptcg_league/eval_pair.py --cand "$ag" --opp "grookey_2617=build/agents/c_grookey" \
    --matches 1000 --workers 0 --seed-base 1400000 --out "$f" --logdir "logs/hiN_2617_${c}" 2>&1 | tail -1
done
echo "===== [$(date +%H:%M)] 全部完成 ====="
