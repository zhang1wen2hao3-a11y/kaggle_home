#!/usr/bin/env bash
# **交叉对战矩阵**：统一协议（同种子基准、座位交替、Bo3、各 400 场）测"谁真的更强"。
# 目的：区分 H1（G8 普遍更强）/ H2（G8 只是克制 db04）/ H3（shuppet 牌+BC 驾驶就够）。
#
#   CELLS="g8bc:port" bash tools/cross_matrix.sh       # 只跑某一格
#   bash tools/cross_matrix.sh                          # 跑默认优先格（4 格，约 40 分钟）
set -uo pipefail
cd "$(dirname "$0")/.."
N="${N:-400}"; SEED="${SEED:-913000}"; W="${WORKERS:-0}"
declare -A A=(
  [port]=build/agents/port_db04
  [g6]=ptcg_league/pools/league/opp_g6_db04
  [g9]=build/agents/g9_db04
  [g7b]=ptcg_league/pools/league/opp_g7b_shuppet
  [g8]=ptcg_league/pools/league/opp_g8_shuppet
  [g8bc]=build/agents/g8_bc_parent
)
# 优先格：① BC 父权重 vs db04（测 H3）② BC父 vs PPO后（G8 的 PPO 到底加了什么）
#        ③ G8 vs G7B（测 H1：是不是普遍更强）④ BC父 vs G6
CELLS="${CELLS:-g8bc:port g8bc:g8 g8:g7b g8bc:g6}"
mkdir -p logs/matrix
for cell in $CELLS; do
  c=${cell%%:*}; o=${cell#*:}
  out="logs/matrix/${c}__vs_${o}.json"
  [ -f "$out" ] && { echo "[skip] $cell"; continue; }
  echo "########## learner=$c  vs  opponent=$o ##########"
  ./tools/py ptcg_league/eval_pair.py --cand "${A[$c]}" --opp "${o}=${A[$o]}" \
      --matches "$N" --workers "$W" --seed-base "$SEED" \
      --out "$out" --logdir "logs/matrix/${c}__${o}" 2>&1 | tail -3
done
echo
echo "===== 交叉矩阵（learner 的胜率）====="
./tools/py - <<'PY'
import glob, json, os
print("%-30s%9s%6s%6s" % ("格（learner vs opp）","胜率","n","异常"))
for f in sorted(glob.glob("logs/matrix/*.json")):
    o=json.load(open(f))["overall"]
    print("%-30s%9.4f%6d%6d" % (os.path.basename(f)[:-5], o["score"], o["n"], o["errors"]))
PY
