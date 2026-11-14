#!/usr/bin/env bash
# **陪练强度阶梯**：先回答"陪练够不够强、学生还能不能学到东西"，再决定 GPU 训练阵容。
#
# 在专用 CPU 机上跑（不占 wishlist 队列）：
#   bash tools/bank_ladder.sh A     # 阶段A：学生 vs 每个候选陪练（6 格，~40 min）
#   bash tools/bank_ladder.sh B     # 阶段B：候选陪练两两互打（15 对，~1.5 h）
#   bash tools/bank_ladder.sh all
#
# 判读（写死）：
#   学生胜率 >95%  ⇒ 该陪练**没有梯度**，扔出陪练池
#   学生胜率 60~90% ⇒ 最理想，留着
#   学生胜率 <50%  ⇒ 金贵，当难关留着
#   阶段B 的排名决定谁占那 4 个 shuppet 槽位（槽位受显存限制，最多 8 bank）
set -uo pipefail
cd "$(dirname "$0")/.."
PHASE="${1:-all}"
M="${M:-200}"
W="${W:-16}"
SEED="${SEED:-950000}"
OUT=data/opp_matrix
mkdir -p "$OUT" logs
LOG="logs/bank_ladder.log"
PY=./tools/py

STUDENT_CAND=build/agents/deck_f5f8_      # 我们的 f5f8 驾驶员（BC 克隆，已过门禁）
# 候选陪练：全部在 0d86（shuppet）上，来源见 docs/opponent_library.md + bc_pool
BANKS=(
  "shbc:build/agents/g8_bc_parent|BC克隆(ミワハルキ,0.9675)"
  "d0d86:build/agents/deck_0d86_|BC克隆(171k帧,另一配方)"
  "shppo:build/agents/g8_shuppet|PPO(0.945)"
  "g7b:build/agents/g7b_shuppet|PPO g7b"
  "g7a:build/agents/g7a_db04_shuppet|PPO g7a"
  "g12:build/agents/g12_db04_dreepy_shuppet|PPO g12"
)

run() {  # run <outfile> <cand> <opp_name=dir>
  local out="$OUT/$1.json" cand="$2" opp="$3"
  if [ -s "$out" ]; then echo "  [skip] $(basename "$out")"; return 0; fi
  echo "  → $(basename "$out")  $(basename "$cand") vs ${opp%%=*}" | tee -a "$LOG"
  $PY ptcg_league/eval_pair.py --cand "$cand" --opp "$opp" --matches "$M" \
      --workers "$W" --seed-base "$SEED" --out "$out" 2>&1 | tail -2 | tee -a "$LOG"
}

if [ "$PHASE" = "A" ] || [ "$PHASE" = "all" ]; then
  echo "===== 阶段A：学生（f5f8）打每个候选陪练 =====" | tee -a "$LOG"
  for row in "${BANKS[@]}"; do
    name="${row%%:*}"; rest="${row#*:}"; dir="${rest%%|*}"
    run "ladder_student_vs_$name" "$STUDENT_CAND" "$name=$dir"
  done
fi

if [ "$PHASE" = "B" ] || [ "$PHASE" = "all" ]; then
  echo "===== 阶段B：候选陪练两两互打 =====" | tee -a "$LOG"
  n=${#BANKS[@]}
  for ((i=0; i<n; i++)); do
    for ((j=i+1; j<n; j++)); do
      a="${BANKS[$i]}"; b="${BANKS[$j]}"
      an="${a%%:*}"; ad="${a#*:}"; ad="${ad%%|*}"
      bn="${b%%:*}"; bd="${b#*:}"; bd="${bd%%|*}"
      run "ladder_${an}_vs_${bn}" "$ad" "$bn=$bd"
    done
  done
fi

echo
echo "===== 汇总（学生视角胜率 / 陪练互打）====="
$PY - "$OUT" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
def score(p):
    try: o = json.load(open(p))["overall"]
    except Exception: return None
    return o.get("score"), o.get("n"), o.get("ci_low"), o.get("ci_high")
rows = []
for p in sorted(glob.glob(os.path.join(d, "ladder_*.json"))):
    s = score(p)
    if not s or s[0] is None: continue
    rows.append((os.path.basename(p)[7:-5], s))
for name, (sc, n, lo, hi) in rows:
    flag = ""
    if name.startswith("student_vs_"):
        flag = "  ← 无梯度，扔掉" if sc > 0.95 else ("  ← 理想" if sc >= 0.60 else "  ← 难关")
    print(f"  {name:28} {sc*100:5.1f}%  n={n:4}  CI[{lo*100:.1f},{hi*100:.1f}]{flag}")
PY
