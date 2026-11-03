#!/usr/bin/env bash
# **新 Specialist 评估**（按 2026-11-03 定的路线）：
#   specialist（新卡组）分别对：G6@db04 / port@db04 / dreepy_drakloak / 真实 meta 其它原型
#
#   bash tools/eval_specialist.sh --cand build/agents/bc_pc --tag bc_pc
#   固定协议：种子 930000、400 场/格、座位交替、Bo3 ⇒ 格子之间可配对
set -uo pipefail
cd "$(dirname "$0")/.."
CAND=""; TAG=""
while [ $# -gt 0 ]; do
  case "$1" in --cand) CAND="$2"; shift 2;; --tag) TAG="$2"; shift 2;; *) shift;; esac
done
[ -n "$CAND" ] || { echo "✗ 缺 --cand"; exit 2; }
TAG="${TAG:-$(basename "$CAND")}"
N="${N:-400}"; SEED="${SEED:-930000}"
OUT="logs/specialist_eval/$TAG"
mkdir -p "$OUT"
declare -A OPP=(
  [g6_db04]=build/agents/g12_db04_dreepy          # 用同卡组的最新 db04 agent 作 G6 系代表
  [port_db04]=build/agents/port_db04
  [dreepy_3eaa]=build/agents/c_dreepy
  [ogerpon_7e39]=build/agents/port_7e39
  [abra_f5f8]=build/agents/c_abra
  [hydrapple]=build/agents/opp_hydrapple
  [thwackey_dipplin]=build/agents/opp_thwackey_dipplin
  [alakazam_dd]=build/agents/opp_alakazam_dd
  [dragapult_ch]=build/agents/opp_dragapult_ch
)
for k in g6_db04 port_db04 dreepy_3eaa ogerpon_7e39 abra_f5f8 hydrapple thwackey_dipplin alakazam_dd dragapult_ch; do
  f="$OUT/vs_$k.json"; [ -f "$f" ] && { echo "[skip] $k"; continue; }
  echo "########## $TAG vs $k ##########"
  ./tools/py ptcg_league/eval_pair.py --cand "$CAND" --opp "$k=${OPP[$k]}" \
      --matches "$N" --workers 0 --seed-base "$SEED" --out "$f" --logdir "$OUT/$k" 2>&1 | tail -2
done
echo
echo "===== $TAG 对位汇总 ====="
./tools/py - "$OUT" <<'PY'
import glob,json,os,sys,math
d=sys.argv[1]
def wil(k,n,z=1.96):
    if n==0: return (0,0,0)
    p=k/n; dd=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return p,max(0,(c-h)/dd),min(1,(c+h)/dd)
print(f"{'对手':22s}{'胜率':>7}{'95% CI':>18}{'n':>5}")
for f in sorted(glob.glob(d+"/vs_*.json")):
    o=json.load(open(f))["overall"]
    print(f"{os.path.basename(f)[3:-5]:22s}{o['score']:>7.3f}{'[%.3f,%.3f]'%(o['ci_low'],o['ci_high']):>18}{o['n']:>5}")
PY
