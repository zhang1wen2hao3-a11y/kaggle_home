#!/usr/bin/env bash
# ② 模型诊断筛选：BC 复制体 vs 端口复制体，**同卡组、同对手、同种子、座位交替**，
#    输出每格 ΔW = W_BC − W_port 与配对 95% CI。
#
# 定位：这是**诊断筛选**，不是"多样性 vs 帧数"的因果实验（两者的玩家数/帧数/训练过程都不同）。
#
#   bash tools/bc_screen.sh            # 等当前 eval 结束后自动跑 12 格
#   CELLS="abra76k:self" bash tools/bc_screen.sh
set -uo pipefail
cd "$(dirname "$0")/.."
N="${N:-200}"; SEED="${SEED:-930000}"
DIR=logs/bc_screen
mkdir -p "$DIR"
declare -A POL=(
  [port_abra]=build/agents/c_abra
  [port_grookey]=build/agents/c_grookey
  [bc_abra76k]=build/agents/bc_abra76k
  [bc_grookey3]=build/agents/bc_grookey3
  [ogerpon]=build/agents/port_7e39
  [db04]=build/agents/port_db04
  [shuppet]=build/agents/opp_shuppet_human      # BC 父权重 ⇒ 已达人类水平
  [dreepy]=build/agents/c_dreepy
)

# 卡组一致性：对每对"BC vs 其端口版本"必须同 md5，否则拒绝跑
check_deck() {
  local a=$1 b=$2
  local ma mb
  ma=$(md5sum "${POL[$a]}/deck.csv" | cut -d' ' -f1)
  mb=$(md5sum "${POL[$b]}/deck.csv" | cut -d' ' -f1)
  [ "$ma" = "$mb" ] || { echo "✗ 卡组不一致：$a($ma) vs $b($mb) —— 会把卡组收益误算成策略收益"; exit 1; }
}
check_deck bc_abra76k port_abra
check_deck bc_grookey3 port_grookey
echo "✓ 卡组一致性通过（BC 与端口版本同卡组）"

run() {  # tag cand oppname
  local out="$DIR/$1.json"
  [ -f "$out" ] && { echo "[skip] $1"; return; }
  echo "########## $1 ##########"
  ./tools/py ptcg_league/eval_pair.py --cand "${POL[$2]}" --opp "$3=${POL[$3]}" \
      --matches "$N" --workers 0 --seed-base "$SEED" --out "$out" --logdir "$DIR/$1" 2>&1 | tail -2
}

# 12 格：每个 BC 版本 5 格（同卡组镜像 + 4 个跨卡组）+ 2 个"端口自对弈"基线
CELLS="${CELLS:-}"
if [ -z "$CELLS" ]; then
  for bc in abra76k:port_abra grookey3:port_grookey; do
    b=${bc%%:*}; p=${bc##*:}
    run "bc_${b}__self_${p}"  "bc_$b" "$p"
    run "bc_${b}__vs_ogerpon" "bc_$b" ogerpon
    run "bc_${b}__vs_db04"    "bc_$b" db04
    run "bc_${b}__vs_shuppet" "bc_$b" shuppet
    run "bc_${b}__vs_dreepy"  "bc_$b" dreepy
  done
  run "port_abra__self"    port_abra     port_abra
  run "port_grookey__self" port_grookey  port_grookey
else
  for c in $CELLS; do
    b=${c%%:*}; o=${c##*:}
    case $b in
      abra76k) p=port_abra ;;
      grookey3) p=port_grookey ;;
      *) p=port_abra ;;
    esac
    if [ "$o" = "self" ]; then run "bc_${b}__self_${p}" "bc_$b" "$p"; else run "bc_${b}__vs_$o" "bc_$b" "$o"; fi
  done
fi

echo
echo "===== ΔW = W_BC − W_port（配对，95% CI）；正 = BC 更好 ====="
./tools/py - <<'PY'
import glob, json, os, statistics, math
D="logs/bc_screen"
def load(tag):
    f=os.path.join(D,tag+".json")
    return json.load(open(f))["overall"] if os.path.exists(f) else None
def paired(a,b):
    pm_a=a.get("per_match") or {}; pm_b=b.get("per_match") or {}
    keys=sorted(set(pm_a)&set(pm_b))
    if len(keys)<30: return None
    d=[pm_a[k]-pm_b[k] for k in keys]
    m=statistics.fmean(d); sd=statistics.pstdev(d); hw=1.96*sd/math.sqrt(len(d))
    return m, m-hw, m+hw, len(keys)
rows=[]
for bc,p in (("abra76k","port_abra"),("grookey3","port_grookey")):
    for opp in (None,"ogerpon","db04","shuppet","dreepy"):
        if opp is None:
            tb=f"bc_{bc}__self_{p}"; tpf=f"port_{p}__self"; label=f"{bc} vs {p}(同卡组镜像)"
        else:
            tb=f"bc_{bc}__vs_{opp}"; tpf=f"{p}__vs_{opp}"; label=f"{bc} vs {opp}"
        A=load(tb); B=load(tpf)
        if not A: continue
        if not B:
            # 跨卡组基线来自 matrix2（同种子 930000、同 200 场）
            mfile=f"logs/matrix2/{'abra' if p=='port_abra' else 'grookey'}__vs__{opp}.json"
            B=json.load(open(mfile))["overall"] if os.path.exists(mfile) else None
        if not B: 
            print(f"{label:34s} 缺基线"); continue
        r=paired(A,B)
        if not r:
            print(f"{label:34s} BC={A['score']:.3f} 基线={B['score']:.3f} （配对键不足）"); continue
        m,lo,hi,n=r
        v="BC显著更好" if lo>0 else ("BC显著更差" if hi<0 else "CI 跨 0")
        rows.append((label,A["score"],B["score"],m,lo,hi,v,n))
print("%-34s%8s%8s%9s%20s  %-12s"%("格","BC","端口","ΔW","95% CI","判定"))
for r in rows:
    print("%-34s%8.3f%8.3f%+8.1fpp%20s  %-12s"%(r[0],r[1],r[2],100*r[3],"[%+.1f,%+.1f]pp"%(100*r[4],100*r[5]),r[6]))
PY
