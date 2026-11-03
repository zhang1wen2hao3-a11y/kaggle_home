#!/usr/bin/env bash
# 本地跑**完整**对手池强度阶梯：池里所有"有卡组"的成员 × 每成员 N 场，对学习者(端口权重@7e39)。
# 直接用仓库里的池目录，不额外占磁盘（不需要 158 MB 的 pool_ladder_kit）。
#
#   bash tools/run_ladder_local.sh                 # 默认 400 场/成员、6 面板
#   N=200 WORKERS=4 bash tools/run_ladder_local.sh
#   MEMBERS="opp_g0001 opp_g5_db04" bash tools/run_ladder_local.sh   # 只补某几个
set -uo pipefail
cd "$(dirname "$0")/.."
N="${N:-400}"; W="${WORKERS:-6}"; SEED="${SEED:-760000}"
LEARNER="build/agents/port_7e39"
P=ptcg_league/pools/league
# 全覆盖：8 个 kit 成员 + G8（shuppet 牌的强驾驶者）
MEMBERS="${MEMBERS:-opp_db04pilot opp_g0001 opp_g0003 opp_g4a opp_g4b opp_g4c \
sp_g0002_7e39_port_plain opp_g5_db04 opp_g8_shuppet}"
mkdir -p logs/ladder_local
[ -d "$LEARNER" ] || { echo "✗ 缺学习者目录 $LEARNER"; exit 1; }
echo "学习者=$LEARNER  每成员 $N 场  并行 $W  种子 $SEED"
echo
for m in $MEMBERS; do
  d="$P/$m"; [ -d "$d" ] || { echo "✗ 缺成员目录 $d"; continue; }
  out="logs/ladder_local/$m.json"
  if [ -f "$out" ]; then
    # 只认"跑满且异常 0"的；半截的重跑
    ok=$(./tools/py - "$out" "$N" <<'PYEOF' 2>/dev/null || echo 0
import json, sys
o=json.load(open(sys.argv[1]))["overall"]
print(1 if o["n"] >= int(sys.argv[2])*0.98 and o["errors"]==0 else 0)
PYEOF
)
    [ "$ok" = 1 ] && { echo "[skip] $m（已完成 n=$N 异常 0）"; continue; }
    echo "[重跑] $m（上次没跑满或异常>0）"; rm -f "$out"
  fi
  echo "########## $m ##########"
  ./tools/py ptcg_league/eval_pair.py --cand "$d" --opp "learner=$LEARNER" \
      --matches "$N" --workers "$W" --seed-base "$SEED" \
      --out "$out" --logdir "logs/ladder_local/$m" 2>&1 | tail -3
done
echo
echo "===== 本地阶梯汇总（p_i = 成员打赢学习者；档位按学习者胜率 1-p_i 判）====="
./tools/py - <<'PY'
import glob, json, os
MAP={"opp_db04pilot":"opp:db04pilot","opp_g0001":"opp:g0001","opp_g0003":"opp:g0003",
     "opp_g4a":"opp:g4a","opp_g4b":"opp:g4b","opp_g4c":"opp:g4c",
     "sp_g0002_7e39_port_plain":"sp:g0002_7e39_port_plain","opp_g5_db04":"opp:g5_db04",
     "opp_g8_shuppet":"opp:g8_shuppet"}
rows=[]
for f in sorted(glob.glob("logs/ladder_local/*.json")):
    k=os.path.basename(f)[:-5]; o=json.load(open(f))["overall"]
    rows.append((MAP.get(k,k), o))
rows.sort(key=lambda kv: kv[1]["score"])
print("%-30s%7s%18s%6s%5s%10s  %s" % ("成员","p_i","95% CI","场次","异常","学习者胜率","档位"))
for aid,o in rows:
    pl=1-o["score"]
    role = "A 强" if pl<0.45 else ("B 中" if pl<0.60 else ("C 偏弱" if pl<0.75 else "D 弱/特殊"))
    print("%-30s%7.3f%18s%6d%5d%10.3f  %s" % (aid,o["score"],
          "[%.3f,%.3f]"%(o["ci_low"],o["ci_high"]),o["n"],o["errors"],pl,role))
json.dump({a:dict(p_i=o["score"],p_learner=round(1-o["score"],4),
                  ci=[o["ci_low"],o["ci_high"]],n=o["n"],errors=o["errors"]) for a,o in rows},
          open("logs/ladder_local/p_i.json","w"), ensure_ascii=False, indent=1)
print("\n→ logs/ladder_local/p_i.json（9 个成员，可直接 pool_set_pi 落库）")
PY
