#!/usr/bin/env bash
# G4 三支的**冻结评估**：每支对端口（同卡组 7e39）400 场 Bo3，**同一批种子** ⇒ 可做配对差。
#
# 预注册口径（下一步.md §25.4）：
#   Δ_AB = p_B − p_A      Δ_BC = p_C − p_B
#   等效界 ±3pp；统计单位 = Bo3 match；配对差用 per_match 逐场相减。
set -uo pipefail
cd /home/zhang/code/kaggle/the-pokemon-company
PY=./tools/py
N="${N:-400}"
SEED="${SEED:-780000}"
PORT=ptcg_league/pools/init/init_port_plain

for b in A B C; do
  out="/tmp/g4_eval_$b.json"
  [ -f "$out" ] && { echo "[skip] $b 已有结果"; continue; }
  echo "########## G4-$b vs 端口 ##########"
  $PY ptcg_league/eval_pair.py --cand "/tmp/ag_g4$b" --opp port="$PORT" \
    --matches "$N" --workers 6 --seed-base "$SEED" --out "$out" 2>&1 | tail -3
done

echo
echo "===== 汇总 + 预注册判定 ====="
$PY - <<'EOF'
import json, statistics
def load(t):
    d = json.load(open(f"/tmp/g4_eval_{t}.json"))
    o = d["overall"]
    return o, (o.get("per_match") or {})
res = {}
for t in "ABC":
    try:
        o, pm = load(t); res[t] = (o, pm)
        print(f"G4-{t}: {o['score']:.4f}  CI[{o['ci_low']:.4f},{o['ci_high']:.4f}]  n={o['n']}  异常{o['errors']}")
    except FileNotFoundError:
        print(f"G4-{t}: 缺结果")
if len(res) < 2:
    raise SystemExit("至少要有两支才能比")
def paired(t1, t2, label):
    if t1 not in res or t2 not in res: return
    pm1, pm2 = res[t1][1], res[t2][1]
    common = sorted(set(pm1) & set(pm2), key=int)
    if len(common) < 30:
        print(f"{label}: 配对不足（{len(common)}）"); return
    d = [pm2[k] - pm1[k] for k in common]
    m = sum(d) / len(d)
    se = statistics.pstdev(d) / len(d) ** 0.5
    lo, hi = m - 1.96 * se, m + 1.96 * se
    equiv = (lo > -0.03) and (hi < 0.03)
    verdict = ("等效（该旋钮无可测收益）" if equiv else
               "显著 >0（该支更好）" if lo > 0 else
               "显著 <0（该支更差）" if hi < 0 else "不显著且未落在等效界内")
    print(f"{label}: {m:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  "
          f"({sum(1 for x in d if x>0)}胜/{sum(1 for x in d if x<0)}负/"
          f"{sum(1 for x in d if x==0)}平)  ⇒ {verdict}")
paired("A", "B", "Δ_AB = p_B − p_A（加冻结 bank 的效果）")
paired("B", "C", "Δ_BC = p_C − p_B（改训练卡组分布的效果，**主假设**）")
paired("A", "C", "Δ_AC = p_C − p_A（两个变化合起来，不可单独归因）")
EOF
