#!/usr/bin/env bash
# BC v1 vs BC v2 vs 端口，**同一副 shuppet 卡组**、同一种子做配对 A/B。
# v1 = bc:deck_0d86（全部帧 171k / 6 epoch，val 73.1%）
# v2 = deck_0d86_w（只用赢家帧 60k / 10 epoch，val 72.6%）
set -uo pipefail
cd /home/zhang/code/kaggle/the-pokemon-company
DECK=decks/pool_field/shuppet_dhelmise__0d86.csv
PY=./tools/py
N=200
SEED=800000

echo "===== 准备：v2 与端口都用 shuppet 卡组 stage ====="
$PY tools/stage_ckpt.py bc_pool/deck_0d86_w.pt --out /tmp/ag_bc_v2 \
    --bundle build/port_plain --deck "$DECK" --force | tail -2
$PY tools/stage_ckpt.py build/port_plain/model.pt --out /tmp/ag_port_shup \
    --bundle build/port_plain --deck "$DECK" --force | tail -2

echo
echo "===== A/B 1：BC v2  vs  端口（同为 shuppet） ====="
$PY ptcg_league/eval_pair.py --cand /tmp/ag_bc_v2 --opp port=/tmp/ag_port_shup \
    --matches $N --workers 5 --seed-base $SEED --out /tmp/ab_v2_port.json | tail -4

echo
echo "===== A/B 2：BC v1  vs  端口（同为 shuppet，同种子） ====="
$PY ptcg_league/eval_pair.py --cand ptcg_league/pools/init/bc_deck_0d86 --opp port=/tmp/ag_port_shup \
    --matches $N --workers 5 --seed-base $SEED --out /tmp/ab_v1_port.json | tail -4

echo
echo "===== A/B 3：BC v2  vs  BC v1（直接对打，同种子） ====="
$PY ptcg_league/eval_pair.py --cand /tmp/ag_bc_v2 --opp v1=ptcg_league/pools/init/bc_deck_0d86 \
    --matches $N --workers 5 --seed-base $SEED --out /tmp/ab_v2_v1.json | tail -4

echo
echo "===== 汇总（同种子配对） ====="
$PY - <<'EOF'
import json
def load(p):
    d = json.load(open(p))
    return d["overall"], (d["overall"].get("per_match") or {})
rows = []
for tag, p in (("v2 vs 端口", "/tmp/ab_v2_port.json"),
               ("v1 vs 端口", "/tmp/ab_v1_port.json"),
               ("v2 vs v1", "/tmp/ab_v2_v1.json")):
    o, _ = load(p)
    rows.append((tag, o))
    print(f"{tag:<12} {o['score']:.4f}  CI[{o['ci_low']:.4f},{o['ci_high']:.4f}]  n={o['n']}  异常{o['errors']}")
# 配对：v2 与 v1 对同一个端口、同一批种子 ⇒ 逐场做差
o2, pm2 = load("/tmp/ab_v2_port.json")
o1, pm1 = load("/tmp/ab_v1_port.json")
common = sorted(set(pm2) & set(pm1), key=int)
if len(common) >= 30:
    d = [pm2[k] - pm1[k] for k in common]
    m = sum(d) / len(d)
    import statistics
    se = (statistics.pstdev(d) / len(d) ** 0.5) if len(d) > 1 else 0.0
    print(f"\n配对差（v2 − v1，对端口同种子 n={len(common)}）：{m:+.4f} "
          f"CI[{m-1.96*se:+.4f},{m+1.96*se:+.4f}]  ({sum(1 for x in d if x>0)}胜/"
          f"{sum(1 for x in d if x<0)}负/{sum(1 for x in d if x==0)}平)")
EOF
