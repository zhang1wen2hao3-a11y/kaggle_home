#!/usr/bin/env bash
# 串行评估 slowking 专才：端口基线 → r1A。跑完自动出对比表。
set -uo pipefail
cd "$(dirname "$0")/.."
N="${N:-200}"; SEED="${SEED:-950000}"
for spec in "port_26c6|meta_port_26c6|端口@slowking" "r1A_26c6|meta_r1A|10M步专才@slowking"; do
  IFS='|' read -r dir tag label <<< "$spec"
  echo "===== $label ====="
  N="$N" SEED="$SEED" bash tools/meta_eval.sh --cand "build/agents/$dir" --tag "$tag"
done
echo
echo "===== 对比 ====="
./tools/py - <<'PY'
import json, glob, os
w = json.load(open("decks/field_real.json"))
arms = {}
for tag in ["meta_port_26c6","meta_r1A"]:
    d = {}
    for f in glob.glob(f"logs/meta/{tag}/vs_*.json"):
        o = json.load(open(f))["overall"]; d[f.split("vs_")[-1][:-5]] = o["score"]
    arms[tag] = d
cells = sorted(arms.get("meta_port_26c6", arms.get("meta_r1A", {})))
print(f"{'格':<16}" + "".join(f"{k.replace('meta_',''):>16}" for k in ["meta_port_26c6","meta_r1A"]) + f"{'Δ':>8}")
for c in cells:
    a = arms.get("meta_port_26c6", {}).get(c); b = arms.get("meta_r1A", {}).get(c)
    print(f"{c:<16}" + (f"{a:>16.3f}" if a is not None else f"{'—':>16}") + (f"{b:>16.3f}" if b is not None else f"{'—':>16}") + (f"{(b-a)*100:>+8.1f}" if a is not None and b is not None else f"{'':>8}"))
def wt(d):
    s = sum(w[c]*v for c,v in d.items() if c in w); n = sum(w[c] for c in d if c in w)
    return s/n if n else 0
print()
for k in ["meta_port_26c6","meta_r1A"]:
    if k in arms: print(f"{k:<20} 加权 = {wt(arms[k]):.4f}")
PY
