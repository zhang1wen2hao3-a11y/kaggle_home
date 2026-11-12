#!/usr/bin/env bash
# **补齐 docs/the-pockmon-battle-choices-win-rate.csv 缺的格子**（本地 eval，可断点续跑）。
#
#   背景：choice-detail 表只覆盖 12×12 = 144 格里的 90 格（老 eval 围着天梯占比 98% 的
#   7 副牌做，冠军牌 0904/26c6/e1d7/e8ee/db04 基本没测过）。对角（a→a 自己打自己）
#   按约定填 "/"，剩下 51 个非对角格要补：
#     A 类 25 格：反向 b→a 已实测，理论上可 1−x 推断 —— 但反向那一格用的可能是
#                 db04→g6_db04 / 26c6→r1A 这类"专精驾驶员"，推算出来的配对和本表口径
#                 （我方 a 的最强驾驶员 vs 对手端口驾驶员）不一致，所以也重跑。
#     B 类 26 格：两侧零数据（e1d7 / e8ee 整行整列），只能跑。
#
# 协议（全部同一口径，便于横向比较）：
#   200 场 Bo3、座位交替、seed-base 970000、双向都先跑（eval_pair 自带座位均衡）
#   我方 = 该行卡组的最强驾驶员（本脚本覆盖的 8 行都是端口权重，详见 OUR）
#   对手 = 端口权重 + 该列卡组（与 choice/panel 表"只差卡组"的口径一致）
#
# 用法：
#   bash tools/fill_winrate.sh            # 跑所有缺格（约 5 min/格，51 格 ≈ 4.3h）
#   N=100 bash tools/fill_winrate.sh      # 快速版
set -uo pipefail
cd "$(dirname "$0")/.."
N="${N:-200}"; SEED="${SEED:-970000}"; OUT="${OUT:-logs/winrate}"
mkdir -p "$OUT"
PY=./tools/py

# 卡组短名 → agent 目录（12 个都核对过：端口权重 + 对应卡组，见 build/agents/*/deck.csv）
declare -A AG=(
  [0904]=build/agents/port_0904
  [0bf5]=build/agents/opp_dunsparce_0bf5
  [0d86]=build/agents/port_0d86
  [2617]=build/agents/c_grookey
  [26c6]=build/agents/port_26c6
  [3eaa]=build/agents/opp_3eaa
  [7e39]=build/agents/port_7e39
  [9d6e]=build/agents/port_on_9d6e
  [db04]=build/agents/port_db04
  [e1d7]=build/agents/port_e1d7
  [e8ee]=build/agents/port_e8ee
  [f5f8]=build/agents/c_abra
)

# 缺格列表：从 detail 表现算（状态 != 实测，且非对角）
LIST=$($PY - <<'PY'
import csv, io, sys
txt = open("docs/the-pockmon-battle-choice-detail.csv", "rb").read().decode("gb18030", errors="replace")
rows = list(csv.DictReader(io.StringIO(txt)))
out = []
for r in rows:
    a, b = r["挑战者"], r["被挑战者"]
    if a == b or r["状态"] == "实测":
        continue
    out.append(f"{a} {b}")
print("\n".join(out))
PY
)
TOT=$(echo "$LIST" | grep -c . )
echo "待跑 $TOT 格 × $N 场 Bo3（seed $SEED）→ $OUT"
i=0
while read -r a b; do
  [ -n "${a:-}" ] || continue
  f="$OUT/${a}_${b}.json"
  i=$((i+1))
  if [ -f "$f" ]; then echo "[$i/$TOT] skip $a→$b（已有）"; continue; fi
  t0=$(date +%s)
  $PY ptcg_league/eval_pair.py --cand "${AG[$a]}" --opp "enemy=${AG[$b]}" \
      --matches "$N" --workers 0 --seed-base "$SEED" --out "$f" --logdir "$OUT/${a}_${b}" \
      > "$OUT/${a}_${b}.log" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  if [ "$rc" = 0 ] && [ -f "$f" ]; then
    s=$($PY -c "import json;o=json.load(open('$f'))['overall'];print(f\"{o['score']:.4f} n={o['n']}\")")
    echo "[$i/$TOT] ✓ $a→$b  $s  (${dt}s)"
  else
    echo "[$i/$TOT] ✗ $a→$b 失败 rc=$rc（看 $OUT/${a}_${b}.log）"
  fi
done <<< "$LIST"
echo "=== 跑完，剩下用 tools/build_winrate_table.py 生成表 ==="
