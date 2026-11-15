#!/usr/bin/env bash
# **路由版提交包**：一个包里放 N 个模型 + 原型族路由表。
#
#   bash tools/make_sub_router.sh <包名> <泛用agent目录> [<原型族>=<专家agent目录> ...]
#   例：bash tools/make_sub_router.sh cand7e39Mix cand7e39S slowking=cand26c6S shuppet=candf5f8S
#
# 规则（用户定的并入阈值，脚本只负责打包，不替你决定要不要加人）：
#   单格增益 ≥5pp 且 Σ覆盖率×胜率增益 超阈值 才值得加专家 —— 见 docs/驾驶员与覆盖规则.md §8
#
# 包结构：main.py(路由版) + route.json + families.json + models/model_*.pt + 单模型包的其余部件
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:?用法: make_sub_router.sh <包名> <泛用agent> [族=专家agent ...]}"; shift
GEN="${1:?缺泛用 agent}"; shift
SPECS=("$@")
DATE=$(date +%H%M)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

[ -f "build/agents/$GEN/model.pt" ] || { echo "✗ 缺 build/agents/$GEN/model.pt"; exit 1; }
# ① 用泛用 agent 的骨架（ptcg/ cg/ data/ deck.csv main.py ...）
cp -a "build/agents/$GEN/." "$TMP/"
rm -f "$TMP/main.py"
cp -f tools/router_main.py "$TMP/main.py"
mkdir -p "$TMP/models"
cp -f "build/agents/$GEN/model.pt" "$TMP/models/model_0.pt"
MODELS='["models/model_0.pt"'
ROUTES="{}"
i=0
for s in "${SPECS[@]}"; do
  fam="${s%%=*}"; ag="${s##*=}"
  [ -f "build/agents/$ag/model.pt" ] || { echo "✗ 缺 build/agents/$ag/model.pt"; exit 1; }
  i=$((i+1))
  cp -f "build/agents/$ag/model.pt" "$TMP/models/model_$i.pt"
  MODELS="$MODELS,\"models/model_$i.pt\""
  ROUTES=$(./tools/py - "$ROUTES" "$fam" "$i" <<'PY'
import json,sys
r=json.loads(sys.argv[1]); r[sys.argv[2]]=int(sys.argv[3]); print(json.dumps(r))
PY
)
  echo "   + 专家 $fam → $ag（model_$i.pt）"
done
MODELS="$MODELS]"
cat > "$TMP/route.json" <<JSON
{"general": 0, "models": $MODELS, "routes": $ROUTES, "max_switches": 2,
 "built": "$DATE", "general_agent": "$GEN", "specs": $(printf '%s\n' "${SPECS[@]}" | ./tools/py -c "import sys,json;print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))")}
JSON
# ② 原型族表（离线算好，运行时不再 glob decks/）
./tools/py - "$TMP/families.json" <<'PY'
import json, sys
sys.path.insert(0, "tools")
from router_core import load_families
fams = load_families()
out = {fam: [[d["hash"], sorted(d["cards"])] for d in decks] for fam, decks in fams.items()}
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False)
print(f"   原型族 {len(out)} 个 / 卡组 {sum(len(v) for v in out.values())} 副")
PY
# ③ 打包（名字长度限制同 make_sub.sh：dist/ptcg_sub_*.tar.gz ≤30 字符）
# ★ Kaggle 卡的是**文件名**长度（不含 dist/）：ptcg_sub_<NAME>.tar.gz ≤ 23 字符 ⇒ NAME ≤ 9
BASE="ptcg_sub_${NAME}.tar.gz"
if [ ${#BASE} -gt 23 ]; then
  NAME=$(echo "$NAME" | cut -c1-9)
  BASE="ptcg_sub_${NAME}.tar.gz"
  echo "⚠️ 包名过长，截断为 $NAME"
fi
[ ${#BASE} -le 23 ] || { echo "✗ 包名太长（${#BASE}）：$BASE"; exit 1; }
OUT="dist/${BASE}"
[ -f "$OUT" ] && [ "${FORCE:-0}" != "1" ] && { echo "✗ $OUT 已存在（FORCE=1 覆盖）"; exit 1; }
tar czhf "$OUT" -C "$TMP" .
n=$((i+1))
printf "✅ %s  %.1f MB  sha16 %s  模型 %d 个（预估常驻 ~%.0f MiB）\n" \
  "$OUT" "$(stat -c%s "$OUT" | awk '{print $1/1048576}')" \
  "$(sha256sum "$OUT" | cut -c1-16)" "$n" "$(echo "$n*17.1" | bc)"
echo "   上传： kaggle competitions submit -c <slug> -f $OUT -m \"$NAME\""
