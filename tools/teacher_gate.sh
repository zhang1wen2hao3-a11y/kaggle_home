#!/usr/bin/env bash
# **门控晋升 h2h**：候选 vs 冻结 teacher，同卡组镜像，返回胜率（供 teacher_loop 判断是否晋升）。
#
#   bash tools/teacher_gate.sh --cand a.bin --teacher t.bin --deck <deck.csv> --n 400 --out /tmp/gate
#
# 为什么是镜像：A 永远只打要提交的那一副牌，teacher 是 A 的旧快照，所以晋升标准 =
# "我现在打这副牌，稳赢我过去的自己"（参考工程 generalist 阶段的 promote-winrate=0.53）。
#
# 注意：镜像 h2h 基线就是 ~0.5（双方同卡组同权重会是平局），所以阈值必须 ≥0.53 才有意义，
# 而且 n 太小会误晋升（n=400 的 CI ≈ ±4.9pp）。建议 n≥400、阈值 0.53~0.55。
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
CAND=""; TEACHER=""; DECK=""; N="${N:-400}"; OUT="/tmp/teacher_gate"; SEED="${SEED:-960000}"
while [ $# -gt 0 ]; do
  case "$1" in
    --cand) CAND="$2"; shift 2;; --teacher) TEACHER="$2"; shift 2;;
    --deck) DECK="$2"; shift 2;; --n) N="$2"; shift 2;;
    --out) OUT="$2"; shift 2;; --seed) SEED="$2"; shift 2;; *) shift;;
  esac
done
[ -n "$CAND" ] && [ -n "$TEACHER" ] || { echo "✗ 需要 --cand 和 --teacher"; exit 2; }
[ -f "$DECK" ] || { echo "✗ 需要 --deck <deck.csv 路径>"; exit 2; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# bin → pt（如果输入是 .bin）
to_pt() {
  local src=$1 dst=$2
  case "$src" in
    *.bin)
      PYTHONPATH=build/port_plain:.:$PYTHONPATH ./tools/py port_repo/native/tools/native_weights.py \
        import "$src" "$dst" >/dev/null 2>&1 || { echo "✗ bin→pt 失败 $src"; exit 1; } ;;
    *) cp -f "$src" "$dst" ;;
  esac
}
# 用现成 agent 目录做模板，只换 model.pt 和 deck.csv
mk_agent() {
  local name=$1 pt=$2 deckcsv=$3
  local d="$TMP/$name"
  cp -r build/agents/port_7e39 "$d"
  rm -f "$d/model.pt"; cp -f "$pt" "$d/model.pt"
  cp -f "$deckcsv" "$d/deck.csv"
}

to_pt "$CAND" "$TMP/cand.pt"
to_pt "$TEACHER" "$TMP/teacher.pt"
mk_agent cand "$TMP/cand.pt" "$DECK"
mk_agent teacher "$TMP/teacher.pt" "$DECK"
mkdir -p "$OUT"
./tools/py ptcg_league/eval_pair.py --cand "$TMP/cand" --opp teacher="$TMP/teacher" \
  --matches "$N" --workers 0 --seed-base "$SEED" --out "$OUT/gate.json" --logdir "$OUT/log" 2>&1 | tail -2
./tools/py - "$OUT/gate.json" <<'PY'
import json, sys
o = json.load(open(sys.argv[1]))["overall"]
print(f"候选 vs 冻结 teacher（镜像）: {o['score']:.4f}  CI[{o['ci_low']:.3f},{o['ci_high']:.3f}]  n={o['n']}")
PY
