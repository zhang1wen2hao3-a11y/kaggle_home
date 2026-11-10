#!/usr/bin/env bash
# 把 build/agents/<名字> 打成 Kaggle 提交用的 tar.gz（-h 解引用软链）。
#   第 3 个参数可选：额外的 deck.csv（默认用 agent 目录里那份）
#   bash tools/make_sub.sh g6_db04 [decks/pool_field/xxx.csv]
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:?用法: tools/make_sub.sh <agent目录名> [deck.csv]}"
DECK="${2:-}"
SRC="build/agents/$NAME"
[ -d "$SRC" ] || { echo "✗ 没有 $SRC"; exit 1; }
[ -f "$SRC/model.pt" ] || { echo "✗ $SRC/model.pt 缺失"; exit 1; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
# 用临时副本，避免污染 agent 目录
cp -a "$SRC/." "$TMP/" 2>/dev/null || true
[ -n "$DECK" ] && cp -f "$DECK" "$TMP/deck.csv"
# ★ Kaggle 对提交文件名长度有限制：ptcg_sub_g8_bc_parent.tar.gz（27 字符）会报
#   "Failed to run Validation Episode"，改成 ptcg_sub_g8_bc.tar.gz（21）就正常。
#   实测 ≤23 字符可用、≥27 字符失败 ⇒ 统一压到 ≤23。
OUT="dist/ptcg_sub_${NAME}.tar.gz"
if [ ${#OUT} -gt 30 ]; then
  SHORT=$(echo "$NAME" | cut -c1-$((30-14)))
  OUT="dist/ptcg_sub_${SHORT}.tar.gz"
  echo "⚠️ 文件名过长（${#OUT}+），已截断为 $(basename "$OUT")"
fi
if [ ${#OUT} -gt 30 ]; then echo "✗ 仍过长：$OUT"; exit 1; fi
# ⚠️ 不覆盖已存在的提交件：dist/ptcg_sub_*.tar.gz 里可能是在榜上打过分的版本，
#    覆盖后 sha 变化、无法追溯（2026-11-05 我自己踩到一次）。
if [ -f "$OUT" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "✗ $OUT 已存在。"
  echo "  若确认要重打（内容等价但 sha 会变），用： FORCE=1 bash tools/make_sub.sh $NAME"
  echo "  建议改为不同名字：NAME 加后缀，例如 ${NAME}_v2"
  exit 1
fi
tar czhf "$OUT" -C "$TMP" .
printf "✅ %s  %.1f MB  sha16 %s  deck=%s 张\n" \
  "$OUT" "$(stat -c%s "$OUT" | awk '{print $1/1048576}')" \
  "$(sha256sum "$OUT" | cut -c1-16)" "$(wc -l < "$TMP/deck.csv")"
echo "   上传： kaggle competitions submit -c <slug> -f $OUT -m \"$NAME\""
