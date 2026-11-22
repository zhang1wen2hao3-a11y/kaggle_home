#!/usr/bin/env bash
# **给一台机器的面板队列追加候选**（等上一批的 done 标记出现后再跑，跑完写自己的标记）
#   bash tools/panel_append.sh <host> "<cand1 cand2>" <等这个标记> <写这个标记> [TESTSET] [TAG] [WORKERS]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
H="${1:?host}"; CANDS="${2:?候选空格分隔}"; WAITF="${3:?等这个标记}"; MK="${4:?写这个标记}"
TS="${5:-decks/testset_meta_real_v2.tsv}"; TAG="${6:-v2}"; W="${7:-24}"
echo "[$(date +%H:%M:%S)] 追加队列 $H: $CANDS（等 $WAITF，最多 240 分钟）"
for i in $(seq 1 480); do [ -f "$WAITF" ] && { echo "[$(date +%H:%M:%S)] ✓ 标记出现，开始"; break; }; sleep 30; done
for c in $CANDS; do
  bash tools/panel_on_host.sh "$c" "$H" "$TS" "$TAG" "$W"
done
echo "APPEND_DONE $H" > "$MK"
echo "[$(date +%H:%M:%S)] 全部完成 → $MK"
