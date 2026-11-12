#!/usr/bin/env bash
# **把评估结果拉回本地**（面板 / 头对头 / 高样本），用于 tools/panel_compare.py 与 panel_ci.py
#   bash tools/pull_eval_results.sh root@1.2.3.4 [端口]
set -uo pipefail
cd "$(dirname "$0")/.."
HOST="${1:?用法: bash tools/pull_eval_results.sh root@HOST [端口]}"
PORT="${2:-22}"
R="${R:-/root/eval_cpu_kit}"
SRC="$HOST:$R/logs/"
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
  --include='meta/***' --include='h2h/***' --include='hiN_*' --exclude='*' "$SRC" logs/
echo "✓ 已拉回："
ls -d logs/meta/*_9d6e 2>/dev/null | sed 's/^/   /'
ls logs/h2h/*.json 2>/dev/null | sed 's/^/   /'
ls logs/hiN_*.json 2>/dev/null | sed 's/^/   /'
