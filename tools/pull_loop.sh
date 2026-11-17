#!/usr/bin/env bash
# **边跑边拉**：队列在跑的时候每 N 秒把结果 rsync 回来——机器随时会被释放，
# 拉晚了结果就没了（2026-11-18 就这样丢掉 33 个格子的结果）。
#   bash tools/pull_loop.sh root@A [root@B ...]        # 前台循环，按 Ctrl-C 结束
#   INTERVAL=120 bash tools/pull_loop.sh root@A        # 自定义间隔（默认 120s）
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
HOSTS=("$@"); [ ${#HOSTS[@]} -eq 0 ] && { echo "用法: bash tools/pull_loop.sh root@HOST [...]"; exit 2; }
INTERVAL="${INTERVAL:-120}"
for round in $(seq 1 240); do
  got=0
  for h in "${HOSTS[@]}"; do
    out=$(timeout 90 rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=8" \
      --include '*/' --include 'vs_*.json' --exclude '*' \
      "$h:/root/eval_cpu_kit/logs/meta/" logs/meta/ 2>&1 | wc -l)
    [ "${out:-0}" -gt 0 ] && got=$((got+out))
  done
  echo "[$(date +%H:%M:%S)] 拉取一轮（本地格数 $(ls logs/meta/*/vs_*.json 2>/dev/null | wc -l)）"
  sleep "$INTERVAL"
done
