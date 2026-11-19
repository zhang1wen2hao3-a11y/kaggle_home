#!/usr/bin/env bash
# **等一串派发跑完 → 拉结果 → 校验 → 关机**（评估机不在时也不会白跑）
#   bash tools/watch_serial_and_stop.sh <host> <dispatch日志> <最少新结果数> [最长等待分钟] [结束标记]
#   结束标记默认 SERIAL_DONE；跑 tools/run_seeds_new_decks.sh 时传 SEEDS_DONE
# 校验规则：日志里出现 SERIAL_DONE；rsync 成功；本次新增的 vs_*.json 数 ≥ 期望；否则**不关机**（留证据）。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
H="${1:?host}"; LOG="${2:?日志}"; NEED="${3:?最少新结果}"; MAXMIN="${4:-240}"; DONE_TAG="${5:-SERIAL_DONE}"
TAG="logs/watch_serial_$(echo "$H" | tr -d '@.')"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$TAG.log"; }

before=$(find logs/meta -name 'vs_*.json' 2>/dev/null | wc -l)
for i in $(seq 1 $((MAXMIN*2))); do
  if grep -q "$DONE_TAG" "$LOG" 2>/dev/null; then say "派发结束（第 $i 次探）"; break; fi
  sleep 30
done
grep -q "$DONE_TAG" "$LOG" 2>/dev/null || { say "⚠️ 等超时，不关机"; exit 1; }

for try in 1 2 3; do
  rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --include '*/' --include 'vs_*.json' --include 'summary.json' \
    --exclude '*' "$H:/root/eval_cpu_kit/logs/meta/" logs/meta/ && break
  say "  rsync 第 $try 次失败，重试"; sleep 10
done
after=$(find logs/meta -name 'vs_*.json' 2>/dev/null | wc -l)
new=$((after-before))
say "本次新增结果 $new 个（期望 ≥$NEED）"
if [ "$new" -lt "$NEED" ]; then say "✗ 结果数不够 ⇒ 保持开机"; exit 2; fi

say "收尾：关 $H"
timeout 30 ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$H" 'sync; nohup shutdown -h now > /dev/null 2>&1 & sleep 2; echo bye' 2>&1 | tail -1 | sed 's/^/  /' | tee -a "$TAG.log"
say DONE
