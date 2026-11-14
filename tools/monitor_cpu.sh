#!/usr/bin/env bash
# 轻量监控：每 5 分钟记一行 CPU 机状态（**不关机**）。
#   bash tools/monitor_cpu.sh root@HOST [端口]
set -uo pipefail
cd "$(dirname "$0")/.."
HOSTS="${1:-root@113.44.139.218}"; PORT="${2:-22}"
# 允许一次传多台（空格分隔）：bash tools/monitor_cpu.sh "root@a root@b"
HOSTS_ARR=($HOSTS)
HOST="${HOSTS_ARR[0]}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=12 -o BatchMode=yes -p $PORT $HOST"
TAG="$(echo "$HOSTS" | tr "@ ." "___")"
LOG="logs/cpu_monitor_${TAG}.log"; mkdir -p logs
for i in $(seq 1 576); do   # 最多 48 小时
  for HOST in "${HOSTS_ARR[@]}"; do
  SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=12 -o BatchMode=yes -p $PORT $HOST"
  if $SSH true 2>/dev/null; then
    st=$($SSH 'cd /root/eval_cpu_kit 2>/dev/null || exit 0
      q=""
      pgrep -f "queue_v[0-9]"    >/dev/null && q="${q}v2,"
      pgrep -f "queue_ceilin[g]" >/dev/null && q="${q}ceil,"
      pgrep -f "queue_lon[g]"    >/dev/null && q="${q}long,"
      pgrep -f "run_in_tabl[e]"  >/dev/null && q="${q}intable,"
      pgrep -f "run_cpu_al[l]"   >/dev/null && q="${q}runall,"
      t=""
      n=$(ls logs/meta/*_s9500*/vs_*.json 2>/dev/null | wc -l)
      done_=$(ls -d logs/meta/*_s9500* 2>/dev/null | while read d; do [ -f "$d/summary.json" ] && echo x; done | wc -l)
      load=$(cut -d" " -f1 /proc/loadavg)
      echo "[$HOST] queues=[$q] 面板格数=$n 完成面板=$done_ load=$load"' 2>/dev/null)
    echo "[$(date +%m-%d\ %H:%M)] $st" >> "$LOG"
    # ★ 每轮都增量拉回结果：机器随时可能被释放，**不能只记状态不拉数据**
    rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
      --include='meta/***' --include='in_table/***' --include='h2h/***' --include='vs9d6e/***' \
      --include='hiN_*' --exclude='*' "$HOST:/root/eval_cpu_kit/logs/" logs/ 2>/dev/null \
      && echo "[$(date +%m-%d\ %H:%M)]   [$HOST] 已增量拉回结果" >> "$LOG"
  else
    echo "[$(date +%m-%d\ %H:%M)] [$HOST] ✗ 机器不可达" >> "$LOG"
  fi
  done
  sleep 300
done
