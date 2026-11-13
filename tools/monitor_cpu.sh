#!/usr/bin/env bash
# 轻量监控：每 5 分钟记一行 CPU 机状态（**不关机**）。
#   bash tools/monitor_cpu.sh root@HOST [端口]
set -uo pipefail
cd "$(dirname "$0")/.."
HOST="${1:-root@113.44.139.218}"; PORT="${2:-22}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=12 -o BatchMode=yes -p $PORT $HOST"
LOG=logs/cpu_monitor.log; mkdir -p logs
for i in $(seq 1 576); do   # 最多 48 小时
  if $SSH true 2>/dev/null; then
    st=$($SSH 'cd /root/eval_cpu_kit 2>/dev/null || exit 0
      q=$(pgrep -f "queue_ceilin[g]" >/dev/null && echo ceil || true)
      l=$(pgrep -f "queue_lon[g]" >/dev/null && echo long || true)
      t=$(pgrep -f "run_in_tabl[e]" >/dev/null && echo intable || true)
      n=$(ls logs/meta/*_s9500*/vs_*.json 2>/dev/null | wc -l)
      done_=$(ls -d logs/meta/*_s9500* 2>/dev/null | while read d; do [ -f "$d/summary.json" ] && echo x; done | wc -l)
      load=$(cut -d" " -f1 /proc/loadavg)
      echo "queues=[$q$l$t] 面板格数=$n 完成面板=$done_ load=$load"' 2>/dev/null)
    echo "[$(date +%m-%d\ %H:%M)] $st" >> "$LOG"
  else
    echo "[$(date +%m-%d\ %H:%M)] ✗ 机器不可达" >> "$LOG"
  fi
  sleep 300
done
