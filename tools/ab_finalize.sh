#!/usr/bin/env bash
# **等 A/B 长跑结束 → 拉回结果 → 关机**（GPU 用完自动收尾）
#
#   bash tools/ab_finalize.sh            # 前台等（会阻塞很久）
#   nohup bash tools/ab_finalize.sh &    # 后台等
#
# 流程：轮询 teacher_loop 进程 → 消失即认为跑完 → rsync 拉回 → md5 校验 → shutdown
# 安全阀：拉回校验不过 **不关机**（留着可以重试）；日志里能看到每步。
set -uo pipefail
cd "$(dirname "$0")/.."
HOST="${HOST:-root@connect.nmb1.seetacloud.com}"; PORT="${PORT:-23268}"
R="${PTCG_REPO:-/root/ptcg_train_kit/port_repo}"
OUT="${AB_OUT:-/root/autodl-tmp/ab_run1}"
LOCAL="${LOCAL:-data/ab_run1}"
SSH="ssh -o StrictHostKeyChecking=no -p $PORT $HOST"
RS="rsync -az --info=progress2 -e \"ssh -o StrictHostKeyChecking=no -p $PORT\""
LOG="$LOCAL/finalize.log"
mkdir -p "$LOCAL"

say() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }

say "=== 等待 A/B 跑完（每 3 分钟查一次）==="
while true; do
  n=$($SSH "ps -eo args | grep -c '[t]eacher_loop.sh slowpoke'" 2>/dev/null || echo 1)
  if [ "${n:-1}" -eq 0 ]; then say "teacher_loop 已退出"; break; fi
  say "还在跑（teacher_loop 进程数=$n）"
  sleep 180
done

say "=== 拉权重（阶段产物 + blend）==="
$SSH "ls -la $OUT/*.bin 2>/dev/null" | tee -a "$LOG"
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" "$HOST:$OUT/"*.bin "$LOCAL/" 2>&1 | tail -3

say "=== 拉日志（run/watch/训练器全量）==="
$SSH "mkdir -p $OUT/logs" 2>/dev/null
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" "$HOST:$OUT/logs/" "$LOCAL/logs/" 2>&1 | tail -2
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" "$HOST:$OUT/run.log" "$LOCAL/" 2>&1 | tail -2
mkdir -p "$LOCAL/train_logs"
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" "$HOST:$R/logs/r*.log" "$LOCAL/train_logs/" 2>&1 | tail -2

say "=== 拉权重快照（experiments/native-r*，只取 .bin 权重，跳过 state_*）==="
for d in $($SSH "ls $R/experiments/ | grep '^native-r[0-9]' | grep -vE 'r1A2|r1A3|r1A4|r1A_db04|r1S1'" 2>/dev/null); do
  say "  $d"
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
    --include='*/' --include='*.bin' --exclude='state_*' --exclude='*' \
    "$HOST:$R/experiments/$d/" "$LOCAL/snapshots/$d/" 2>&1 | tail -1
done

say "=== 校验 ==="
remote_md5=$($SSH "md5sum $OUT/*.bin 2>/dev/null | sort" | awk '{print $1}' | sort)
local_md5=$(md5sum "$LOCAL"/*.bin 2>/dev/null | awk '{print $1}' | sort)
if [ "$remote_md5" = "$local_md5" ]; then
  say "✓ 权重 md5 全部一致（$(echo "$local_md5" | wc -l) 个文件）"
  ok=1
else
  say "✗ md5 不一致！远端 $(echo "$remote_md5"|wc -l) 个 / 本地 $(echo "$local_md5"|wc -l) 个 ⇒ 先不关机"
  ok=0
fi
du -sh "$LOCAL" | tee -a "$LOG"

if [ "$ok" = 1 ] && [ "${KEEP_GPU:-0}" != "1" ]; then
  say "=== 关机 ==="
  $SSH "sync; (shutdown -h now || poweroff) 2>&1" | tail -2
  sleep 20
  $SSH "echo alive" 2>/dev/null && say "⚠️ 还活着，可能关机命令被拒" || say "✓ GPU 已关"
else
  say "=== 不关机（校验失败或 KEEP_GPU=1）==="
fi
