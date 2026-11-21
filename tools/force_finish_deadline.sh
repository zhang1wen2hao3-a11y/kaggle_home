#!/usr/bin/env bash
# **兜底强制收尾**：到点先尽力拉一次结果，然后**无条件关机**。
#
# 用户 2026-11-22 要求：主收尾脚本若因"结果没拉全 / 上传失败"卡住，机器不能一直开着。
#   · 主链（watch_serial_and_stop / finish_via_wrapper）= 正常路径：拉全 + 校验通过 → 关机；
#   · 本脚本 = 最后防线：到点再拉一次（失败也记日志），随即 `shutdown -h now`。
#
# CPU 例：
#   HOST=root@1.2.3.4 PORT=22 WAIT_MIN=40 \
#   PULL_CMD="rsync -az -e 'ssh -o StrictHostKeyChecking=no' --include '*/' --include 'vs_*.json' --exclude '*' root@1.2.3.4:/root/eval_cpu_kit/logs/meta/ logs/meta/" \
#   bash tools/force_finish_deadline.sh
#
# GPU 例（只拉 final）：
#   HOST=root@host PORT=18932 WAIT_MIN=55 \
#   PULL_CMD='ssh -p 18932 -o StrictHostKeyChecking=no root@host "cat /root/3eaa_r1/3eaa_r1.bin" > models/ppo/cand3EAAR1.bin' \
#   bash tools/force_finish_deadline.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
HOST="${HOST:?HOST=root@x.x.x.x}"; PORT="${PORT:-22}"; WAIT_MIN="${WAIT_MIN:?WAIT_MIN=分钟}"
PULL_CMD="${PULL_CMD:-}"; LOG="${LOG:-logs/force_finish_$(echo "$HOST" | tr -d '@.').log}"
SSHOPT="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -p $PORT"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "兜底已挂：$HOST:$PORT 在 $WAIT_MIN 分钟后强制关机（主链若先完成会先关，本兜底随之失效）"
sleep $(( WAIT_MIN * 60 ))

if [ -n "$PULL_CMD" ]; then
  say "到点：最后尝试拉一次结果"
  if eval "$PULL_CMD" >> "$LOG" 2>&1; then say "  ✓ 最后一次拉取成功"; else say "  ⚠️ 最后一次拉取失败（仍照常关机，缺口见本日志）"; fi
fi
say "强制关机 $HOST"
timeout 40 ssh $SSHOPT "$HOST" 'sync; nohup shutdown -h now >/dev/null 2>&1 & sleep 2; echo bye' >> "$LOG" 2>&1
say "DONE"
