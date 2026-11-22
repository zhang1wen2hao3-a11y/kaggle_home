#!/usr/bin/env bash
# **多轮训练的收尾**：等每个 run 的 final.bin 落地 → 拉回 → md5 双向校验 → 转 .pt；全部齐了才关机。
#   WRAP=/tmp/ssh3m SCP=/tmp/scp3m GPU="-p 38237 root@root@host" \
#     RUNS="9180_r1:cand9180R1  cb25_r1:candCB25R1  049c_r1:cand049CR1" bash tools/finish_multi_gpu.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
: "${WRAP:?}" "${SCP:?}" "${GPU:?}" "${RUNS:?}"
SIZEF=18205408
LOG=logs/finish_multi_gpu.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
rsh(){ "$WRAP" "$@"; }
mkdir -p models/ppo/gpu_multi
fail=0
for spec in $RUNS; do
  RUN="${spec%%:*}"; NAME="${spec##*:}"
  FINAL="/root/$RUN/$RUN.bin"
  say "等 $RUN（最多 150 分钟）"
  stable=0; prev=-1
  for i in $(seq 1 300); do
    sz=$(rsh $GPU "stat -c %s $FINAL 2>/dev/null" 2>/dev/null | tr -d '[:space:]')
    case "${sz:-}" in ''|*[!0-9]*) sz=0;; esac
    if [ "$sz" = "$SIZEF" ] && [ "$sz" = "$prev" ]; then stable=$((stable+1)); else stable=0; fi
    prev=$sz
    { [ "$sz" = "$SIZEF" ] && [ "$stable" -ge 1 ]; } && { say "  ✓ $RUN 写完"; break; }
    sleep 30
  done
  [ "${sz:-0}" = "$SIZEF" ] || { say "  ✗ 等不到 $FINAL"; fail=1; continue; }
  ok=0
  for t in 1 2 3; do
    "$SCP" "$(echo $GPU | awk '{print $NF}'):$FINAL" "models/ppo/gpu_multi/$RUN.bin" >/dev/null 2>&1
    [ "$(stat -c %s "models/ppo/gpu_multi/$RUN.bin" 2>/dev/null)" = "$SIZEF" ] && { ok=1; break; }
    say "  第 $t 次拉取失败，重试"; sleep 5
  done
  [ "$ok" = 1 ] || { say "  ✗ $RUN 拉取失败"; fail=1; continue; }
  lf=$(md5sum "models/ppo/gpu_multi/$RUN.bin" | cut -d' ' -f1)
  rf=$(rsh $GPU "md5sum $FINAL | cut -d' ' -f1" 2>/dev/null | tr -d '[:space:]')
  [ "$lf" = "$rf" ] || { say "  ✗ $RUN md5 不一致（本地 $lf / 远端 $rf）"; fail=1; continue; }
  cp -f "models/ppo/gpu_multi/$RUN.bin" "models/ppo/$NAME.bin"
  PYTHONPATH=build/port_plain:. ./tools/py tools/native_weights.py import "models/ppo/$NAME.bin" "models/ppo/$NAME.pt" 2>&1 | tail -1
  say "  ✓ $NAME.pt（md5 $lf）"
done
if [ "$fail" = 0 ]; then
  say "═══ 全部校验通过，关 GPU（唯一一次关机）═══"
  rsh $GPU 'sync; nohup shutdown -h now > /dev/null 2>&1 & sleep 2; echo bye' 2>&1 | tail -1
else
  say "⚠️ 有失败项，保持开机"
fi
say DONE
