#!/usr/bin/env bash
# **盯 GPU 训练 → 跑完拉回权重并校验 md5 → 关机器**
#
#   bash tools/watch_gpu_then_pull.sh root@connect.nmb1.seetacloud.com 23268 cand7e39S,candE1D7S
#
# 设计：训练脚本本身**不关机**（避免产物没拉就断电），由这里在"两个 run 都出了 .bin
# 且没有训练进程"之后：rsync → md5 逐个对远端校验 → 全部一致才 shutdown。
# 任何一个不一致就不关，留机器等人工处理。
set -uo pipefail
cd "$(dirname "$0")/.."
HOST="${1:?用法: bash tools/watch_gpu_then_pull.sh root@HOST [端口] [run名,逗号分隔]}"
PORT="${2:-22}"
RUNS="${3:-cand7e39S,candE1D7S}"
R=/root/ptcg_train_kit/port_repo
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes -p $PORT $HOST"
LOG=logs/watch_gpu.log; mkdir -p logs data/ab_run3
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "盯 $HOST:$PORT  等待 runs=$RUNS 全部出权重"

for i in $(seq 1 360); do          # 最多 6 小时
  if ! $SSH true 2>/dev/null; then
    [ $((i % 6)) = 1 ] && say "（机器不可达，第 $i 次探测）"
    sleep 60; continue
  fi
  running=$($SSH "pgrep -f 'train_nativ[e]' >/dev/null && echo yes || echo no" 2>/dev/null)
  ready=yes
  for r in ${RUNS//,/ }; do
    $SSH "test -f /root/autodl-tmp/$r/$r.bin" 2>/dev/null || ready=no
  done
  say "第 $i 次：running=$running  ready=$ready"
  if [ "$running" = "no" ] && [ "$ready" = "yes" ]; then
    say "训练结束且权重齐 ✅ 开始拉回"
    ok=yes
    for r in ${RUNS//,/ }; do
      rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
        "$HOST:/root/autodl-tmp/$r/$r.bin" "data/ab_run3/$r.bin" || ok=no
      rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
        "$HOST:$R/logs/$r.log" "data/ab_run3/$r.train.log" 2>/dev/null || true
      l=$(md5sum "data/ab_run3/$r.bin" 2>/dev/null | cut -d' ' -f1)
      rm_=$($SSH "md5sum /root/autodl-tmp/$r/$r.bin" 2>/dev/null | cut -d' ' -f1)
      if [ -n "$l" ] && [ "$l" = "$rm_" ]; then
        say "  ✓ $r.bin md5 $l"
      else
        say "  ✗ $r.bin md5 不一致 local=$l remote=$rm_"; ok=no
      fi
    done
    if [ "$ok" = "yes" ]; then
      say "全部校验通过 → 关机"
      $SSH "sync; nohup shutdown -h now >/dev/null 2>&1 &" || true
      sleep 20
      if $SSH true 2>/dev/null; then say "⚠️ 关机命令已发但机器仍可连，请人工确认"; else say "✓ 机器已关机"; fi
      exit 0
    else
      say "✗ 有文件不一致，**不关机**，等人工处理"
      exit 1
    fi
  fi
  sleep 60
done
say "✗ 6 小时内没等到，退出（不关机）"
exit 1
