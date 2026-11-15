#!/usr/bin/env bash
# **盯 GPU 训练 → 进度入日志 → 跑完拉回全部产物（md5 校验）→ 关机器**
#
#   bash tools/watch_gpu_pull_shutdown.sh root@HOST [端口] [远端输出目录] [run名]
#
# 与 tools/watch_gpu_then_pull.sh 的区别：这个专用于"native 训练线"的目录结构
#   · 快照落在 <repo>/experiments/native-<run>/<step>.bin（CHECKPOINT_INTERVAL 控制）
#   · 最终模型落在 $OUT/<....>.bin
#   · 训练日志 <repo>/logs/<run>.log（每 epoch 一行，含 SPS / lgw / kl）
#
# 关机纪律：**只有**「训练进程消失 + 最终 .bin 存在 + 每个拉回文件的 md5 与远端一致」才关机；
# 任何一条不满足就只记日志、留着机器等人工看，绝不带着未校验的产物断电。
set -uo pipefail
cd "$(dirname "$0")/.."
HOST="${1:?用法: bash tools/watch_gpu_pull_shutdown.sh root@HOST [端口] [OUT目录] [run名]}"
PORT="${2:-22}"
OUT="${3:-/root/f5f8_shup4m}"
RUN="${4:-deck_f5f8__db04}"
R=/root/ptcg_train_kit/port_repo
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes -p $PORT $HOST"
LOG=logs/gpu_monitor.log
LOCAL="models/ppo/gpu_$RUN"
mkdir -p logs "$LOCAL"
say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }

say "盯 $HOST:$PORT  run=$RUN  OUT=$OUT（每 60s 一轮）"
for i in $(seq 1 720); do            # 最多 12 小时
  if ! $SSH true 2>/dev/null; then
    [ $((i % 10)) = 1 ] && say "（机器不可达，第 $i 次探测）"
    sleep 60; continue
  fi
  info=$($SSH "tail -1 $R/logs/$RUN.log 2>/dev/null | cut -c1-200" 2>/dev/null)
  running=$($SSH "pgrep -f 'train_nativ[e]' >/dev/null && echo yes || echo no" 2>/dev/null)
  nsnap=$($SSH "ls $R/experiments/native-$RUN/*.bin 2>/dev/null | wc -l" 2>/dev/null)
  final=$($SSH "ls $OUT/*.bin 2>/dev/null | head -1" 2>/dev/null)
  [ $((i % 5)) = 1 ] && say "第 $i 轮 running=$running 快照=${nsnap:-0} 末行=${info#epoch }"
  if [ "$running" = "no" ] && [ -n "$final" ]; then
    say "✅ 训练进程已退出且最终模型存在 → 开始拉产物"
    ok=yes
    # ① 最终模型 + 快照 + 训练日志
    rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" "$HOST:$OUT/" "$LOCAL/" || ok=no
    rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
      "$HOST:$R/experiments/native-$RUN/" "$LOCAL/snapshots/" || ok=no
    rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
      "$HOST:$R/logs/$RUN.log" "$LOCAL/train.log" 2>/dev/null || true
    # ② md5 逐个对远端校验
    for f in $(cd "$LOCAL" && find . -name '*.bin' -printf '%P\n' 2>/dev/null); do
      l=$(md5sum "$LOCAL/$f" | cut -d' ' -f1)
      # ★ 快照从 `native-$RUN/` 根目录拉（rsync 目标是 snapshots/ 子目录），所以校验路径
      #   必须用 basename：早先写成 `$R/experiments/native-$RUN/$f`（$f 含 `snapshots/` 前缀）
      #   会全部对不上 ⇒ 误判"校验失败"而不关机（2026-11-16 实际踩到）。
      b=$(basename "$f")
      r=$($SSH "md5sum $OUT/$f 2>/dev/null | cut -d' ' -f1; md5sum $R/experiments/native-$RUN/$b 2>/dev/null | cut -d' ' -f1" 2>/dev/null | grep -c "$l")
      if [ "$r" -ge 1 ]; then say "  ✓ $f md5 $l"; else say "  ✗ $f md5 不一致（local=$l）"; ok=no; fi
    done
    nb=$(find "$LOCAL" -name '*.bin' | wc -l)
    say "  拉回 $nb 个 .bin 到 $LOCAL（校验 $([ "$ok" = yes ] && echo 通过 || echo 失败)）"
    if [ "$ok" = yes ]; then
      say "═══ 关机（产物已校验）═══"
      $SSH "sync; shutdown -h now" 2>/dev/null
      say "已下发关机命令。" ; exit 0
    else
      say "⚠️ 校验失败 ⇒ **不关机**，请人工检查"; exit 1
    fi
  fi
  sleep 60
done
say "⚠️ 12 小时到点仍未完成，退出监控（不关机）"
exit 2
