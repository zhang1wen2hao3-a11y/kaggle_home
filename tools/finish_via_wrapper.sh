#!/usr/bin/env bash
# **GPU 收尾（密码登录版）**：等训练出 final.bin → 拉回 → md5 双向校验 → 转 .pt → 关机
#
# 为什么另写一个：有些机器只有密码登录（公钥被 sshd StrictModes 拒），而 finish_run_generic.sh
# 里的 ssh/rsync 是按公钥写的。这里改为"所有远端调用都走一个外部 ssh 包装脚本"，
# 包装脚本自己用 askpass 喂密码 —— 不改动原脚本，也不在容器里碰 systemd/shutdown 的定时功能。
#
#   GPU_SSH_WRAP=/tmp/ssh3 GPU="-p 38237 root@host" FINAL=/root/9d6e_r3/9d6e_r3.bin \
#     RUN=9d6e_r3 NAME=cand9D6ER3 DECK=dunsparce_abra__9d6e bash tools/finish_via_wrapper.sh
#
# ⚠️ 全程只用 `shutdown -h now`（唯一一次、且在 md5 校验通过之后）。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
WRAP="${GPU_SSH_WRAP:?需要 ssh 包装脚本路径}"
GPU="${GPU:?需要 '-p PORT root@host'}"
FINAL="${FINAL:?}"; RUN="${RUN:?}"; NAME="${NAME:?}"; DECK="${DECK:?}"
PORT=$(echo "$GPU" | sed -n 's/.*-p *\([0-9]\+\).*/\1/p'); PORT=${PORT:-22}
RHOST=$(echo "$GPU" | awk '{print $NF}')
SIZEF=18205408
LOG="logs/finish_${NAME}.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
rsh() { "$WRAP" "$@"; }

say "等 $FINAL 落地（每 30s 探，最多 120 分钟）"
for i in $(seq 1 240); do
  sz=$(rsh $GPU "stat -c %s $FINAL 2>/dev/null" 2>/dev/null | tr -d '[:space:]')
  run=$(rsh $GPU 'pgrep -f "native/train_native" 2>/dev/null | wc -l' 2>/dev/null | tr -d '[:space:]')
  case "${sz:-}" in ''|*[!0-9]*) sz=0;; esac
  case "${run:-}" in ''|*[!0-9]*) run=-1;; esac        # -1 = 连不上（不等于训练结束）
  say "  第 $i 次：final=${sz}B 训练进程=${run}"
  [ "$sz" = "$SIZEF" ] && [ "$run" = "0" ] && break
  sleep 30
done
[ "${sz:-0}" = "$SIZEF" ] || { say "✗ 等不到 $FINAL（最后一次 ${sz:-0}B）"; exit 1; }

mkdir -p "models/ppo/gpu_$RUN"
rsync -az --timeout=300 --rsh="$WRAP" "$RHOST:$FINAL" "models/ppo/gpu_$RUN/$RUN.bin" \
  || { say "✗ 拉 final 失败"; exit 1; }
rsync -az --timeout=300 --rsh="$WRAP" "$RHOST:/root/ptcg_train_kit/port_repo/experiments/native-$RUN/" \
  "models/ppo/gpu_$RUN/snapshots/" 2>/dev/null || true
[ "$(stat -c %s "models/ppo/gpu_$RUN/$RUN.bin")" = "$SIZEF" ] || { say "✗ 大小不对"; exit 1; }
lf=$(md5sum "models/ppo/gpu_$RUN/$RUN.bin" | cut -d' ' -f1)
rf=$(rsh $GPU "md5sum $FINAL | cut -d' ' -f1" 2>/dev/null | tr -d '[:space:]')
[ "$lf" = "$rf" ] || { say "✗ md5 不一致（本地 $lf / 远端 $rf）"; exit 1; }
say "✓ final md5 $lf"

cp "models/ppo/gpu_$RUN/$RUN.bin" "models/ppo/$NAME.bin"
PYTHONPATH=build/port_plain:. ./tools/py tools/native_weights.py import "models/ppo/$NAME.bin" "models/ppo/$NAME.pt" 2>&1 | tail -1
say "✓ models/ppo/$NAME.pt（大小 $(stat -c %s "models/ppo/$NAME.pt") B）"

say "═══ 关 GPU（产物已校验，唯一一次关机）═══"
rsh $GPU 'sync; nohup shutdown -h now > /dev/null 2>&1 & sleep 2; echo bye' 2>&1 | tail -1 | sed 's/^/  /' | tee -a "$LOG"
say "DONE（面板评估等你在 CPU 机器上开好我再来）"
