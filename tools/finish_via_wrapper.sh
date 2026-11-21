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

# ★ 2026-11-22 重要更正：`pgrep -f "native/train_native"` 会**匹配到执行它的那个 shell 自己**
#   （远程命令的 cmdline 里就含这个字符串）⇒ 训练早就退出了，计数仍恒为 1，
#   我据此误判成"训练还在跑收尾 eval，挂了一小时"，实际训练写完 final 后立刻就退出了。
#   必须写成中括号形式 `[n]ative/train_native`，让模式本身不匹配自己的命令行。
say "等 $FINAL 落地（每 30s 探，最多 120 分钟）"
# ★ 2026-11-22 教训：原来要求"文件到位 **且** 训练进程退出"才收尾。
#   实测训练写完 final.bin 后进程还会挂着做收尾 eval（整整挂了一小时）⇒ 条件永远不满足，
#   白白多跑了 85 分钟的机器。现在改成：**文件大小对 + 连续两次探测大小不变** 就认为写完（进程爱挂多久挂多久）。
stable=0; prev=-1
for i in $(seq 1 240); do
  sz=$(rsh $GPU "stat -c %s $FINAL 2>/dev/null" 2>/dev/null | tr -d '[:space:]')
  run=$(rsh $GPU 'pgrep -f "[n]ative/train_native" 2>/dev/null | wc -l' 2>/dev/null | tr -d '[:space:]')
  case "${sz:-}" in ''|*[!0-9]*) sz=0;; esac
  case "${run:-}" in ''|*[!0-9]*) run=-1;; esac        # -1 = 连不上（不等于训练结束）
  if [ "$sz" = "$SIZEF" ] && [ "$sz" = "$prev" ]; then stable=$((stable+1)); else stable=0; fi
  say "  第 $i 次：final=${sz}B 训练进程=${run} 稳定=${stable}"
  prev=$sz
  { [ "$sz" = "$SIZEF" ] && [ "$stable" -ge 1 ]; } && { say "✓ final 已写完且大小稳定，立刻收尾（不等进程退出）"; break; }
  sleep 30
done
[ "${sz:-0}" = "$SIZEF" ] || { say "✗ 等不到 $FINAL（最后一次 ${sz:-0}B）"; exit 1; }

mkdir -p "models/ppo/gpu_$RUN"
# ★ 2026-11-22 两版血泪：`rsync` 在某些机器上 300 秒传不完 18 MB 会超时；
#   `ssh cat` 管道在这台 nmb2 上会把 18,205,408 B 截成 17.58 MB（md5 对不上）。
#   ⇒ 优先用 scp（SCP_WRAP=<scp 包装脚本>），拉完**必须比 size + md5**。
ok=0
for try in 1 2 3; do
  if [ -n "${SCP_WRAP:-}" ]; then
    "$SCP_WRAP" "$RHOST:$FINAL" "models/ppo/gpu_$RUN/$RUN.bin" >/dev/null 2>&1
  else
    rsh $GPU "cat $FINAL" > "models/ppo/gpu_$RUN/$RUN.bin" 2>/dev/null
  fi
  if [ "$(stat -c %s "models/ppo/gpu_$RUN/$RUN.bin" 2>/dev/null)" = "$SIZEF" ]; then ok=1; break; fi
  say "  第 $try 次拉取失败（大小不符），重试"; sleep 5
done
[ "$ok" = 1 ] || { say "✗ 拉 final 失败"; exit 1; }
# 续训用的 state 也顺手拉（失败不影响主流程）
rsh $GPU "cat /root/ptcg_train_kit/port_repo/experiments/native-$RUN/state_latest.bin" \
  > "models/ppo/gpu_$RUN/state_latest.bin" 2>/dev/null || true
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
