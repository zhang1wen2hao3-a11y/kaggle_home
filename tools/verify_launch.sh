#!/usr/bin/env bash
# **启动验证纪律**（2026-11-17 用户要求固化）：任何"在远端起任务"的动作，起完必须
# 在 10~20 秒内确认三件事，**不要用长 sleep 等结果**：
#   ① 远端**真有进程**在跑（不是脚本瞬间退出了）
#   ② 结果目录开始出现产物（或在 N 次探测内开始增长）
#   ③ 日志里没有立刻出现的致命错误（assert/traceback/MISSING/不存在）
# 任一条不满足 → 立刻打印远端日志尾部并非零退出（让人马上修，而不是等 30 分钟才发现白跑）。
#
#   bash tools/verify_launch.sh root@HOST <探针命令> [关键日志glob]
# 例：
#   bash tools/verify_launch.sh root@1.2.3.4 'pgrep -fc "ptcg_league/[e]val_pair"' '/root/dispatch_*.log'
#
# 环境变量：INTERVAL=12（秒）、ROUNDS=6（默认最多 72 秒）、ALLOW_ZERO=0
#   探针命令需输出一个**数字**（进程数 / 产物数）；0 视为"还没起来"，>0 视为成功。
set -uo pipefail
# ★ 2026-11-22 修 bug：探针用 `2>&1 | head -1` 取第一行，而新机器第一次连会先打印
#   "Warning: Permanently added ... to the list of known hosts."（走 stderr，被 2>&1 并进 stdout）
#   ⇒ 第一行是警告而不是进程数 ⇒ 被当成 0 ⇒ 明明在跑也判"没起来"，把还在跑的面板 kill 掉。
#   现在：stderr 丢掉，只取纯数字行，并统一 LogLevel=ERROR。
HOST="${1:?用法: bash tools/verify_launch.sh <host> <探针命令> [日志glob]}"
PROBE="${2:?需要探针命令，输出一个数字}"
LOGLOB="${3:-/root/*.log}"
INTERVAL="${INTERVAL:-12}"
ROUNDS="${ROUNDS:-6}"
ALLOW_ZERO="${ALLOW_ZERO:-0}"
die() {
  echo "✗ 启动验证失败：$*"
  echo "--- 远端日志尾部（$LOGLOB）---"
  timeout 20 ssh -o BatchMode=yes -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 "$HOST" \
    "for f in $LOGLOB; do echo \"== \$f\"; tail -12 \"\$f\"; done" 2>&1 | tail -40
  exit 2
}
for i in $(seq 1 "$ROUNDS"); do
  out=$(timeout 20 ssh -o BatchMode=yes -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 "$HOST" "$PROBE" 2>/dev/null | tr -d '[:space:]' | grep -oE '[0-9]+' | head -1)
  case "$out" in ''|*[!0-9]*) out=0;; esac
  echo "  探第 $i 次（每 ${INTERVAL}s）：$out"
  if [ "$out" -gt 0 ]; then echo "✓ 启动确认（$out）"; exit 0; fi
  # 顺手查致命错误关键字，有就直接失败（不用等满轮次）
  err=$(timeout 20 ssh -o BatchMode=yes -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 "$HOST" \
        "for f in $LOGLOB; do grep -lE 'Traceback|AssertionError|MISSING_AGENT|不存在' \"\$f\" 2>/dev/null; done" 2>/dev/null | head -1)
  [ -n "$err" ] && die "日志里已有致命错误：$err"
  sleep "$INTERVAL"
done
[ "$ALLOW_ZERO" = 1 ] && { echo "（允许 0，放行）"; exit 0; }
die "连续 $ROUNDS 次探针为 0（${ROUNDS}x${INTERVAL}s）"
