#!/usr/bin/env bash
# **兜底定时关机**：给一台机器挂一个"到点必关"的 systemd 定时关机，作为看门狗失效时的第二道保险。
#
# 为什么需要（2026-11-19 用户要求）：看门狗依赖"日志标记 + 结果数量校验"，
# 一旦调度器被中断、或 scp/ssh 因主机指纹失败，脚本会**退出但不关机**，机器就空转挂着（已经发生过两次）。
# systemd 的定时关机由机器自己执行，不受我这边的进程/网络影响。
#
#   bash tools/arm_deadline_shutdown.sh root@123.249.80.18 02:00
#   bash tools/arm_deadline_shutdown.sh -p 18932 root@connect.westc.seetacloud.com 02:00
#
# 查：ssh <host> 'shutdown --show'     （或看 /run/systemd/shutdown/scheduled）
# 撤：ssh <host> 'shutdown -c'
#
# 注意：到点关机是**无条件**的。所以时间要留在"任务一定跑得完"之后；
#       任务提前完成时，正常的收尾脚本会先关机，定时器随之作废（机器都关了）。
set -uo pipefail
SSHOPT="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"

PORT=22
if [ "${1:-}" = "-p" ]; then PORT="$2"; shift 2; fi
H="${1:?用法: arm_deadline_shutdown.sh [-p PORT] host HH:MM}"
WHEN="${2:?给个时间，如 02:00}"

echo "→ 在 $H 挂定时关机 $WHEN（端口 $PORT）"
ssh $SSHOPT -p "$PORT" "$H" "shutdown -h $WHEN 2>&1 | tail -2; echo '--- 已排程确认'; shutdown --show 2>/dev/null || cat /run/systemd/shutdown/scheduled 2>/dev/null | xxd | head -2"
