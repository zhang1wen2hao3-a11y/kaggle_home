#!/usr/bin/env bash
# **v2 面板批量收尾**：等本机所有候选跑完 → 拉回 → 校验格数 → 关机（校验不过不关）。
#   bash tools/finish_panels_v2.sh root@HOST "candA,candB,candC" logs/v2_123.done 17
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
H="${1:?host}"; CANDS="${2:?候选逗号分隔}"; DONE="${3:?完成标记文件}"; EXP="${4:-17}"
TAG=v2; LOG=logs/finish_panels_v2.log
SSHOPT="-o BatchMode=yes -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "等 $DONE（最多 180 分钟）"
for i in $(seq 1 360); do [ -f "$DONE" ] && { say "✓ 标记出现"; break; }; sleep 30; done
say "拉取 $H 结果"
timeout 300 rsync -az --include '*/' --include 'vs_*.json' --exclude '*' -e "ssh $SSHOPT" \
  "$H:/root/eval_cpu_kit/logs/meta/" logs/meta/ || say "⚠️ 拉取失败"
ok=1
for c in ${CANDS//,/ }; do
  n=$(ls "logs/meta/${c}_${TAG}_s950000"/vs_*.json 2>/dev/null | wc -l)
  say "  $c: $n/$EXP 格"
  [ "$n" -ge "$((EXP-2))" ] || ok=0
done
if [ "$ok" = 1 ]; then
  say "═══ 校验通过，关 $H ═══"
  timeout 40 ssh $SSHOPT "$H" 'sync; nohup shutdown -h now > /dev/null 2>&1 & sleep 2; echo bye' 2>&1 | tail -1
else
  say "✗ 格数不足 ⇒ 保持开机（需要人工看）"
  exit 2
fi
say "DONE $H"
