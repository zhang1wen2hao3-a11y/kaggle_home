#!/usr/bin/env bash
# **一台 CPU 上串行评测多个候选（v2 口径 17 格）+ 统一收尾关机**
#   bash tools/eval_batch_v2.sh <host> "candA candB candC" [TESTSET] [WORKERS]
#
# 做的事：① 每个候选：推权重 → 建 agent 目录 → 跑 17 格 → 拉回；
#         ② 全部跑完 → 校验每格 17 个 json → 一次性关机；
#         ③ 另外挂一道**固定时点**兜底（本脚本打印建议的 AT 时间）。
# 特点：候选权重必须已经拉到 models/ppo/<候选>.pt（GPU 收尾脚本负责这件事）。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
H="${1:?host}"; CANDS="${2:?候选列表（空格分隔）}"; TS="${3:-decks/testset_meta_real_v2.tsv}"; W="${4:-24}"
SSHOPT="-o BatchMode=yes -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
LOG=logs/eval_batch_v2.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "评测开始：$H ← $CANDS（口径 $TS，workers $W）"
for c in $CANDS; do
  # 等权重落地（GPU 收尾可能晚几分钟）
  for i in $(seq 1 60); do [ -s "models/ppo/$c.pt" ] && break; sleep 30; done
  if [ ! -s "models/ppo/$c.pt" ]; then say "✗ $c 权重没等到，跳过"; continue; fi
  bash tools/panel_on_host.sh "$c" "$H" "$TS" v2 "$W" 2>&1 | tail -4 | sed "s/^/  [$c] /"
done

say "拉取全部结果"
for t in 1 2 3; do
  timeout 300 rsync -az --include '*/' --include 'vs_*.json' --exclude '*' -e "ssh $SSHOPT" \
    "$H:/root/eval_cpu_kit/logs/meta/" logs/meta/ && break
  say "  第 $t 次拉取失败，重试"; sleep 5
done
ok=1
for c in $CANDS; do
  n=$(ls "logs/meta/${c}_v2_s950000"/vs_*.json 2>/dev/null | wc -l)
  say "  $c: $n/17 格"
  [ "$n" -ge 16 ] || ok=0
done
if [ "$ok" = 1 ]; then
  say "═══ 全部校验通过，关 $H ═══"
  timeout 40 ssh $SSHOPT "$H" 'sync; nohup shutdown -h now >/dev/null 2>&1 & sleep 2; echo bye' 2>&1 | tail -1
else
  say "✗ 有候选格数不足 ⇒ 保持开机"
fi
say BATCH_DONE
