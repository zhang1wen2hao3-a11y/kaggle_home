#!/usr/bin/env bash
# **一条命令把评估部署到一台新 CPU 机并起队列**
#
#   bash tools/deploy_eval_cpu.sh root@1.2.3.4 [端口]      # 默认 22
#
# 做的事：推 dist/eval_cpu_kit.tar.gz → 解包 → setup.sh（装 agent / 自检）→
#         后台跑 tools/queue_remaining.sh（幂等，已完成格自动跳过）→ 打印拉回命令
set -uo pipefail
cd "$(dirname "$0")/.."
HOST="${1:?用法: bash tools/deploy_eval_cpu.sh root@HOST [端口]}"
PORT="${2:-22}"
R="${R:-/root/eval_cpu_kit}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PORT $HOST"
TAR="dist/eval_cpu_kit.tar.gz"

[ -f "$TAR" ] || { echo "✗ 缺 $TAR，先 bash tools/pack_eval_cpu.sh"; exit 1; }
echo "=== ① 连通性 + 机器规格 ==="
$SSH 'nproc; free -g | awk "/^Mem:/{print \"内存(G)\", \$2}"; python3 -V' || { echo "✗ 连不上"; exit 1; }

echo "=== ② 推包（$(du -h "$TAR" | cut -f1)）==="
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" "$TAR" "$HOST:/root/" || exit 1

echo "=== ③ 解包 + setup ==="
$SSH "cd /root && rm -rf eval_cpu_kit && tar xzf eval_cpu_kit.tar.gz && cd eval_cpu_kit && bash setup.sh" | tail -8

echo "=== ④ 起队列（后台，幂等）==="
$SSH "cd $R && mkdir -p logs && setsid nohup bash tools/queue_remaining.sh > logs/queue_all.log 2>&1 & sleep 5; tail -2 $R/logs/queue_all.log"

cat <<EOF

✓ 已部署到 $HOST:$R
  看进度：  ssh -p $PORT $HOST 'tail -3 $R/logs/queue_all.log; ls $R/logs/meta/*_9d6e/vs_*.json | wc -l'
  拉结果：  bash tools/pull_eval_results.sh $HOST $PORT
EOF
