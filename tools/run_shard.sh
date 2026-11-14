#!/usr/bin/env bash
# /root/run_shard.sh <all|0|1> —— CPU 机上的队列入口（由 orchestrate_cpu.sh 自动推送）
set -uo pipefail
cd /root/eval_cpu_kit || exit 1
mode="${1:-all}"
mkdir -p logs
echo "$mode" > logs/.runner_mode
exec bash tools/run_wishlist.sh "$mode"
