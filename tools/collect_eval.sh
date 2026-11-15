#!/usr/bin/env bash
# **把多台机器的结果收回来 + 合并 + 刷表**（跑完/中途都可以跑，幂等）
#
#   bash tools/collect_eval.sh root@A root@B root@C
#
# 做三件事：
#   ① 从每台机器 rsync logs/meta（只补缺，不覆盖本地已有）
#   ② 把每台的 data/opp_matrix（单格测量）也拉回来
#   ③ 调 refresh_tables.sh（母表 → 两套交付表 → 名字校验 → 方案A测试集）
# 然后打印"队列里还剩几个没跑完"。
set -uo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { echo "用法: bash tools/collect_eval.sh root@A [root@B ...]"; exit 2; }
R=/root/eval_cpu_kit
for h in "$@"; do
  port=22; host=$h
  case "$h" in *:*) host=${h%:*}; port=${h##*:};; esac
  echo "== 收 $host:$port"
  rsync -az --ignore-existing -e "ssh -o StrictHostKeyChecking=no -p $port" \
    "$host:$R/logs/meta/" logs/meta/ 2>/dev/null
  rsync -az --ignore-existing -e "ssh -o StrictHostKeyChecking=no -p $port" \
    "$host:$R/data/opp_matrix/" data/opp_matrix/ 2>/dev/null
done
echo
bash tools/refresh_tables.sh --quiet 2>&1 | tail -4
echo
./tools/py tools/eval_queue.py status
