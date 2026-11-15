#!/usr/bin/env bash
# **长期跑机守夜**：定时拉结果 → 刷表 → 提交 git，直到远端把活干完且没有新文件。
#
#   bash tools/watch_and_snapshot.sh root@A root@B      # 默认每 5 分钟一轮
#   INTERVAL=300 MAXROUND=48 bash tools/watch_and_snapshot.sh root@A
#
# 为什么要有它：机器随时可能被释放（2026-11-15 就掉了一台），结果不能等收工才拉；
# 而且"拉回来"必须同时"刷表 + 提交"，否则下次谁也不知道表是按哪批证据出的。
set -uo pipefail
cd "$(dirname "$0")/.."
HOSTS=("$@"); [ ${#HOSTS[@]} -eq 0 ] && HOSTS=(root@113.44.216.143 root@1.92.97.36)
INTERVAL=${INTERVAL:-300}
MAXROUND=${MAXROUND:-48}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
quiet_rounds=0
for r in $(seq 1 "$MAXROUND"); do
  bash tools/pull_results.sh "${HOSTS[@]}" > "$TMP/pull.log" 2>&1
  landed=$(grep -o '落地 [0-9]*' "$TMP/pull.log" | awk '{s+=$2} END{print s+0}')
  bash tools/refresh_tables.sh --quiet > "$TMP/refresh.log" 2>&1 || echo "⚠️ 刷表失败（见 $TMP/refresh.log）"
  rr=$(ls data/opp_matrix/rr_*.json 2>/dev/null | wc -l)
  pil=$(ls logs/meta/*_A_s950000/summary.json 2>/dev/null | wc -l)
  bash tools/git_snapshot.sh "自动快照 $(date +%m-%d\ %H:%M)：循环赛 $rr 格，方案A 面板 $pil 份" > "$TMP/git.log" 2>&1
  cmsg=$(tail -1 "$TMP/git.log")
  echo "[$(date +%H:%M)] 新增文件 $landed 个 | 循环赛 $rr 格 | 方案A面板 $pil 份 | $cmsg"
  if [ "$landed" -eq 0 ]; then quiet_rounds=$((quiet_rounds+1)); else quiet_rounds=0; fi
  # 收工条件：连续 3 轮没有新文件（远端活干完 / 机器掉了）
  [ "$quiet_rounds" -ge 3 ] && { echo "连续 3 轮无新结果 ⇒ 收工"; break; }
  sleep "$INTERVAL"
done
echo "WATCH_DONE"
