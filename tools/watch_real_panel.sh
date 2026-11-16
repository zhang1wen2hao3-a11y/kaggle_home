#!/usr/bin/env bash
# **三台机器的真实口径面板：监视 + 自动收尾**（每 90 秒一轮）
#   bash tools/watch_real_panel.sh [host1 host2 host3]
# 每轮做：查进度 → 有新文件就拉回本地 → 全部跑完则聚合 + 刷表 + 提交 git
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
HOSTS=("$@"); [ ${#HOSTS[@]} -eq 0 ] && HOSTS=(root@113.44.216.143 root@1.92.97.36 root@113.44.139.218)
TOTAL_CELLS=${TOTAL_CELLS:-14}      # 2 候选 × 7 新格
for round in $(seq 1 80); do
  done_n=0; busy=0; line=""
  for h in "${HOSTS[@]}"; do
    info=$(timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
      'cd /root/eval_cpu_kit 2>/dev/null; n=$(ls logs/meta/*_real_s950000/vs_*.json 2>/dev/null|wc -l); r=$(pgrep -fc "ptcg_league/eval_pair" || echo 0); echo "$n $r"' 2>/dev/null)
    n=${info%% *}; r=${info##* }
    case "$n" in ''|*[!0-9]*) n=0;; esac
    case "$r" in ''|*[!0-9]*) r=0;; esac
    done_n=$((done_n+n)); busy=$((busy+r))
    line="$line ${h#root@}:$n格/$([ "$r" -gt 0 ] && echo 跑 || echo 闲)"
  done
  echo "[$(date +%H:%M:%S)]$line | 合计 $done_n/$TOTAL_CELLS"
  bash tools/pull_results.sh "${HOSTS[@]}" > /tmp/pull_real.log 2>&1
  if [ "$busy" -eq 0 ] && [ "$done_n" -ge "$TOTAL_CELLS" ]; then
    echo "===== 全部格子跑完，开始聚合 ====="
    ./tools/py tools/real_panel_agg.py candf5f8G candf5f8S 2>&1 | tail -50
    bash tools/refresh_tables.sh --quiet 2>&1 | tail -2
    bash tools/git_snapshot.sh "真实份额口径面板（17 格）跑完：candf5f8G / candf5f8S" 2>&1 | tail -1
    echo WATCH_REAL_DONE; exit 0
  fi
  if [ "$busy" -eq 0 ] && [ "$done_n" -lt "$TOTAL_CELLS" ]; then
    idle=$((idle+1)); [ "$idle" -ge 3 ] && { echo "⚠️ 三台都闲着但只完成 $done_n/$TOTAL_CELLS ⇒ 停下等人看"; exit 1; }
  else idle=0; fi
  sleep 90
done
echo "WATCH_REAL_TIMEOUT"
