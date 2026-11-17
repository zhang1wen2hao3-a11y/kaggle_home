#!/usr/bin/env bash
# 等 0bf5 队列跑完 → 拉结果 → 校验 → 关机（失败则不关机）
set -uo pipefail
cd /home/zhang/code/kaggle/the-pokemon-company || exit 1
for i in $(seq 1 240); do
  if grep -q QUEUE_0BF5_DONE logs/queue_0bf5_box.log 2>/dev/null; then break; fi
  sleep 30
done
echo "[$(date +%H:%M)] 队列结束，开始收尾"
./tools/py tools/real_panel_agg.py cand0bf5S 2>&1 | tail -22
./tools/py tools/real_panel_agg.py opp_dunsparce_0bf5 2>&1 | tail -4
./tools/py tools/cell_gap_diagnosis.py --cand cand0bf5S 2>&1 | tail -24
bash tools/refresh_tables.sh --quiet 2>&1 | tail -2
bash tools/git_snapshot.sh "cand0bf5S 真实口径门禁（17格+4格第二种子）+ 逐格差距诊断" 2>&1 | tail -1
bash tools/finish_box.sh root@113.44.139.218 38
echo WATCH_0BF5_SHUTDOWN_DONE
