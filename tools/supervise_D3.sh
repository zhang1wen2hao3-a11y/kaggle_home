#!/usr/bin/env bash
# 监督 candf5f8D 第三种子：机器掉线则本地补跑剩余格；跑完聚合 + 刷表 + 提交
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
HOST=root@113.44.139.218
CELLS="ogerpon_7e39 grookey_2617 abra_e8ee abra_f5f8 dunsparce_9d6e dunsparce_0bf5 dreepy_3eaa champ_0904 champ_26c6 champ_e1d7"
TAG=candf5f8D_real_s950002
dead=0
for i in $(seq 1 60); do
  n=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 $HOST "ls /root/eval_cpu_kit/logs/meta/$TAG/vs_*.json 2>/dev/null | wc -l" 2>/dev/null | tr -d '[:space:]')
  case "${n:-}" in ''|*[!0-9]*) n="";; esac
  if [ -z "$n" ]; then
    dead=$((dead+1)); echo "[$(date +%H:%M:%S)] 机器不可达（连续 $dead 次）"
    if [ "$dead" -ge 2 ]; then
      echo "⇒ 改本地补跑剩余格"
      timeout 90 rsync -az -e "ssh -o BatchMode=yes" --include '*/' --include 'vs_*.json' --exclude '*' \
        "$HOST:/root/eval_cpu_kit/logs/meta/" logs/meta/ 2>/dev/null
      bash tools/eval_candidate_local.sh candf5f8D "$CELLS" 2>&1 | tail -4
      break
    fi
    sleep 30; continue
  fi
  dead=0
  echo "[$(date +%H:%M:%S)] 远端完成 $n/10"
  [ "$n" -ge 10 ] && break
  sleep 30
done
timeout 120 rsync -az -e "ssh -o BatchMode=yes" --include '*/' --include 'vs_*.json' --exclude '*' \
  "$HOST:/root/eval_cpu_kit/logs/meta/" logs/meta/ 2>/dev/null
echo "本地现有 $(ls logs/meta/$TAG/vs_*.json 2>/dev/null | wc -l)/10"
bash tools/refresh_tables.sh --quiet 2>&1 | tail -1
./tools/py tools/real_panel_agg.py candf5f8D 2>&1 | tail -6
bash tools/git_snapshot.sh "candf5f8D 第三种子（n→450）：母表/交付表重刷" 2>&1 | tail -1
echo SUPERVISE_D3_DONE
