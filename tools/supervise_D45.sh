#!/usr/bin/env bash
# 监督 candf5f8D 第四/五种子：掉线则本地补跑；跑完刷表 + 聚合 + 提交
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
HOST=root@113.44.139.218
CELLS="ogerpon_7e39 grookey_2617 abra_e8ee dunsparce_9d6e dunsparce_0bf5 champ_0904 champ_26c6 champ_e1d7"
dead=0
for i in $(seq 1 90); do
  n=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 $HOST "ls /root/eval_cpu_kit/logs/meta/candf5f8D_real_s95000{3,4}/vs_*.json 2>/dev/null | wc -l" 2>/dev/null | tr -d '[:space:]')
  case "${n:-}" in ''|*[!0-9]*) n="";; esac
  if [ -z "$n" ]; then
    dead=$((dead+1)); echo "[$(date +%H:%M:%S)] 不可达（连续 $dead 次）"
    if [ "$dead" -ge 2 ]; then
      timeout 90 rsync -az -e "ssh -o BatchMode=yes" --include '*/' --include 'vs_*.json' --exclude '*' \
        "$HOST:/root/eval_cpu_kit/logs/meta/" logs/meta/ 2>/dev/null
      bash tools/eval_candidate_local.sh candf5f8D "$CELLS" 2>&1 | tail -3
      break
    fi
    sleep 30; continue
  fi
  dead=0; echo "[$(date +%H:%M:%S)] 远端完成 $n/16"
  [ "$n" -ge 16 ] && break
  sleep 30
done
timeout 150 rsync -az -e "ssh -o BatchMode=yes" --include '*/' --include 'vs_*.json' --exclude '*' \
  "$HOST:/root/eval_cpu_kit/logs/meta/" logs/meta/ 2>/dev/null
bash tools/refresh_tables.sh --quiet 2>&1 | tail -1
./tools/py -c "
import csv
rows=[r for r in csv.DictReader(open('docs/the-pockmon-battle-choice-in-table.csv',encoding='gb18030')) if r['驾驶员']=='candf5f8D']
print('母表 candf5f8D:', [(r['被挑战者'], r['胜率'], 'n='+r['样本场次']) for r in rows])
for f in ('docs/the-pockmon-battle-choices-win-rate-pilot.csv','docs/the-pockmon-battle-choices-win-rate-pilot_top.csv'):
    for r in csv.DictReader(open(f,encoding='gb18030')):
        if r['卡组']=='f5f8': print(f.split('/')[-1],'→',r['最佳驾驶员'],r['面板场地加权'])
"
bash tools/git_snapshot.sh "candf5f8D 第四/五种子（n=450 覆盖 8 格）：刷表" 2>&1 | tail -1
echo SUPERVISE_D45_DONE
