#!/usr/bin/env bash
# 等 GPU 训练落地 → 拉回最终 .bin（md5 校验）→ 转 .pt
set -uo pipefail
cd /home/zhang/code/kaggle/the-pokemon-company
H="-P 38237 root@connect.nmb2.seetacloud.com"
OUT=/root/f5f8_gap4m/candf5f8S_db04.bin
for i in $(seq 1 60); do
  sz=$(ssh -o BatchMode=yes $H "stat -c %s $OUT 2>/dev/null" 2>/dev/null)
  if [ "${sz:-0}" = "18205408" ]; then
    sleep 20   # 等它写完
    sz2=$(ssh -o BatchMode=yes $H "stat -c %s $OUT" 2>/dev/null)
    [ "${sz2:-0}" = "18205408" ] && break
  fi
  sleep 60
done
echo "[$(date +%H:%M)] 训练结束，拉取权重"
ssh -o BatchMode=yes $H "md5sum $OUT"
scp -q $H:$OUT /tmp/candf5f8G.bin || { echo "拉取失败"; exit 1; }
md5sum /tmp/candf5f8G.bin
mkdir -p models/ppo
cp /tmp/candf5f8G.bin models/ppo/candf5f8G.bin
PYTHONPATH=build/port_plain:. ./tools/py tools/native_weights.py import models/ppo/candf5f8G.bin models/ppo/candf5f8G.pt 2>&1 | tail -2
ls -l models/ppo/candf5f8G.bin models/ppo/candf5f8G.pt
echo "PULL_DONE"
