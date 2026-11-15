#!/usr/bin/env bash
# 等 watch_gpu_pull_shutdown 把 final .bin 拉到本地 → 转成 candf5f8G.{bin,pt}（md5 校验）
set -uo pipefail
cd /home/zhang/code/kaggle/the-pokemon-company || exit 1
SRC=models/ppo/gpu_candf5f8S_db04/candf5f8S_db04.bin
for i in $(seq 1 120); do [ -s "$SRC" ] && [ "$(stat -c %s "$SRC")" = 18205408 ] && break; sleep 30; done
[ -s "$SRC" ] || { echo "没等到 $SRC"; exit 1; }
sleep 5
cp "$SRC" models/ppo/candf5f8G.bin
PYTHONPATH=build/port_plain:. ./tools/py tools/native_weights.py import models/ppo/candf5f8G.bin models/ppo/candf5f8G.pt 2>&1 | tail -2
ls -l models/ppo/candf5f8G.bin models/ppo/candf5f8G.pt
echo "CONVERT_DONE $(md5sum models/ppo/candf5f8G.pt | cut -c1-12)"
