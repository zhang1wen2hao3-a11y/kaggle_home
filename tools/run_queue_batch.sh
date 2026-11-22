#!/usr/bin/env bash
# **批量轮训队列**（在一台 GPU 上串行跑多副牌，每副 4M 步）
#   bash tools/run_queue_batch.sh 9180_r1:unseen__91808263:PORT cb25_r1:beldum-metang__cb25:PORT 049c_r1:dreepy-drakloak__049c:PORT
#   起点由 INIT_<run> 指定（默认端口权重）；bank 统一用现役最强 6 个。
#   ★ 只训练，不关机；收尾由 tools/finish_multi_gpu.sh 统一做（拉齐 3 个 bin + md5 + 转 pt + 关机）。
set -uo pipefail
cd /root/ptcg_train_kit/port_repo || { echo "✗ 不在套件目录"; exit 1; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PTCG_REPO=/root/ptcg_train_kit/port_repo PYTHON=/root/miniconda3/bin/python3
BANKS=/root/cand9D6ER4.bin,/root/cand3EAAR1.bin,/root/candf5f8E.bin,/root/ph3_2617_v2.bin,/root/cand7e39S.bin,/root/cand9D6ER3.bin
BANKDECKS=dunsparce_abra__9d6e,dreepy_drakloak__3eaa,abra_kadabra__f5f8,grookey_thwackey__2617,teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__9d6e
for spec in "$@"; do
  RUN="${spec%%:*}"; rest="${spec#*:}"; DECK="${rest%%:*}"; START="${rest#*:}"
  echo "──────── [$(date +%H:%M:%S)] 开始 $RUN（$DECK，起点=$START）"
  ANCHOR="$DECK" INIT="$START" BANKS="$BANKS" BANKDECKS="$BANKDECKS" \
  LEAGUE_PCT=0.08 TRACK="$DECK,dunsparce_abra__9d6e,dreepy_drakloak__3eaa" \
  T="${T:-4000000}" LR="${LR:-1e-4}" OUT="/root/$RUN" RUN="$RUN" CG=1 CHECKPOINT_INTERVAL=16 \
  PYTHONUNBUFFERED=1 bash tools/db04_line.sh > "/root/$RUN.log" 2>&1
  echo "──────── [$(date +%H:%M:%S)] 完成 $RUN"
done
echo "QUEUE_BATCH_DONE" > /root/queue_batch.done
