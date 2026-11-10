#!/usr/bin/env bash
# 每轮训练产物 → 自动评测：GPU 机上的 bin 拉回来、转 pt、推到评估机、跑 300 场、拉结果。
#
# 用法：
#   bash tools/dual_round_eval.sh A1 A        # A 侧第 1 轮 → 对手用 3eaa 的 BC 克隆
#   bash tools/dual_round_eval.sh B2 B        # B 侧第 2 轮 → 对手用 26c6 的 BC 克隆
#
# 命名：GPU 上的产物是 /root/autodl-tmp/dual_kl2/dk_r<轮><侧>_blend.bin
#      侧 = A（26c6）/ B（3eaa）
set -uo pipefail
cd "$(dirname "$0")/.."
TAG=$1; SIDE=$2
GPU="root@connect.nmb1.seetacloud.com"; GP=23268
OUTDIR=/root/autodl-tmp/dual_kl2
case "$SIDE" in
  A) DECK=slowpoke_slowking__26c6; OPP=ag_bc_3eaa ;;   # 26c6 头 vs 3eaa 克隆
  B) DECK=dreepy_drakloak__3eaa;   OPP=ag_bc_26c6 ;;   # 3eaa 头 vs 26c6 克隆
  *) echo "✗ SIDE 只能是 A 或 B"; exit 1 ;;
esac
ROUND=${TAG:1:1}

echo "① 确认 GPU 产物 dk_r${ROUND}${SIDE}_blend.bin"
ssh -o BatchMode=yes -p $GP "$GPU" "ls -l $OUTDIR/dk_r${ROUND}${SIDE}_blend.bin" || exit 1

echo "② GPU 上转 .pt"
ssh -o BatchMode=yes -p $GP "$GPU" "cd /root/ptcg_train_kit/port_repo && PYTHONPATH=/root/bc2626/build/port_plain:. \
  /root/miniconda3/bin/python3 native/tools/native_weights.py import \
  $OUTDIR/dk_r${ROUND}${SIDE}_blend.bin /root/kl/dk_${TAG}.pt" | tail -1

echo "③ 拉回本地"
scp -q -P $GP "$GPU:/root/kl/dk_${TAG}.pt" /tmp/dk_${TAG}.pt && md5sum /tmp/dk_${TAG}.pt

echo "④ 推到评估机并评测"
bash tools/evnode.sh push dk_${TAG} /tmp/dk_${TAG}.pt "$DECK" | tail -1
bash tools/evnode.sh run "ag_dk_${TAG}" "bc=build/agents/$OPP" 300 990001 "/tmp/ev_dk_${TAG}.json" | tail -3
