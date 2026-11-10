#!/usr/bin/env bash
# 训练机上的批量 BC 队列：一次把准备好的几副牌全部训完。
#
# 用法（在训练机 /root/bc2626 下）：
#   bash tools/train_queue.sh bc_0904p bc_e1d7p bc_26c6p
#   bash tools/train_queue.sh --epochs 15 bc_0904p
#
# 约定：data/<name>.npz + data/<name>.f32 已存在；权重输出到 out/<name>.pt。
# 每副训完立即打印 val 一致率，训完一副就产出一副（不等全部结束）。
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
EPOCHS=12
LR=2e-4
BS=256
NAMES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --epochs) EPOCHS="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --bs) BS="$2"; shift 2;;
    *) NAMES+=("$1"); shift;;
  esac
done
[ ${#NAMES[@]} -eq 0 ] && { echo "用法: bash tools/train_queue.sh [--epochs N] <name>..."; exit 1; }

mkdir -p out
FAILED=()
for n in "${NAMES[@]}"; do
  npz="data/$n.npz"
  if [ ! -f "$npz" ]; then echo "✗ 缺 $npz，跳过"; FAILED+=("$n"); continue; fi
  if [ ! -f "data/$n.f32" ]; then echo "✗ 缺 data/$n.f32，跳过"; FAILED+=("$n"); continue; fi
  echo "===== $n 开始 $(date +%H:%M:%S)  $(du -h "data/$n.f32" | cut -f1) ====="
  $PY -u tools/bc_train.py --npz "$npz" --device cuda --epochs "$EPOCHS" \
      --lr "$LR" --bs "$BS" --out "out/$n.pt" 2>&1 | tail -6
  if [ -f "out/$n.pt" ]; then
    echo "✅ $n → out/$n.pt  $(stat -c%s "out/$n.pt") B"
  else
    echo "✗ $n 训练未产出"; FAILED+=("$n")
  fi
done

echo "===== 队列结束 $(date +%H:%M:%S) ====="
ls -l out/*.pt 2>/dev/null | tail -8
[ ${#FAILED[@]} -gt 0 ] && echo "⚠️ 失败/跳过：${FAILED[*]}"
