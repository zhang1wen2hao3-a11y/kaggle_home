#!/usr/bin/env bash
# **把 A/B 长跑需要的文件推到 GPU 机**（GPU 开机后跑一次）。
#   bash tools/push_ab_kit.sh
# 推的东西：改过的脚本、choice 表、覆盖率权重、ph3 权重的 .bin。
set -uo pipefail
cd "$(dirname "$0")/.."
HOST="${HOST:-root@connect.nmb1.seetacloud.com}"; PORT="${PORT:-23268}"
R="${PTCG_REPO:-/root/ptcg_train_kit/port_repo}"
SSH="ssh -o StrictHostKeyChecking=no -p $PORT $HOST"

echo "=== 0) 连通性 ==="
$SSH 'echo ok; nvidia-smi --query-gpu=name --format=csv,noheader' || { echo "✗ 连不上"; exit 1; }

echo "=== 1) 文本文件（脚本 / choice 表 / 权重配置）==="
for f in tools/db04_line.sh tools/teacher_loop.sh tools/teacher_gate.sh tools/choice_to_banks.py \
         tools/weight_dist.py tools/launch_ab.sh tools/kl_blend.py \
         docs/the-pockmon-battle-choice.csv decks/weights_field_b05.txt; do
  [ -f "$f" ] || { echo "  ⚠️ 本地缺 $f，跳过"; continue; }
  case "$f" in
    decks/*) dst="$R/decks/$(basename "$f")" ;;
    docs/*)  dst="$R/docs/$(basename "$f")" ;;
    *)       dst="$R/$(basename "$f")" ;;
  esac
  if cat "$f" | $SSH "mkdir -p \$(dirname $dst) && cat > $dst" ; then
    echo "  ✓ $f → $dst"
  else
    echo "  ✗ $f 推送失败"
  fi
done
$SSH "chmod +x $R/tools/*.sh 2>/dev/null || true"

echo "=== 2) 权重 .bin（ph3 三个 + port_w 校验）==="
for b in ph3_2617_v1 ph3_2617_v2 ph3_9d6e_v2; do
  src="models/weights/ph3/$b.bin"
  [ -f "$src" ] || { echo "  ⚠️ 本地缺 $src"; continue; }
  md5s=$(md5sum "$src" | cut -d' ' -f1)
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" "$src" "$HOST:/root/kl/$b.bin" && \
    far=$($SSH "md5sum /root/kl/$b.bin | cut -d' ' -f1") && \
    { [ "$md5s" = "$far" ] && echo "  ✓ /root/kl/$b.bin (md5 校验通过)" || echo "  ✗ /root/kl/$b.bin md5 不一致"; }
done

echo "=== 3) 远端自检 ==="
$SSH "cd $R && ls -l tools/teacher_loop.sh tools/choice_to_banks.py decks/weights_field_b05.txt docs/the-pockmon-battle-choice.csv 2>&1 | awk '{print \"  \", \$5, \$9}'; ls /root/kl/*.bin | tr '\n' ' '; echo"
echo
echo "✓ 推送完成。GPU 机上执行： bash tools/launch_ab.sh   （或先 --dry 看命令）"
