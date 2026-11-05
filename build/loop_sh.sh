#!/usr/bin/env bash
# **无人值守循环**：训练 → 评估 → 判定 → 换边/继续 → 直到两侧都停摆
#
#   bash loop.sh <state名> [最多轮数]
#   bash loop.sh <state名> --dry        # 自测：不训练，复制 init 当产物
#
# 前置：state 已建好（tools/dual_train.py init ...），训练仓库已 setup 过
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
NAME="${1:?用法: bash loop.sh <state名> [轮数] [--dry]}"; shift || true
MAX=8; DRY=0
for a in "$@"; do case "$a" in --dry) DRY=1;; [0-9]*) MAX="$a";; esac; done
PY=${PYTHON:-python3}
STUB=""; [ "$DRY" = 1 ] && STUB="--stub"

# ★ 单实例锁：两个循环同时跑同一个 state 会各自推进 round、互相覆盖，
#   state 直接花掉（实测过一次误留的旧循环 + 新循环并发，同一轮出现两个分数）。
mkdir -p logs
LOCK="logs/loop_${NAME}.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "✗ 已有循环在跑 $NAME（锁 $LOCK）"
  echo "   看它是否还活着: ps -o pid,etime,args -p \$(cat logs/loop_${NAME}.pid 2>/dev/null)"
  echo "   确认已死后删锁: rm -f $LOCK"
  exit 1
fi
echo "$$" > "logs/loop_${NAME}.pid"

echo "===== 循环开始 $NAME 最多 $MAX 轮 $([ $DRY = 1 ] && echo '（DRY 模式）') ====="
for i in $(seq 1 "$MAX"); do
  echo
  echo "───────── 第 $i 轮 ─────────"
  if $PY tools/dual_train.py stop --names "$NAME" >/dev/null 2>&1; then
    echo "已满足停止条件，循环结束"; break
  fi
  # ① 训练
  if ! $PY tools/dual_train.py train --names "$NAME" --repo port_repo \
        --out "$HERE/runs" --dwf decks/weights_real.txt $STUB; then
    echo "✗ 训练失败，循环中止"; exit 1
  fi
  # ② 取本轮产物路径（train 已复制到 models/ppo/）
  BIN=$($PY - "$NAME" <<'PY'
import json,sys,os
st=json.load(open(f"ptcg_league/state/dual_{sys.argv[1]}.json"))
side=st["turn"]; rnd=st["round"]+1
print(f"models/ppo/{sys.argv[1]}_r{rnd}{side}.bin")
PY
)
  echo "[loop] 产物 = $BIN"
  # ③ 评估 + 判定
  if ! $PY tools/dual_train.py collect --names "$NAME" --bin "$BIN" \
        ; then
    echo "✗ 评估失败，循环中止"; exit 1
  fi
done

echo
echo "===== 结束状态 ====="
$PY tools/dual_train.py status --names "$NAME"
