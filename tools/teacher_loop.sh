#!/usr/bin/env bash
# **冻结 teacher + 门控晋升 + 两网轮训**（升级版 dual_kl_loop，对齐参考工程 two-net 结构）。
#
# 结构：
#   A = 舰牌专精，永远打 <ship_deck>（最终提交的就是 A）
#   B = 通才，打池子里所有牌（对手）
#   teacher = A 的冻结快照，两个用途：①KL 回拉目标 ②晋升 h2h 参照
#
# 每轮：
#   ① 训 A：learner=A 打舰牌；对手 = 冻结 B（打所有牌，league 主）+ 自对弈池
#   ② KL 回拉 A → teacher（α，零-CUDA 近似，防漂）
#   ③ 门控晋升：镜像 h2h A vs teacher ≥ B0+3pp 才把 teacher 换成 A（B0=起点镜像基线）
#   ④ 训 B：learner=B 打所有牌（P_ANCHOR_LEARNER=0）；对手 = 冻结 A 绑舰牌
#
# 用法（GPU 机，repo 根）：
#   bash tools/teacher_loop.sh <ship_deck> <initA.bin> <initB.bin> [轮数] [每步] [α] [--dry]
# 例：
#   bash tools/teacher_loop.sh grookey_thwackey__2617 /root/kl/port_w.bin /root/kl/port_w.bin 6 2000000 0.10
#
# ⚠️ v1 已知假设（上机先冒烟）：
#   - BANKDECKS 留空 = 冻结 bank 打池子所有牌（league-opp 抽 pool）
#   - P_ANCHOR_LEARNER=0 时 ANCHOR 只是占位（pool 解析用），learner 抽 pool
#   - teacher_gate.sh / eval_pair 在本机可用（GPU 机 96 核，直接本地跑 h2h）
set -uo pipefail

SHIP="$1"; INIT_A="$2"; INIT_B="$3"
ROUNDS="${4:-6}"; STEPS="${5:-2000000}"; ALPHA="${6:-0.10}"
DRY=0; for a in "$@"; do [ "$a" = "--dry" ] && DRY=1; done
PROMOTE_MARGIN="${PROMOTE_MARGIN:-0.03}"   # 相对镜像基线 B0 的晋升余量（3pp）
N_GATE="${N_GATE:-400}"

if [ "$STEPS" -lt 32768 ]; then echo "✗ STEPS < 32768 会全 NaN"; exit 2; fi

R="${PTCG_REPO:-/root/ptcg_train_kit/port_repo}"
OUT="${OUT:-/root/autodl-tmp/teacher_loop}"; mkdir -p "$OUT"
LOG="$OUT/logs"; mkdir -p "$LOG"
PY="${PYTHON:-/root/miniconda3/bin/python3}"
KL_PY="${KL_PY:-$R/tools/kl_blend.py}"
BUNDLE="${BUNDLE:-/root/bc2626/build/port_plain}"
SHIP_CSV="${SHIP_CSV:-$R/decks/pool_field/$SHIP.csv}"

A="$INIT_A"; B="$INIT_B"; TEACHER="$INIT_A"

# 镜像基线 B0：起点时 A vs teacher（同一个文件）先量一次座位偏置
gate() { # gate <cand.bin> <teacher.bin> -> 胜率(0~1)
  bash tools/teacher_gate.sh --cand "$1" --teacher "$2" --deck "$SHIP_CSV" --n "$N_GATE" --out "$OUT/gate" 2>/dev/null \
    | ./tools/py -c "import sys;print(sys.stdin.read())" 2>/dev/null \
    | grep -oE "[0-9]+\.[0-9]{4}" | head -1
}
B0=$(gate "$A" "$TEACHER"); B0=${B0:-0.5}
echo "镜像基线 B0 = $B0（晋升阈值 = $B0 + $PROMOTE_MARGIN）"

promote() { awk -v w="$1" -v b="$B0" -v m="$PROMOTE_MARGIN" 'BEGIN{ exit !(w+0 >= b+m) }'; }

for i in $(seq 1 "$ROUNDS"); do
  # ── ① 训 A：learner=A 打舰牌，对手=冻结 B 打所有牌 ──
  echo "===== 第 $i 轮 A：训舰牌专精（对手=B 冻结，打所有牌）====="
  cd "$R"
  TRAIN_A="env PYTHON=$PY INIT=$A OUT=$OUT RUN=r${i}A T=$STEPS THREADS=8 \
    LEAGUE_PCT=0.6 SEED=$((424242 + i)) ANCHOR=$SHIP TRACK=$SHIP \
    DWF=decks/weights_field_b05.txt BANKS=$B BANKDECKS= \
    bash tools/db04_line.sh"
  if [ "$DRY" = 1 ]; then echo "  $TRAIN_A"; else eval "$TRAIN_A" > "$LOG/r${i}A.log" 2>&1 || { echo "  ✗ A 训练失败，看 $LOG/r${i}A.log"; exit 1; }; fi
  A_NEW="$OUT/r${i}A.bin"
  # ② KL 回拉 A → teacher
  A_BLEND="$OUT/r${i}A_blend.bin"
  BLEND_A="env PYTHONPATH=$R PTCG_NW=$R/native/tools/native_weights.py PTCG_BUNDLE=$BUNDLE \
    $PY $KL_PY --new-bin $A_NEW --init-bin $TEACHER --alpha $ALPHA --out-bin $A_BLEND"
  if [ "$DRY" = 1 ]; then echo "  $BLEND_A"; else eval "$BLEND_A" > "$LOG/r${i}A_blend.log" 2>&1 || echo "  ⚠️ KL 回拉失败，继续"; fi
  [ -f "$A_BLEND" ] && A="$A_BLEND" || A="$A_NEW"
  # ③ 门控晋升
  W=$(gate "$A" "$TEACHER")
  echo "  A vs teacher 镜像胜率 = $W （阈值 $B0+$PROMOTE_MARGIN）"
  if [ -n "$W" ] && promote "$W"; then
    TEACHER="$A"; echo "  ✅ 晋升：teacher ← A"
  else
    echo "  · 未达晋升阈值，teacher 不变"
  fi

  # ── ④ 训 B：learner=B 打所有牌，对手=冻结 A 绑舰牌 ──
  echo "===== 第 $i 轮 B：训通才（对手=A 冻结，打舰牌）====="
  TRAIN_B="env PYTHON=$PY INIT=$B OUT=$OUT RUN=r${i}B T=$STEPS THREADS=8 \
    LEAGUE_PCT=0.6 SEED=$((424242 + i + 1000)) ANCHOR=$SHIP TRACK=$SHIP \
    DWF=decks/weights_field_b05.txt BANKS=$A BANKDECKS=$SHIP \
    P_ANCHOR_LEARNER=0.0 bash tools/db04_line.sh"
  if [ "$DRY" = 1 ]; then echo "  $TRAIN_B"; else eval "$TRAIN_B" > "$LOG/r${i}B.log" 2>&1 || { echo "  ✗ B 训练失败，看 $LOG/r${i}B.log"; exit 1; }; fi
  B="$OUT/r${i}B.bin"
done

echo "===== 完成。最终 A（舰牌专精）= $A；teacher = $TEACHER"
echo "评估 A：拉回本地 → 建 agent → 12 格 meta_eval（与端口同机同种子对照）"
