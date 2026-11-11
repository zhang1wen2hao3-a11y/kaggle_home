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
# ★ 动态切换（默认开）：A 的覆盖率加权对位胜率涨 THRESH_PP 就杀 A、切 B；
#   B 的胜率回到 baseline 附近就杀 B、结束本轮。设 DYN=0 退回固定步数。
# ★ 训练模式
#   MODE=ab     —— 纯交替（默认，对齐参考工程 two-net finetune）：
#                  A 训时对手 = 当前 B（打覆盖率牌池），100% 行都是冻结对手
#                  B 训时对手 = 当前 A（绑舰牌），同样 100%
#                  两侧交替，某一侧涨 THRESH_PP 就换边；跑满 ROUNDS 轮停止。
#   MODE=matrix —— 旧的混合方案：对手 = choice 矩阵驾驶员（覆盖率分槽）+ 51% 自对弈。
MODE="${MODE:-ab}"
# ★ 多轮时把"上一轮的增强模型"带进新一轮：
#   A 阶段：矩阵驾驶员之外，额外挂 B 的当前权重若干槽（不钉牌 → 按覆盖率抽牌）
#   B 阶段：对手 A 用**当前 A**（第 1 轮就是刚训完的 A），不再固定 init
B_SLOTS="${B_SLOTS:-2}"        # A 阶段里 B 占几个 bank 槽
B_OPP="${B_OPP:-A}"            # B 阶段的对手用 A 还是 init
DYN="${DYN:-1}"
THRESH_PP="${THRESH_PP:-3.0}"   # 对齐参考工程 cross-gate-delta 0.03
# 权重快照间隔（epoch；1 epoch = 32768 步）：
#   DYN 模式杀训练会丢"最后一个未落盘的 epoch 段"，所以默认加密到 20 epoch（655k 步）；
#   固定步数模式默认 100 epoch（3.28M 步）。每个快照都能用 tools/weight_dist.py 量位移。
if [ "$DYN" = 1 ]; then _DEF_CKPT=20; else _DEF_CKPT=100; fi
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-$_DEF_CKPT}"

if [ "$STEPS" -lt 32768 ]; then echo "✗ STEPS < 32768 会全 NaN"; exit 2; fi

R="${PTCG_REPO:-/root/ptcg_train_kit/port_repo}"
OUT="${OUT:-/root/autodl-tmp/teacher_loop}"; mkdir -p "$OUT"
LOG="$OUT/logs"; mkdir -p "$LOG"
PY="${PYTHON:-/root/miniconda3/bin/python3}"
KL_PY="${KL_PY:-$R/tools/kl_blend.py}"
BUNDLE="${BUNDLE:-/root/bc2626/build/port_plain}"
SHIP_CSV="${SHIP_CSV:-$R/decks/pool_field/$SHIP.csv}"

A="$INIT_A"; B="$INIT_B"; TEACHER="$INIT_A"

# ── A 阶段的对手驾驶员：查 choice 矩阵（对手牌 → 最强驾驶员），按覆盖率分配 bank 槽位 ──
#    引擎硬限制：league_pct * n_banks < 0.5 ⇒ 指定驾驶员最多覆盖 49% 的对局，
#    其余 51% 仍是自对弈（learner 自己打覆盖率抽的牌）。
SHIP_HASH="${SHIP##*__}"
SHIP_HASH="${SHIP_HASH:0:4}"
PLAN_BANKS=""; PLAN_DECKS=""; PLAN_PCT=""; PLAN_WEIGHTS=""
if [ -f tools/choice_to_banks.py ] && [ -f docs/the-pockmon-battle-choice.csv ]; then
  PLAN=$(./tools/py tools/choice_to_banks.py --a-deck "$SHIP_HASH" --emit-env 2>/dev/null || true)
  if [ -n "$PLAN" ]; then
    eval "$PLAN"
    PLAN_BANKS="$BANKS"; PLAN_DECKS="$BANKDECKS"; PLAN_PCT="$LEAGUE_PCT"
    PLAN_WEIGHTS="${BANK_WEIGHTS:-}"
  fi
fi
if [ -z "$PLAN_BANKS" ]; then
  echo "⚠️ choice 取不到 $SHIP_HASH 这一行，A 阶段退回单 bank（旧行为）"
  PLAN_BANKS="$B"; PLAN_DECKS=""; PLAN_PCT="0.49"
fi
echo "A 阶段对手驾驶员 bank: $(echo "$PLAN_BANKS" | tr ',' '\n' | wc -l) 个槽位, league_pct=$PLAN_PCT"

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
  A_NEW=""; B_NEW=""      # ★ 每轮必须重置，否则上一轮的值会漏到这一轮（曾导致 B 阶段对手停在旧 A）
  # ── ① 训 A：learner=A 打舰牌；对手 = 矩阵驾驶员 + 当前 B（多轮时才挂 B）──
  if [ "$MODE" = "ab" ]; then
    A_BANKS="$B"; A_DECKS=""; A_PCT="1.0"; A_WEIGHTS="1.0"
    echo "  [round $i] MODE=ab：A 的对手 = 当前 B（$(basename "$B")），100% 行，B 按覆盖率抽牌"
  else
  A_BANKS="$PLAN_BANKS"; A_DECKS="$PLAN_DECKS"; A_WEIGHTS="$PLAN_WEIGHTS"
  N_MATRIX=0
  [ -n "$PLAN_BANKS" ] && N_MATRIX=$(echo "$PLAN_BANKS" | tr ',' '\n' | wc -l)
  if [ "$i" -ge 2 ] || [ "$(md5sum "$B" 2>/dev/null | cut -d' ' -f1)" != "$(md5sum "$INIT_B" 2>/dev/null | cut -d' ' -f1)" ]; then
    extra_banks=""; extra_decks=""; extra_w=""
    for _k in $(seq 1 "$B_SLOTS"); do
      extra_banks="${extra_banks:+$extra_banks,}$B"
      extra_decks="${extra_decks:+$extra_decks,}-"      # "-" = 不钉牌，按覆盖率先验抽牌
      extra_w="${extra_w:+$extra_w,}1.0"               # B 打全池，权重记 1.0
    done
    A_BANKS="${A_BANKS:+$A_BANKS,}$extra_banks"
    A_DECKS="${A_DECKS:+$A_DECKS,}$extra_decks"
    A_WEIGHTS="${A_WEIGHTS:+$A_WEIGHTS,}$extra_w"
    N_TOTAL=$((N_MATRIX + B_SLOTS))
    A_PCT=$(awk -v q=0.49 -v n="$N_TOTAL" 'BEGIN{printf "%.5f", q/n}')
    echo "  [round $i] A 阶段对手 = $N_MATRIX 个矩阵槽 + $B_SLOTS 个当前B槽（league_pct=$A_PCT）"
  else
    A_PCT="$PLAN_PCT"
    echo "  [round $i] A 阶段对手 = $N_MATRIX 个矩阵槽（B 还是 init，不额外挂）"
  fi
  fi
  echo "===== 第 $i 轮 A：训舰牌专精（对手=B 冻结，打所有牌）====="
  cd "$R"
  TRAIN_A="env PYTHON=$PY INIT=$A OUT=$OUT RUN=r${i}A T=$STEPS THREADS=8 \
    LEAGUE_PCT=${A_PCT:-$PLAN_PCT} SEED=$((424242 + i)) ANCHOR=$SHIP TRACK=$SHIP \
    DWF=decks/weights_field_b05.txt BANKS=$A_BANKS BANKDECKS=$A_DECKS \
    CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL \
    bash tools/db04_line.sh"
  if [ "$DRY" = 1 ]; then
    echo "  $TRAIN_A"
  elif [ "$DYN" = 1 ]; then
    # 后台训 + 动态触发器：涨 THRESH_PP 就杀掉训练
    echo "  [DYN] 启动 A 训练并挂触发器（阈值 +${THRESH_PP}pp）"
    eval "$TRAIN_A" > "$LOG/r${i}A.log" 2>&1 &
    TRAIN_PID=$!
    PY_WATCH="${PY_WATCH:-}"; [ -z "$PY_WATCH" ] && { [ -x ./tools/py ] && PY_WATCH=./tools/py || PY_WATCH="$PY"; }
    $PY_WATCH tools/ab_watch.py --log "$R/logs/r${i}A.log" --mode A \
        --weights "${A_WEIGHTS:-$PLAN_WEIGHTS}" --thresh-pp "$THRESH_PP" \
        --kill-pattern "run-name r${i}A" >> "$LOG/r${i}A.watch.log" 2>&1
    wait "$TRAIN_PID" 2>/dev/null || true
    # 被杀时没有 final.bin，取最新快照
    A_NEW=$(ls -t "$R/experiments/native-r${i}A/"*.bin 2>/dev/null | grep -v state | head -1)
    [ -n "$A_NEW" ] && cp -f "$A_NEW" "$OUT/r${i}A.bin" && A_NEW="$OUT/r${i}A.bin"
    echo "  [DYN] A 阶段结束于 $(basename "${A_NEW:-none}")"
  else
    eval "$TRAIN_A" > "$LOG/r${i}A.log" 2>&1 || { echo "  ✗ A 训练失败，看 $LOG/r${i}A.log"; exit 1; }
  fi
  A_NEW="${A_NEW:-$OUT/r${i}A.bin}"
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
  if [ "$B_OPP" = "init" ]; then B_OPP_BIN="$INIT_A"; else B_OPP_BIN="$A"; fi
  echo "===== 第 $i 轮 B：训通才（对手=A 用 $(basename "$B_OPP_BIN")，打舰牌）====="
  TRAIN_B="env PYTHON=$PY INIT=$B OUT=$OUT RUN=r${i}B T=$STEPS THREADS=8 \
    LEAGUE_PCT=$([ "$MODE" = "ab" ] && echo 1.0 || echo 0.49) SEED=$((424242 + i + 1000)) ANCHOR=$SHIP TRACK=$SHIP \
    DWF=decks/weights_field_b05.txt BANKS=$B_OPP_BIN BANKDECKS=$SHIP \
    P_ANCHOR_LEARNER=0.0 CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL \
    bash tools/db04_line.sh"
  if [ "$DRY" = 1 ]; then
    echo "  $TRAIN_B"
  elif [ "$DYN" = 1 ]; then
    echo "  [DYN] 启动 B 训练并挂触发器（B 追回失分即停）"
    eval "$TRAIN_B" > "$LOG/r${i}B.log" 2>&1 &
    TRAIN_PID=$!
    PY_WATCH="${PY_WATCH:-}"; [ -z "$PY_WATCH" ] && { [ -x ./tools/py ] && PY_WATCH=./tools/py || PY_WATCH="$PY"; }
    $PY_WATCH tools/ab_watch.py --log "$R/logs/r${i}B.log" --mode B \
        --thresh-pp "$THRESH_PP" --kill-pattern "run-name r${i}B" >> "$LOG/r${i}B.watch.log" 2>&1
    wait "$TRAIN_PID" 2>/dev/null || true
    B_NEW=$(ls -t "$R/experiments/native-r${i}B/"*.bin 2>/dev/null | grep -v state | head -1)
    [ -n "$B_NEW" ] && cp -f "$B_NEW" "$OUT/r${i}B.bin"
    echo "  [DYN] B 阶段结束于 $(basename "${B_NEW:-none}")"
  else
    eval "$TRAIN_B" > "$LOG/r${i}B.log" 2>&1 || { echo "  ✗ B 训练失败，看 $LOG/r${i}B.log"; exit 1; }
  fi
  B="$OUT/r${i}B.bin"
done

echo "===== 完成。最终 A（舰牌专精）= $A；teacher = $TEACHER"
echo "评估 A：拉回本地 → 建 agent → 12 格 meta_eval（与端口同机同种子对照）"
