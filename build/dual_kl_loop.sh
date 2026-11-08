#!/usr/bin/env bash
# **双向换边 + 权重回拉**：两个克隆互相当对手，轮流学，直到两边都不涨。
#
# 复用（不重写）：
#   * db04_line.sh   —— 训练执行（已有的配方/预检/硬门槛）
#   * kl_blend.py    —— 每轮把权重往该侧 BC 初始拉回 α（KL 的零-CUDA 近似）
#   * 训练器自己的 `win_vs_league` —— 免费的"这一侧打对面打得怎样"，不用额外跑评估
#
# 轮次安排（A=26c6 侧，B=3eaa 侧）：
#   1: A 学，B 冻结      2: B 学，A(轮1产物) 冻结
#   3: A 学，B(轮2产物) 冻结   4: B 学，A(轮3产物) 冻结   ...
#
#   bash dual_kl_loop.sh <deckA> <initA.bin> <deckB> <initB.bin> [轮数] [每步] [α] [--dry]
#
# 例：
#   bash dual_kl_loop.sh slowpoke_slowking__26c6 /root/kl/bc_26c6.bin \
#        dreepy_drakloak__3eaa /root/kl/bc_3eaa.bin 6 100000 0.25
set -uo pipefail

DECK_A="$1"; INIT_A="$2"; DECK_B="$3"; INIT_B="$4"
ROUNDS="${5:-6}"; STEPS="${6:-100000}"; ALPHA="${7:-0.25}"
DRY=0; for a in "$@"; do [ "$a" = "--dry" ] && DRY=1; done
LEAGUE_PCT="${LEAGUE_PCT:-0.45}"
PATIENCE="${PATIENCE:-2}"          # 某侧连续几轮不涨就认为它停了

# ★ T 必须 ≥ 训练器一个 epoch（32768），否则产出全 NaN（实测 3,234,061 个 NaN 参数）
if [ "$STEPS" -lt 32768 ]; then
  echo "✗ STEPS=$STEPS < 32768 ⇒ 训练器产出全 NaN 权重。改成 ≥ 32768"; exit 2
fi

R=/root/ptcg_train_kit/port_repo
OUT=/root/autodl-tmp/dual_kl
LOG=/root/autodl-tmp/dual_kl_logs
mkdir -p "$OUT" "$LOG" 2>/dev/null || true
PY=/root/miniconda3/bin/python3
KL_PY="${KL_PY:-/root/kl/tools/kl_blend.py}"
BUNDLE="${BUNDLE:-/root/bc2626/build/port_plain}"

cur_a="$INIT_A"; cur_b="$INIT_B"
flat_a=0; flat_b=0; prev_a=""; prev_b=""

# 从训练器日志里取这一轮的"对冻结对手胜率"（免费指标，不用另跑评估）
win_vs_league() {  # win_vs_league <repo_log>
  grep -oE "lgw [-0-9.]+" "$1" 2>/dev/null | tail -1 | awk '{print $2}'
}
fails() {         # 浮点比较：a < b 且 b 有效
  awk -v a="$1" -v b="$2" 'BEGIN{ if (a=="" || b=="" || a=="-1.000") exit 1; exit !(a+0 < b+0) }'
}

echo "===== 双向换边 KL-PPO $(date +%H:%M:%S)"
echo "  A: $DECK_A   init=$(basename "$INIT_A")"
echo "  B: $DECK_B   init=$(basename "$INIT_B")"
echo "  $ROUNDS 轮 × $STEPS 步，每轮回拉 α=$ALPHA，league_pct=$LEAGUE_PCT，patience=$PATIENCE"
[ "$DRY" = 1 ] && echo "  ⚠️ DRY：只打印命令，不执行"

for i in $(seq 1 "$ROUNDS"); do
  if [ $((i % 2)) = 1 ]; then
    SIDE=A; L_DECK="$DECK_A"; L_INIT="$cur_a"; OPP_BIN="$cur_b"; OPP_DECK="$DECK_B"; ANCHOR_INIT="$INIT_A"
  else
    SIDE=B; L_DECK="$DECK_B"; L_INIT="$cur_b"; OPP_BIN="$cur_a"; OPP_DECK="$DECK_A"; ANCHOR_INIT="$INIT_B"
  fi
  RUN="dk_r${i}${SIDE}"
  echo "----- 第 $i 轮：$SIDE 学（$L_DECK），对手=$OPP_DECK  init=$(basename "$L_INIT")"

  TRAIN_CMD="cd $R && env PYTHON=$PY INIT=$L_INIT OUT=$OUT RUN=$RUN T=$STEPS THREADS=8 \
LEAGUE_PCT=$LEAGUE_PCT SEED=$((424242 + i)) ANCHOR=$L_DECK TRACK=$OPP_DECK,$L_DECK \
DWF=decks/weights_real.txt BANKS=$OPP_BIN BANKDECKS=$OPP_DECK bash tools/db04_line.sh"
  BLEND_CMD="env PYTHONPATH=$R PTCG_NW=$R/native/tools/native_weights.py PTCG_BUNDLE=$BUNDLE \
$PY $KL_PY --new-bin $OUT/$RUN.bin --init-bin $ANCHOR_INIT --alpha $ALPHA --out-bin $OUT/${RUN}_blend.bin"

  if [ "$DRY" = 1 ]; then
    echo "    train: $TRAIN_CMD"
    echo "    blend: $BLEND_CMD"
    [ "$SIDE" = A ] && cur_a="$OUT/${RUN}_blend.bin" || cur_b="$OUT/${RUN}_blend.bin"
    continue
  fi

  ( eval "$TRAIN_CMD" > "$LOG/$RUN.log" 2>&1 )
  sz=$(stat -c%s "$OUT/$RUN.bin" 2>/dev/null || echo 0)
  if [ "$sz" != "18205408" ]; then
    echo "    ✗ 训练产物异常 size=$sz"; tail -15 "$LOG/$RUN.log"; exit 1
  fi
  eval "$BLEND_CMD" > "$LOG/$RUN.blend.log" 2>&1 || {
    echo "    ✗ 回拉失败"; tail -10 "$LOG/$RUN.blend.log"; exit 1; }
  NPY=$(/root/miniconda3/bin/python3 -c "
import numpy as np; a=np.fromfile('$OUT/${RUN}_blend.bin',dtype=np.float32); print(int(np.isnan(a).sum()))" 2>/dev/null || echo 1)
  if [ "$NPY" != "0" ]; then
    echo "    ✗ 回拉后含 $NPY 个 NaN，停止"; exit 1
  fi

  W=$(win_vs_league "$R/logs/$RUN.log")
  echo "    对冻结对手胜率 lgw=$W  产物=$OUT/${RUN}_blend.bin  $(date +%H:%M:%S)"
  if [ "$SIDE" = A ]; then
    fails "$W" "$prev_a" && flat_a=$((flat_a+1)) || flat_a=0
    prev_a="$W"; cur_a="$OUT/${RUN}_blend.bin"
  else
    fails "$W" "$prev_b" && flat_b=$((flat_b+1)) || flat_b=0
    prev_b="$W"; cur_b="$OUT/${RUN}_blend.bin"
  fi
  echo "    flat: A=$flat_a B=$flat_b"
  if [ "$flat_a" -ge "$PATIENCE" ] && [ "$flat_b" -ge "$PATIENCE" ]; then
    echo "===== 🛑 两侧各连续 $PATIENCE 轮没涨（A lgw=$prev_a, B lgw=$prev_b）⇒ 停"
    break
  fi
done

[ "$DRY" = 1 ] && { echo "===== DRY 结束"; exit 0; }
cp -f "$cur_a" "$OUT/final_A.bin" 2>/dev/null
cp -f "$cur_b" "$OUT/final_B.bin" 2>/dev/null
echo "===== 完成 $(date +%H:%M:%S)"
echo "    A 侧：$OUT/final_A.bin      B 侧：$OUT/final_B.bin"
echo "    回本地：scp <host>:$OUT/final_A.bin models/ppo/dualA.bin ; 同法取 final_B.bin"
