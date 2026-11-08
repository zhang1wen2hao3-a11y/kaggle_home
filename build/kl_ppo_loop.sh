#!/usr/bin/env bash
# **PPO + 权重回拉**（KL-to-init 的零-CUDA 近似）—— 把 BC 克隆练成"会赢的真人代表"。
#
# 为什么这么做：真 KL-to-init 要改 CUDA 损失（引擎有 frozen-bank 前向，可行但要在 GPU 上
# 编译调试）。这里用**多段短训 + 每段把权重往 BC 初始拉回 α** 达到同样效果：
#
#     W ← (1-α)·W_ppo + α·W_bc_init
#
# α 越大越钉在"真人风格"上；每段只训 STEPS 步，避免一次跑远。
#
#   bash kl_ppo_loop.sh <学习者卡组> <BC初始.bin> <对手克隆.bin> <对手卡组> [轮数] [每轮步数] [α]
#
# 例：让 26c6 克隆（miwa 风格）对着 3eaa 克隆（Dipam 风格）练，复现真实 0.93
#   bash kl_ppo_loop.sh slowpoke_slowking__26c6 /root/kl/bc_26c6.bin /root/kl/bc_3eaa.bin \
#        dreepy_drakloak__3eaa 16 25000 0.15
set -uo pipefail
ANCHOR="$1"; INIT_BIN="$2"; OPP_BIN="$3"; OPP_DECK="$4"
ROUNDS="${5:-4}"; STEPS="${6:-100000}"; ALPHA="${7:-0.25}"
# ★ T 必须 ≥ 训练器的一个 epoch（32768 步），否则 **产出的权重全是 NaN**
#   （实测：T=25000/500 → 3,234,061 个 NaN 参数；T=40000 → 0 个）
if [ "$STEPS" -lt 32768 ]; then
  echo "✗ STEPS=$STEPS < 32768（一个 epoch）⇒ 训练器会产出全 NaN 权重。改成 ≥ 32768"
  exit 2
fi
LEAGUE_PCT="${LEAGUE_PCT:-0.45}"

R=/root/ptcg_train_kit/port_repo
OUT=/root/autodl-tmp/kl
LOG=/root/autodl-tmp/kl_logs
mkdir -p "$OUT" "$LOG"
PY=/root/miniconda3/bin/python3
KL_PY="${KL_PY:-/root/kl/tools/kl_blend.py}"
BUNDLE="${BUNDLE:-/root/bc2626/build/port_plain}"

echo "===== KL-PPO 循环 $(date +%H:%M:%S)"
echo "  学习者卡组 = $ANCHOR"
echo "  BC 初始    = $INIT_BIN"
echo "  对手克隆   = $OPP_BIN @ $OPP_DECK （league_pct=$LEAGUE_PCT）"
echo "  $ROUNDS 轮 × $STEPS 步，每轮回拉 α=$ALPHA"

CUR="$INIT_BIN"
for i in $(seq 1 "$ROUNDS"); do
  RUN="kl_r${i}"
  echo "===== 第 $i 轮 start $(date +%H:%M:%S)  init=$(basename "$CUR")"
  ( cd "$R" && env PYTHON="$PY" INIT="$CUR" OUT="$OUT" RUN="$RUN" T="$STEPS" \
      THREADS=8 LEAGUE_PCT="$LEAGUE_PCT" SEED=$((424242 + i)) \
      ANCHOR="$ANCHOR" TRACK="$OPP_DECK,$ANCHOR" DWF=decks/weights_real.txt \
      BANKS="$OPP_BIN" BANKDECKS="$OPP_DECK" \
      bash tools/db04_line.sh > "$LOG/$RUN.log" 2>&1 )
  sz=$(stat -c%s "$OUT/$RUN.bin" 2>/dev/null || echo 0)
  if [ "$sz" != "18205408" ]; then
    echo "✗ 第 $i 轮产物异常（size=$sz），停止"
    tail -20 "$LOG/$RUN.log"
    exit 1
  fi
  # 回拉：W ← (1-α)·本轮 + α·BC 初始
  PYTHONPATH="$R" PTCG_NW="$R/native/tools/native_weights.py" PTCG_BUNDLE="$BUNDLE" "$PY" "$KL_PY" \
      --new-bin "$OUT/$RUN.bin" --init-bin "$INIT_BIN" --alpha "$ALPHA" \
      --out-bin "$OUT/${RUN}_blend.bin" > "$LOG/$RUN.blend.log" 2>&1 \
    || { echo "✗ 回拉失败"; tail -10 "$LOG/$RUN.blend.log"; exit 1; }
  CUR="$OUT/${RUN}_blend.bin"
  echo "===== 第 $i 轮 done $(date +%H:%M:%S)  → $CUR"
done

cp -f "$CUR" "$OUT/final_kl.bin"
echo "===== 完成 $(date +%H:%M:%S)  最终 $OUT/final_kl.bin"
echo "   回本地： scp <host>:$OUT/final_kl.bin ./models/ppo/"
