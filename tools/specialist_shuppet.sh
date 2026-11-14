#!/usr/bin/env bash
# **shuppet 特攻**：把「强 shuppet 驾驶员」的权重提到 LEAGUE_PCT，微调 4M 步。
#
# 为什么需要这个脚本（evidence）：
#   * 我们所有 12 panel 里打 0d86 的格子都是 98~100%，它们的对手驾驶员全是 `init` 端口
#     —— 而端口驾驶 shuppet 是**塌缩**的（opponent_library §D.3③：0d86 端口 0.0025）。
#     ⇒ 权重再大也没有梯度：学习信号已经饱和。
#   * 换成强 shuppet 驾驶员（`g8_shuppet` 系 checkpoint）后，天梯/本地都是被压着打
#     （ag_g8_shuppet vs g6_db04 = 0.945）。
#   ⇒ 特攻的正确定义：**强驾驶员 × 高权重**，而不是"0d86 这个卡组 × 高权重"。
#
# 用法（GPU 机，cd 到训练仓库后）：
#   ANCHOR=abra_kadabra__f5f8 bash tools/specialist_shuppet.sh          # f5f8 特攻（默认）
#   ANCHOR=dunsparce_abra__9d6e LEAGUE_PCT=1.0 bash tools/specialist_shuppet.sh
#   BANKS_ONLY_SHUP=1 ANCHOR=dunsparce_abra__db04 bash tools/specialist_shuppet.sh   # 纯 shuppet 对手
#
# 预注册判据（跑完照这个判，不事后改）：
#   B0 = 训练前候选 vs `g8_shuppet`@0d86 的胜率（3 seed 池化，n≥600/格）
#   B1 = 训练后同一格
#   胜 → B1 − B0 ≥ +15pp  **且** 12 panel 场权重分下降 ≤ 2pp  **且** 同牌镜像（对端口）≥ 0.45
#   否则算失败，不改判据。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PTCG_REPO="${PTCG_REPO:-/root/ptcg_train_kit/port_repo}"

ANCHOR="${ANCHOR:-abra_kadabra__f5f8}"
RUN="${RUN:-$(echo "$ANCHOR" | sed 's/.*__//')_shup}"
T="${T:-4000000}"                 # 4M 步（≈ 122 epoch）
LR="${LR:-2e-4}"
LEAGUE_PCT="${LEAGUE_PCT:-0.6}"    # 60% 的行给 shuppet 对手席
INIT="${INIT:-native/pt_model_field.bin}"
OUT="${OUT:-/root/$RUN}"
CKPT="${CHECKPOINT_INTERVAL:-50}"

SHUP_DECK="shuppet_dhelmise__0d86"
# 4 个「会打 shuppet」的 checkpoint（18,205,408 B，与 pt_model_field.bin 同架构 ⇒ 可当 bank）
SHUP_BANKS="${SHUP_BANKS:-/root/kl/g8_shuppet_bc.bin,/root/kl/g7b_shuppet.bin,/root/kl/g7a_db04_shuppet.bin,/root/kl/g12_db04_dreepy_shuppet.bin}"
GEN_BANKS="${GEN_BANKS:-/root/league_g0003/bank_0.bin,/root/league_g0003/bank_1.bin}"

if [ "${BANKS_ONLY_SHUP:-0}" = 1 ]; then
  BANKS="$SHUP_BANKS"; BANKDECKS="$SHUP_DECK,$SHUP_DECK,$SHUP_DECK,$SHUP_DECK"
else
  # 4 shuppet + 2 通用 = 6 bank（>8 会触发 cudaGraphInstantiate 失败，见 tools/db04_line.sh 注释）
  BANKS="$SHUP_BANKS,$GEN_BANKS"
  BANKDECKS="$SHUP_DECK,$SHUP_DECK,$SHUP_DECK,$SHUP_DECK,-,-"
fi

cat <<EOF
=========== shuppet 特攻 ===========
学习者卡组 ANCHOR : $ANCHOR
OUT               : $OUT
INIT              : $INIT
步数 T            : $T   (LR=$LR, snapshot every $CKPT epoch)
LEAGUE_PCT        : $LEAGUE_PCT
BANKS             : $BANKS
BANKDECKS         : $BANKDECKS
-----------------------------------
预注册判据（需 GPU 跑完后在 CPU 机核对）：
  1) vs g8_shuppet@0d86：B0 → B1，要求 **B1 − B0 ≥ +15pp**（3 seed 池化，n≥600）
  2) 12 panel 场权重分：下降 **≤ 2pp**
  3) 同牌镜像（对端口驱动同卡组）：**≥ 0.45**
  不满足任意一条 ⇒ 该 run 算失败，不得直接提交。
====================================
EOF
[ "${CHECK:-1}" = 1 ] || { echo "（CHECK=0，提前退出）"; exit 0; }

ANCHOR="$ANCHOR" TRACK="$ANCHOR,shuppet_dhelmise__0d86,slowpoke_slowking__26c6" \
BANKS="$BANKS" BANKDECKS="$BANKDECKS" LEAGUE_PCT="$LEAGUE_PCT" \
T="$T" LR="$LR" INIT="$INIT" OUT="$OUT" CG=1 CHECKPOINT_INTERVAL="$CKPT" \
bash "$HERE/db04_line.sh"
