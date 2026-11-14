#!/usr/bin/env bash
# **shuppet 特攻**：把「强 shuppet 驾驶员」的权重提到指定比例，微调 4M 步。
#
# 为什么需要（evidence）：
#   * 我们所有 12 panel 里打 0d86 的格子都是 98~100%，它们的对手驾驶员全是 `init` 端口
#     —— 而端口驾驶 shuppet 是**塌缩**的（opponent_library §D.3③：0d86 端口 0.0025）。
#     ⇒ 权重再大也没有梯度：学习信号已经饱和。
#   * 换成强 shuppet 驾驶员后立刻见真章：f5f8@端口 vs `g8_shuppet` 只有 11.5%；
#     而 f5f8 的 BC 克隆（`bc:deck_f5f8`）vs `g8_bc_parent` 有 87.5%（n=200）。
#   ⇒ 特攻的正确定义：**强驾驶员 × 高权重**，不是"0d86 这个卡组 × 高权重"。
#
# ★ bank 语义（别搞混）：
#   · `BANKDECKS` 里写了卡组名 = **绑定**：这个 bank 永远打那副牌（身份是它是谁）
#   · `BANKDECKS` 里写 `-`      = **不绑定**：这个 bank 打的是按 `decks/weights_real.txt`
#     （187 副真实野区分布）**抽出来的卡组** —— 这才叫"通用 bank"
#   · "每副牌配最强驾驶员"是另一件事，由 `tools/choice_to_banks.py --strong` 生成
#     （bank 与卡组一一绑定，且 bank 权重就是该牌的最佳驾驶员）
#   · `--league-pct` 是**每个 bank** 的行占比，约束是 `pct × bank 数 ≤ 1.0`
#     ⇒ 本脚本用 `LEAGUE_TOTAL`（冻结对手总占比，默认 0.5）反算，禁止直接填 0.6 这种数
#
# 用法（GPU 机，cd 到训练仓库后）：
#   ANCHOR=abra_kadabra__f5f8 INIT=/root/bc_f5f8/deck_f5f8_.bin bash tools/specialist_shuppet.sh
#   LEAGUE_TOTAL=1.0 ... bash tools/specialist_shuppet.sh      # 纯交替模式（无自对弈信号，慎用）
#   BANKS_ONLY_SHUP=1 ... bash tools/specialist_shuppet.sh     # 6 bank → 4 bank
#
# 预注册判据（跑完照这个判，不事后改）：
#   B0 = 训练前候选 vs `g8_bc_parent`@0d86（**持有型考官，故意不进 bank**）
#   B1 = 训练后同一格（3 seed 池化，n≥600）
#   胜 → B1 − B0 ≥ +5pp  **且** 12 panel 场权重分下降 ≤ 2pp  **且** 镜像（对端口）≥ 0.55
#   否则算失败 ⇒ 只登记成对手 bank，不得当候选驾驶员提交。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PTCG_REPO="${PTCG_REPO:-/root/ptcg_train_kit/port_repo}"

ANCHOR="${ANCHOR:-abra_kadabra__f5f8}"
RUN="${RUN:-$(echo "$ANCHOR" | sed 's/.*__//')_shup}"
T="${T:-4000000}"                   # 4M 步（≈122 epoch）
LR="${LR:-2e-4}"
LEAGUE_TOTAL="${LEAGUE_TOTAL:-0.5}" # 冻结对手占总行比例（>0.5 变纯交替模式）
INIT="${INIT:-native/pt_model_field.bin}"
OUT="${OUT:-/root/$RUN}"
CKPT="${CHECKPOINT_INTERVAL:-50}"
DWF="${DWF:-decks/weights_real.txt}" # 通用 bank 的抽牌分布（187 副真实野区）

SHUP_DECK="shuppet_dhelmise__0d86"
# 4 个「会打 shuppet」的 checkpoint（18,205,408 B，与 pt_model_field.bin 同架构 ⇒ 可当 bank）
SHUP_BANKS="${SHUP_BANKS:-/root/kl/g8_shuppet_bc.bin,/root/kl/g7b_shuppet.bin,/root/kl/g7a_db04_shuppet.bin,/root/kl/g12_db04_dreepy_shuppet.bin}"
# 2 个通用 bank：不绑卡组，按 DWF 抽牌（= 早先联赛世代的通用对手）
GEN_BANKS="${GEN_BANKS:-/root/league_g0003/bank_0.bin,/root/league_g0003/bank_1.bin}"
# 可选：额外的**绑定** bank（例：slowking 26c6 用 r1A —— 顶分段 54.6% 的对手，
# 但默认 6 bank 里一个都没有；填上它能把"打 slowking"也纳入训练）
EXTRA_BANKS="${EXTRA_BANKS:-}"      # 例：/root/kl/r1A_slowking.bin
EXTRA_DECKS="${EXTRA_DECKS:-}"      # 与 EXTRA_BANKS 逐位对应：slowpoke_slowking__26c6

if [ "${BANKS_ONLY_SHUP:-0}" = 1 ]; then
  BANKS="$SHUP_BANKS"; BANKDECKS="$SHUP_DECK,$SHUP_DECK,$SHUP_DECK,$SHUP_DECK"
  N_SHUP=4; N_GEN=0
else
  BANKS="$SHUP_BANKS,$GEN_BANKS"
  BANKDECKS="$SHUP_DECK,$SHUP_DECK,$SHUP_DECK,$SHUP_DECK,-,-"
  N_SHUP=4; N_GEN=2
fi
N_EXTRA=0
if [ -n "$EXTRA_BANKS" ]; then
  BANKS="$BANKS,$EXTRA_BANKS"
  BANKDECKS="$BANKDECKS,$EXTRA_DECKS"
  N_EXTRA=$(awk -F, '{print NF}' <<< "$EXTRA_BANKS")
fi
N=$((N_SHUP + N_GEN + N_EXTRA))
[ "$N" -le 8 ] || { echo "✗ bank 数 $N > 8：2080 Ti 上 cudaGraphInstantiate 会失败（见 db04_line.sh 注释）"; exit 2; }
# pct × bank 数 ≤ 1.0（trainer 会断言）；这里反算并取 4 位小数向下取
LEAGUE_PCT=$(awk -v l="$LEAGUE_TOTAL" -v n="$N" 'BEGIN{printf "%.4f", l/n}')
SHUP_SHARE=$(awk -v p="$LEAGUE_PCT" -v s="$N_SHUP" 'BEGIN{printf "%.1f", p*s*100}')
TOTAL_SHARE=$(awk -v p="$LEAGUE_PCT" -v n="$N" 'BEGIN{printf "%.1f", p*n*100}')
FAIL=$(awk -v p="$LEAGUE_PCT" -v n="$N" 'BEGIN{print (p*n > 1.0 + 1e-9) ? 1 : 0}')
[ "$FAIL" = 0 ] || { echo "✗ LEAGUE_TOTAL=$LEAGUE_TOTAL 太大：pct×bank 数=$TOTAL_SHARE% > 100%"; exit 2; }

cat <<EOF
=========== shuppet 特攻 ===========
学习者卡组 ANCHOR : $ANCHOR
OUT               : $OUT
INIT              : $INIT
步数 T            : $T   (LR=$LR, snapshot every $CKPT epoch)
bank              : $N 个（$N_SHUP 个绑 $SHUP_DECK + $N_GEN 个不绑/通用 + $N_EXTRA 个额外绑定）
额外绑定           : ${EXTRA_DECKS:-无}
通用 bank 抽牌分布 : $DWF
--league-pct      : $LEAGUE_PCT / bank   ⇒ 冻结对手占 $TOTAL_SHARE% 行，
                    其中 **vs shuppet 占 $SHUP_SHARE%** 行
-----------------------------------
预注册判据：
  1) vs g8_bc_parent@0d86（持有型，不在 bank 里）：B0 → B1，要求 **≥ +5pp**（3 seed 池化 n≥600）
  2) 12 panel 场权重分：下降 **≤ 2pp**
  3) 镜像（对 port@同牌）：**≥ 0.55**
  不满足任意一条 ⇒ 算失败，只登记成对手 bank，不得直接提交。
====================================
EOF
[ "${CHECK:-1}" = 1 ] || { echo "（CHECK=0，提前退出）"; exit 0; }

ANCHOR="$ANCHOR" TRACK="$ANCHOR,shuppet_dhelmise__0d86,slowpoke_slowking__26c6" \
BANKS="$BANKS" BANKDECKS="$BANKDECKS" LEAGUE_PCT="$LEAGUE_PCT" \
T="$T" LR="$LR" INIT="$INIT" OUT="$OUT" CG=1 CHECKPOINT_INTERVAL="$CKPT" DWF="$DWF" \
bash "$HERE/db04_line.sh"
