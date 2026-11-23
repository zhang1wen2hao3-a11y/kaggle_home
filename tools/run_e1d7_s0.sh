#!/usr/bin/env bash
# **S0：GAE λ=1.0（窗口内纯 MC，去 TD 偏差）**（2026-11-23）
#
# 依据：Ng/Harada/Russell 1999 与 RUDDER 都指出"延迟奖励 + TD 目标会带来偏差与信息指数衰减"。
#   我们 γ 已是 1.0，但 λ=0.95 仍带 TD 尾巴，且 GAE 在 horizon=64 处被截断 ⇒ 一局 100~150 个
#   决策的终端信号传不回去 ⇒ 策略趋于保守（实测：贴能 −25%、攻击 −22%、撤退 +73%）。
#   S0 只改一个旋钮：λ 0.95 → 1.0，其余（起点/配方/预算/种子）与对照完全一致。
#
# 选 `e1d7` 的理由：**这是我们与 1400+ 真人在策略上差异最大的一副牌**
#   （对照 ミワハルキ 1594：贴能 75%、攻击 78%、撤退 173%，同一份 60 张牌、同一对手 7e39）
#   ⇒ 训练后能不能把这些比例拉回来，是本次实验的主指标（tools/style_gap.py 量化）。
#
# 唯一变量：GAE_LAMBDA=1.0     对照：candE1D7R4（λ=0.95，43.2% v2 面板）
set -uo pipefail
cd /root/ptcg_train_kit/port_repo || { echo "✗ 不在 GPU 套件目录"; exit 1; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PTCG_REPO=/root/ptcg_train_kit/port_repo PYTHON=/root/miniconda3/bin/python3

NEED="/root/candE1D7R4.bin /root/cand9D6ER4.bin /root/cand3EAAR1.bin /root/candf5f8E.bin /root/ph3_2617_v2.bin /root/cand7e39S.bin"
MISS=""; for f in $NEED; do [ -s "$f" ] || MISS="$MISS $f"; done
if [ -n "$MISS" ]; then echo "✗ 缺权重：$MISS" >&2; exit 1; fi

export GAE_LAMBDA="${GAE_LAMBDA:-1.0}"     # ★ 本实验唯一变量
echo "S0：GAE_LAMBDA=$GAE_LAMBDA（γ 默认 1.0）"

export ANCHOR=z_ogerpon_toolbox_miwa__e1d7
export INIT="${INIT:-/root/candE1D7R4.bin}"
export BANKS=/root/cand7e39S.bin,/root/cand9D6ER4.bin,/root/cand3EAAR1.bin,/root/candf5f8E.bin,/root/ph3_2617_v2.bin,/root/candE1D7R4.bin
export BANKDECKS=teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__9d6e,dreepy_drakloak__3eaa,abra_kadabra__f5f8,grookey_thwackey__2617,z_ogerpon_toolbox_miwa__e1d7
export LEAGUE_PCT=0.08
export TRACK=z_ogerpon_toolbox_miwa__e1d7,teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__9d6e
export T="${T:-2000000}" LR="${LR:-1e-4}" OUT="${OUT:-/root/e1d7_s0}" RUN="${RUN:-e1d7_s0}" CG=1 CHECKPOINT_INTERVAL=8
export PYTHONUNBUFFERED=1
bash tools/db04_line.sh
