#!/usr/bin/env bash
# **S4：entropy bonus 调低到 0.002（对齐 12th place 的配方）**（2026-11-23）
#
# 背景：我们的训练器 `--ent-coef` 默认 **0.01**，而 pkmn-kaggle 的 12 名方案用的是 **0.002**（低 5 倍）。
#   他们用同样的纯 ±1 终局奖励跑出 1186 分；我们怀疑"熵太高 → 探索噪声大 →
#   rollout 里随机撤退变多、价值学习被摊薄"，从而学出"宁可撤退"的保守习惯。
#   另一侧假设是反的（熵太低 → 陷入保守局部最优），所以这是一次**方向性对照**：
#   若 S4 的撤退率下降 ⇒ 是"熵太高"的锅；若上升或不变 ⇒ 需要反方向（S5: ent 0.03）。
#
# 唯一变量：ENT_COEF=0.002      对照：candE1D7R4（ent 0.01，43.2% v2 面板）
# 注：推理侧**不加温度** —— 我们提交用的是 argmax（temp=0），加温度只会引入随机性，不是修复。
set -uo pipefail
cd /root/ptcg_train_kit/port_repo || { echo "✗ 不在 GPU 套件目录"; exit 1; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PTCG_REPO=/root/ptcg_train_kit/port_repo PYTHON=/root/miniconda3/bin/python3

NEED="/root/candE1D7R4.bin /root/cand9D6ER4.bin /root/cand3EAAR1.bin /root/candf5f8E.bin /root/ph3_2617_v2.bin /root/cand7e39S.bin"
MISS=""; for f in $NEED; do [ -s "$f" ] || MISS="$MISS $f"; done
if [ -n "$MISS" ]; then echo "✗ 缺权重：$MISS" >&2; exit 1; fi

export ENT_COEF="${ENT_COEF:-0.002}"
echo "S4：ENT_COEF=$ENT_COEF（对照 candE1D7R4 用默认 0.01）"

export ANCHOR=z_ogerpon_toolbox_miwa__e1d7
export INIT="${INIT:-/root/candE1D7R4.bin}"
export BANKS=/root/cand7e39S.bin,/root/cand9D6ER4.bin,/root/cand3EAAR1.bin,/root/candf5f8E.bin,/root/ph3_2617_v2.bin,/root/candE1D7R4.bin
export BANKDECKS=teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__9d6e,dreepy_drakloak__3eaa,abra_kadabra__f5f8,grookey_thwackey__2617,z_ogerpon_toolbox_miwa__e1d7
export LEAGUE_PCT=0.08
export TRACK=z_ogerpon_toolbox_miwa__e1d7,teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__9d6e
export T="${T:-2000000}" LR="${LR:-1e-4}" OUT="${OUT:-/root/e1d7_s4}" RUN="${RUN:-e1d7_s4}" CG=1 CHECKPOINT_INTERVAL=8
export PYTHONUNBUFFERED=1
bash tools/db04_line.sh
