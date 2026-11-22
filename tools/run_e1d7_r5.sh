#!/usr/bin/env bash
# **e1d7_r5：第一种"偏向性训练"实验（形状奖励）**（2026-11-22）
#
# 背景：同一对位（e1d7 打 7e39）对比 1594 分真人，我们的驾驶员系统性"更慢"：
#   贴能 75%（19.2 vs 25.7）、攻击 78%（6.8 vs 8.7）、撤退 173%（5.2 vs 3.0）。
# 这一轮在**纯胜负奖励之外**加三项形状奖励，看能不能把节奏拉回人类水平。
#
# ★ 开关设计（用户 2026-11-22 要求）：普通训练与偏向训练**共用一个二进制**，
#   只靠环境变量切换，默认全 0 = 与以前完全一致：
#     SHAPE_ATTACH=0.01 SHAPE_ATTACK=0.01 SHAPE_RETREAT=-0.02   # 偏向模式
#     （不设 = 普通模式）
#
# 判据（预注册）：① 体检表三项转绿（贴能 ≥85%、攻击 ≥90%、撤退 ≤130% 的人类基线）；
#                ② v2 面板 ≥47%（起点 candE1D7R4 = 43.2%）。
set -uo pipefail
cd /root/ptcg_train_kit/port_repo || { echo "✗ 不在 GPU 套件目录"; exit 1; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PTCG_REPO=/root/ptcg_train_kit/port_repo PYTHON=/root/miniconda3/bin/python3

NEED="/root/candE1D7R4.bin /root/cand9D6ER4.bin /root/cand3EAAR1.bin /root/candf5f8E.bin /root/ph3_2617_v2.bin /root/cand7e39S.bin"
MISS=""; for f in $NEED; do [ -s "$f" ] || MISS="$MISS $f"; done
if [ -n "$MISS" ]; then echo "✗ 缺权重：$MISS" >&2; exit 1; fi

export SHAPE_ATTACH="${SHAPE_ATTACH:-0.01}"
export SHAPE_ATTACK="${SHAPE_ATTACK:-0.01}"
export SHAPE_RETREAT="${SHAPE_RETREAT:--0.02}"
echo "形状奖励：贴能 +$SHAPE_ATTACH / 攻击 +$SHAPE_ATTACK / 撤退 $SHAPE_RETREAT"

export ANCHOR=z_ogerpon_toolbox_miwa__e1d7
export INIT="${INIT:-/root/candE1D7R4.bin}"
export BANKS=/root/cand9D6ER4.bin,/root/cand3EAAR1.bin,/root/candf5f8E.bin,/root/ph3_2617_v2.bin,/root/cand7e39S.bin,/root/cand9D6ER3.bin
export BANKDECKS=dunsparce_abra__9d6e,dreepy_drakloak__3eaa,abra_kadabra__f5f8,grookey_thwackey__2617,teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__9d6e
export LEAGUE_PCT=0.08
export TRACK=z_ogerpon_toolbox_miwa__e1d7,teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__9d6e,dreepy_drakloak__3eaa
export T="${T:-4000000}" LR="${LR:-1e-4}" OUT="${OUT:-/root/e1d7_r5}" RUN="${RUN:-e1d7_r5}" CG=1 CHECKPOINT_INTERVAL=16
export PYTHONUNBUFFERED=1
bash tools/db04_line.sh
