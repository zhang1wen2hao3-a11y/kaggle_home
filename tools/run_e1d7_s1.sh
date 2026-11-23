#!/usr/bin/env bash
# **S1：奖品稠密奖励（reward_win=0）**（2026-11-23）
#
# 依据：`env.c` 里早就实现了"每取一张奖励卡/击倒一只就给 ∝ 差值 的奖励"，等价于
#   Φ(s)=双方奖励卡差 的**势函数差分**（ΣF  telescopes 到 Φ(s_T)−Φ(s_0)，终端清零）
#   ⇒ 按 Ng et al. 1999，这是**策略不变**的稠密化，正是"稀疏奖励→保守"的最直接解法。
#   **我们从来没跑过这个模式**（`reward_win` 一直硬编码 1）。
#
# 唯一变量：REWARD_WIN=0       对照：candE1D7R4（终端 ±1，43.2% v2 面板）
# 主指标同 S0：与 1400+ 真人的五项动作比例（tools/style_gap.py）
set -uo pipefail
cd /root/ptcg_train_kit/port_repo || { echo "✗ 不在 GPU 套件目录"; exit 1; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PTCG_REPO=/root/ptcg_train_kit/port_repo PYTHON=/root/miniconda3/bin/python3

NEED="/root/candE1D7R4.bin /root/cand9D6ER4.bin /root/cand3EAAR1.bin /root/candf5f8E.bin /root/ph3_2617_v2.bin /root/cand7e39S.bin"
MISS=""; for f in $NEED; do [ -s "$f" ] || MISS="$MISS $f"; done
if [ -n "$MISS" ]; then echo "✗ 缺权重：$MISS" >&2; exit 1; fi

export REWARD_WIN="${REWARD_WIN:-0}"       # ★ 本实验唯一变量（0 = 奖品稠密奖励）
[ "$REWARD_WIN" = 0 ] && echo "S1：REWARD_WIN=0 ⇒ 启用奖品稠密奖励（势函数=奖励卡差）"

export ANCHOR=z_ogerpon_toolbox_miwa__e1d7
export INIT="${INIT:-/root/candE1D7R4.bin}"
export BANKS=/root/cand7e39S.bin,/root/cand9D6ER4.bin,/root/cand3EAAR1.bin,/root/candf5f8E.bin,/root/ph3_2617_v2.bin,/root/candE1D7R4.bin
export BANKDECKS=teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__9d6e,dreepy_drakloak__3eaa,abra_kadabra__f5f8,grookey_thwackey__2617,z_ogerpon_toolbox_miwa__e1d7
export LEAGUE_PCT=0.08
export TRACK=z_ogerpon_toolbox_miwa__e1d7,teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__9d6e
export T="${T:-2000000}" LR="${LR:-1e-4}" OUT="${OUT:-/root/e1d7_s1}" RUN="${RUN:-e1d7_s1}" CG=1 CHECKPOINT_INTERVAL=8
export PYTHONUNBUFFERED=1
bash tools/db04_line.sh
