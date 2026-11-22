#!/usr/bin/env bash
# **5b8f_r1：轮训队列主队列 #7**（2026-11-22）
# 目标牌：dreepy-drakloak__5b8f（多龙巴鲁托系，真场份额 2.7%，外部天花板 RtoABC2 1311）
#   · 投资分 47.2，只有 BC 权重（bc_5b8f = bc_3eaa 的权重套 5b8f 的牌）⇒ 弱起点
#   · 按今天标定：弱起点单轮 +20~40pp 面板 / +300 分 LB（a47a 187.5→571、8fe3 BC→51.3%）
# bank = 现役最强 6 个非端口驾驶员；7e39 只留 1 槽（专精 bank 已被证明教不动那格）
set -uo pipefail
cd /root/ptcg_train_kit/port_repo || { echo "✗ 不在 GPU 套件目录"; exit 1; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PTCG_REPO=/root/ptcg_train_kit/port_repo PYTHON=/root/miniconda3/bin/python3

NEED="/root/bc_5b8f.bin /root/cand9D6ER4.bin /root/cand3EAAR1.bin /root/candf5f8E.bin /root/ph3_2617_v2.bin /root/cand7e39S.bin"
MISS=""; for f in $NEED; do [ -s "$f" ] || MISS="$MISS $f"; done
if [ -n "$MISS" ]; then echo "✗ 缺权重，先从本地 models/ppo 推：$MISS" >&2; exit 1; fi

export ANCHOR=dreepy-drakloak__5b8f
export INIT="${INIT:-/root/bc_5b8f.bin}"
export BANKS=/root/cand9D6ER4.bin,/root/cand3EAAR1.bin,/root/candf5f8E.bin,/root/ph3_2617_v2.bin,/root/cand7e39S.bin,/root/cand9D6ER3.bin
export BANKDECKS=dunsparce_abra__9d6e,dreepy_drakloak__3eaa,abra_kadabra__f5f8,grookey_thwackey__2617,teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__9d6e
export LEAGUE_PCT=0.08
export TRACK=dreepy-drakloak__5b8f,dunsparce_abra__9d6e,dreepy_drakloak__3eaa,grookey_thwackey__2617
export T=4000000 LR="${LR:-1e-4}" OUT="${OUT:-/root/5b8f_r1}" RUN="${RUN:-5b8f_r1}" CG=1 CHECKPOINT_INTERVAL=16
export PYTHONUNBUFFERED=1
bash tools/db04_line.sh
