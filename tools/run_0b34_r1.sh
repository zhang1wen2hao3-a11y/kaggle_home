#!/usr/bin/env bash
# **0b34_r1：轮训队列主队列 #6**（2026-11-22）
# 目标牌：tealmaskogerponex-forestofvitality__0b34（青绿面具厄鬼椪 ex + 生机之森 + Hydrapple ex 草系）
#   · 真场份额 2.7%，外部天花板 ktr 1334.3（投资分 47.7，未训牌里最高）
#   · 我们只有端口驾驶员 init_0b34（面板 40.5%）⇒ 弱起点，按今天标定预期 +10~30pp
# bank = 现役最强 6 个非端口驾驶员（7e39 只放 1 槽：专精 bank 已被证明教不动那一格）
set -uo pipefail
cd /root/ptcg_train_kit/port_repo || { echo "✗ 不在 GPU 套件目录"; exit 1; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PTCG_REPO=/root/ptcg_train_kit/port_repo PYTHON=/root/miniconda3/bin/python3

NEED="/root/cand9D6ER4.bin /root/cand9D6ER3.bin /root/cand3EAAR1.bin /root/candf5f8E.bin /root/ph3_2617_v2.bin /root/cand7e39S.bin"
MISS=""; for f in $NEED; do [ -s "$f" ] || MISS="$MISS $f"; done
[ -s "${INIT:-/root/ptcg_train_kit/port_repo/native/pt_model_field.bin}" ] || MISS="$MISS ${INIT:-port}"
if [ -n "$MISS" ]; then echo "✗ 缺权重：$MISS" >&2; exit 1; fi

export ANCHOR=tealmaskogerponex-forestofvitality__0b34
export INIT="${INIT:-/root/ptcg_train_kit/port_repo/native/pt_model_field.bin}"
export BANKS=/root/cand9D6ER4.bin,/root/cand9D6ER3.bin,/root/cand3EAAR1.bin,/root/candf5f8E.bin,/root/ph3_2617_v2.bin,/root/cand7e39S.bin
export BANKDECKS=dunsparce_abra__9d6e,dunsparce_abra__9d6e,dreepy_drakloak__3eaa,abra_kadabra__f5f8,grookey_thwackey__2617,teal_mask_ogerpon_ex_applin__7e39
export LEAGUE_PCT=0.08
export TRACK=tealmaskogerponex-forestofvitality__0b34,dunsparce_abra__9d6e,dreepy_drakloak__3eaa,teal_mask_ogerpon_ex_applin__7e39
export T=4000000 LR="${LR:-1e-4}" OUT="${OUT:-/root/0b34_r1}" RUN="${RUN:-0b34_r1}" CG=1 CHECKPOINT_INTERVAL=16
export PYTHONUNBUFFERED=1
bash tools/db04_line.sh
