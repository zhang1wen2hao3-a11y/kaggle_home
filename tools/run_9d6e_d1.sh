#!/usr/bin/env bash
# **9d6e 蒸馏 D1：BC 蒸馏初始化 + 短 PPO**（2026-11-22）
#
# 依据：9d6e 的"逐格最强变体"oracle 比最好单变体高 +4.0pp（72.2% → 76.1%），
#       其中一大半来自 2617 格（cand9D6ER4 领先次优 26.6pp）。
#       把这 4 个"真专家格"的走法先 BC 蒸馏成一个学生（val 一致率 82.9% vs 端口 64.7%），
#       再用 PPO 微调，目标是超过 72.2%。
#
#   INIT = /root/distill_9d6e.bin（BC 蒸馏产物）
#   T    = 2M 步（短程，判"蒸馏初始化有没有用"，不是长跑）
set -uo pipefail
cd /root/ptcg_train_kit/port_repo || { echo "✗ 不在 GPU 套件目录"; exit 1; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PTCG_REPO=/root/ptcg_train_kit/port_repo PYTHON=/root/miniconda3/bin/python3

NEED="/root/distill_9d6e.bin /root/cand9D6ER4.bin /root/cand3EAAR1.bin /root/candf5f8E.bin /root/ph3_2617_v2.bin /root/cand7e39S.bin /root/cand9D6ER3.bin"
MISS=""; for f in $NEED; do [ -s "$f" ] || MISS="$MISS $f"; done
if [ -n "$MISS" ]; then echo "✗ 缺权重：$MISS" >&2; exit 1; fi

export ANCHOR=dunsparce_abra__9d6e
export INIT="${INIT:-/root/distill_9d6e.bin}"
export BANKS=/root/cand9D6ER4.bin,/root/cand3EAAR1.bin,/root/candf5f8E.bin,/root/ph3_2617_v2.bin,/root/cand7e39S.bin,/root/cand9D6ER3.bin
export BANKDECKS=dunsparce_abra__9d6e,dreepy_drakloak__3eaa,abra_kadabra__f5f8,grookey_thwackey__2617,teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__9d6e
export LEAGUE_PCT=0.08
export TRACK=dunsparce_abra__9d6e,grookey_thwackey__2617,abra_kadabra__f5f8,dunsparce_abra__9d6e
export T="${T:-2000000}" LR="${LR:-5e-5}" OUT="${OUT:-/root/9d6e_d1}" RUN="${RUN:-9d6e_d1}" CG=1 CHECKPOINT_INTERVAL=8
export PYTHONUNBUFFERED=1
bash tools/db04_line.sh
