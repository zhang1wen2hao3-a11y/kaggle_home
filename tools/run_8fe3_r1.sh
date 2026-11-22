#!/usr/bin/env bash
# **8fe3_r1：轮训队列主队列 #5（2026-11-22 备好，等 GPU）**
# 目标牌：dreepy-drakloak__8fe3（真场份额 4.1%，外部天花板 tyr1218 1318.5）
#   · 投资分 54.4，是"从没训练过"的牌里最高的一副
#   · 现在只有 BC 权重（bc_8fe3，实际是 bc_3eaa 的权重套 8fe3 的牌 ⇒ 等于弱起点）；
#     按今天的标定，弱起点单轮能涨 20~40pp 面板 / +300 分 LB（a47a 187.5→518.6）
#
# bank = 现役最强 6 个非端口驾驶员。注意 **7e39 只放 1 槽**：
#   2026-11-22 的终止实验证明"7e39 专精 bank 教不动 7e39 这一格"，堆槽位只是浪费样本。
#
# 起点：bc_8fe3（BC 权重）；判据：面板 ≥ 上一版 +5pp（BC 起点约 25~35%，所以要 ≥40%）
set -uo pipefail
cd /root/ptcg_train_kit/port_repo || { echo "✗ 不在 GPU 套件目录"; exit 1; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PTCG_REPO=/root/ptcg_train_kit/port_repo PYTHON=/root/miniconda3/bin/python3

NEED="/root/bc_8fe3.bin /root/cand9D6ER4.bin /root/cand3EAAR1.bin /root/candf5f8E.bin /root/ph3_2617_v2.bin /root/cand7e39S.bin"
MISS=""; for f in $NEED; do [ -s "$f" ] || MISS="$MISS $f"; done
if [ -n "$MISS" ]; then echo "✗ 缺权重，先从本地 models/ppo 推：$MISS" >&2; exit 1; fi

export ANCHOR=dreepy-drakloak__8fe3
export INIT="${INIT:-/root/bc_8fe3.bin}"
export BANKS=/root/cand9D6ER4.bin,/root/cand3EAAR1.bin,/root/candf5f8E.bin,/root/ph3_2617_v2.bin,/root/cand7e39S.bin,/root/candE1D7R4.bin
export BANKDECKS=dunsparce_abra__9d6e,dreepy_drakloak__3eaa,abra_kadabra__f5f8,grookey_thwackey__2617,teal_mask_ogerpon_ex_applin__7e39,z_ogerpon_toolbox_miwa__e1d7
export LEAGUE_PCT=0.08
export TRACK=dreepy-drakloak__8fe3,dunsparce_abra__9d6e,dreepy_drakloak__3eaa,grookey_thwackey__2617
export T=4000000 LR="${LR:-1e-4}" OUT="${OUT:-/root/8fe3_r1}" RUN="${RUN:-8fe3_r1}" CG=1 CHECKPOINT_INTERVAL=16
export PYTHONUNBUFFERED=1
bash tools/db04_line.sh
