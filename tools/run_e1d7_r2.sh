#!/usr/bin/env bash
# **e1d7 第二轮专项（2026-11-19 计划，尚未启动）**
#
# 为什么改 bank：`e1d7_r1` 的 8 个 bank 里塞了 4 个 dunsparce 系（0bf5×2 / 9d6e / db04），
# 但 2026-11-18 抓的榜前 20 队 434 局回放显示：**前 20 队几乎不用 dunsparce**（0bf5 仅 14 局 28.6%），
# 真正的高频对手是 dreepy 系（3eaa 177 局、8fe3 48、049c 29、5b8f 32）、beldum 系（a47a 105、f91d 37、cb25 30）、
# 0904（38 局）、7e39（21 局）。而 e1d7 自己是 144 局 66.0% —— 唯一"大量使用 + 高胜率"的牌。
#
# run_e1d7_r2.sh <GPU 主机>（脚本在 GPU 机器上跑，本文件只是模板/记录）
# 用法（在 GPU 机器上）：
#   ANCHOR=z_ogerpon_toolbox_miwa__e1d7 INIT=/root/candE1D7T.bin OUT=/root/e1d7_r2 RUN=e1d7_r2 \
#     bash tools/run_e1d7_r2.sh
set -uo pipefail
cd /root/ptcg_train_kit/port_repo || exit 1
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PTCG_REPO=/root/ptcg_train_kit/port_repo PYTHON=/root/miniconda3/bin/python3
export ANCHOR="${ANCHOR:-z_ogerpon_toolbox_miwa__e1d7}"
export INIT="${INIT:-/root/candE1D7T.bin}"
PORT=/root/ptcg_train_kit/port_repo/native/pt_model_field.bin
# 8 个 bank：按"榜前 20 队的真实对局频率"配，不再按 dunsparce
export BANKS=$PORT,$PORT,$PORT,$PORT,$PORT,/root/candE1D7T.bin,/root/a10_dk_r1A.bin,/root/deck_0d86_.bin
export BANKDECKS=dreepy_drakloak__3eaa,dreepy_drakloak__3eaa,z_dragapult_dreepy_petit__0904,dreepy-drakloak__8fe3,dreepy-drakloak__049c,beldum-metang__cb25,z_ogerpon_toolbox_miwa__e1d7,slowpoke_slowking__26c6
export LEAGUE_PCT=0.06
export TRACK=z_ogerpon_toolbox_miwa__e1d7,dreepy_drakloak__3eaa,z_dragapult_dreepy_petit__0904,dreepy-drakloak__8fe3,beldum-metang__cb25
export T=4000000 LR="${LR:-1e-4}" OUT="${OUT:-/root/e1d7_r2}" RUN="${RUN:-e1d7_r2}" CG=1 CHECKPOINT_INTERVAL=16
export PYTHONUNBUFFERED=1
bash tools/db04_line.sh
