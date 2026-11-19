#!/usr/bin/env bash
# **e1d7 第二轮专项（2026-11-19 路线结论的落地版）**
#
# 为什么这么配 bank（每条都有数据出处）：
#   · e1d7 是唯一"榜前 20 队大量使用且高胜率"的牌：144 局 66.0%，天花板 1600.4（榜一 knshnb，同卡组哈希 e1d754a7）
#   · 我们同牌已能打到 LB 峰值 992.1（candE1D7T）⇒ 起点用它，不从头训
#   · 上一轮 e1d7_r1 把 4 个 bank 给了 dunsparce 系，但榜前 20 队里 dunsparce 只有 0bf5 14 局 28.6%
#     ⇒ 那 4 个槽位浪费。这一轮换成真实场覆盖率最高的牌（docs/必须卡组清单.csv 的对手覆盖率）：
#        dreepy_drakloak__3eaa 18.3% / beldum_metang__a47a 10.8% / z_dragapult_dreepy_petit__0904 3.6%
#        dreepy_drakloak__ea39 3.4% / beldum_metang__f91d 3.2% / grookey_thwackey__2617 1.5%
#     + 3eaa 给两份（覆盖最高）+ 镜像 e1d7（镜像格是我们唯一确定的短板：真实回放里对同系 0/4）
#   · ⚠️ 这些牌必须都在 `native/ptcg_tables_field.bin`（187 行卡组表）里，否则 --bank-decks 绑定失败。
#     7e39 / 26c6 / 8fe3 / 049c / 0b34 目前不在 blob 中，等重建 blob（native/tools/export_tables.py）后再用。
#
# 判据（预注册）：与 candE1D7T **同一快照**的 LB 对比，Δ ≥ +30 分才算赢；否则 e1d7 线只守不攻。
#
# 用法（GPU 机器上、kit 目录里）：
#   INIT=/root/candE1D7T.bin OUT=/root/e1d7_r2 RUN=e1d7_r2 bash tools/run_e1d7_r2.sh
set -uo pipefail
cd /root/ptcg_train_kit/port_repo || exit 1
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PTCG_REPO=/root/ptcg_train_kit/port_repo PYTHON=/root/miniconda3/bin/python3

export ANCHOR=z_ogerpon_toolbox_miwa__e1d7
export INIT="${INIT:-/root/candE1D7T.bin}"
PORT=/root/ptcg_train_kit/port_repo/native/pt_model_field.bin
export BANKS=$PORT,$PORT,$PORT,$PORT,$PORT,$PORT,/root/candE1D7T.bin,$PORT
export BANKDECKS=dreepy_drakloak__3eaa,dreepy_drakloak__3eaa,beldum_metang__a47a,z_dragapult_dreepy_petit__0904,dreepy_drakloak__ea39,beldum_metang__f91d,dreepy_drakloak__3eaa,grookey_thwackey__2617
export LEAGUE_PCT=0.06
export TRACK=z_ogerpon_toolbox_miwa__e1d7,dreepy_drakloak__3eaa,beldum_metang__a47a,z_dragapult_dreepy_petit__0904,dreepy_drakloak__ea39
export T=4000000 LR="${LR:-1e-4}" OUT="${OUT:-/root/e1d7_r2}" RUN="${RUN:-e1d7_r2}" CG=1 CHECKPOINT_INTERVAL=16
export PYTHONUNBUFFERED=1
bash tools/db04_line.sh
