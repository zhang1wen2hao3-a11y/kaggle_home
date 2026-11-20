#!/usr/bin/env bash
# **9d6e_r3：用"我们自己的最强驾驶员"当陪练**（2026-11-21 决策）
#
# 决策依据（全部来自实测，见 docs/训练路线结论 §6.5）：
#   1. 9d6e 家族打**端口**对手 67.6%，但打**我们自己的强驾驶员**只掉到 37.8%(2617)/46.9%(f5f8)；
#      而真实场的对手比端口强得多（榜前队 1300+）⇒ 瓶颈是"没跟强对手练过"，不是"牌不好"。
#   2. 逐格加权缺口只有 0.6pp（0904 0.25 / 0382 0.16 / 2617 0.12）⇒ 继续对着端口微调没有油水。
#   3. 所以这一轮把 bank 全部换成**非 init**：cand7e39S / ph3_2617_v2 / candE1D7R2 / candf5f8E / b9d6e / bc_3eaa。
#
# 前置条件（必须先在本地做，10 分钟）：
#   port_repo/native/tools/export_tables.py --pool decks/pool_field native/ptcg_tables_field.bin
#   —— 现在 blob 只有 187 行，7e39 / f5f8 / 91808263 都不在里面，bank 绑不上。
#
# 判据（预注册）：强对强复测（n≥450）里 `9d6e 打 2617` ≥50%（现 37.8%）、`打 f5f8` ≥55%（现 46.9%），
#                 且 `打 7e39` 不退化（现 90%）、端口口径面板总分不低于 66%。
#
# 用法（GPU 机器上、kit 目录里）：
#   INIT=/root/ab9d6e_r2A.bin OUT=/root/9d6e_r3 RUN=9d6e_r3 bash tools/run_9d6e_r3.sh
set -uo pipefail
cd /root/ptcg_train_kit/port_repo || exit 1
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PTCG_REPO=/root/ptcg_train_kit/port_repo PYTHON=/root/miniconda3/bin/python3

export ANCHOR=dunsparce_abra__9d6e
export INIT="${INIT:-/root/ab9d6e_r2A.bin}"
# 8 个 bank：全部是"非 init"的强驾驶员（这是与 r1/r2 最大的区别）
export BANKS=/root/cand7e39S.bin,/root/cand7e39S.bin,/root/ph3_2617_v2.bin,/root/ph3_2617_v2.bin,/root/candE1D7R2.bin,/root/candf5f8E.bin,/root/b9d6e.bin,/root/bc_3eaa.bin
export BANKDECKS=teal_mask_ogerpon_ex_applin__7e39,teal_mask_ogerpon_ex_applin__7e39,grookey_thwackey__2617,grookey_thwackey__2617,z_ogerpon_toolbox_miwa__e1d7,abra_kadabra__f5f8,dunsparce_abra__9d6e,dreepy_drakloak__3eaa
export LEAGUE_PCT=0.10
export TRACK=dunsparce_abra__9d6e,teal_mask_ogerpon_ex_applin__7e39,grookey_thwackey__2617,abra_kadabra__f5f8
export T=4000000 LR="${LR:-1e-4}" OUT="${OUT:-/root/9d6e_r3}" RUN="${RUN:-9d6e_r3}" CG=1 CHECKPOINT_INTERVAL=16
export PYTHONUNBUFFERED=1
bash tools/db04_line.sh
