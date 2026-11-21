#!/usr/bin/env bash
# **e1d7_r4：7e39 专精轮**（2026-11-22，依据 docs/我方与高分选手逐格缺口_2026-11-22.md）
#
# 为什么这么配 bank
# ----------------
# 逐格缺口实测：1600 分选手（やる気元気ミワハルキ 1594）用**同一副 e1d7** 打 `7e39` 是 **21/22 = 95.5%**，
# 而我们的 candE1D7R3 只有 **38.0%**；7e39 占真实场 **34.9%**（全场最高）。
# 这一格修上去，e1d7 加权面板就能从 38.7% → ~46%（+7pp），是眼下性价比最高的一件事。
# 所以 bank 里 **7e39 占 3 槽**（对手越常出现，策略越会被逼着学会它的打法），另配
# 9d6e（打 7e39 88.7% 的牌）/ 2617（打 7e39 78%）/ 3eaa 作为"会打 7e39 的老师"。
#
# 起点：candE1D7R3（38.7%）。判据：7e39 格 ≥50%（现在 38.0%）、加权面板 ≥43.7%（现在 38.7%）。
set -uo pipefail
cd /root/ptcg_train_kit/port_repo || { echo "✗ 不在 GPU 套件目录"; exit 1; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PTCG_REPO=/root/ptcg_train_kit/port_repo PYTHON=/root/miniconda3/bin/python3

NEED="/root/cand7e39S.bin /root/cand9D6ER4.bin /root/ph3_2617_v2.bin /root/cand3EAAR1.bin /root/candE1D7R3.bin"
MISS=""; for f in $NEED; do [ -s "$f" ] || MISS="$MISS $f"; done
if [ -n "$MISS" ]; then echo "✗ 缺权重：$MISS" >&2; exit 1; fi

export ANCHOR=z_ogerpon_toolbox_miwa__e1d7
export INIT="${INIT:-/root/candE1D7R3.bin}"
export BANKS=/root/cand7e39S.bin,/root/cand7e39S.bin,/root/cand7e39S.bin,/root/cand9D6ER4.bin,/root/ph3_2617_v2.bin,/root/cand3EAAR1.bin
export BANKDECKS=teal_mask_ogerpon_ex_applin__7e39,teal_mask_ogerpon_ex_applin__7e39,teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__9d6e,grookey_thwackey__2617,dreepy_drakloak__3eaa
export LEAGUE_PCT=0.08
export TRACK=z_ogerpon_toolbox_miwa__e1d7,teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__9d6e,grookey_thwackey__2617
export T=4000000 LR="${LR:-1e-4}" OUT="${OUT:-/root/e1d7_r4}" RUN="${RUN:-e1d7_r4}" CG=1 CHECKPOINT_INTERVAL=16
export PYTHONUNBUFFERED=1
bash tools/db04_line.sh
