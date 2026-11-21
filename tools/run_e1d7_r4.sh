#!/usr/bin/env bash
# **e1d7_r4：e1d7 的"专精格"抢救轮**（2026-11-22 备选，队列里 e1d7 已标平台期，跑之前先看 a47a 结果）
#
# 为什么还要给 e1d7 一轮
# ----------------------
# 同口径拆格（docs/训练路线结论_2026-11-19.md §7）：
#   · e1d7 打 **9d6e 家族**（份额 19.9%）加权只有 13.3%，而 3eaa 有 37.0% ⇒ 这是它总分卡在 38.7% 的主因；
#   · r3 的 bank 里虽然有 9d6e×2 + f5f8，但家里 7e39 也占一槽 ⇒ 家份额被稀释，9d6e 家族只从 9.2% 爬到 13.3%（+0.8pp 总分）。
# 所以 r4 把 bank **全部**换成"赢过 9d6e 家族"的驾驶员（9d6e×2 / 0bf5 / f5f8×2 / 3eaa），
# 目标是把这个 19.9% 的格子从 13% 推到 30%+（总分 ≈ +3.5pp）。若 r4 仍不过闸门 ⇒ e1d7 按平台期处理，不再投钱。
#
# 起点：candE1D7R3（38.7%）
set -uo pipefail
cd /root/ptcg_train_kit/port_repo || { echo "✗ 不在 GPU 套件目录"; exit 1; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PTCG_REPO=/root/ptcg_train_kit/port_repo PYTHON=/root/miniconda3/bin/python3

NEED="/root/cand9D6ER4.bin /root/cand9D6ER3.bin /root/cand0bf5S.bin /root/candf5f8E.bin /root/candf5f8G.bin /root/cand3EAAR1.bin /root/candE1D7R3.bin"
MISS=""; for f in $NEED; do [ -s "$f" ] || MISS="$MISS $f"; done
if [ -n "$MISS" ]; then echo "✗ 缺权重：$MISS" >&2; exit 1; fi

export ANCHOR=z_ogerpon_toolbox_miwa__e1d7
export INIT="${INIT:-/root/candE1D7R3.bin}"
export BANKS=/root/cand9D6ER4.bin,/root/cand9D6ER3.bin,/root/cand0bf5S.bin,/root/candf5f8E.bin,/root/candf5f8G.bin,/root/cand3EAAR1.bin
export BANKDECKS=dunsparce_abra__9d6e,dunsparce_abra__9d6e,dunsparce_abra__0bf5,abra_kadabra__f5f8,abra_kadabra__f5f8,dreepy_drakloak__3eaa
export LEAGUE_PCT=0.08
export TRACK=z_ogerpon_toolbox_miwa__e1d7,dunsparce_abra__9d6e,dunsparce_abra__0bf5,abra_kadabra__f5f8
export T=4000000 LR="${LR:-1e-4}" OUT="${OUT:-/root/e1d7_r4}" RUN="${RUN:-e1d7_r4}" CG=1 CHECKPOINT_INTERVAL=16
export PYTHONUNBUFFERED=1
bash tools/db04_line.sh
