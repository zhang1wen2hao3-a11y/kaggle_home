#!/usr/bin/env bash
# **S3：把"可攻击性/威胁度"写进势函数 Φ**（2026-11-23）
#
# 为什么要这一项：S2 的 Φ 只有「奖励卡差 + 能量差 + 血量差」三个量，
#   而实测我们的驾驶员 **撤退率是人类 2.1 倍**（0.53 vs 0.25 每回合）——
#   在 S2 的 Φ 下，撤退几乎不吃亏（把充好能量的 active 换下去，能量还在场上）。
#   S3 加入**凸的威胁度项**：threat = min(active 能量,8)² + 0.25·Σ_bench 能量，
#   撤退会立刻把 threat 打下去 ⇒ 在 Φ 里自然变"亏"，而它仍是势函数差分，
#   按 Ng 定理**最优策略不变**（只是把学习信号变密、方向变对）。
#
# 与 S2 的权重差别：为让四项合起来仍 ≈ ±1 量级，S3 把前两项调小、把 attackable 作为主项。
#   PBRS_PRIZE=0.05  PBRS_ENERGY=0.01  PBRS_DAMAGE=0.0005  PBRS_ATTACKABLE=0.01
set -uo pipefail
cd /root/ptcg_train_kit/port_repo || { echo "✗ 不在 GPU 套件目录"; exit 1; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PTCG_REPO=/root/ptcg_train_kit/port_repo PYTHON=/root/miniconda3/bin/python3

NEED="/root/candE1D7R4.bin /root/cand9D6ER4.bin /root/cand3EAAR1.bin /root/candf5f8E.bin /root/ph3_2617_v2.bin /root/cand7e39S.bin"
MISS=""; for f in $NEED; do [ -s "$f" ] || MISS="$MISS $f"; done
if [ -n "$MISS" ]; then echo "✗ 缺权重：$MISS" >&2; exit 1; fi
if ! strings native/puffer_ptcg/_C*.so 2>/dev/null | grep -q pbrs_attackable; then
  echo "⚠️ _C*.so 里没有 pbrs_attackable ⇒ 先用改过的源码重编译（bash native/build_native.sh）"
fi

export PBRS_PRIZE="${PBRS_PRIZE:-0.05}"
export PBRS_ENERGY="${PBRS_ENERGY:-0.01}"
export PBRS_DAMAGE="${PBRS_DAMAGE:-0.0005}"
export PBRS_ATTACKABLE="${PBRS_ATTACKABLE:-0.01}"
echo "S3：PBRS — 奖励卡 $PBRS_PRIZE / 能量 $PBRS_ENERGY / 血量 $PBRS_DAMAGE / **可攻击性 $PBRS_ATTACKABLE**（终局 ±1 保留）"

export ANCHOR=z_ogerpon_toolbox_miwa__e1d7
export INIT="${INIT:-/root/candE1D7R4.bin}"
export BANKS=/root/cand7e39S.bin,/root/cand9D6ER4.bin,/root/cand3EAAR1.bin,/root/candf5f8E.bin,/root/ph3_2617_v2.bin,/root/candE1D7R4.bin
export BANKDECKS=teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__9d6e,dreepy_drakloak__3eaa,abra_kadabra__f5f8,grookey_thwackey__2617,z_ogerpon_toolbox_miwa__e1d7
export LEAGUE_PCT=0.08
export TRACK=z_ogerpon_toolbox_miwa__e1d7,teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__9d6e
export T="${T:-2000000}" LR="${LR:-1e-4}" OUT="${OUT:-/root/e1d7_s3}" RUN="${RUN:-e1d7_s3}" CG=1 CHECKPOINT_INTERVAL=8
export PYTHONUNBUFFERED=1
bash tools/db04_line.sh
