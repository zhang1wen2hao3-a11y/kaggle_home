#!/usr/bin/env bash
# **S2：势函数型奖励塑形（PBRS，Ng/Harada/Russell 1999）**（2026-11-23）
#
# 与 S1 的区别（这是关键）：
#   · S1 把奖励**换成**奖品奖励（等价于 Φ=奖励卡差 的差分，但终局 ±1 被换掉）；
#   · S2 **保留终局 ±1**，另外每一步加 F = Φ(s') − Φ(s)：
#         Φ = w_p·(对手剩余奖励卡 − 我方剩余奖励卡)
#           + w_e·(我方场上附着能量 − 对手)
#           + w_d·(我造成的伤害 − 对手造成的伤害)      （以 p0 视角算，天然对称）
#     按定理，最优策略**不变**（Q 只被平移 −Φ(s)），只是把"延迟奖励"变成逐步可学的稠密信号。
#   · 我上一轮那套 `SHAPE_ATTACH/ATTACK/RETREAT` 是**非势函数**塑形 —— 会改变最优策略，
#     所以它失败（−2.7pp）是理论预期之内的，不能拿它否证"稠密化"。
#
# 权重默认值让一局的总势差 ≈ ±1（与终局 ±1 同量级）：
#   PBRS_PRIZE=0.10（最多 ±0.6） PBRS_ENERGY=0.02（最多 ±0.4） PBRS_DAMAGE=0.0005（最多 ±0.5）
set -uo pipefail
cd /root/ptcg_train_kit/port_repo || { echo "✗ 不在 GPU 套件目录"; exit 1; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PTCG_REPO=/root/ptcg_train_kit/port_repo PYTHON=/root/miniconda3/bin/python3

NEED="/root/candE1D7R4.bin /root/cand9D6ER4.bin /root/cand3EAAR1.bin /root/candf5f8E.bin /root/ph3_2617_v2.bin /root/cand7e39S.bin"
MISS=""; for f in $NEED; do [ -s "$f" ] || MISS="$MISS $f"; done
if [ -n "$MISS" ]; then echo "✗ 缺权重：$MISS" >&2; exit 1; fi
# 势函数必须在环境里生效 —— 检查二进制是否带 PBRS 支持
if ! strings /root/ptcg_train_kit/port_repo/native/puffer_ptcg/_C*.so 2>/dev/null | grep -q pbrs_prize; then
  echo "⚠️ 当前 _C*.so 里没有 pbrs_* ⇒ 需要先用改过的 env.c/binding.c 重新编译（bash native/build_native.sh）"
fi

export PBRS_PRIZE="${PBRS_PRIZE:-0.10}"
export PBRS_ENERGY="${PBRS_ENERGY:-0.02}"
export PBRS_DAMAGE="${PBRS_DAMAGE:-0.0005}"
echo "S2：PBRS 权重 — 奖励卡差 $PBRS_PRIZE / 能量差 $PBRS_ENERGY / 血量差 $PBRS_DAMAGE（终局 ±1 保留）"

export ANCHOR=z_ogerpon_toolbox_miwa__e1d7
export INIT="${INIT:-/root/candE1D7R4.bin}"
export BANKS=/root/cand7e39S.bin,/root/cand9D6ER4.bin,/root/cand3EAAR1.bin,/root/candf5f8E.bin,/root/ph3_2617_v2.bin,/root/candE1D7R4.bin
export BANKDECKS=teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__9d6e,dreepy_drakloak__3eaa,abra_kadabra__f5f8,grookey_thwackey__2617,z_ogerpon_toolbox_miwa__e1d7
export LEAGUE_PCT=0.08
export TRACK=z_ogerpon_toolbox_miwa__e1d7,teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__9d6e
export T="${T:-2000000}" LR="${LR:-1e-4}" OUT="${OUT:-/root/e1d7_s2}" RUN="${RUN:-e1d7_s2}" CG=1 CHECKPOINT_INTERVAL=8
export PYTHONUNBUFFERED=1
bash tools/db04_line.sh
