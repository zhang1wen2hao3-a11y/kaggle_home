#!/usr/bin/env bash
# **a47a_r1：轮训队列主队列 #3（2026-11-22 待跑）**
# 目标牌：beldum_metang__a47a（战斗铁哑铃·金属怪）
#   · 真实场份额 10.8%（**第二高**，仅次于 3eaa 18.3%）
#   · 外部天花板 SeaGoat 1277.9（a47a 上 144 局 31.5% 胜率；榜前队在这副牌上也不好过）
#   · 我们目前**只有端口基线 init_a47a**（真实 17 格面板 0.8% 自评）⇒ 典型"弱起点"，按今天的规律单轮能赚 10–18pp
#   · 投资分 73.7（第 3）
# 风险提示（写进脚本免得以后忘）：beldum 系列天梯用端口只 187 分、榜前队 37% 胜率 —— 这副牌本身可能"弱"。
#   所以这一轮同时是**判定轮**：若面板仍 < 12%，就照淘汰机制降级，不再投第二轮。
#
# bank = 现役最强 6 个非端口驾驶员（9d6e_r4 / 9d6e_r3 / 3eaa_r1 / f5f8E / 2617 / 7e39S）
# 跑法：bash tools/run_a47a_r1.sh        （机器上先放好 bins，缺了脚本会直接报出来）
set -uo pipefail
cd /root/ptcg_train_kit/port_repo || { echo "✗ 不在 GPU 套件目录，先 scp 上去"; exit 1; }
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PTCG_REPO=/root/ptcg_train_kit/port_repo PYTHON=/root/miniconda3/bin/python3

# ── 预检：bank 权重一个都不能少（少一个就等于偷偷用弱对手练）──
NEED="/root/cand9D6ER4.bin /root/cand9D6ER3.bin /root/cand3EAAR1.bin /root/candf5f8E.bin /root/ph3_2617_v2.bin /root/cand7e39S.bin"
MISS=""
for f in $NEED; do [ -s "$f" ] || MISS="$MISS $f"; done
[ -s "${INIT:-/root/ptcg_train_kit/port_repo/native/pt_model_field.bin}" ] || MISS="$MISS ${INIT:-/root/ptcg_train_kit/port_repo/native/pt_model_field.bin}"
if [ -n "$MISS" ]; then
  echo "✗ 缺权重，先从本地 models/ppo 推上来：$MISS" >&2
  exit 1
fi

export ANCHOR=beldum_metang__a47a
export INIT="${INIT:-/root/ptcg_train_kit/port_repo/native/pt_model_field.bin}"          # 起点＝端口（初始化权重，等价 init_a47a）
export BANKS=/root/cand9D6ER4.bin,/root/cand9D6ER3.bin,/root/cand3EAAR1.bin,/root/candf5f8E.bin,/root/ph3_2617_v2.bin,/root/cand7e39S.bin
export BANKDECKS=dunsparce_abra__9d6e,dunsparce_abra__9d6e,dreepy_drakloak__3eaa,abra_kadabra__f5f8,grookey_thwackey__2617,teal_mask_ogerpon_ex_applin__7e39
export LEAGUE_PCT=0.08
export TRACK=beldum_metang__a47a,dunsparce_abra__9d6e,dreepy_drakloak__3eaa,grookey_thwackey__2617
export T=4000000 LR="${LR:-1e-4}" OUT="${OUT:-/root/a47a_r1}" RUN="${RUN:-a47a_r1}" CG=1 CHECKPOINT_INTERVAL=16
export PYTHONUNBUFFERED=1
bash tools/db04_line.sh
