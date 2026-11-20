#!/usr/bin/env bash
# **对称补测·追加轮次**（把每对从 150 局补到 450 局）
#
# 为什么：`tools/sym_pair_eval.sh` 跑的是第一轮（seed 950200，150 局）。
# 交付表的格级选人要求"池化 n≥400"，所以要再补两轮（每轮 150 局）到 450。
#
#   bash tools/sym_pair_eval2.sh root@<host> <轮次=1|2> [每对局数=150] [workers=24]
#
# 输出：logs/meta/sym/<A>__vs__<B>__r<轮次>.json（与第一轮的 <A>__vs__<B>.json 分开存，便于池化）
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
HOST="${1:?host，如 root@123.249.80.18}"
ROUND="${2:?轮次 1 或 2（对应 seed 950201 / 950202）}"
MATCHES="${3:-150}"
WORKERS="${4:-24}"
SEED=$((950200 + ROUND))
OUTDIR=logs/meta/sym
mkdir -p "$OUTDIR"

PAIRS=(
  "cand7e39S   7e39   ab9d6e_r1A 9d6e   11行13列"
  "bc_3eaa     3eaa   ab9d6e_r2A 9d6e    8行13列"
  "candE1D7R2  e1d7   ab9d6e_r2A 9d6e   14行13列"
  "ab9d6e_r2A  9d6e   ph3_2617_v2 2617  12行7列"
  "ab9d6e_r2A  9d6e   cand0bf5S  0bf5   12行4列"
  "cand7e39S   7e39   candf5f8E   f5f8   11行18列"
  "cand0bf5S   0bf5   candf5f8E   f5f8   （单向）"
  "ph3_2617_v2 2617   cand7e39S  7e39    6行12列"
  "cand7e39S   7e39   ph3_2617_v2 2617  11行7列"
  "candE1D7R2  e1d7   cand7e39S  7e39   14行12列"
  "cand7e39S   7e39   candE1D7R2 e1d7   11行16列"
)
SSHOPT="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

echo "[$(date +%H:%M:%S)] 轮次 $ROUND（seed $SEED）→ $HOST，${#PAIRS[@]} 对 × $MATCHES 局"
for spec in "${PAIRS[@]}"; do
  read -r A A_d B B_d coord <<< "$spec"
  out="$OUTDIR/${A}__vs__${B}__r${ROUND}.json"
  [ -s "$out" ] && { echo "[skip] $A vs $B r$ROUND"; continue; }
  echo "[$(date +%H:%M:%S)] → $A($A_d) vs $B($B_d)  $coord"
  ssh $SSHOPT "$HOST" "cd /root/eval_cpu_kit && setsid nohup python3 ptcg_league/eval_pair.py \
     --cand build/agents/$A --opp $B_d=build/agents/$B --matches $MATCHES --workers $WORKERS \
     --seed-base $SEED --out /root/eval_cpu_kit/logs/meta/sym_r${ROUND}/${A}__vs__${B}.json \
     > /root/eval_cpu_kit/logs/sym_${A}_${B}_r${ROUND}.log 2>&1 & sleep 2; echo started"
  # 等这一对跑完（单机串行，避免互相抢核）
  for i in $(seq 1 120); do
    # ★ pgrep -c 无匹配时会打印 0 且退出码 1 ⇒ `|| echo 0` 会输出 "0\n0"，
    #   拼起来变成 "00" ≠ "0"，等待循环永远不结束（2026-11-20 踩到，两台机器各卡住 5 分钟）。
    n=$(ssh $SSHOPT "$HOST" 'pgrep -fc "[e]val_pair" 2>/dev/null || true' 2>/dev/null | head -1 | tr -d '[:space:]')
    [ -z "${n:-}" ] && n=0
    [ "${n:-0}" = "0" ] && break
    sleep 15
  done
  rsync -az --timeout=60 -e "ssh $SSHOPT" \
    "$HOST:/root/eval_cpu_kit/logs/meta/sym_r${ROUND}/${A}__vs__${B}.json" "$out" 2>/dev/null \
    && echo "[$(date +%H:%M:%S)]   ✓ 已拉回 $(basename "$out")" || echo "[$(date +%H:%M:%S)]   ✗ 拉取失败"
done
echo "[$(date +%H:%M:%S)] SYM_R${ROUND}_DONE"
