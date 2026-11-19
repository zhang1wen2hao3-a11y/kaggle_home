#!/usr/bin/env bash
# **串行派发**：多个候选/多组格子排队跑，一个 dispatch_panel 跑完再跑下一个。
# 为什么需要：同时起两个 dispatch_panel 会互相抢"空闲机"判定，两个队列都不推进
# （2026-11-18 实测：e1d7T 面板停在 8/17，另一个队列在推）。
#
#   bash tools/serial_dispatch.sh HOST CAND:cells
# ★ 一律带 --local-after 0：绝不让评估回落到开发机本地 CPU（用户 2026-11-19 明确要求） [CAND:cells ...]
#   例：bash tools/serial_dispatch.sh root@1.2.3.4 \
#         "ab9d6e_r2A:out_77a5 out_ea39" "candE1D7T:champ_0904 abra_f5f8"
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
HOST="${1:?host}"; shift

# ★ 兜底定时关机（用户 2026-11-19 要求）：DEADLINE=02:00 时，先在目标机挂 systemd 定时关机，
#   这样即使看门狗失效/网络断了，机器到点也会自己关。
if [ -n "${DEADLINE:-}" ]; then
  bash tools/arm_deadline_shutdown.sh "$HOST" "$DEADLINE" >/dev/null 2>&1 || true
fi
for spec in "$@"; do
  CAND="${spec%%:*}"; CELLS="${spec#*:}"
  echo "[$(date +%H:%M:%S)] === $CAND ($CELLS)"
  ./tools/py tools/dispatch_panel.py --cands "$CAND" --cells $CELLS --seeds 950000 \
     --matches 150 --workers 24 --interval 15 --hosts "$HOST" --local-after 0 2>&1 | tail -2
done
echo "[$(date +%H:%M:%S)] SERIAL_DONE"
