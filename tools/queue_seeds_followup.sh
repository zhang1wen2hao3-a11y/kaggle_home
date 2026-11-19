#!/usr/bin/env bash
# **接续派发**：等上一批跑完（日志出现结束标记）后，接着给同一台机器派下一批。
#
# 为什么要这个：`dispatch_panel` 一次只认一个队列，两个队列同时跑会互相抢"空闲机"判定
# （2026-11-18 实测两个都不推进）；而 `watch_serial_and_stop.sh` 一旦看到结束标记就会关机，
# 所以"还要继续干活"的时候必须先把自动关机 watcher 撤掉，改用本脚本串起来。
#
#   bash tools/queue_seeds_followup.sh <host> <上一批日志> <结束标记> <新牌文件[:agent]> [新牌2...]
#
# 例：等 123 的反向跑完，接着补 a47a（新牌 × 17 格 正向 + 17 个现有驾驶员 × a47a 反向）
#   bash tools/queue_seeds_followup.sh root@123.249.80.18 logs/seeds_123_rev.log SEEDS_DONE beldum_metang__a47a:init_a47a
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
HOST="${1:?host}"; LOG="${2:?上一批日志}"; TAG="${3:?结束标记}"; shift 3
DECKS=("$@")
[ ${#DECKS[@]} -gt 0 ] || { echo "✗ 没给新牌"; exit 1; }

CELLS17="ogerpon_7e39 dunsparce_0bf5 grookey_2617 out_91808263 dreepy_3eaa champ_e1d7 out_07be out_3f45 out_77a5 champ_0904 abra_f5f8 out_ea39 out_9bc9d655 out_0382 dunsparce_9d6e abra_e8ee champ_26c6"
EXIST_AGENTS="port_7e39 opp_dunsparce_0bf5 c_grookey port_on_91808263 opp_3eaa port_e1d7 port_on_07be port_on_3f45 port_on_77a5 port_0904 candf5f8E port_on_ea39 port_on_9bc9d655 port_on_0382 port_on_9d6e port_e8ee r1A_26c6"
MATCHES="${MATCHES:-200}"

echo "[$(date +%H:%M:%S)] 等 $LOG 出现 $TAG …"
for i in $(seq 1 480); do
  grep -q "$TAG" "$LOG" 2>/dev/null && { echo "[$(date +%H:%M:%S)] 上一批结束"; break; }
  sleep 30
done
grep -q "$TAG" "$LOG" 2>/dev/null || { echo "✗ 等超时，什么都不派"; exit 1; }

for spec in "${DECKS[@]}"; do
  deck="${spec%%:*}"; agent="${spec##*:}"
  h4=$(echo "$deck" | sed 's/.*__//')
  echo "[$(date +%H:%M:%S)] 正向 $agent × 17 格"
  bash tools/serial_dispatch.sh "$HOST" "$agent:$CELLS17" || true
  echo "[$(date +%H:%M:%S)] 反向 17 个现有驾驶员 × 新格 $h4"
  ./tools/py tools/dispatch_panel.py --cands $EXIST_AGENTS --cells "$h4" \
     --cell-agents "$h4=build/agents/$agent" --seeds 950000 --matches "$MATCHES" \
     --workers 24 --interval 15 --hosts "$HOST" --local-after 0 2>&1 | tail -2
done
echo "[$(date +%H:%M:%S)] FOLLOWUP_DONE"
