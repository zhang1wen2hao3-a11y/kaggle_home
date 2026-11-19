#!/usr/bin/env bash
# **给新接入的卡组补种子**（2026-11-19：5b8f / 049c / 1917 / 0b34 / 71b2 / cb25 共 6 副）
#
# 为什么需要：维护表（the-pockmon-battle-choice*.csv）是从母表重算的，母表只认"评估结果"。
# 新牌刚接进表时，18×18 的胜率矩阵里它那一行/列全是空的 —— 这就是"种子不够"。
# 每副新牌要补：正向 17 格（新牌当挑战者打现有 17 格）+ 反向 17 格（现有牌打新牌）= 34 格；
# 6 副 × 34 = 204 格，每格 MIN_N=200 局。
#
#   bash tools/run_seeds_new_decks.sh <host> fwd     # 只跑正向（102 格，约 1.7 h / 32 核）
#   bash tools/run_seeds_new_decks.sh <host> rev     # 只跑反向（102 格）
#   bash tools/run_seeds_new_decks.sh <host> all     # 全部（204 格，约 3.5 h / 32 核）
#
# 三台并行 ≈ 1.2 h。跑完务必：rsync 结果 → 校验数量 → 关机（tools/watch_serial_and_stop.sh）。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
HOST="${1:?第一个参数是评估机，如 root@113.44.139.218}"
MODE="${2:-all}"
MATCHES="${MATCHES:-200}"

# 新牌 → 它的驾驶员 agent（缺模型时由 tools/materialize_required_drivers.py 补的 BC/init）
NEW="5b8f:bc_5b8f 049c:bc_049c 1917:bc_1917 0b34:init_0b34 71b2:init_71b2 cb25:init_cb25"
# 现有 17 格（真实份额测试集的格名）
CELLS17="ogerpon_7e39 dunsparce_0bf5 grookey_2617 out_91808263 dreepy_3eaa champ_e1d7 out_07be out_3f45 out_77a5 champ_0904 abra_f5f8 out_ea39 out_9bc9d655 out_0382 dunsparce_9d6e abra_e8ee champ_26c6"
# 反向用的现有牌驾驶员（端口口径，和矩阵里大部分单元同一口径）
EXIST_AGENTS="port_7e39 opp_dunsparce_0bf5 c_grookey port_on_91808263 opp_3eaa port_e1d7 port_on_07be port_on_3f45 port_on_77a5 port_0904 candf5f8E port_on_ea39 port_on_9bc9d655 port_on_0382 port_on_9d6e port_e8ee r1A_26c6"
NEWCELLS="5b8f 049c 1917 0b34 71b2 cb25"

case "$MODE" in
  fwd|all)
    for spec in $NEW; do
      agent="${spec##*:}"
      echo "[$(date +%H:%M:%S)] 正向 $agent × 17 格"
      bash tools/serial_dispatch.sh "$HOST" "$agent:$CELLS17" || exit 1
    done
    ;;
esac
case "$MODE" in
  rev|all)
    echo "[$(date +%H:%M:%S)] 反向 17 个现有驾驶员 × 6 个新格"
    ./tools/py tools/dispatch_panel.py --cands $EXIST_AGENTS --cells $NEWCELLS \
       --cell-agents "5b8f=build/agents/bc_5b8f,049c=build/agents/bc_049c,1917=build/agents/bc_1917,0b34=build/agents/init_0b34,71b2=build/agents/init_71b2,cb25=build/agents/init_cb25" \
       --seeds 950000 --matches "$MATCHES" --workers 24 --interval 15 --hosts "$HOST" --local-after 0 2>&1 | tail -3
    ;;
esac
echo "[$(date +%H:%M:%S)] SEEDS_DONE ($MODE, matches=$MATCHES)"
