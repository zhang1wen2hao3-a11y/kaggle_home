#!/usr/bin/env bash
# **剩余评估队列（幂等，可断点续跑）**：在任何一台装好 eval kit 的 CPU 机上跑。
#
#   目标机上：  bash tools/queue_remaining.sh
#
# 已经完成的不重跑（meta_eval / h2h 都按输出 json 判断）。当前待补：
#   1) r2B@9d6e 12 格面板
#   2) 头对头：r2A vs r1A、r1A vs 端口、r2B vs 端口、r2A vs r2B（9d6e 同牌，400 场）
#   3) 高样本复核：ogerpon_7e39（权重 0.347）单格 n=1000，r1A / r2A / port 各一份
#      —— 因为 r1A 的 +2.4pp 里有 +1.4pp 压在这一格（n=200 时 CI±6.5pp）
set -uo pipefail
cd "$(dirname "$0")/.."

need() { [ -e "$1" ] || { echo "✗ 缺 $1 —— 先在目标机解包 dist/eval_cpu_kit.tar.gz 并跑 setup.sh"; exit 1; }; }
for f in tools/meta_eval.sh tools/h2h.sh build/agents/port_on_9d6e build/agents/port_7e39; do need "$f"; done
for k in r1A r1B r2A r2B; do
  [ -d "build/agents/${k}_9d6e" ] || need "models_ppo/$k.pt"
  [ -d "build/agents/${k}_9d6e" ] || bash tools/make_agent_dir.sh "build/agents/${k}_9d6e" "models_ppo/$k.pt" decks/pool_field/dunsparce_abra__9d6e.csv
done

echo "===== [$(date +%H:%M)] ① r2B@9d6e 12 格面板 ====="
bash tools/meta_eval.sh --cand build/agents/r2B_9d6e --tag r2B_9d6e 2>&1 | tail -2

echo "===== [$(date +%H:%M)] ② 头对头（400 场）====="
bash tools/h2h.sh build/agents/r2A_9d6e build/agents/r1A_9d6e     400 r2A_vs_r1A_9d6e 2>&1 | tail -1
bash tools/h2h.sh build/agents/r1A_9d6e build/agents/port_on_9d6e 400 r1A_vs_port_9d6e 2>&1 | tail -1
bash tools/h2h.sh build/agents/r2B_9d6e build/agents/port_on_9d6e 400 r2B_vs_port_9d6e 2>&1 | tail -1
bash tools/h2h.sh build/agents/r2A_9d6e build/agents/r2B_9d6e     400 r2A_vs_r2B_9d6e 2>&1 | tail -1

echo "===== [$(date +%H:%M)] ③ 高样本复核 ogerpon_7e39（n=1000）====="
for c in r1A r2A port; do
  ag="build/agents/${c}_9d6e"; [ "$c" = port ] && ag=build/agents/port_on_9d6e
  [ -f "logs/hiN_7e39_${c}.json" ] && { echo "  [skip] $c（已有）"; continue; }
  echo "  --- $c"
  ./tools/py ptcg_league/eval_pair.py --cand "$ag" --opp "ogerpon_7e39=build/agents/port_7e39" \
    --matches 1000 --workers 0 --seed-base 1000000 --out "logs/hiN_7e39_${c}.json" \
    --logdir "logs/hiN_7e39_${c}" 2>&1 | tail -1
done
echo "===== [$(date +%H:%M)] 全部完成 ====="
