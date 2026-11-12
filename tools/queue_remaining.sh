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
  [ -d "build/agents/ab9d6e_$k" ] || need "models_ppo/ab9d6e_$k.pt"
  [ -d "build/agents/ab9d6e_$k" ] || bash tools/make_agent_dir.sh "build/agents/ab9d6e_$k" "models_ppo/ab9d6e_$k.pt" decks/pool_field/dunsparce_abra__9d6e.csv
done

echo "===== [$(date +%H:%M)] ① r2B@9d6e 12 格面板 ====="
bash tools/meta_eval.sh --cand build/agents/ab9d6e_r2B --tag ab9d6e_r2B 2>&1 | tail -2

echo "===== [$(date +%H:%M)] ② 决定性对照：port vs r1A 各再跑 2 个种子（合并后 n≈600/格）====="
# 为什么不是"只把 ogerpon 一格加大样本"：+2.4pp 的 CI 是 [-1.0, +5.9]，
# 半宽 ∝ 1/√n ⇒ 整面板再跑 2 个独立种子（等价 3× 样本）才能把半宽压到 ~1.4pp。
# 只加一格只是砍掉一个方差项，其他 11 格的噪声还在。
for sd in 950001 950002; do
  for spec in "meta_port_9d6e:build/agents/port_on_9d6e" "ab9d6e_r1A:build/agents/ab9d6e_r1A"; do
    tag="${spec%%:*}"; ag="${spec##*:}"
    t="${tag}_s${sd}"
    [ -f "logs/meta/$t/summary.json" ] && { echo "  [skip] $t（已完成）"; continue; }
    echo "  --- $t"
    SEED=$sd bash tools/meta_eval.sh --cand "$ag" --tag "$t" 2>&1 | tail -1
  done
done

echo "===== [$(date +%H:%M)] ③ 头对头（400 场）====="
bash tools/h2h.sh build/agents/ab9d6e_r2A build/agents/ab9d6e_r1A     400 ab9d6e_r2A_vs_r1A 2>&1 | tail -1
bash tools/h2h.sh build/agents/ab9d6e_r1A build/agents/port_on_9d6e 400 ab9d6e_r1A_vs_port 2>&1 | tail -1
bash tools/h2h.sh build/agents/ab9d6e_r2B build/agents/port_on_9d6e 400 ab9d6e_r2B_vs_port 2>&1 | tail -1
bash tools/h2h.sh build/agents/ab9d6e_r2A build/agents/ab9d6e_r2B     400 ab9d6e_r2A_vs_r2B 2>&1 | tail -1

echo "===== [$(date +%H:%M)] ④ ogerpon_7e39 单格高样本（n=1000，作为补充证据）====="
for c in r1A r2A port; do
  ag="build/agents/ab9d6e_$c"; [ "$c" = port ] && ag=build/agents/port_on_9d6e
  [ -f "logs/hiN_7e39_${c}.json" ] && { echo "  [skip] $c（已有）"; continue; }
  echo "  --- $c"
  ./tools/py ptcg_league/eval_pair.py --cand "$ag" --opp "ogerpon_7e39=build/agents/port_7e39" \
    --matches 1000 --workers 0 --seed-base 1000000 --out "logs/hiN_7e39_${c}.json" \
    --logdir "logs/hiN_7e39_${c}" 2>&1 | tail -1
done
echo "===== [$(date +%H:%M)] 全部完成 ====="
echo
echo "合并多种子看显著性："
echo "  ./tools/py tools/panel_ci.py --base meta_port_9d6e --base meta_port_9d6e_s950001 --base meta_port_9d6e_s950002 \\"
echo "      --pool 'ab9d6e_r1A=ab9d6e_r1A,ab9d6e_r1A_s950001,ab9d6e_r1A_s950002' "
