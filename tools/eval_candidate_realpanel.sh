#!/usr/bin/env bash
# **给某个候选跑"真实口径 17 格面板"**（结果已训练好、只差评测时用这一条）
#
#   bash tools/eval_candidate_realpanel.sh candf5f8D [host1 host2 ...]
#   默认机器：root@113.44.216.143 root@1.92.97.36 root@113.44.139.218
#   可选：SEED2_CELLS="out_ea39 out_07be ..." 给这些格补第二种子（n→300）
#
# 做的事：推 .pt → 各机 make_agent_dir → 中央调度器分片跑 → 拉回 vs_*.json → 真实口径聚合 → 提交 git
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
NAME="${1:?用法: bash tools/eval_candidate_realpanel.sh <候选名> [hosts...]}"; shift || true
HOSTS=("$@"); [ ${#HOSTS[@]} -eq 0 ] && HOSTS=(root@113.44.216.143 root@1.92.97.36 root@113.44.139.218)
CELLS="ogerpon_7e39 dunsparce_0bf5 grookey_2617 out_91808263 dreepy_3eaa champ_e1d7 out_07be out_3f45 out_77a5 champ_0904 abra_f5f8 out_ea39 out_9bc9d655 out_0382 dunsparce_9d6e abra_e8ee champ_26c6"
SEED2_CELLS="${SEED2_CELLS:-out_ea39 out_07be dreepy_3eaa out_0382 out_77a5}"
[ -s "models/ppo/$NAME.pt" ] || { echo "✗ 缺 models/ppo/$NAME.pt"; exit 1; }

up=0
for h in "${HOSTS[@]}"; do
  scp -q "models/ppo/$NAME.pt" "$h:/root/eval_cpu_kit/stage_$NAME.pt" || { echo "✗ 推 $h 失败（机器可能已被释放）"; continue; }
  scp -q tools/driver_registry.py "$h:/root/eval_cpu_kit/tools/"
  timeout 150 ssh -o BatchMode=yes "$h" "cd /root/eval_cpu_kit && bash tools/make_agent_dir.sh build/agents/$NAME stage_$NAME.pt decks/pool_field/abra_kadabra__f5f8.csv 2>&1 | tail -1" | sed "s/^/  [$h] /"
  up=$((up+1))
done
[ "$up" -gt 0 ] || { echo "✗ 没有任何机器可达，先把机器开起来再跑这条"; exit 1; }

echo "== 17 格真实口径面板（seed 950000）"
# ★ 启动验证（2026-11-17 纪律）：派完 20 秒内必须确认远端真有进程，否则立刻报错退出。
#   之前"测试集 agent 路径写成裸名"就是这样白等了两轮 30 分钟。
./tools/py tools/dispatch_panel.py --cands "$NAME" --cells $CELLS --seeds 950000 \
   --matches 150 --workers 24 --interval 20 --hosts "${HOSTS[@]}" > "logs/dispatch_${NAME}_s950000.log" 2>&1 &
DISPATCH_PID=$!
for h in "${HOSTS[@]}"; do
  bash tools/verify_launch.sh "$h" 'pgrep -fc "ptcg_league/[e]val_pair" || true' "/root/dispatch_${NAME}_*.log" \
    || { echo "✗ $h 上派活没跑起来，见上（日志 logs/dispatch_${NAME}_s950000.log）"; kill $DISPATCH_PID 2>/dev/null; exit 2; }
done
wait $DISPATCH_PID
tail -4 "logs/dispatch_${NAME}_s950000.log"
if [ -n "$SEED2_CELLS" ]; then
  echo "== 补第二种子（n→300）"
  ./tools/py tools/dispatch_panel.py --cands "$NAME" --cells $SEED2_CELLS --seeds 950001 \
     --matches 150 --workers 24 --interval 20 --hosts "${HOSTS[@]}" 2>&1 | tail -3
fi
for h in "${HOSTS[@]}"; do
  timeout 90 rsync -az -e "ssh -o BatchMode=yes" --include '*/' --include 'vs_*.json' --exclude '*' \
    "$h:/root/eval_cpu_kit/logs/meta/" logs/meta/ 2>/dev/null
done
./tools/py tools/real_panel_agg.py "$NAME" 2>&1 | tail -25
bash tools/git_snapshot.sh "$NAME 真实口径面板结果" 2>&1 | tail -1
echo EVAL_CANDIDATE_DONE
