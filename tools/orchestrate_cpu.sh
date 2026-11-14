#!/usr/bin/env bash
# **两台 CPU 机的调度器**：分片并行、掉线自动接管、全丢则等新机器
#
#   bash tools/orchestrate_cpu.sh "root@A root@B"
#
# 每 5 分钟一轮：
#   ① 探测哪些机器还活着（都不活就等，不退出）
#   ② 把本地 logs/meta（已完成面板）同步给活着的机器 → 谁都跳过已完成的
#   ③ 按存活数决定每个跑什么：
#        2 台 → 按顺序分 shard 0 / shard 1（互不重复）
#        1 台 → 跑 all（把另一台欠的活全接过来）
#   ④ 模式不对/没在跑 → 杀掉重启（幂等，已完成自动跳过）
#   ⑤ 反向拉回结果，并刷新母表与两张交付表
#   ⑥ 愿望单跑完后自动接 run_in_table.sh（内部对局 38 对）
set -uo pipefail
cd "$(dirname "$0")/.."
HOSTS=(${1:?用法: bash tools/orchestrate_cpu.sh "root@A root@B"})
R=/root/eval_cpu_kit
LOG=logs/orchestrator.log; mkdir -p logs
say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" >> "$LOG"; }

probe() { ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes "$1" true 2>/dev/null; }

start_runner() {   # start_runner <host> <mode>
  local h=$1 mode=$2
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes "$h" \
    "pkill -f 'run_wishlis[t]' 2>/dev/null; pkill -f 'meta_eva[l].sh' 2>/dev/null; sleep 2; \
     setsid nohup bash /root/run_shard.sh $mode > /dev/null 2>&1 < /dev/null &" 2>/dev/null
}

for round in $(seq 1 720); do          # 60 小时
  alive=()
  for h in "${HOSTS[@]}"; do probe "$h" && alive+=("$h"); done
  if [ ${#alive[@]} -eq 0 ]; then
    [ $((round % 6)) = 1 ] && say "⚠️ 两台都不可达，等你开新机器（第 $round 轮）"
    sleep 300; continue
  fi
  say "存活 ${#alive[@]} 台：${alive[*]}"

  # ② 同步已完成面板给每台
  for h in "${alive[@]}"; do
    rsync -az --ignore-existing -e "ssh -o StrictHostKeyChecking=no" logs/meta/ "$h:$R/logs/meta/" 2>/dev/null
  done

  # ③④ 分配并确保 runner 在跑
  if [ ${#alive[@]} -eq 1 ]; then modes=("all"); else modes=(0 1); fi
  for i in "${!alive[@]}"; do
    h="${alive[$i]}"; want="${modes[$i]:-all}"
    have=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes "$h" \
             "cat $R/logs/.runner_mode 2>/dev/null; pgrep -f 'run_wishlis[t]' >/dev/null && echo RUNNING" 2>/dev/null | tr '\n' ' ')
    if echo "$have" | grep -q "RUNNING" && echo "$have" | grep -q "\b$want\b"; then
      say "  $h 正常在跑 mode=$want"
    else
      say "  $h 需要(重)启动 mode=$want（当前: ${have:-无}）"
      start_runner "$h" "$want"
    fi
  done

  # ⑤ 拉回结果 + 刷新表
  for h in "${alive[@]}"; do
    rsync -az -e "ssh -o StrictHostKeyChecking=no" \
      --include='meta/***' --include='in_table/***' --include='h2h/***' --include='vs9d6e/***' --include='hiN_*' \
      --exclude='*' "$h:$R/logs/" logs/ 2>/dev/null
  done
  # ★ 固定流程（规则见 docs/驾驶员与覆盖规则.md）：母表 → 两口径交付表 → 名字校验 → 方案A测试集
  bash tools/refresh_tables.sh --quiet > /dev/null 2>&1 || say "  ⚠️ refresh_tables 失败（表未刷新）"
  done_panels=$(ls -d logs/meta/*_s9500* 2>/dev/null | while read d; do [ -f "$d/summary.json" ] && echo x; done | wc -l)
  say "  本地已完成面板 $done_panels；表已刷新"

  # ⑥ 愿望单跑完 → 接内部对局
  if [ ! -f logs/.wishlist_done ] && [ "$done_panels" -ge 100 ]; then
    say "✅ 愿望单基本跑完（$done_panels 面板）→ 触发 run_in_table"
    for h in "${alive[@]}"; do
      ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes "$h" \
        "cd $R && setsid nohup bash -c 'cd $R && exec bash tools/run_in_table.sh >> logs/in_table_shard.log 2>&1' > /dev/null 2>&1 < /dev/null &" 2>/dev/null
    done
    touch logs/.wishlist_done
  fi
  sleep 300
done
