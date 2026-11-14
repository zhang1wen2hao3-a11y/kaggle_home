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
#
# **机器掉线/交接的约定（用户定的规则）**
#   · 探测不到 = 大概率被关机 ⇒ 记一条"失联"日志；它的活由活着的那台用 mode=all 全接（幂等、不丢）
#   · 活着但**忙**（正在跑面板/手动评测）⇒ 不动它，下一轮再看
#   · 活着且空 ⇒ 立刻给它派活（2 台存活时分 shard 0/1，只有 1 台时跑 all）
#   · 新机器（没装过环境）⇒ 自动推包 + 装依赖(numpy/torch) + setup.sh，不用手工介入
#   · 两台都失联 ⇒ 只记一条"等明天开机器"，然后慢速轮询（10 分钟一次）
set -uo pipefail
cd "$(dirname "$0")/.."
HOSTS=(${1:?用法: bash tools/orchestrate_cpu.sh "root@A root@B"})
R=/root/eval_cpu_kit
LOG=logs/orchestrator.log; mkdir -p logs
say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" >> "$LOG"; }

probe() { ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes "$1" true 2>/dev/null; }

BUSY_PAT="run_wishlis[t]|meta_ev[a]l|eval_pai[r]|bank_ladde[r]|run_in_tabl[e]|queue_remainin[g]"
busy() {   # busy <host> -> 0=忙
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes "$1" \
      "pgrep -f '$BUSY_PAT' >/dev/null" 2>/dev/null
}

ensure_setup() {   # 新机器/被重置 ⇒ 补齐运行环境（**幂等、绝不 rm -rf 已有目录**）
  local h=$1
  local st
  st=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes "$h" \
       "test -d $R/build && echo KIT; python3 -c 'import torch' 2>/dev/null && echo DEPS; \
        test -f $R/tools/run_wishlist.sh && echo RUNNER; test -f /root/run_shard.sh && echo SHARD" \
       2>/dev/null | tr '\n' ' ')
  if ! echo "$st" | grep -q DEPS; then
    say "  🛠  $h 缺 numpy/torch ⇒ 自动安装（清华源 + 阿里云 wheel）"
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -o BatchMode=yes "$h" "
      (pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy filelock fsspec typing-extensions sympy networkx jinja2 markupsafe mpmath &&
       pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple 'https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.14.1%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl') \
       > /root/pip.log 2>&1 && echo DEPS_OK" 2>/dev/null | tail -1 | sed "s/^/    $h /" | tee -a "$LOG"
  fi
  if ! echo "$st" | grep -q KIT; then
    say "  🛠  $h 没有评估包 ⇒ 推 $(du -h dist/eval_cpu_kit.tar.gz | cut -f1) 并解包 + setup"
    rsync -az -e "ssh -o StrictHostKeyChecking=no" dist/eval_cpu_kit.tar.gz "$h:/root/" 2>/dev/null
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes "$h" \
        "cd /root && tar xzf eval_cpu_kit.tar.gz && cd eval_cpu_kit && bash setup.sh > /root/setup.log 2>&1 && echo SETUP_OK" \
        2>/dev/null | tail -1 | sed "s/^/    $h /" | tee -a "$LOG"
  else
    # 包在、但可能缺脚本（老包）⇒ 只补文件，不动目录
    if ! echo "$st" | grep -q RUNNER; then
      scp -q -o StrictHostKeyChecking=no "$(dirname "$0")/run_wishlist.sh" "$h:$R/tools/" 2>/dev/null
      scp -q -o StrictHostKeyChecking=no "$(dirname "$0")/bank_ladder.sh" "$h:$R/tools/" 2>/dev/null
      say "  🛠  $h 补 run_wishlist.sh / bank_ladder.sh（不重解包）"
    fi
    rsync -az -e "ssh -o StrictHostKeyChecking=no" logs/wishlist.tsv decks/testset_meta_A.tsv \
      "$h:$R/logs/" 2>/dev/null
    ssh -o StrictHostKeyChecking=no -o BatchMode=yes "$h" "cp -f $R/logs/testset_meta_A.tsv $R/decks/ 2>/dev/null; true" 2>/dev/null
  fi
  if ! echo "$st" | grep -q SHARD; then
    scp -q -o StrictHostKeyChecking=no "$(dirname "$0")/run_shard.sh" "$h:/root/run_shard.sh" 2>/dev/null
    ssh -o StrictHostKeyChecking=no -o BatchMode=yes "$h" "chmod +x /root/run_shard.sh" 2>/dev/null
  fi
}

note_state() {   # 记录 up/down 转变（只记变化，不刷屏）
  local h=$1 st=$2 f=logs/.host_state prev
  touch "$f"
  prev=$(awk -F= -v k="$h" '$1==k{print $2}' "$f")
  [ "$prev" = "$st" ] && return 0
  grep -v "^$h=" "$f" > "$f.tmp" 2>/dev/null; mv "$f.tmp" "$f"
  echo "$h=$st" >> "$f"
  case "$h=$prev→$st" in
    *"→down") say "  ⚠️ $h **失联**（大概是被关机了）—— 它的活由剩下那台接管" ;;
    *"→up")   say "  ✅ $h 上线" ;;
  esac
}

start_runner() {   # start_runner <host> <mode>
  local h=$1 mode=$2
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes "$h" \
    "pkill -f 'run_wishlis[t]' 2>/dev/null; pkill -f 'meta_eva[l].sh' 2>/dev/null; sleep 2; \
     setsid nohup bash /root/run_shard.sh $mode > /dev/null 2>&1 < /dev/null &" 2>/dev/null
}

for round in $(seq 1 720); do          # 60 小时
  alive=()
  for h in "${HOSTS[@]}"; do
    if probe "$h"; then alive+=("$h"); note_state "$h" up; else note_state "$h" down; fi
  done
  if [ ${#alive[@]} -eq 0 ]; then
    if [ $((round % 3)) = 1 ]; then
      say "⚠️ 两台都不可达（都关机了）—— 等你明天开机器；本地表保持最新，工作可恢复"
    fi
    sleep 600; continue
  fi
  say "存活 ${#alive[@]} 台：${alive[*]}"

  # ② 同步已完成面板给每台
  for h in "${alive[@]}"; do
    rsync -az --ignore-existing -e "ssh -o StrictHostKeyChecking=no" logs/meta/ "$h:$R/logs/meta/" 2>/dev/null
  done

  # ③④ 分配并确保 runner 在跑
  # 存活 1 台 → all（全跑）；存活 N 台 → 按主机顺序分 shard 0..N-1（队列 shard 列 = 行号 % N）
  modes=()
  for i in "${!alive[@]}"; do
    if [ ${#alive[@]} -eq 1 ]; then modes+=("all"); else modes+=("$i"); fi
  done
  for i in "${!alive[@]}"; do
    h="${alive[$i]}"; want="${modes[$i]:-all}"
    ensure_setup "$h"
    have=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes "$h" \
             "cat $R/logs/.runner_mode 2>/dev/null; pgrep -f 'run_wishlis[t]' >/dev/null && echo RUNNING" 2>/dev/null | tr '\n' ' ')
    if echo "$have" | grep -q "RUNNING" && echo "$have" | grep -q "\b$want\b"; then
      say "  $h 正常在跑 mode=$want"
    elif busy "$h"; then
      say "  $h 空着但**有别的任务在跑**（面板/手动评测）⇒ 不动它" 
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
