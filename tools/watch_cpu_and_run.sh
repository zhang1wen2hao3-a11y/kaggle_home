#!/usr/bin/env bash
# **盯住 CPU 评估机：等它上线 → 一键部署 → 把欠的对局跑完 → 拉回结果刷新表**
#
#   bash tools/watch_cpu_and_run.sh root@113.44.139.218 [端口]
#
# 设计成可以挂着跑几小时：每 60 秒探一次，机器一闪断就等它回来继续（所有步骤都幂等）。
# - 机器是全新实例（kit 没了）⇒ 自动 pack + 推 + setup（含 4 个历史最佳驾驶员）
# - kit 在但不全 ⇒ 只补缺的
# - 跑的是 tools/run_in_table.sh（38 对 × 2 种子 × 200 场，断点续跑）
# - 结束后本地 ./tools/py tools/build_in_table.py 刷新表
set -uo pipefail
cd "$(dirname "$0")/.."
HOST="${1:?用法: bash tools/watch_cpu_and_run.sh root@HOST [端口]}"
PORT="${2:-22}"
R="${R:-/root/eval_cpu_kit}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=12 -o BatchMode=yes -p $PORT $HOST"
LOG="${LOG:-logs/watch_cpu.log}"
mkdir -p logs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "开始盯 $HOST:$PORT"
for i in $(seq 1 720); do          # 12 小时内每 60 秒探一次
  if ! $SSH true 2>/dev/null; then
    [ $((i % 10)) = 1 ] && say "（等待机器上线…第 $i 次探测）"
    sleep 60; continue
  fi
  say "机器在线（第 $i 次探测）：$( $SSH 'hostname; nproc' 2>/dev/null | tr '\n' ' ' )"

  # ① 需要时打包
  if ! $SSH "test -d $R/build/port_plain" 2>/dev/null; then
    say "远端没有 kit ⇒ 本地打包"
    bash tools/pack_eval_cpu.sh >/dev/null 2>&1 || { say "✗ 打包失败"; exit 1; }
    say "推包 $(du -h dist/eval_cpu_kit.tar.gz | cut -f1)"
    rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" dist/eval_cpu_kit.tar.gz "$HOST:/root/" || { say "✗ 推包失败"; exit 1; }
    $SSH "cd /root && rm -rf eval_cpu_kit && tar xzf eval_cpu_kit.tar.gz && cd eval_cpu_kit && bash setup.sh" | tail -6 | sed 's/^/    /'
  else
    say "远端已有 kit"
    # 缺驾驶员就补（4 个 pt 一起推，幂等）
    if ! $SSH "test -d $R/build/agents/g6_db04" 2>/dev/null; then
      say "补推 4 个驾驶员的 model.pt"
      $SSH "mkdir -p $R/models_pilot"
      for k in g6_db04 ph3_2617_v1 r1A_26c6 g8_bc_parent; do
        rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" "build/agents/$k/model.pt" "$HOST:$R/models_pilot/$k.pt" || say "  ✗ $k"
      done
      $SSH "cd $R && declare -A PL=( [g6_db04]=dunsparce_abra__db04 [ph3_2617_v1]=grookey_thwackey__2617 [r1A_26c6]=slowpoke_slowking__26c6 [g8_bc_parent]=shuppet_dhelmise__0d86 );
        for k in \"\${!PL[@]}\"; do bash tools/make_agent_dir.sh \"build/agents/\$k\" \"models_pilot/\$k.pt\" \"decks/pool_field/\${PL[\$k]}.csv\" >/dev/null; done; ls build/agents | wc -l" | sed 's/^/    agent 数: /'
    fi
  fi

  # ② 推脚本 + 缺对清单（每次都同步，脚本可能改过）
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
    tools/run_in_table.sh tools/run_cpu_all.sh tools/build_in_table.py tools/meta_eval.sh tools/h2h.sh \
    tools/bo3_eval.py tools/make_agent_dir.sh "$HOST:$R/tools/" 2>/dev/null
  # ★ tools/py 必须是目标机自己的可移植壳；推本地那份会把 /home/zhang/... 的绝对路径带过去，
  #   导致所有评估报"✗ 找不到 /home/zhang/miniforge3/envs/py310/bin/python"（2026-11-14 踩过）
  $SSH "cd $R && printf '#!/usr/bin/env bash\nexec \"\${PYTHON:-python3}\" \"\$@\"\n' > tools/py && chmod +x tools/py" 2>/dev/null
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" logs/in_table_missing.txt "$HOST:$R/logs/" 2>/dev/null

  # ③ 起跑（后台，幂等）
  if $SSH "pgrep -f 'run_cpu_al[l]' >/dev/null" 2>/dev/null; then
    say "队列已在跑，继续等"
  else
    say "启动 tools/run_cpu_all.sh（内部对局 → 补种子 → 高样本格）"
    $SSH "cd $R && mkdir -p logs/in_table && setsid nohup bash tools/run_cpu_all.sh > logs/cpu_all.log 2>&1 & sleep 3; pgrep -f 'run_cpu_al[l]' | head -1" | sed 's/^/    pid /' 
  fi

  # ④ 等它跑完（最多 4 小时/轮）
  for j in $(seq 1 240); do
    if ! $SSH true 2>/dev/null; then say "机器掉了，回到外层等待"; break; fi
    if ! $SSH "pgrep -f 'run_cpu_al[l]' >/dev/null" 2>/dev/null; then
      n=$($SSH "ls $R/logs/in_table/*.json 2>/dev/null | wc -l" 2>/dev/null | tr -dc '0-9')
      say "补跑结束（远端 $n 个结果文件）⇒ 拉回"
      rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" --include='in_table/***' --include='meta/***' --include='hiN_*' --exclude='*' "$HOST:$R/logs/" logs/ 2>/dev/null
      ./tools/py tools/build_in_table.py 2>&1 | tail -4 | sed 's/^/    /'
      say "✓ 完成，退出"
      exit 0
    fi
    sleep 60
  done
done
say "✗ 12 小时内机器一直没上线"
exit 1
