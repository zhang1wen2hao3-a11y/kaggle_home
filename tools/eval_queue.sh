#!/usr/bin/env bash
# **评测串行队列**：一次只跑一个臂，跑完再上下一个（避免互相抢 CPU）。
#
#   bash tools/eval_queue.sh start      # 后台启动（有锁，重复启动会被拒）
#   bash tools/eval_queue.sh stop       # 写入停止标记（当前臂跑完后退出）
#   bash tools/eval_queue.sh status     # 看队列进度
#   bash tools/eval_queue.sh run        # 前台串行执行（由 start 内部调用）
#
# 排序原则：先便宜后昂贵、先基线后变体；已完成的臂自动跳过（可断点续跑）。
# 串行保证：每个臂开始前都等待所有 eval_pair.py 进程结束——队列外的任务也不会被抢占。
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
LOCK=/tmp/eval_queue.lock
STOP=/tmp/eval_queue.stop
LOG=$ROOT/logs/eval_queue.log
mkdir -p "$ROOT/logs"

QUEUE=(
  "panel|build/agents/port_db04|port_db04|与 g6_db04 配对的端口基线"
  "meta|build/agents/c_grookey|meta_port_2617|12 格基线（2617 卡组）"
  "meta|build/agents/ph3_2617_v2|meta_ph3_2617_v2|训练@2617 INIT=port_w"
  "meta|build/agents/ph3_2617_v1|meta_ph3_2617_v1|训练@2617 INIT=g6_db04"
  "meta|build/agents/port_on_9d6e|meta_port_9d6e|12 格基线（9d6e 卡组）"
  "meta|build/agents/ph3_9d6e_v2|meta_ph3_9d6e_v2|训练@9d6e INIT=port_w"
  "meta|build/agents/g6_db04|meta_g6_db04|正例（天梯 1060）放到统一尺子"
)

wait_idle() {
  while pgrep -f "eval_pair\.py" > /dev/null 2>&1; do sleep 15; done
}

do_run() {
  echo $$ > "$LOCK"
  echo "[$(date +%T)] ===== 队列开始（$$）=====" >> "$LOG"
  for item in "${QUEUE[@]}"; do
    if [ -f "$STOP" ]; then echo "[$(date +%T)] 收到停止标记，退出" >> "$LOG"; break; fi
    IFS="|" read -r kind cand tag note <<< "$item"
    wait_idle
    echo "[$(date +%T)] ===== $tag ($kind) $note =====" >> "$LOG"
    if [ "$kind" = "panel" ]; then
      N=200 SEED=940000 bash tools/panel_eval.sh --cand "$cand" --tag "$tag" >> "$LOG" 2>&1
    else
      N=200 SEED=950000 bash tools/meta_eval.sh --cand "$cand" --tag "$tag" >> "$LOG" 2>&1
    fi
    echo "[$(date +%T)] ----- $tag 结束 (exit=$?) -----" >> "$LOG"
  done
  rm -f "$LOCK"
  echo "[$(date +%T)] 队列全部完成" >> "$LOG"
}

case "${1:-status}" in
  start)
    if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
      echo "✗ 队列已在运行（pid $(cat "$LOCK")）"; exit 1
    fi
    rm -f "$STOP"
    nohup bash "$ROOT/tools/eval_queue.sh" run > /dev/null 2>&1 &
    echo "✓ 队列已启动（日志 $LOG，pid $!）"
    ;;
  run) do_run ;;
  stop) touch "$STOP"; echo "✓ 已写入停止标记（当前臂跑完后退出）" ;;
  status)
    echo "--- 锁 ---"
    if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then echo "运行中 pid $(cat "$LOCK")"; else echo "未运行"; fi
    echo "--- 停止标记 ---"; [ -f "$STOP" ] && echo "存在" || echo "无"
    echo "--- 当前评测进程 ---"
    n_p=$(pgrep -f "eval_pair\.py" 2>/dev/null | wc -l)
    [ "$n_p" -gt 0 ] && echo "有（$n_p 个）" || echo "无"
    echo "--- 进度 ---"
    for item in "${QUEUE[@]}"; do
      IFS="|" read -r kind cand tag note <<< "$item"
      if [ "$kind" = "panel" ]; then d="$ROOT/logs/panel/$tag"; total=7; else d="$ROOT/logs/meta/$tag"; total=12; fi
      done_n=$(ls "$d"/vs_*.json 2>/dev/null | wc -l)
      mark="·"; [ "$done_n" -ge "$total" ] && mark="✓"
      printf "  %s %-22s %2d/%d  %s\n" "$mark" "$tag" "$done_n" "$total" "$note"
    done
    ;;
  *) sed -n '2,12p' "$0"; exit 1;;
esac
