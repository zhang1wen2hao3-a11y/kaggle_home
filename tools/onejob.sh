#!/usr/bin/env bash
# **一次只跑一个重任务 + 内存天花板 90%**
#
# 本机只有 7.5 GB。两条硬规则：
#   1. 同一个 named lock 只能有一个持有者（第二个直接退出，不排队、不互相拖慢）；
#   2. 整体内存占用不许越过 `CEIL_PCT`（默认 90%）——
#      启动前不够就拒绝启动；跑起来越界就每 10 秒报警（只报警，不杀进程：杀错的代价更大）。
#
# 用法：
#   tools/onejob.sh bc   -- ./tools/py -u tools/bc_train.py ... --max-frames 60000
#   tools/onejob.sh eval -- ./tools/py ptcg_league/run_league.py next --resume
#
# 环境变量：CEIL_PCT=90 / MIN_MB=<启动所需最小可用内存> / LOCK_DIR=/tmp/ptcg_locks
set -uo pipefail

CEIL_PCT="${CEIL_PCT:-90}"
LOCK_DIR="${LOCK_DIR:-/tmp/ptcg_locks}"
mkdir -p "$LOCK_DIR"

mem() {  # → "total_mb avail_mb"
  awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{printf "%d %d", t/1024, a/1024}' /proc/meminfo
}

read -r MEM_TOTAL MEM_AVAIL <<< "$(mem)"
USED_PCT=$(( (MEM_TOTAL - MEM_AVAIL) * 100 / MEM_TOTAL ))
CEIL_MB=$(( MEM_TOTAL * CEIL_PCT / 100 ))
MIN_MB="${MIN_MB:-$(( MEM_TOTAL * (100 - CEIL_PCT) / 100 ))}"

name="${1:?用法: onejob.sh <name> -- <cmd...>}"
shift
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "✗ 没给命令"; exit 2; }

if [ "$MEM_AVAIL" -lt "$MIN_MB" ]; then
  echo "✗ 现在已用 ${USED_PCT}%（可用 ${MEM_AVAIL} MB），开新任务会破 ${CEIL_PCT}% 天花板。占用的进程："
  ps -eo pid,rss,args --sort=-rss | head -6 | cut -c1-110
  exit 75
fi

lock="$LOCK_DIR/$name.lock"
exec 9>"$lock"
got=0
for _ in $(seq 1 20); do            # 最多等 20 s：上一个任务的残留 fd 通常几秒内释放
  if flock -n 9; then got=1; break; fi
  sleep 1
done
if [ "$got" = 0 ]; then
  echo "✗ [$name] 已被 pid $(cat "$lock.pid" 2>/dev/null || echo '?') 占用 —— 一次只跑一个重任务"
  ps -eo pid,rss,args --sort=-rss | head -6 | cut -c1-110
  exit 76
fi
echo $$ > "$lock.pid"
echo "$*" > "$lock"
echo "[onejob:$name] pid=$$ 已用 ${USED_PCT}%（可用 ${MEM_AVAIL} MB；天花板 ${CEIL_PCT}% = ${CEIL_MB} MB）"

"$@" 9>&- &
child=$!

# 看门狗：越界只报警，不杀
(
  n=0
  while kill -0 "$child" 2>/dev/null; do
    sleep 10
    kill -0 "$child" 2>/dev/null || break
    read -r t a <<< "$(mem)"
    u=$(( (t - a) * 100 / t ))
    if [ "$u" -gt "$CEIL_PCT" ] && [ "$n" -lt 30 ]; then
      n=$((n+1))
      echo "[onejob:$name] $u% > ${CEIL_PCT}%（可用 ${a} MB）—— 考虑停掉非必要进程" >&2
    fi
  done
) 9>&- &
wd=$!

t0=$(date +%s)
wait "$child"
rc=$?
kill "$wd" 2>/dev/null
wait "$wd" 2>/dev/null
t1=$(date +%s)
read -r t2 a2 <<< "$(mem)"
echo "[onejob:$name] 退出 rc=$rc 用时 $((t1-t0))s  收尾已用 $(( (t2-a2)*100/t2 ))%（启动时 ${USED_PCT}%）"
rm -f "$lock.pid"
exit $rc
