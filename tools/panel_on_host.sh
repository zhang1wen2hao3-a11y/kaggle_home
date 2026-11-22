#!/usr/bin/env bash
# **在一台 CPU 机上跑某个候选的 17 格面板**（可指定测试集口径 / 标签 / 并发数）
#
#   bash tools/panel_on_host.sh <候选> <host> [TESTSET] [TAG] [WORKERS]
#   例：bash tools/panel_on_host.sh cand0904R1 root@123.249.80.18 decks/testset_meta_real_v2.tsv v2 12
#
# 做的事：① 把测试集里 17 格的对手 agent 同步到远端（缺哪个推哪个，md5 校验）
#         ② 推候选权重 + 建 agent 目录  ③ 调度分片跑  ④ 拉回结果（tag 隔离）  ⑤ 打结果表
# ★ 不关机：关机由调用方在所有并发任务结束后统一做（tools/finish_box.sh）。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
NAME="${1:?候选名}"; H="${2:?host}"; TS="${3:-decks/testset_meta_real.tsv}"; TAG="${4:-real}"; W="${5:-12}"
KIT=/root/eval_cpu_kit
SSHOPT="-o BatchMode=yes -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
CELLS=$(awk -F'\t' '!/^#/ && NF>=3 && $1!="tail_mix"{printf "%s ", $1}' "$TS")
echo "[$(date +%H:%M:%S)] $NAME @ $H  口径=$TS 标签=$TAG workers=$W 格数=$(echo $CELLS | wc -w)"

push_md5() {  # $1 本地 $2 远端绝对路径
  local lm rm
  lm=$(md5sum "$1" | cut -d' ' -f1)
  rm=$(timeout 60 ssh $SSHOPT "$H" "md5sum '$2' 2>/dev/null | cut -d' ' -f1" | tr -d '[:space:]')
  [ "$lm" = "$rm" ] && return 0
  for i in 1 2 3; do
    scp -q $SSHOPT "$1" "$H:$2" 2>/dev/null
    rm=$(timeout 60 ssh $SSHOPT "$H" "md5sum '$2' 2>/dev/null | cut -d' ' -f1" | tr -d '[:space:]')
    [ "$lm" = "$rm" ] && return 0
    sleep 4
  done
  return 1
}

# ① 对手 agent 全部就位（v1 + v2 两套测试集用到的格 agent 都要有）
#   ★ 2026-11-22 两个坑（113 因此空转 25 分钟）：
#     ① 以前把 model.pt 直接 scp 到 build/agents/<n>/model.pt —— 目录不存在时 scp 静默失败；
#     ② make_agent_dir.sh 会 `rm -rf` 目标目录再 cp，所以**绝不能**拿目标目录里的 model.pt 当源
#        （等于删掉再拷自己 ⇒ 权重丢失）。统一改成：先推 $KIT/stage_<n>.pt，再 make_agent_dir。
for TSX in decks/testset_meta_real.tsv "$TS"; do
  awk -F'\t' '!/^#/ && NF>=3 && $1!="tail_mix"{print $2}' "$TSX" | sort -u | while read -r dirp; do
    case "$dirp" in build/agents/*) ;; *) continue;; esac
    nm=${dirp#build/agents/}
    timeout 40 ssh $SSHOPT "$H" "ls $KIT/build/agents/$nm/main.py" >/dev/null 2>&1 && continue
    [ -f "build/agents/$nm/model.pt" ] || { echo "  ⚠️ 本地缺 $nm/model.pt（该格会被跳过）"; continue; }
    deck=$(./tools/py -c "import sys;sys.path.insert(0,'tools');import glob;import driver_registry as d;s=d.deck_of('$nm') or '';c=glob.glob(f'decks/pool_field/*__{s}.csv') if s else [];print(c[0] if c else '')")
    [ -n "$deck" ] || { echo "  ⚠️ 推断不出 $nm 的牌表，跳过"; continue; }
    push_md5 "build/agents/$nm/model.pt" "$KIT/stage_$nm.pt" || { echo "  ✗ $nm 权重推不上去"; continue; }
    timeout 180 ssh $SSHOPT "$H" "cd $KIT && bash tools/make_agent_dir.sh build/agents/$nm stage_$nm.pt $deck 2>&1 | tail -1" | sed 's/^/  /'
  done
done

# ② 候选自己
push_md5 "models/ppo/$NAME.pt" "$KIT/stage_$NAME.pt" || { echo "✗ $NAME 权重推送失败"; exit 1; }
DECK=$(./tools/py - "$NAME" <<'PY'
import glob,sys
sys.path.insert(0,'tools')
try:
    import driver_registry as d; s=d.deck_of(sys.argv[1]) or ''
except Exception: s=''
c=(glob.glob(f'decks/pool_field/*__{s}.csv') if s else [])
print(c[0] if c else '')
PY
)
timeout 150 ssh $SSHOPT "$H" "cd $KIT && bash tools/make_agent_dir.sh build/agents/$NAME stage_$NAME.pt \"$DECK\" 2>&1 | tail -1" | sed 's/^/  /'

# ③ 跑（tag 隔离，不覆盖 v1 结果）
./tools/py tools/dispatch_panel.py --cands "$NAME" --cells $CELLS --seeds 950000 \
  --matches 150 --workers "$W" --interval 20 --hosts "$H" --testset "$TS" --tag "$TAG" \
  > "logs/dispatch_${NAME}_${TAG}.log" 2>&1
tail -2 "logs/dispatch_${NAME}_${TAG}.log"

# ④ 拉回
timeout 180 rsync -az --include '*/' --include 'vs_*.json' --exclude '*' -e "ssh $SSHOPT" \
  "$H:$KIT/logs/meta/" logs/meta/ || echo "  ⚠️ 拉取失败"
n=$(ls "logs/meta/${NAME}_${TAG}_s950000"/vs_*.json 2>/dev/null | wc -l)
echo "[$(date +%H:%M:%S)] $NAME 结果 $n/$(echo $CELLS | wc -w) 格"
# ⑤ 汇总
./tools/py tools/panel_rank.py --cands "$NAME" --tag "$TAG" 2>/dev/null | tail -3 || true
echo "[$(date +%H:%M:%S)] PANEL_HOST_DONE $NAME"
