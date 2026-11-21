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

# ★ 2026-11-22 修 bug：牌表以前**写死 f5f8** ⇒ 给 e1d7/0904 的候选建 agent 时用的是 f5f8 的牌，
#   权重是自己的、牌是别人的，整份面板是错的。现在按注册表 deck 字段反查池子文件名；DECK=... 可覆盖。
if [ -z "${DECK:-}" ]; then
  DECK=$(./tools/py - "$NAME" <<'PY'
import glob, sys
sys.path.insert(0, 'tools')
try:
    import driver_registry as d
    short = d.deck_of(sys.argv[1]) or ''
except Exception:
    short = ''
cands = (glob.glob(f'decks/pool_field/*__{short}.csv') if short else []) + \
        (glob.glob(f'decks/*__{short}.csv') if short else [])
print(cands[0] if cands else '')
PY
)
fi
[ -s "${DECK:-/nonexistent}" ] || { echo "✗ 推断不出 $NAME 的牌表（注册表 deck 字段 / decks/pool_field/*__<deck>.csv）—— 用 DECK=... 指定"; exit 1; }
echo "== 牌表 $DECK"

up=0
SSHOPT="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
# ★ 2026-11-22：20 MB 的 .pt 推一次偶尔会 "lost connection"（网抖），原来没有重试 ⇒ 整条面板直接放弃。
#   现在：① 远端 md5 已一致就跳过；② 失败重试 3 次；③ 推完必校验 md5。
push_file() {  # $1=本地文件 $2=远端绝对路径
  local lm rm t
  lm=$(md5sum "$1" | cut -d' ' -f1)
  rm=$(timeout 60 ssh $SSHOPT "$h" "md5sum '$2' 2>/dev/null | cut -d' ' -f1" | tr -d '[:space:]')
  [ "$lm" = "$rm" ] && { echo "  已就位（md5 一致，跳过推送）"; return 0; }
  for t in 1 2 3; do
    scp -q $SSHOPT -o ServerAliveInterval=15 "$1" "$h:$2" 2>/dev/null
    rm=$(timeout 60 ssh $SSHOPT "$h" "md5sum '$2' 2>/dev/null | cut -d' ' -f1" | tr -d '[:space:]')
    [ "$lm" = "$rm" ] && return 0
    echo "  第 $t 次推送失败/校验不符，重试"; sleep 5
  done
  return 1
}

for h in "${HOSTS[@]}"; do
  push_file "models/ppo/$NAME.pt" "/root/eval_cpu_kit/stage_$NAME.pt" || { echo "✗ 推 $h 失败（3 次重试后仍校验不符，机器可能已被释放）"; continue; }
  scatter=$(scp -q $SSHOPT tools/driver_registry.py "$h:/root/eval_cpu_kit/tools/" 2>&1) || echo "  ⚠ registry 没推上去：$scatter"
  timeout 150 ssh $SSHOPT "$h" "cd /root/eval_cpu_kit && bash tools/make_agent_dir.sh build/agents/$NAME stage_$NAME.pt \"$DECK\" 2>&1 | tail -1" | sed "s/^/  [$h] /"
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

# ★ 收尾关机（2026-11-18 用户要求）：**所有拉取与校验完成之后**才关评估机。
#   SHUTDOWN_CPU=0 可关闭该行为；校验不过（坏文件/格数不足）则不关机并告警。
if [ "${SHUTDOWN_CPU:-1}" = 1 ]; then
  EXPECT=$(( $(echo $CELLS | wc -w) + $(echo ${SEED2_CELLS:-} | wc -w) ))
  for h in "${HOSTS[@]}"; do
    echo "== 收尾：$h（期望 ${EXPECT} 格）"
    bash tools/finish_box.sh "$h" "$EXPECT" || echo "⚠️ $h 关机前校验未通过，保持开机"
  done
fi
echo EVAL_CANDIDATE_DONE
