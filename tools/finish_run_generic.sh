#!/usr/bin/env bash
# **通用收尾**：等 GPU 训练出模型 → 校验并拉回 → 转 .pt → 关 GPU → 推 CPU → 真实口径面板 → 聚合
#
# 必填环境变量：
#   GPU_HOST  ssh 参数（含 -p 端口，**小写 -p**）  FINAL  远端 final.bin 路径
#   RUN       训练 run 名（experiments/native-$RUN、logs/$RUN.log 用它）  NAME  新候选名
# 可选：CELLS（默认 17 格）、SEED2_CELLS（默认空）、CPU_HOSTS（默认三台）
#
#   GPU_HOST="-p 21760 -o ConnectTimeout=15 root@host" FINAL=/root/x/y.bin RUN=candf5f8D_db04 \
#     NAME=candf5f8E bash tools/finish_run_generic.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
: "${GPU_HOST:?}" "${FINAL:?}" "${RUN:?}" "${NAME:?}"
R=/root/ptcg_train_kit/port_repo
CELLS="${CELLS:-ogerpon_7e39 dunsparce_0bf5 grookey_2617 out_91808263 dreepy_3eaa champ_e1d7 out_07be out_3f45 out_77a5 champ_0904 abra_f5f8 out_ea39 out_9bc9d655 out_0382 dunsparce_9d6e abra_e8ee champ_26c6}"
SEED2_CELLS="${SEED2_CELLS:-}"
read -r -a HOSTS <<< "${CPU_HOSTS:-root@113.44.216.143 root@1.92.97.36 root@113.44.139.218}"
LOG="logs/finish_${NAME}.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "等训练落地（每 30s 探，最多 90 分钟）"
for i in $(seq 1 180); do
  sz=$(timeout 25 ssh -o BatchMode=yes $GPU_HOST "stat -c %s $FINAL 2>/dev/null" 2>/dev/null | tr -d '[:space:]')
  run=$(timeout 20 ssh -o BatchMode=yes $GPU_HOST 'pgrep -fc "train_[n]ative" 2>/dev/null || true' 2>/dev/null | head -1 | tr -d '[:space:]')
  case "${sz:-}" in ''|*[!0-9]*) sz=0;; esac
  case "${run:-}" in ''|*[!0-9]*) run=0;; esac
  say "  第 $i 次：final=${sz}B 训练进程=$run"
  [ "$sz" = 18205408 ] && [ "$run" = 0 ] && break
  sleep 30
done
[ "${sz:-0}" = 18205408 ] || { say "✗ 等不到 $FINAL"; exit 1; }

mkdir -p "models/ppo/gpu_$RUN"
rsync -az -e "ssh -o BatchMode=yes ${GPU_HOST%% *} $(echo "$GPU_HOST" | grep -q -- '-p ' && echo "-p $(echo "$GPU_HOST" | sed 's/.*-p \([0-9]*\).*/\1/')")" \
  "${GPU_HOST##* }:$FINAL" "models/ppo/gpu_$RUN/$RUN.bin"
rsync -az -e "ssh -o BatchMode=yes -p $(echo "$GPU_HOST" | sed 's/.*-p \([0-9]*\).*/\1/')" \
  "${GPU_HOST##* }:$R/experiments/native-$RUN/" "models/ppo/gpu_$RUN/snapshots/"
lf=$(md5sum "models/ppo/gpu_$RUN/$RUN.bin" | cut -d' ' -f1)
rf=$(timeout 25 ssh -o BatchMode=yes $GPU_HOST "md5sum $FINAL | cut -d' ' -f1")
[ "$lf" = "$rf" ] || { say "✗ final md5 不一致"; exit 1; }
say "✓ final md5 $lf"

cp "models/ppo/gpu_$RUN/$RUN.bin" "models/ppo/$NAME.bin"
PYTHONPATH=build/port_plain:. ./tools/py tools/native_weights.py import "models/ppo/$NAME.bin" "models/ppo/$NAME.pt" 2>&1 | tail -1
say "✓ models/ppo/$NAME.pt"

if ! grep -q "\"$NAME\"" tools/driver_registry.py; then
  python3 - "$NAME" <<'PY'
import sys
name=sys.argv[1]; p='tools/driver_registry.py'; s=open(p).read()
add=('    "%s": dict(agent="%s", weights="models/ppo/%s.pt",\n'
     '                   remote=None, deck="f5f8", fidelity=None,\n'
     '                   note="（自动登记，评测后补结论）"),\n') % (name,name,name)
anchor='    "candf5f8D": dict('
open(p,'w').write(s.replace(anchor, add+anchor))
PY
  say "✓ 已登记 $NAME"
fi

say "═══ 关 GPU（产物已校验）═══"
timeout 30 ssh -o BatchMode=yes $GPU_HOST 'sync; nohup shutdown -h now > /dev/null 2>&1 & sleep 2; echo bye' 2>&1 | tail -1 | sed 's/^/  /' | tee -a "$LOG"

bash tools/make_agent_dir.sh "build/agents/$NAME" "models/ppo/$NAME.pt" decks/pool_field/abra_kadabra__f5f8.csv 2>&1 | tail -1 | tee -a "$LOG"
for h in "${HOSTS[@]}"; do
  scp -q "models/ppo/$NAME.pt" "$h:/root/eval_cpu_kit/stage_$NAME.pt" || { say "  ✗ 推 $h 失败（可能已释放）"; continue; }
  scp -q tools/driver_registry.py "$h:/root/eval_cpu_kit/tools/"
  timeout 150 ssh -o BatchMode=yes "$h" "cd /root/eval_cpu_kit && bash tools/make_agent_dir.sh build/agents/$NAME stage_$NAME.pt decks/pool_field/abra_kadabra__f5f8.csv 2>&1 | tail -1" | sed "s/^/  [$h] /" | tee -a "$LOG"
  bash tools/verify_launch.sh "$h" 'ls -d /root/eval_cpu_kit/build/agents/'"$NAME"' 2>/dev/null | wc -l' '/root/setup.log' | tee -a "$LOG"
done

say "== 17 格真实口径面板（seed 950000）"
./tools/py tools/dispatch_panel.py --cands "$NAME" --cells $CELLS --seeds 950000 --matches 150 \
   --workers 24 --interval 20 --hosts "${HOSTS[@]}" 2>&1 | tee -a "$LOG" | tail -3
if [ -n "$SEED2_CELLS" ]; then
  say "== 补第二种子：$SEED2_CELLS"
  ./tools/py tools/dispatch_panel.py --cands "$NAME" --cells $SEED2_CELLS --seeds 950001 --matches 150 \
     --workers 24 --interval 20 --hosts "${HOSTS[@]}" 2>&1 | tee -a "$LOG" | tail -3
fi
for h in "${HOSTS[@]}"; do
  timeout 90 rsync -az -e "ssh -o BatchMode=yes" --include '*/' --include 'vs_*.json' --exclude '*' \
    "$h:/root/eval_cpu_kit/logs/meta/" logs/meta/ 2>/dev/null
done
./tools/py tools/real_panel_agg.py "$NAME" 2>&1 | tee -a "$LOG" | tail -22
bash tools/git_snapshot.sh "$NAME 真实口径面板结果" 2>&1 | tail -1 | tee -a "$LOG"
say "DONE"
