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
DECK="${DECK:?需要 DECK（卡组文件基名，如 dunsparce_abra__0bf5）}"
# ★ 2026-11-21：有些 GPU 机器只有密码登录（公钥被 sshd StrictModes 拒）。
#   先用 askpass 建一个 ControlMaster 长连接，再把 -o ControlPath=... 通过 SSHEXTRA 传进来，
#   这样脚本里所有 ssh/rsync 都能复用那条已认证的 socket，不必改登录方式。
SSHEXTRA="${SSHEXTRA:-}"
R=/root/ptcg_train_kit/port_repo
CELLS="${CELLS:-ogerpon_7e39 dunsparce_0bf5 grookey_2617 out_91808263 dreepy_3eaa champ_e1d7 out_07be out_3f45 out_77a5 champ_0904 abra_f5f8 out_ea39 out_9bc9d655 out_0382 dunsparce_9d6e abra_e8ee champ_26c6}"
SEED2_CELLS="${SEED2_CELLS:-}"
read -r -a HOSTS <<< "${CPU_HOSTS:-root@113.44.216.143 root@1.92.97.36 root@113.44.139.218}"
LOG="logs/finish_${NAME}.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "等训练落地（每 30s 探，最多 90 分钟）"
for i in $(seq 1 180); do
  sz=$(timeout 25 ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $SSHEXTRA $GPU_HOST "stat -c %s $FINAL 2>/dev/null" 2>/dev/null | tr -d '[:space:]')
  run=$(timeout 20 ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $SSHEXTRA $GPU_HOST 'pgrep -fc "train_[n]ative" 2>/dev/null || true' 2>/dev/null | head -1 | tr -d '[:space:]')
  case "${sz:-}" in ''|*[!0-9]*) sz=0;; esac
  case "${run:-}" in ''|*[!0-9]*) run=0;; esac
  say "  第 $i 次：final=${sz}B 训练进程=$run"
  [ "$sz" = 18205408 ] && [ "$run" = 0 ] && break
  sleep 30
done
[ "${sz:-0}" = 18205408 ] || { say "✗ 等不到 $FINAL"; exit 1; }

mkdir -p "models/ppo/gpu_$RUN"
# 端口/主机分开取，rsync 用 -e "ssh ... -p PORT"（早先那版拼接出错 ⇒ final.bin 根本没拉下来，
# md5 比对必然失败，白丢一次训练产物；这次改成显式 PORT/HOSTNAME 再拼）。
SSHOPT=$(echo "$GPU_HOST" | sed -n 's/.*-p *\([0-9]\+\).*/\1/p'); SSHOPT=${SSHOPT:-22}
RHOST=$(echo "$GPU_HOST" | awk '{print $NF}')
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $SSHEXTRA -p $SSHOPT" "$RHOST:$FINAL" "models/ppo/gpu_$RUN/$RUN.bin" \
  || { say "✗ 拉 final 失败：$FINAL"; exit 1; }
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $SSHEXTRA -p $SSHOPT" "$RHOST:$R/experiments/native-$RUN/" "models/ppo/gpu_$RUN/snapshots/"
[ "$(stat -c %s "models/ppo/gpu_$RUN/$RUN.bin")" = 18205408 ] || { say "✗ 拉回的 final 大小不对"; exit 1; }
lf=$(md5sum "models/ppo/gpu_$RUN/$RUN.bin" | cut -d' ' -f1)
rf=$(timeout 25 ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $SSHEXTRA $GPU_HOST "md5sum $FINAL | cut -d' ' -f1")
[ "$lf" = "$rf" ] || { say "✗ final md5 不一致"; exit 1; }
say "✓ final md5 $lf"

cp "models/ppo/gpu_$RUN/$RUN.bin" "models/ppo/$NAME.bin"
PYTHONPATH=build/port_plain:. ./tools/py tools/native_weights.py import "models/ppo/$NAME.bin" "models/ppo/$NAME.pt" 2>&1 | tail -1
say "✓ models/ppo/$NAME.pt"

if ! grep -q "\"$NAME\"" tools/driver_registry.py; then
  python3 - "$NAME" <<'PY'
import sys
name=sys.argv[1]; deck=sys.argv[2]; p='tools/driver_registry.py'; s=open(p).read()
add=('    "%s": dict(agent="%s", weights="models/ppo/%s.pt",\n'
     '                   remote=None, deck="%s", fidelity=None,\n'
     '                   note="（自动登记，评测后补结论）"),\n') % (name,name,name,deck)
anchor='    "candf5f8D": dict('
open(p,'w').write(s.replace(anchor, add+anchor))
PY
  say "✓ 已登记 $NAME"
fi

say "═══ 关 GPU（产物已校验）═══"
timeout 30 ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $SSHEXTRA $GPU_HOST 'sync; nohup shutdown -h now > /dev/null 2>&1 & sleep 2; echo bye' 2>&1 | tail -1 | sed 's/^/  /' | tee -a "$LOG"

# ★ 2026-11-19 用户要求：GPU 跑完**立刻停**；CPU 机器可能已经按计划关掉，
#   这时不要回落到本地跑面板（用户明确禁止开发机跑评估）。SKIP_CPU=1 ⇒ 只做"拉权重→md5→转 pt→关机"。
if [ "${SKIP_CPU:-0}" = 1 ]; then
  say "SKIP_CPU=1：GPU 已关，权重已落 models/ppo/$NAME.bin/.pt，面板等有 CPU 机器时再跑"
  exit 0
fi

bash tools/make_agent_dir.sh "build/agents/$NAME" "models/ppo/$NAME.pt" "decks/pool_field/$DECK.csv" 2>&1 | tail -1 | tee -a "$LOG"
REACHABLE=()
for h in "${HOSTS[@]}"; do
  scp -q "models/ppo/$NAME.pt" "$h:/root/eval_cpu_kit/stage_$NAME.pt" || { say "  ✗ 推 $h 失败（可能已释放）"; continue; }
  REACHABLE+=("$h")
  scp -q tools/driver_registry.py "$h:/root/eval_cpu_kit/tools/"
  # ★ 2026-11-18 修：原来这里硬编码 decks/pool_field/abra_kadabra__f5f8.csv，
  #   非 f5f8 候选（如 e1d7 专项）会被铺成错卡组 ⇒ 现在统一用 $DECK
  timeout 150 ssh -o BatchMode=yes "$h" "cd /root/eval_cpu_kit && bash tools/make_agent_dir.sh build/agents/$NAME stage_$NAME.pt decks/pool_field/$DECK.csv 2>&1 | tail -1" | sed "s/^/  [$h] /" | tee -a "$LOG"
  bash tools/verify_launch.sh "$h" 'pgrep -fc "build/agents/'"$NAME"'" || true' "/root/dispatch_${NAME}_*.log" | tee -a "$LOG"
done

if [ ${#REACHABLE[@]} -eq 0 ]; then
  say "⚠️ 没有任何远程评估机可用 ⇒ **改在本地跑**"
  bash tools/eval_candidate_local.sh "$NAME" "$CELLS" ${SEED2_CELLS:+"$SEED2_CELLS"}
  ./tools/py tools/real_panel_agg.py "$NAME" 2>&1 | tee -a "$LOG" | tail -22
say "== 逐格差距诊断（我们 vs 高分玩家，判断是"我们偏弱"还是"对手偏弱需精修"）"
./tools/py tools/cell_gap_diagnosis.py --cand "$NAME" 2>&1 | tee -a "$LOG" | tail -24
  bash tools/git_snapshot.sh "$NAME 真实口径面板结果（本地兜底跑）" 2>&1 | tail -1 | tee -a "$LOG"
  say "DONE(local)"; exit 0
fi
HOSTS=("${REACHABLE[@]}")

say "== 17 格真实口径面板（seed 950000）"
./tools/py tools/dispatch_panel.py --cands "$NAME" --cells $CELLS --seeds 950000 --matches 150 \
   --workers 24 --interval 20 --hosts "${HOSTS[@]}" 2>&1 | tee -a "$LOG" | tail -3
if [ -n "$SEED2_CELLS" ]; then
  say "== 补第二种子：$SEED2_CELLS"
  ./tools/py tools/dispatch_panel.py --cands "$NAME" --cells $SEED2_CELLS --seeds 950001 --matches 150 \
     --workers 24 --interval 20 --hosts "${HOSTS[@]}" 2>&1 | tee -a "$LOG" | tail -3
fi
for h in "${HOSTS[@]}"; do
  timeout 90 rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $SSHEXTRA" --include '*/' --include 'vs_*.json' --exclude '*' \
    "$h:/root/eval_cpu_kit/logs/meta/" logs/meta/ 2>/dev/null
done
./tools/py tools/real_panel_agg.py "$NAME" 2>&1 | tee -a "$LOG" | tail -22
say "== 逐格差距诊断（我们 vs 高分玩家，判断是"我们偏弱"还是"对手偏弱需精修"）"
./tools/py tools/cell_gap_diagnosis.py --cand "$NAME" 2>&1 | tee -a "$LOG" | tail -24
bash tools/git_snapshot.sh "$NAME 真实口径面板结果" 2>&1 | tail -1 | tee -a "$LOG"
# ★ 收尾关机（2026-11-18 用户要求）：CPU 评估机在所有拉取/聚合/提交之后才关。
#   GPU 已经在上面"md5 校验通过"后立即关掉（同样属于"拉取结束再关"）。
if [ "${SHUTDOWN_CPU:-1}" = 1 ]; then
  EXPECT=$(( $(echo $CELLS | wc -w) + $(echo ${SEED2_CELLS:-} | wc -w) ))
  for h in "${HOSTS[@]}"; do
    say "收尾：$h（期望 ${EXPECT} 格）"
    bash tools/finish_box.sh "$h" "$EXPECT" || say "⚠️ $h 关机前校验未通过，保持开机"
  done
fi
say "DONE"
