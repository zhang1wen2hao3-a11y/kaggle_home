#!/usr/bin/env bash
# **新机器一键就绪 + 评测**：等 preflight → 推 payload → 铺 agent → 派活（含 20s 启动验证）→ 聚合
#   HOSTS="root@A root@B" CAND=candf5f8E bash tools/bootstrap_and_eval.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
read -r -a HOSTS <<< "${HOSTS:?需要 HOSTS}"; CAND="${CAND:?需要 CAND}"
CELLS="${CELLS:-ogerpon_7e39 dunsparce_0bf5 grookey_2617 out_91808263 dreepy_3eaa champ_e1d7 out_07be out_3f45 out_77a5 champ_0904 abra_f5f8 out_ea39 out_9bc9d655 out_0382 dunsparce_9d6e abra_e8ee champ_26c6}"
SEED2_CELLS="${SEED2_CELLS:-abra_f5f8 out_ea39 out_0382 out_07be dreepy_3eaa}"
LOG=logs/bootstrap_${CAND}.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

for h in "${HOSTS[@]}"; do
  say "等 $h 的 preflight（每 15s 探一次）"
  for i in $(seq 1 120); do
    ok=$(timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
      'python3 -c "import torch,numpy" 2>/dev/null && [ -d /root/eval_cpu_kit/build/agents ] && echo Y || echo N' 2>/dev/null | tail -1)
    [ "$ok" = Y ] && break
    sleep 15
  done
  [ "$ok" = Y ] || { say "✗ $h 环境没就绪，跳过"; continue; }
  say "  $h 就绪 → 推 payload"
  scp -q decks/testset_meta_real.tsv "$h:/root/eval_cpu_kit/decks/"
  scp -q tools/real_panel_run.sh tools/driver_registry.py tools/mk_newcells.sh "$h:/root/eval_cpu_kit/tools/"
  scp -q decks/pool_field/unseen__91808263.csv decks/pool_field/unseen__9bc9d655.csv "$h:/root/eval_cpu_kit/decks/pool_field/"
  scp -q "models/ppo/$CAND.pt" "$h:/root/stage_$CAND.pt"
  timeout 200 ssh -o BatchMode=yes "$h" "cd /root/eval_cpu_kit && bash tools/mk_newcells.sh 2>&1|tail -1 && \
     bash tools/make_agent_dir.sh build/agents/$CAND /root/stage_$CAND.pt decks/pool_field/abra_kadabra__f5f8.csv 2>&1|tail -1 && \
     echo agents=\$(ls build/agents|wc -l)" 2>&1 | sed "s/^/  [$h] /" | tee -a "$LOG"
done
READY=()
for h in "${HOSTS[@]}"; do
  n=$(timeout 15 ssh -o BatchMode=yes "$h" "ls -d /root/eval_cpu_kit/build/agents/$CAND 2>/dev/null | wc -l" 2>/dev/null | tr -d '[:space:]')
  [ "${n:-0}" = 1 ] && READY+=("$h")
done
[ ${#READY[@]} -gt 0 ] || { say "✗ 没有机器就绪 ⇒ 按纪律改**本地**跑"; bash tools/eval_candidate_local.sh "$CAND" "$CELLS" "$SEED2_CELLS" 2>&1|tail -3; ./tools/py tools/real_panel_agg.py "$CAND" 2>&1|tail -20; exit 0; }
say "就绪机器：${READY[*]}"

say "== 17 格真实口径面板（seed 950000）"
./tools/py tools/dispatch_panel.py --cands "$CAND" --cells $CELLS --seeds 950000 --matches 150 \
   --workers 24 --interval 20 --hosts "${READY[@]}" > "logs/dispatch_${CAND}_s950000.log" 2>&1 &
PID=$!
for h in "${READY[@]}"; do
  bash tools/verify_launch.sh "$h" 'pgrep -fc "build/agents/'"$CAND"'" 2>/dev/null || true' "/root/dispatch_${CAND}_*.log" | tee -a "$LOG"
done
wait $PID; tail -2 "logs/dispatch_${CAND}_s950000.log" | tee -a "$LOG"

say "== 第二种子：$SEED2_CELLS"
./tools/py tools/dispatch_panel.py --cands "$CAND" --cells $SEED2_CELLS --seeds 950001 --matches 150 \
   --workers 24 --interval 20 --hosts "${READY[@]}" 2>&1 | tail -2 | tee -a "$LOG"

for h in "${READY[@]}"; do
  timeout 120 rsync -az -e "ssh -o BatchMode=yes" --include '*/' --include 'vs_*.json' --exclude '*' \
    "$h:/root/eval_cpu_kit/logs/meta/" logs/meta/ 2>/dev/null
done
./tools/py tools/real_panel_agg.py "$CAND" 2>&1 | tee -a "$LOG" | tail -22
bash tools/refresh_tables.sh --quiet 2>&1|tail -1|tee -a "$LOG"
bash tools/git_snapshot.sh "$CAND 真实口径面板（17格+五格第二种子）" 2>&1|tail -1|tee -a "$LOG"
if [ "${SHUTDOWN_CPU:-1}" = 1 ]; then
  EXPECT=$(( $(echo $CELLS | wc -w) + $(echo ${SEED2_CELLS:-} | wc -w) ))
  for h in "${READY[@]}"; do
    say "收尾：$h（期望 ${EXPECT} 格）"
    bash tools/finish_box.sh "$h" "$EXPECT" || say "⚠️ $h 关机前校验未通过，保持开机"
  done
fi
say BOOTSTRAP_DONE
