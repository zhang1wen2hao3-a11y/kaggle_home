#!/usr/bin/env bash
# **GPU 训练 → 拉回 → 推到 CPU 机 → 起门禁评估 → 关 GPU**
#
#   bash tools/gpu_train_handoff.sh <gpu_host> <gpu_port> <run_name> <规范名> <out_dir> <cpu_host>
#
# 例：bash tools/gpu_train_handoff.sh root@connect.nmb2.seetacloud.com 38237 deck_26c6__db04 cand26c6S /root/slowking_shup4m root@113.44.216.143
#
# 做的事（全自动，日志在 logs/gpu_handoff.log）：
#   ① 每 60s 探一次训练：打印 epoch/step/SPS/lgw 进度
#   ② 训练进程退出且最终 .bin 存在 → rsync 拉回（最终模型 + 全部快照 + stdout 日志）
#   ③ 逐个 md5 与远端比对（**快照在 experiments/native-<run>/ 下，注意两个源目录**）
#   ④ 校验通过 → 用规范名落盘：models/ppo/<规范名>.pt + .bin；不合格就停，不关机
#   ⑤ 推到 CPU 机：建 agent 目录 + 起门禁评估（3 个单格 + 12 格面板 × 3 种子）
#   ⑥ 完成后关 GPU 机（只在③④都通过时）
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${1:?gpu host}"; GPORT="${2:?gpu port}"; RUN="${3:?run name}"; NAME="${4:?正式名}"
OUT="${5:-/root/run_out}"; CPU="${6:?cpu host}"; DECK="${7:?卡组文件基名，如 slowpoke_slowking__26c6}"
CPU_PORT="${CPU_PORT:-22}"
R=/root/ptcg_train_kit/port_repo
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p $GPORT $GPU"
LOG=logs/gpu_handoff.log; mkdir -p logs
say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }

say "盯 $GPU:$GPORT  run=$RUN  正式名=$NAME  → 回传后推到 $CPU 跑门禁"

for i in $(seq 1 240); do          # 最多 4 小时
  if ! $SSH true 2>/dev/null; then
    [ $((i % 10)) = 1 ] && say "（GPU 不可达，第 $i 次）"
    sleep 60; continue
  fi
  running=$($SSH "pgrep -f 'train_nativ[e]' >/dev/null && echo yes || echo no")
  last=$($SSH "tail -1 $R/logs/$RUN.log 2>/dev/null | cut -c1-110")
  final=$($SSH "ls $OUT/*.bin 2>/dev/null | head -1")
  if [ "$running" = "yes" ]; then
    [ $((i % 5)) = 1 ] && say "训练中：$last"
    sleep 60; continue
  fi
  if [ -z "$final" ]; then
    say "⚠️ 训练进程已退出但 $OUT 下没有 .bin —— 不关机，等人看"; exit 1
  fi
  say "✅ 训练结束（$last）→ 开始拉产物"

  # ② 拉回（run 名以最终 .bin 的文件名为准，别用传进来的 RUN —— 训练器有时会用 INIT 名做前缀）
  OKRUN=$(basename "$final"); OKRUN="${OKRUN%.bin}"
  say "  实际 run 名推断为 $OKRUN"
  mkdir -p "models/ppo/gpu_$NAME" "models/ppo/gpu_$NAME/snapshots"
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p $GPORT" "$GPU:$OUT/" "models/ppo/gpu_$NAME/" || true
  for cand in "$OKRUN" "$RUN"; do
    rsync -az -e "ssh -o StrictHostKeyChecking=no -p $GPORT" \
      "$GPU:$R/experiments/native-$cand/" "models/ppo/gpu_$NAME/snapshots/" 2>/dev/null && break
  done
  for cand in "$OKRUN" "$RUN"; do
    rsync -az -e "ssh -o StrictHostKeyChecking=no -p $GPORT" \
      "$GPU:$R/logs/$cand.log" "models/ppo/gpu_$NAME/train.log" 2>/dev/null && break
  done
  [ -s "models/ppo/gpu_$NAME/train.log" ] || say "  ⚠️ 训练日志没拉到（不影响权重，仅影响追溯）"
  [ "$(ls models/ppo/gpu_$NAME/snapshots/*.bin 2>/dev/null | wc -l)" -gt 0 ] || say "  ⚠️ 快照没拉到（只有最终模型；想按快照挑最好的，下次改 --checkpoint-interval 名字匹配）"

  # ③ md5 逐个校验（两个源目录都要比）
  ok=yes
  for f in $(cd "models/ppo/gpu_$NAME" && find . -name '*.bin' -printf '%P\n' 2>/dev/null); do
    l=$(md5sum "models/ppo/gpu_$NAME/$f" | cut -d' ' -f1)
    r=$($SSH "md5sum $OUT/${f##*/} 2>/dev/null | cut -d' ' -f1; md5sum $R/experiments/native-$RUN/${f##*/} 2>/dev/null | cut -d' ' -f1")
    echo "$r" | grep -qx "$l" && say "  ✓ $f" || { say "  ✗ $f md5 不一致"; ok=no; }
  done
  [ "$ok" = yes ] || { say "⚠️ md5 校验失败 ⇒ 停，不落盘不关机"; exit 1; }

  # ④ 用规范名落盘（.pt 是评测用的 torch 格式，.bin 是训练器用的 native blob）
  cp -f "models/ppo/gpu_$NAME/${final##*/}" "models/ppo/$NAME.bin"
  PYTHONPATH=build/port_plain:. ./tools/py tools/native_weights.py import "models/ppo/$NAME.bin" "models/ppo/$NAME.pt" \
    2>&1 | tail -1 | sed 's/^/  /' | tee -a "$LOG"
  [ -f "models/ppo/$NAME.pt" ] || { say "✗ 转 .pt 失败"; exit 1; }
  say "  → models/ppo/$NAME.{bin,pt} 就绪"

  # ⑤ 推 CPU 机 + 建 agent + 起门禁
  say "推 $NAME.pt 到 $CPU 并起门禁评估"
  scp -q -o StrictHostKeyChecking=no -P "$CPU_PORT" "models/ppo/$NAME.pt" "$CPU:/root/eval_cpu_kit/models_ceiling/$NAME.pt"
  ssh -o StrictHostKeyChecking=no -p "$CPU_PORT" "$CPU" "cd /root/eval_cpu_kit &&
      bash tools/make_agent_dir.sh build/agents/$NAME models_ceiling/$NAME.pt decks/pool_field/$DECK.csv >/dev/null 2>&1;
      ls build/agents/$NAME/model.pt" 2>/dev/null | tail -1 | sed 's/^/  /' | tee -a "$LOG"

  cat > /tmp/gate_$NAME.sh <<EOS
#!/usr/bin/env bash
cd /root/eval_cpu_kit || exit 1
./tools/py ptcg_league/eval_pair.py --cand build/agents/$NAME --opp shbc=build/agents/g8_bc_parent \\
   --matches 200 --workers 24 --seed-base 950000 --out data/opp_matrix/${NAME}_vs_shbc.json 2>&1 | tail -1
./tools/py ptcg_league/eval_pair.py --cand build/agents/$NAME --opp f5f8=build/agents/deck_f5f8_ \\
   --matches 200 --workers 24 --seed-base 950000 --out data/opp_matrix/${NAME}_vs_f5f8.json 2>&1 | tail -1
./tools/py ptcg_league/eval_pair.py --cand build/agents/$NAME --opp port=build/agents/port_26c6 \\
   --matches 200 --workers 24 --seed-base 950000 --out data/opp_matrix/${NAME}_mirror.json 2>&1 | tail -1
for sd in 950000 950001 950002; do
  [ -f "logs/meta/${NAME}_s\${sd}/summary.json" ] && continue
  TS=decks/testset_meta.tsv N=150 SEED=\$sd bash tools/meta_eval.sh --cand build/agents/$NAME --tag "${NAME}_s\${sd}" 2>&1 | tail -1
done
echo GATE_DONE
EOS
  scp -q -o StrictHostKeyChecking=no -P "$CPU_PORT" "/tmp/gate_$NAME.sh" "$CPU:/root/gate_$NAME.sh"
  ssh -o StrictHostKeyChecking=no -p "$CPU_PORT" "$CPU" \
      "chmod +x /root/gate_$NAME.sh; setsid nohup bash /root/gate_$NAME.sh > /root/gate_$NAME.log 2>&1 < /dev/null & sleep 5; echo started" \
      2>/dev/null | tail -1 | sed 's/^/  /' | tee -a "$LOG"

  # ⑥ 关机（产物已校验 + 已推 CPU）
  say "═══ 产物已校验并推到 CPU ⇒ 关 GPU 机 ═══"
  $SSH "sync; nohup shutdown -h now > /dev/null 2>&1 & sleep 2; echo bye" 2>/dev/null | tail -1 | sed 's/^/  /' | tee -a "$LOG"
  say "完成。门禁日志：$CPU:/root/gate_$NAME.log"
  exit 0
done
say "⚠️ 4 小时未完成，退出（不关机）"; exit 2
