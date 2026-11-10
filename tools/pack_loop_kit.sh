#!/usr/bin/env bash
# 打**无人值守循环整包**：训练 + 评估 + 判定 + 换边，一个包装到 GPU 机自己跑完。
#
#   bash tools/pack_loop_kit.sh
#   → dist/ptcg_loop_kit.tar.gz
#
# 布局刻意做成"我们仓库的瘦身副本"，因为 eval_pair/bo3_eval 依赖
# ROOT/tools/bo3_eval.py + build/agents/<名> 这套相对路径。
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
STAGE="$ROOT/build/loop_kit/ptcg_loop"
rm -rf "$ROOT/build/loop_kit"; mkdir -p "$STAGE"

say() { printf "\n[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }

say "1/6 训练仓库（含两个 C 补丁）"
cp -a port_repo "$STAGE/port_repo"
rm -rf "$STAGE/port_repo/experiments" "$STAGE/port_repo/logs" \
       "$STAGE/port_repo/native/build" "$STAGE/port_repo/model_wide.pt"
find "$STAGE/port_repo" -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true

say "2/6 评估器 + 引擎骨架"
mkdir -p "$STAGE/tools" "$STAGE/ptcg_league" "$STAGE/framework" "$STAGE/build"
for t in bo3_eval.py dual_train.py make_agent_dir.sh db04_line.sh patch_bank_decks.py; do
  cp -f "tools/$t" "$STAGE/tools/"
done
# 运行期可用 PYTHON=... 覆盖（打包机写死会在换环境时踩坑）
printf '#!/usr/bin/env bash\nexec "${PYTHON:-python3}" "$@"\n' > "$STAGE/tools/py"
chmod +x "$STAGE/tools/py"
cp -f ptcg_league/eval_pair.py "$STAGE/ptcg_league/"
cp -f framework/replay.py "$STAGE/framework/" 2>/dev/null || true
cp -f framework/__init__.py "$STAGE/framework/" 2>/dev/null || true
# 包根也要有 cg/ 与 data/：bo3_eval 的 play_match 在**父进程**里 `from cg.game import ...`，
# 而 eval_pair 把 ROOT 加进 sys.path ⇒ 需要 ROOT/cg 存在（本地仓库根凭 .gitignore 里的副本）
cp -a build/port_plain/cg "$STAGE/cg"
cp -a build/port_plain/data "$STAGE/data"
cp -f build/port_plain/EN_Card_Data.csv build/port_plain/r2e_manifest.json "$STAGE/"
# 骨架：cg/ ptcg/ data/ main.py ...（agent 目录靠它建软链）
cp -a build/port_plain "$STAGE/build/port_plain"
rm -rf "$STAGE/build/port_plain/model.pt"   # 20 MB，下面按需放

# dual_train.train 会调 <repo>/tools/db04_line.sh —— 必须铺进训练仓库
mkdir -p "$STAGE/port_repo/tools"
cp -f tools/db04_line.sh tools/patch_bank_decks.py "$STAGE/port_repo/tools/"

say "3/6 面板对手（15 个，权重硬链接去重）"
mkdir -p "$STAGE/build/agents"
UPLINK="$STAGE/build/port_plain/model_port.pt"
cp -f build/port_plain/model.pt "$UPLINK"
N=0
while IFS=$'\t' read -r rank deck8 name share adir; do
  case "$rank" in \#*|"") continue;; esac
  [ -f "$adir/deck.csv" ] || { echo "  ⚠️ 跳过（无 deck.csv）$adir"; continue; }
  d="$STAGE/build/agents/$(basename "$adir")"
  mkdir -p "$d"
  for l in EN_Card_Data.csv cg data main.py ptcg r2e_manifest.json; do
    ln -sfn "../../port_plain/$l" "$d/$l"
  done
  cp -f "$adir/deck.csv" "$d/deck.csv"
  # 权重去重：与端口权重相同的用硬链接，不同的单独拷
  if cmp -s "$adir/model.pt" build/port_plain/model.pt; then ln -f "$UPLINK" "$d/model.pt"
  else cp -f "$adir/model.pt" "$d/model.pt"; fi
  N=$((N+1))
done < data/panel_top15.tsv
echo "  面板对手 $N 个"

say "4/6 卡组池 + 权重 + 起始模型"
mkdir -p "$STAGE/decks/pool_field" "$STAGE/models/ppo" "$STAGE/state" "$STAGE/logs/dual" "$STAGE/runs"
cp -f decks/pool_field/*.csv "$STAGE/decks/pool_field/"
cp -f decks/weights_real.txt "$STAGE/decks/"
for m in models/ppo/g6_db04.bin bc_pool/grookey.bin; do
  [ -f "$m" ] && cp -f "$m" "$STAGE/models/ppo/" && echo "  起始模型 $(basename $m)"
done

say "5/6 loop.sh + RUNBOOK"
cp -f build/loop_sh.sh "$STAGE/loop.sh"; chmod +x "$STAGE/loop.sh"
cp -f build/loop_RUNBOOK.md "$STAGE/RUNBOOK.md"

say "6/6 打包"
( cd "$ROOT/build/loop_kit" && tar czf "$ROOT/dist/ptcg_loop_kit.tar.gz" ptcg_loop )
ls -la "$ROOT/dist/ptcg_loop_kit.tar.gz" | awk '{printf "  %.1f MB\n", $5/1048576}'
sha256sum "$ROOT/dist/ptcg_loop_kit.tar.gz" | cut -c1-16
