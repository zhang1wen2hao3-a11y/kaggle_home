#!/usr/bin/env bash
# 打**评估机专用**的包（Ubuntu 22.04、32 核、无 GPU）。
#
# 为什么单独打：训练机那套（ptcg_train_kit）里塞着 port_repo / CUDA / 卡组训练脚本，
# 评估机一个都用不上；而评估只需要「引擎 + 推理骨架 + 面板对手 + 卡组 + eval_pair」。
# 评估机上不需要 nvcc、不需要编译、不需要 1.4 GB 的训练仓库。
#
#   bash tools/pack_eval_kit.sh                 # 默认：5 个 FIELD 对手 + g6_db04 候选
#   bash tools/pack_eval_kit.sh --cand g6_db04 --cand r1A_db04_g6
#
# 产出一份 dist/ptcg_eval_kit.tar.gz，评估机上：
#   tar xzf ptcg_eval_kit.tar.gz && cd ptcg_eval && bash setup.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
STAGE="dist/eval_kit_stage/ptcg_eval"
OUT="dist/ptcg_eval_kit.tar.gz"

CANDS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --cand) CANDS+=("$2"); shift 2;;
    *) echo "✗ 未知参数 $1"; exit 1;;
  esac
done
[ ${#CANDS[@]} -eq 0 ] && CANDS=(g6_db04)

rm -rf "$(dirname "$STAGE")"; mkdir -p "$STAGE"

echo "① 引擎与推理骨架（build/port_plain，28 MB）"
mkdir -p "$STAGE/build"
cp -a build/port_plain "$STAGE/build/port_plain"
# 根目录这份是给**父进程**用的：eval_pair 在主进程里 import cg.game 才能驱动对局
for f in cg data EN_Card_Data.csv r2e_manifest.json; do
  cp -a "build/port_plain/$f" "$STAGE/$f"
done

echo "② 评估器与工具"
mkdir -p "$STAGE/ptcg_league" "$STAGE/tools" "$STAGE/decks" "$STAGE/build/agents" \
         "$STAGE/models" "$STAGE/panel/models"
cp -f ptcg_league/eval_pair.py "$STAGE/ptcg_league/"
for f in tools/bo3_eval.py tools/make_agent_dir.sh tools/py; do cp -f "$f" "$STAGE/tools/"; done
chmod +x "$STAGE/tools/py" "$STAGE/tools/make_agent_dir.sh"

echo "③ 卡组池（187 副 + 对手权重）"
cp -a decks/pool_field "$STAGE/decks/pool_field"
cp -f decks/weights_real.txt "$STAGE/decks/" 2>/dev/null || true

# ── ④ 面板对手：同一个 model.pt 被打包一次，按 sha 去重 ─────────────────────
# 实测：5 个 FIELD 对手里有 4 个的 model.pt 都指向 build/port_plain/model.pt（同一个
# 策略、只是卡组不同），逐个复制会白塞 60 MB。
echo "④ 面板对手（去重后只装不同的 model.pt）"
: > "$STAGE/panel/agents.tsv"
for spec in \
  "ogerpon_7e39=build/agents/port_7e39" \
  "abra_f5f8=build/agents/c_abra" \
  "grookey_2617=build/agents/c_grookey" \
  "shuppet_0d86=build/agents/opp_shuppet_human" \
  "dreepy_3eaa=build/agents/c_dreepy" ; do
  name="${spec%%=*}"; dir="${spec##*=}"
  [ -f "$dir/model.pt" ] || { echo "✗ 缺 $dir/model.pt"; exit 1; }
  real=$(readlink -f "$dir/model.pt")
  tag=$(md5sum "$real" | cut -c1-8)
  m="$STAGE/panel/models/$tag.pt"
  [ -f "$m" ] || cp -f "$real" "$m"
  # ★ deck 必须按**对手名**存：4 个对手共用同一个 model.pt（同一个 port 策略、只是
  #   牌不同），按 model tag 存会互相覆盖 —— 实测让 ogerpon/abra/grookey 全在打
  #   dreepy 的牌，基线胜率从 0.66 掉到 0.27，而表面上毫无报错。
  mkdir -p "$STAGE/panel/decks"
  cp -f "$dir/deck.csv" "$STAGE/panel/decks/$name.csv"
  printf '%s\t%s\t%s\n' "$name" "$tag" "$(basename "$real")" >> "$STAGE/panel/agents.tsv"
  echo "   $name → $tag ($(basename "$real"))"
done

# ── ⑤ 候选模型 ─────────────────────────────────────────────────────────────
echo "⑤ 候选模型"
for c in "${CANDS[@]}"; do
  pt="models/ppo/$c.pt"
  if [ ! -f "$pt" ]; then
    bin="models/ppo/$c.bin"
    [ -f "$bin" ] || { echo "✗ 没有 $pt 也没有 $bin"; exit 1; }
    echo "   $c: 由 .bin 现转 .pt"
    PYTHONPATH="build/port_plain:$ROOT" ./tools/py port_repo/native/tools/native_weights.py import "$bin" "$pt" >/dev/null || {
      echo "✗ bin→pt 失败: $c"; exit 1; }
  fi
  cp -f "$pt" "$STAGE/models/$c.pt"
  # 候选用哪副牌：取 dual state 里该侧 deck（拿不到就按 db04 兜底）
  deck=$(./tools/py - "$c" <<'PY'
import json, glob, os, sys
c = sys.argv[1]
for f in glob.glob("ptcg_league/state/dual_*.json"):
    st = json.load(open(f))
    for side in ("A", "B"):
        m = os.path.basename(st["sides"][side]["model"])
        if m.startswith(c):
            print(st["sides"][side]["deck"]); raise SystemExit
print("dunsparce_abra__db04")
PY
)
  printf '%s\t%s\n' "$c" "$deck" >> "$STAGE/models/cands.tsv"
  echo "   $c @ $deck"
done

# ── ⑥ setup.sh：评估机上一条命令铺好 agent 目录 ────────────────────────────
cat > "$STAGE/setup.sh" <<'EOS'
#!/usr/bin/env bash
# 评估机初始化：把 panel/models/*.pt 铺成标准 agent 目录。
# 为什么不在打包机上铺：agent 目录里的 cg/ptcg/data/main.py 是**绝对路径**软链，
# 跨机器就是死链；必须在目标机上重新 ln。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
PY=${PYTHON:-python3}
echo "=== 环境自检 ==="
$PY -V || { echo "✗ 没有 $PY"; exit 1; }
$PY -c "import torch,numpy;print('torch',torch.__version__,'numpy',numpy.__version__)" \
  || { echo "✗ 缺 torch/numpy：pip install --index-url https://download.pytorch.org/whl/cpu torch numpy"; exit 1; }
[ -f build/port_plain/cg/libcg.so ] || { echo "✗ build/port_plain/cg/libcg.so 缺失（包不完整）"; exit 1; }
echo "=== 铺面板对手 ==="
while IFS=$'\t' read -r name tag _; do
  [ -z "$name" ] && continue
  bash tools/make_agent_dir.sh "build/agents/$name" "panel/models/$tag.pt" "panel/decks/$name.csv" >/dev/null
  echo "   ✓ build/agents/$name"
done < panel/agents.tsv
echo "=== 铺候选 ==="
while IFS=$'\t' read -r c deck; do
  [ -z "$c" ] && continue
  bash tools/make_agent_dir.sh "build/agents/ag_$c" "models/$c.pt" "decks/pool_field/$deck.csv" >/dev/null
  echo "   ✓ build/agents/ag_$c  (deck=$deck)"
done < models/cands.tsv
echo
echo "✅ 就绪。跑一次配对评估："
echo "   $PY ptcg_league/eval_pair.py --cand build/agents/ag_${c:-g6_db04} \\"
while IFS=$'\t' read -r name _ _; do [ -n "$name" ] && echo "       --opp $name=build/agents/$name \\"; done < panel/agents.tsv
echo "       --matches 1250 --workers 0 --seed-base 950000 --out /root/ev.json --logdir /root/ev_log"
EOS
chmod +x "$STAGE/setup.sh"

echo "⑥ 打包"
tar czf "$OUT" -C "$(dirname "$STAGE")" ptcg_eval
printf "✅ %s  %.1f MB  sha16 %s\n" "$OUT" \
  "$(stat -c%s "$OUT" | awk '{print $1/1048576}')" "$(sha256sum "$OUT" | cut -c1-16)"
echo "   评估机： tar xzf $(basename "$OUT") && cd ptcg_eval && bash setup.sh"
