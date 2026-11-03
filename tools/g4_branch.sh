#!/usr/bin/env bash
# **G4 三分支受控对照**（在 GPU 机器上跑）：只变"对手卡组分布 + 对手池"，其余全同。
#
# | 分支 | 对手卡组分布 | 对手 bank | 目的 |
# |---|---|---|---|
# | A | weights_real（混合 185 副） | 无 | 旧方案基线 |
# | B | weights_real（混合） | 2 个冻结 bank | 行为多样性 |
# | C | weights_only_shuppet（~100% shuppet） | **与 B 完全相同** | 卡组专精训练 |
#
# 受控要求（不满足就别跑）：同一个 init.bin / 同一个 --timesteps / lr / seed / AG / 评估协议。
#
# 依据（审计 §24）：训练器**没有**"bank 绑牌"语义，卡组权重是**全局 per-run**；
# 所以"专精"只能靠整轮把分布压到一个原型，且权重文件必须给 anchor 族留非零权重（否则 REJECTED）。
#
# 用法（云端）：
#   bash tools/g4_branch.sh                     # 默认 40 万步 / 512 agents，三支全跑
#   BRANCHES="C"  bash tools/g4_branch.sh        # 只跑 C
#   AG=768 T=1300000 bash tools/g4_branch.sh     # 加档
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${PTCG_REPO:-$HOME/work/ptcg_train_kit/port_repo}"
[ -f "$REPO/native/build_native.sh" ] || { echo "✗ 找不到训练仓库：$REPO（用 PTCG_REPO= 指定）"; exit 1; }
cd "$REPO"

PY="${PYTHON:-python3}"
T="${T:-400000}"                 # 短预算先筛（4090 ≈3 分钟 / 2080Ti ≈5–6 分钟）
AG="${AG:-512}"                  # 512 = 与 gen1–3 同量级；VRAM ≈17 GB / 22.5 GB
MB="${MB:-512}"
LR="${LR:-2e-4}"
SEED="${SEED:-424242}"
THREADS="${THREADS:-8}"          # 9 核留 1 个给系统
ANCHOR="teal_mask_ogerpon_ex_applin__7e39"
# 只观测、不影响采样（anchor 只放一副，否则 p_anchor_learner 会把学习者打散到多副）
TRACK="${TRACK:-shuppet_banette__f2af,shuppet_dhelmise__0d86,shuppet_dhelmise__fe82,teal_mask_ogerpon_ex_applin__7e39}"
OUT="${OUT:-/root/g4}"
BANKS="${BANKS:-/root/league_g0003/bank_0.bin,/root/league_g0003/bank_1.bin}"
BRANCHES="${BRANCHES:-A B C}"
mkdir -p "$OUT" logs

# 预检
for f in native/build_native.sh native/ptcg_tables_field.bin native/pt_model_field.bin \
         decks/weights_real.txt decks/weights_only_shuppet.txt; do
  [ -f "$f" ] || { echo "✗ 缺 $f"; exit 1; }
done
[ "$(stat -c%s native/pt_model_field.bin)" = "18205408" ] \
  || { echo "✗ native/pt_model_field.bin 不是 18,205,408 B（坏包，会导致 init.bin has 0 bytes）"; exit 1; }
if [ -n "$BANKS" ]; then
  IFS=, read -ra _b <<< "$BANKS"
  for b in "${_b[@]}"; do [ -f "$b" ] || { echo "✗ bank 不存在：$b"; exit 1; }; done
fi
echo "===== G4  T=$T AG=$AG LR=$LR SEED=$SEED THREADS=$THREADS ====="
echo "banks = ${BANKS:-（无）}"
echo "输出  = $OUT"
echo

vram_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }

run_branch() {  # $1=名字  $2=卡组权重  $3=banks(可空)
  local tag="$1" wf="$2" bnks="$3"
  local rd="experiments/native-g4_$tag"
  rm -rf "$rd"; mkdir -p "$rd"
  cp -f native/pt_model_field.bin "$rd/init.bin"
  [ -s "$rd/init.bin" ] || { echo "✗ [$tag] init.bin 拷贝失败"; return 1; }

  local used; used=$(vram_used)
  if [ "${used:-0}" -gt 4000 ]; then
    echo "⚠️ [$tag] 显存已被占 ${used} MiB，可能残留进程："
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv
    echo "   先 kill -9，否则会 CUDA OOM"; return 1
  fi

  echo "########## [G4-$tag] 卡组权重 = $(basename "$wf") ##########"
  PYTHONPATH=data:.:native PTCG_TABLES=native/ptcg_tables_field.bin OMP_WAIT_POLICY=PASSIVE \
  "$PY" native/train_native.py --run-name "g4_$tag" \
    --timesteps "$T" --no-anneal-lr \
    --total-agents "$AG" --minibatch "$MB" --num-threads "$THREADS" \
    --lr "$LR" --seed "$SEED" \
    --deck-pool decks/pool_field --deck-weights-file "$wf" \
    --anchor-decks "$ANCHOR" --track-decks "$TRACK" \
    --p-anchor-learner 1.0 --p-anchor-self-opp 0.0 --p-anchor-league-opp 0.0 \
    ${bnks:+--league-ckpts "$bnks" --league-pct 0.08} \
    2>&1 | tee "logs/g4_$tag.log" | grep -E "deck weights|REJECTED|epoch 5 |epoch 10 |final" | head -6

  if [ -f "$rd/final.bin" ]; then
    cp -f "$rd/final.bin" "$OUT/g4_$tag.bin"
    echo "✅ [$tag] → $OUT/g4_$tag.bin ($(stat -c%s "$OUT/g4_$tag.bin") B)"
  else
    echo "✗ [$tag] 没有 final.bin，看 logs/g4_$tag.log"
  fi
  echo
}

for b in $BRANCHES; do
  case "$b" in
    A) run_branch A decks/weights_real.txt "" ;;
    B) run_branch B decks/weights_real.txt "$BANKS" ;;
    C) run_branch C decks/weights_only_shuppet.txt "$BANKS" ;;
    *) echo "未知分支 $b" ;;
  esac
done

echo "===== 汇总（受控对照的关键记录）====="
printf '%-6s %-34s %-22s %-34s %s\n' 分支 卡组分布 ESS行 "实际分布(tl/to)" 模型
for b in $BRANCHES; do
  ess=$(grep -m1 -oE "ESS [0-9.]+" "logs/g4_$b.log" 2>/dev/null || echo "ESS ?")
  nact=$(grep -m1 -oE "[0-9]+/185 active" "logs/g4_$b.log" 2>/dev/null || echo "?")
  wf=$(grep -m1 -oE "deck-weights-file [^ ]+" "logs/g4_$b.log" 2>/dev/null | awk '{print $2}' || echo "?")
  tlto=$(grep -oE "tl[0-3]:[0-9.]+/to[0-3]:[0-9.]+" "logs/g4_$b.log" 2>/dev/null | tail -1 || echo "?")
  sz=$(stat -c%s "$OUT/g4_$b.bin" 2>/dev/null || echo 0)
  printf '%-6s %-34s %-22s %-34s %s\n' "G4-$b" "$(basename "${wf:-?}")" "$nact $ess" "${tlto:-?}" "${sz} B"
done
echo
echo "===== 硬门槛检查（缺证据则该支作废）====="
for b in $BRANCHES; do
  if grep -qE "to[0-3]:[0-9.]+" "logs/g4_$b.log" 2>/dev/null; then
    echo "  ✅ G4-$b 有实际分布证据"
  else
    echo "  ❌ G4-$b 没有 tl/to 字段 ⇒ 按预注册规则，该支不进胜率分析"
  fi
done
echo
echo "三个 final.bin 拉回本机后用**同一套冻结评估**打分（对端口 400 场 + 固定评测集），"
echo "再决定谁值得完整预算。注意：三支的 AG/T/LR/SEED/评估协议必须完全相同。"
