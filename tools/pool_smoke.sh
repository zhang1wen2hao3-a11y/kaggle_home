#!/usr/bin/env bash
# **对手池 smoke test**（在 GPU 机器上跑，只 rollout、不训练 PPO）
#
# 只回答一个问题：**训练时对手座位实际抽到的是什么卡组**。
# 不估计强度、不看胜率结论 —— 胜率只用来验"有没有异常"。
#
# 依据（审计，见 下一步.md §24）：
#   · 训练器**没有**"bank 绑牌"语义（`docs/training.md:131`），卡组一律从全局权重抽；
#   · 卡组权重是**全局 per-run**（`T->deck_cw` 在 tables blob 里，`env.c:778`），不是逐 bank；
#   · 因此"专精"只能靠**整轮把分布压到一个原型**（`decks/weights_only_<原型>.txt`）。
#
# 两组对照：
#   A 原始混合   decks/weights_real.txt              → 应复现旧的抽牌分布
#   B 专精抽牌   decks/weights_only_shuppet.txt      → 应只抽到 shuppet 族
#
# 用法（云端）：
#   bash tools/pool_smoke.sh            # 默认每组 300 局
#   EPISODES=600 bash tools/pool_smoke.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${PTCG_REPO:-$HOME/work/ptcg_train_kit/port_repo}"
[ -f "$REPO/native/build_native.sh" ] || { echo "✗ 找不到训练仓库：$REPO（用 PTCG_REPO= 指定）"; exit 1; }
cd "$REPO"

PY="${PYTHON:-python3}"
EPISODES="${EPISODES:-300}"
OUT="${OUT:-$REPO/logs/smoke}"
mkdir -p "$OUT"
TBL=native/ptcg_tables_field.bin
POOL=decks/pool_field

# bank 必须是**权重 blob**（.bin），不是卡组。默认用包里的两个 shuppet 脑，缺就退化为纯抽牌验证。
BANKS_DEFAULT=""
for cand in "$HERE"/bank_*.bin ../bc_pool/agent_sue124_.bin ../bc_pool/deck_0d86_.bin; do
  [ -f "$cand" ] && BANKS_DEFAULT="$BANKS_DEFAULT,$(cd "$(dirname "$cand")" && pwd)/$(basename "$cand")"
done
BANKS_DEFAULT="${BANKS_DEFAULT#,}"
echo "===== 对手池 smoke test  EPISODES=$EPISODES（极短真训练，不用 rollout-only）====="
echo "repo   = $REPO"
echo "输出   = $OUT"
echo

run_group() {  # $1=组名  $2=权重文件
  local tag="$1" wf="$2"
  local dm="$OUT/deckmat_$tag"
  local rd="experiments/native-smoke_$tag"
  mkdir -p "$rd"
  # ⚠️ 必须无条件覆盖 + 校验大小：`[ -f ] || cp` 会把上次残留的 0 字节 init.bin 当"已存在"跳过，
  #   然后训练器报 "init.bin has 0 bytes" 直接退出（实测踩过）。
  cp -f native/pt_model_field.bin "$rd/init.bin"
  [ -s "$rd/init.bin" ] || { echo "✗ 拷贝 init.bin 失败（$rd/init.bin 是空/不存在）"; exit 1; }
  rm -f "$dm".*
  echo "########## [$tag] 卡组权重 = $(basename "$wf") ##########"
  PYTHONPATH=data:.:native PTCG_TABLES="$TBL" OMP_WAIT_POLICY=PASSIVE \
  PTCG_DECK_MATRIX="$dm" PTCG_DECK_MATRIX_EVERY="$EPISODES" \
  "$PY" native/train_native.py \
    --run-name "smoke_$tag" \
    --timesteps "$((EPISODES * 400))" \
    --no-anneal-lr \
    --total-agents 128 --minibatch 128 \
    --deck-pool "$POOL" --deck-weights-file "$wf" \
    --anchor-decks "teal_mask_ogerpon_ex_applin__7e39" --p-anchor-learner 1.0 \
    --p-anchor-self-opp 0.0 --p-anchor-league-opp 0.0 \
    ${BANKS:+--league-ckpts "$BANKS"} ${BANKS:+--league-pct 0.06} \
    --deck-matrix-every "${MATRIX:-0}" \
    --seed 424242 \
    2>&1 | tee "$OUT/$tag.log" | grep -E "epoch|deck weights|league|episodes|saved" | tail -12
  echo
}

# 两个权重文件必须已经在**训练仓库**里（kit 只带 weights_real）
for wf in decks/weights_real.txt decks/weights_only_shuppet.txt; do
  if [ ! -f "$wf" ]; then
    for cand in "$HERE/../decks/$(basename "$wf")" "$HERE/$(basename "$wf")"; do
      [ -f "$cand" ] && cp -f "$cand" "$wf" && echo "已放入 $wf（来自 $cand）"
    done
  fi
  [ -f "$wf" ] || { echo "✗ 缺 $wf"; echo "   从本机补：scp -P <端口> decks/$(basename "$wf") <用户>@<主机>:$REPO/decks/"; exit 1; }
done

BANKS="${BANKS:-$BANKS_DEFAULT}"
[ -n "$BANKS" ] || echo "⚠️ 没找到 bank 权重 blob ⇒ 只验证抽牌分布（够用：本轮就为验它）"

SMOKE_GROUPS="${SMOKE_GROUPS:-A_mixed B_only}"
case " $SMOKE_GROUPS " in *" A_mixed "*) run_group A_mixed decks/weights_real.txt ;; esac
case " $SMOKE_GROUPS " in *" B_only "*)  run_group B_only  decks/weights_only_shuppet.txt ;; esac

echo "===== 验收（看原生报的 deck weights 行：ESS/min/max 由**实际生效**的累积权重算出）====="
for tag in A_mixed B_only; do
  echo "[$tag]"
  grep -m1 "deck weights ->" "$OUT/$tag.log" 2>/dev/null | sed 's/^/  /' || echo "  （没有该行：训练没真正启动，看日志尾）"
  grep -m1 "epoch 1 " "$OUT/$tag.log" 2>/dev/null | grep -oE "ep_len [0-9]+|mir [0-9.]+|fwr [0-9.]+|d0 [0-9.]+" | sed 's/^/  /'
done
echo
echo "判定标准："
echo "  A_mixed ESS 应 ~29、max ~0.058（＝旧混合分布）"
echo "  B_only  ESS 应 ~3、  max ~0.33（＝只剩 shuppet 三副，专精抽牌生效）"
echo "  B_only 若 ESS 仍 ~29 ⇒ 权重文件没生效，禁止启动 G4"
echo
echo
echo "产物：$OUT/（每组一个 .log + 一份 deck matrix）"
echo "对照 tools/deck_matrix.py 可把矩阵翻成可读的原型对局表。"
