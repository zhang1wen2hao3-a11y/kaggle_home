#!/usr/bin/env bash
# 11（v2）— 从**一次性物化**的 bc_all.npz 按元数据切出多个 BC 脑子并训练。
# 架构：物化只做一次（~2.3h, ~22GB），之后每个脑子 = meta 过滤 + 几分钟训练。
# 用法：
#   bash tools/bc_pool.sh
#   TARGETS="deck:0d86 deck:26c6" EPOCHS=6 bash tools/bc_pool.sh
#   WINNERS=1 bash tools/bc_pool.sh        # 只用赢家帧
set -uo pipefail
cd "$(dirname "$0")/.."
PY=./tools/py
NPZ="${NPZ:-data/bc_all.npz}"
POOL="${POOL:-bc_pool}"
EPOCHS="${EPOCHS:-6}"
LR="${LR:-2e-4}"
DEV="${DEV:-cuda}"
WINNERS="${WINNERS:-0}"
TARGETS="${TARGETS:-deck:0d86 deck:26c6 deck:f5f8 agent:ミワハルキ agent:RtoABC2 agent:Raihan agent:sue124}"

mkdir -p "$POOL" logs
REG="$POOL/registry.tsv"
[ -f "$REG" ] || printf 'name\tfilter\tbin_sha16\tdate\n' > "$REG"
[ -f "$NPZ" ] || { echo "✗ 缺 $NPZ（物化还没完成？）"; exit 1; }

for spec in $TARGETS; do
  key="${spec%%:*}"; val="${spec#*:}"
  safe=$(echo "$val" | tr -c 'A-Za-z0-9_.' '_')
  name="${key}_${safe}"
  out_bin="$POOL/${name}.bin"
  echo; echo "################ [$name] $spec $(date +%T) ################"
  if [ -f "$out_bin" ]; then echo "已存在，跳过"; continue; fi

  FLAG=(--only-deck "$val")
  [ "$key" = "agent" ] && FLAG=(--only-agent "$val")
  [ "$WINNERS" = "1" ] && FLAG+=(--winners-only)

  echo "--- 端口基线（同一子集）---"
  $PY -u tools/bc_train.py --npz "$NPZ" --baseline --device "$DEV" "${FLAG[@]}" 2>&1 | tail -4

  echo "--- 训练 ---"
  $PY -u tools/bc_train.py --npz "$NPZ" --epochs "$EPOCHS" --lr "$LR" \
      --out "$POOL/${name}.pt" --device "$DEV" "${FLAG[@]}" 2>&1 | tail -12

  if [ ! -f "$POOL/${name}.pt" ]; then echo "⚠️ 无 .pt，跳过"; continue; fi

  ( cd port_repo && PYTHONPATH=data:.:native ../$PY native/tools/native_weights.py export \
      "../$POOL/${name}.pt" "../$out_bin" ) 2>&1 | tail -1
  [ -f "$out_bin" ] || { echo "⚠️ export 失败"; continue; }

  sha=$(sha256sum "$out_bin" | cut -c1-16)
  printf '%s\t%s\t%s\t%s\n' "$name" "$spec" "$sha" "$(date +%F)" >> "$REG"
  echo "✅ $name → $out_bin  sha=$sha"
done

echo; echo "################ 全部完成 $(date +%T) ################"
ls -lh "$POOL" | tail -20
