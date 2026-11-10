#!/usr/bin/env bash
# 交叉矩阵跑批：候选（按其卡组出战）× 主卡组（用该卡组的端口驾驶当对手）。
# 每格记录「候选的胜率」= 该候选对该主卡组的压力 q_kj（供调度器的覆盖公式用）。
#
#   bash tools/matrix_run.sh remote     # 在评估机侧跑（远程 16 面板，400 场/格）
#   bash tools/matrix_run.sh local      # 在本地跑（本地 6 面板，300 场/格）
#
# 两侧共享 data/scheduler/matrix_progress.tsv，已完成的格子自动跳过 ⇒ 可随时中断续跑。
set -uo pipefail
cd "$(dirname "$0")/.."

SIDE="${1:?用法: matrix_run.sh remote|local}"
PROG=data/scheduler/matrix_progress.tsv
mkdir -p data/scheduler
[ -f "$PROG" ] || printf 'candidate\topp_deck\tcand_win\tn\tmachine\n' > "$PROG"

# 7 个主卡组 → 该卡组的代表候选（现成最强 agent）
declare -A CAND_AGENT=(
  [7e39]=port_7e39  [0bf5]=port_0bf5  [2617]=port_2617  [0d86]=g8_bc_parent
  [9d6e]=port_9d6e  [f5f8]=port_f5f8  [3eaa]=port_3eaa
)
# 7 个主卡组 → 对手侧 agent（评估机 kit 里的名字）
declare -A OPP_AGENT=(
  [7e39]=ogerpon_7e39  [0bf5]=dunsparce_0bf5  [2617]=grookey_2617  [0d86]=shuppet_0d86
  [9d6e]=dunsparce_9d6e  [f5f8]=abra_f5f8  [3eaa]=dreepy_3eaa
)
DECKS=(7e39 0bf5 2617 0d86 9d6e f5f8 3eaa)

if [ "$SIDE" = remote ]; then MATCHES=400; CAND_PREFIX="ag_"; else MATCHES=300; CAND_PREFIX=""; fi
# 远程 kit 里候选就是那 7 个面板 agent（按卡组各一个）；本地用 build/agents 里的端口/BC
declare -A CAND_REMOTE=(
  [7e39]=ogerpon_7e39 [0bf5]=dunsparce_0bf5 [2617]=grookey_2617 [0d86]=shuppet_0d86
  [9d6e]=dunsparce_9d6e [f5f8]=abra_f5f8 [3eaa]=dreepy_3eaa
)
idx=0

done_cell() { grep -qs "^$1	$2	" "$PROG"; }

n=0
for c in "${DECKS[@]}"; do
  for o in "${DECKS[@]}"; do
    cid="${c}_x_${o}"
    idx=$((idx+1))
    # 分片：远程跑 idx%3!=0（约 2/3），本地跑 idx%3==0（约 1/3）
    if [ "$SIDE" = remote ] && [ $((idx % 3)) -eq 0 ]; then continue; fi
    if [ "$SIDE" = local ] && [ $((idx % 3)) -ne 0 ]; then continue; fi
    done_cell "$c" "$o" && continue
    n=$((n+1))
    echo "----- [$SIDE] $cid ($(date +%H:%M:%S)) #$n -----"
    if [ "$SIDE" = remote ]; then
      out="/tmp/mx_${cid}.json"
      bash tools/evnode.sh run "ag_${CAND_REMOTE[$c]}" "${o}=build/agents/${OPP_AGENT[$o]}" \
        "$MATCHES" 990001 "$out" 2>&1 | grep "vs " | tail -1
    else
      out="/tmp/mx_${cid}.json"
      PTCG_PANELS=6 /home/zhang/miniforge3/envs/py310/bin/python ptcg_league/eval_pair.py \
        --cand "build/agents/${CAND_AGENT[$c]}" --opp "${o}=build/agents/${LOCAL_OPP[$o]}" \
        --matches "$MATCHES" --seed-base 990001 --out "$out" --logdir "/tmp/mxlog_$cid" \
        2>&1 | grep "候选" | tail -1
    fi
    if [ -f "$out" ]; then
      win=$(/home/zhang/miniforge3/envs/py310/bin/python - "$out" <<'PY'
import json,sys
o=json.load(open(sys.argv[1]))['overall']
pm=o.get('per_match'); v=pm if isinstance(pm,float) else o.get('score',0)
print(f"{v:.4f}\t{o.get('n',0)}")
PY
)
      printf '%s\t%s\t%s\t%s\n' "$c" "$o" "$win" "$SIDE" >> "$PROG"
    fi
  done
done
echo "MATRIX_${SIDE}_DONE $(date +%H:%M:%S)  已完成 $(($(wc -l < "$PROG")-1))/49 格"
