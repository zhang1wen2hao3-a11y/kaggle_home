#!/usr/bin/env bash
# 补齐对手池里**缺驾驶者**的高频原型。
#
# 依据（393 局真实天梯，data/episodes*.csv 去重后的分布）：
#   teal_mask_ogerpon 29.5%（我们胜率 0.741）—— 有端口驾驶者，但只有它一个
#   abra_kadabra      15.3%（0.500）—— 有 bc:deck_f5f8 ✅
#   grookey_thwackey  12.0%（0.532）—— **没有驾驶者**，13,329 帧可训
#   dunsparce_abra    10.2%（0.375）—— **没有驾驶者**，只有 5,099 帧（不够，先跳过）
#   dreepy_drakloak    8.9%（0.543）—— 有 bc:agent_Raihan ✅
#   shuppet_dhelmise   5.9%（**0.043**）—— 有两个驾驶者 ✅（池里最该留的对手）
#
# 所以这一轮只补两个：grookey（12%，没有）和 Ogerpon 第二个驾驶者（29.5%，只有端口）。
# 不用 winners-only（EXP-0158 已证伪：丢一半数据抵不过标签噪声的收益）。
set -uo pipefail
cd /home/zhang/code/kaggle/the-pokemon-company
PY=./tools/py
ONEJOB=./tools/onejob.sh
mkdir -p logs bc_pool

run_one() {  # $1=arch hash  $2=名字  $3=max-frames
  local arch="$1" name="$2" cap="$3"
  if [ -f "bc_pool/${name}.bin" ]; then
    echo "[skip] ${name} 已存在"
    return 0
  fi
  echo
  echo "########## ${name}（deck:${arch}, max-frames ${cap}）$(date +%T) ##########"
  # 名字以 agent_ 开头 ⇒ 按玩家过滤，否则按卡组 hash 过滤
  case "$name" in
    agent_*) FILT=(--only-agent "$arch") ;;
    *)       FILT=(--only-deck "$arch") ;;
  esac
  $ONEJOB bc -- $PY -u tools/bc_train.py --npz data/bc_all.npz \
      --epochs 8 --lr 2e-4 --bs 256 --device cuda \
      "${FILT[@]}" --max-frames "$cap" \
      --out "bc_pool/${name}.pt"
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "bc_pool/${name}.pt" ]; then
    echo "✗ ${name} 训练失败 rc=$rc"; return $rc
  fi
  ( cd port_repo && PYTHONPATH=data:.:native ../$PY native/tools/native_weights.py export \
      "../bc_pool/${name}.pt" "../bc_pool/${name}.bin" ) 2>&1 | tail -1
  echo "✅ ${name} → bc_pool/${name}.bin"
}

echo "===== 对手池补驾驶员开始 $(date +%F' '%T) ====="
# grookey 12% 真实占比、没驾驶者、13,329 帧
run_one 2617 deck_2617_grookey 12000
# ★ 最强玩家 ミワハルキ（LB 1441，254,526 帧 = 全语料 48.5%）—— 这是唯一可能造出
#   "比 BC 克隆强"的候选。前两次都因为 OOM / 5 小时 I/O 没跑成，现在有
#   分块抽样 + 合法性批 512 + 页缓存回收，预计 1.5 小时。
run_one "やる気元気ミワハルキ 2nd ver." agent_miwa_full 80000

echo
echo "===== 重建登记表 ====="
$PY tools/bc_pool_registry.py | tail -12
echo "===== 完成 $(date +%T) ====="
