#!/usr/bin/env bash
# **CPU 机的一次性总队列**（幂等，机器随时断都能续）：
#   ① 补 choice-in-table 缺的自家对局（38 对 × 2 种子）
#   ② 给还只有单种子数据的驾驶员补种子（ab9d6e_r1B / r2B）
#   ③b r1B vs 端口 头对头（上次机器断线没拿到）
#   ③c 候选『打 9d6e』：我方换 7e39/2617/0d86/e8ee，对手固定端口@9d6e
#   ④ 评估 blend 版（决定下一轮还要不要做 kl_blend；可删）
set -uo pipefail
cd "$(dirname "$0")/.."
echo "===== [$(date +%H:%M)] ① 内部对局 ====="
bash tools/run_in_table.sh 2>&1 | tail -3

echo "===== [$(date +%H:%M)] ② 单种子驾驶员补种子 ====="
for spec in "ab9d6e_r1B:build/agents/ab9d6e_r1B" "ab9d6e_r2B:build/agents/ab9d6e_r2B"; do
  tag="${spec%%:*}"; ag="${spec##*:}"
  [ -d "$ag" ] || { echo "  ✗ 缺 $ag，跳过"; continue; }
  for sd in 950001 950002; do
    t="${tag}_s${sd}"
    [ -f "logs/meta/$t/summary.json" ] && { echo "  [skip] $t"; continue; }
    echo "  --- $t"
    SEED=$sd bash tools/meta_eval.sh --cand "$ag" --tag "$t" 2>&1 | tail -1
  done
done

echo "===== [$(date +%H:%M)] ③b 头对头：ab9d6e_r1B vs 端口（9d6e，400 场）====="
if [ ! -f logs/h2h/ab9d6e_r1B_vs_port_9d6e.json ] && [ -d build/agents/ab9d6e_r1B ]; then
  bash tools/h2h.sh build/agents/ab9d6e_r1B build/agents/port_on_9d6e 400 ab9d6e_r1B_vs_port_9d6e 2>&1 | tail -1
else
  echo "  [skip] 已有或缺 agent"
fi

echo "===== [$(date +%H:%M)] ③c 关键实验：候选『打 9d6e』（我方换牌，对手固定端口@9d6e）====="
# 面板一直只测"候选拿 9d6e 打"，从没测过"候选拿别的牌打 9d6e"——
# 后者才是 r1B/r2B 的训练任务，也是要填 choice 表 9d6e 列所需的证据。
# 先做权重最高的 4 副场地牌 × {r1B, r2B}，每格 2 种子 × 200 场。
OPP9=build/agents/port_on_9d6e
declare -A FULL=(
  [7e39]=teal_mask_ogerpon_ex_applin__7e39 [2617]=grookey_thwackey__2617
  [0d86]=shuppet_dhelmise__0d86             [e8ee]=abra_kadabra__e8ee )
mkdir -p logs/vs9d6e
for cand in r1B r2B; do
  pt="models_ppo/ab9d6e_${cand}.pt"
  [ -f "$pt" ] || { echo "  ✗ 缺 $pt，跳过 $cand"; continue; }
  for dk in 7e39 2617 0d86 e8ee; do
    tmp="build/agents/cand_${cand}_on_${dk}"
    [ -d "$tmp" ] || bash tools/make_agent_dir.sh "$tmp" "$pt" "decks/pool_field/${FULL[$dk]}.csv" >/dev/null \
      || { echo "  ✗ 铺 $cand@$dk 失败"; continue; }
    for spec in "1500000:" "1600000:__s2"; do
      sb="${spec%%:*}"; suf="${spec##*:}"
      f="logs/vs9d6e/${cand}_on_${dk}${suf}.json"
      [ -f "$f" ] && { echo "  [skip] $cand@$dk$suf"; continue; }
      echo "  --- ${cand}@${dk} vs 端口@9d6e$suf"
      ./tools/py ptcg_league/eval_pair.py --cand "$tmp" --opp "enemy=$OPP9" \
        --matches 200 --workers 0 --seed-base "$sb" --out "$f" \
        --logdir "logs/vs9d6e/${cand}_on_${dk}${suf}" 2>&1 | tail -1
    done
  done
done

# ③（9d6e→2617 高样本 n=1000）已删除：该格 n=600 的 CI 已是 ±4.0pp、区间整体在 0.5 以下，
# 再加 1000 场只能把加权精度改善 0.15pp，不成比例。

echo "===== [$(date +%H:%M)] ④ KL 回拉（blend）到底有没有伤性能？3 种子面板 ====="
# r1A_blend 是第 1 轮 B 实际面对的对手；它=0.9*r1A+0.1*端口，位移 1.791%→1.612%。
# 我们**从未**评估过 blend 版本的强度，而下一轮还要不要做 kl_blend 取决于这个答案。
for spec in "ab9d6e_r1A_blend:models_ppo/ab9d6e_r1A_blend.pt"; do
  tag="${spec%%:*}"; pt="${spec##*:}"
  [ -f "$pt" ] || { echo "  ✗ 缺 $pt，跳过"; continue; }
  ag="build/agents/$tag"
  [ -d "$ag" ] || bash tools/make_agent_dir.sh "$ag" "$pt" decks/pool_field/dunsparce_abra__9d6e.csv >/dev/null 2>&1
  for sd in 950000 950001 950002; do
    t="${tag}_s${sd}"
    [ -f "logs/meta/$t/summary.json" ] && { echo "  [skip] $t"; continue; }
    echo "  --- $t"
    SEED=$sd bash tools/meta_eval.sh --cand "$ag" --tag "$t" 2>&1 | tail -1
  done
done

echo "===== [$(date +%H:%M)] 全部完成 ====="
