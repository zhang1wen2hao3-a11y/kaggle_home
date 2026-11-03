#!/usr/bin/env bash
# **对手池强度阶梯**：池里每个成员，按其**自己的卡组**、对学习者（= 当前 best 的 7e39）各打 200 场。
#
# 为什么要它：角色划分（强/中/特殊）必须建立在 p_i 上，而 p_i 就是这里的胜率。
# 之前只零散测过 3 个对局，等于凭感觉分类。
# 这也是训练里 bank 的真实处境：bank 打自己的牌，学习者打 anchor 牌。
set -uo pipefail
cd /home/zhang/code/kaggle/the-pokemon-company
PY=./tools/py
ONEJOB=./tools/onejob.sh
N=${N:-200}
SEED=760000
LEARNER=ptcg_league/pools/league/lg_boot_port_plain   # 端口权重 @ 7e39

if [ ! -d "$LEARNER" ]; then LEARNER=ptcg_league/pools/init/init_port_plain; fi
echo "学习者（anchor 卡组 7e39）= $LEARNER"
echo "每对 $N 场 Bo3，种子 $SEED（各成员同种子 ⇒ 可配对）"
echo

mkdir -p logs/ladder
for d in ptcg_league/pools/init/bc_*; do
  name=$(basename "$d")
  case "$name" in *0d86_w*) continue ;; esac   # 已判废的 winners-only 版本，不入阶梯
  out="logs/ladder/${name}.json"
  if [ -f "$out" ]; then echo "[skip] $name"; continue; fi
  echo "########## $name ##########"
  # ⚠️ 不要在这里再套 onejob：外层已经用 `onejob ladder` 拿过同名锁了，
  #    内层再拿会 rc=76（锁不可重入）——第一版就是这样，8 个成员全瞬间失败。
  $PY ptcg_league/eval_pair.py \
      --cand "$d" --opp learner="$LEARNER" \
      --matches "$N" --workers 5 --seed-base "$SEED" --out "$out" 2>&1 | tail -3
done

echo
echo "===== 强度阶梯汇总（成员胜率 = p_i）====="
$PY - <<'EOF'
import glob, json, os
rows = []
for f in sorted(glob.glob("logs/ladder/*.json")):
    d = json.load(open(f))
    o = d["overall"]
    rows.append((os.path.basename(f)[:-5], o["score"], o["ci_low"], o["ci_high"], o["n"], o["errors"]))
rows.sort(key=lambda r: -r[1])
print(f"{'成员':<26}{'p_i':>7}{'95% CI':>18}{'场次':>6}{'异常':>5}  角色建议")
for name, s, lo, hi, n, e in rows:
    if s < 0.45:   role = "A 强对手（训练策略胜率 55–100%）"
    elif s < 0.55: role = "B 中等（50/50）"
    elif s < 0.75: role = "C 中等偏弱"
    else:          role = "D 弱/特殊（胜率 <25%）"
    print(f"{name:<26}{s:>7.3f}{f'[{lo:.3f},{hi:.3f}]':>18}{n:>6}{e:>5}  {role}")
print("\n注：这里的 p_i = 该成员**打赢学习者**的比例；训练里学习者对它的胜率 = 1 − p_i。")
EOF
