#!/usr/bin/env bash
# **口径校准：逐格把对手从"端口"换成"最强驾驶员"，只跑一次对比就够**（2026-11-20 用户纠正）
#
# 用户原话：为什么不是某一个格子 A->B 和 B->A 目前都不是 init 的时候，
#   确认一下这两个不是 init 的驾驶员互相对战一下，确认下有胜率提升就行，
#   而是需要把所有候选对手都各打 150 局才行？
#
# 对，全因子是浪费。校准只需**每格一次**：
#   · 该格已有"参考候选 vs 端口对手"的旧读数（历史面板结果）；
#   · 新读数 = 参考候选 vs 该格**已选定的强对手**（testset_meta_real.tsv 第 2 列）；
#   · 两者相减 = 这一格的口径偏移；整行加权 = 面板的乐观量。
# 若某格换了强对手后**反而更好打**（Δ>0），说明那个"强对手"其实不硬 ⇒ 该格退回端口（审计表里记一笔）。
#
#   REF=p9d6e bash tools/calibrate_strong_testset.sh root@<host> [每格局数=150]
#
# 机器要求：目标机上要有 REF 与该格对手的 agent 目录（build/agents/<名字>）。
# 跑完输出：logs/meta/<REF>_strong_s950000/vs_*.json 与 docs/口径校准_<REF>.csv
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
HOST="${1:?第一个参数是评估机，如 root@123.249.80.18}"
MATCHES="${2:-150}"
REF="${REF:-p9d6e}"
TS=decks/testset_meta_real.tsv
WORKERS="${WORKERS:-24}"
DEADLINE="${DEADLINE:-02:00}"

bash tools/arm_deadline_shutdown.sh "$HOST" "$DEADLINE" >/dev/null 2>&1 || true

CELLS=(); AGENTS=()
while IFS=$'\t' read -r cell agent rest; do
  [ "${cell:0:1}" = "#" ] && continue
  [ -z "${cell:-}" ] && continue
  [ "$cell" = "tail_mix" ] && continue
  CELLS+=("$cell"); AGENTS+=("$agent")
done < "$TS"
echo "参考候选 $REF；待校准 ${#CELLS[@]} 格；每格 $MATCHES 局"

# ① 参考候选在本机的 agent 目录（远端也要有）
[ -d "build/agents/$REF" ] || { echo "✗ 本地缺 build/agents/$REF"; exit 1; }

# ② 逐格跑：REF vs 该格强对手（-opp 用完整 agent 路径）
for i in "${!CELLS[@]}"; do
  cell="${CELLS[$i]}"; agent="${AGENTS[$i]}"
  out="logs/meta/${REF}_strong_s950000/vs_${cell}.json"
  mkdir -p "$(dirname "$out")"
  [ -s "$out" ] && { echo "[skip] $cell"; continue; }
  echo "→ $cell（对手 ${agent##*/}）"
  ./tools/py ptcg_league/eval_pair.py --cand "build/agents/$REF" --opp "$cell=$agent" \
      --matches "$MATCHES" --workers "$WORKERS" --seed-base 950100 --out "$out" 2>&1 | tail -1
done
echo "本地直跑完毕（若本地跑不动，用 dispatch_panel 派到评估机，见文末提示）"

# ③ 出对比表：新口径 vs 旧（端口）口径
./tools/py - "$REF" <<'PY'
import csv, glob, json, os, sys
ref = sys.argv[1]
W = {}
for ln in open('decks/testset_meta_real.tsv'):
    if ln.startswith('#') or not ln.strip(): continue
    f = ln.rstrip('\n').split('\t')
    if f[0] == 'tail_mix': continue
    W[f[0]] = float(f[4])
def pooled(paths):
    k = n = 0
    for p in paths:
        try: o = json.load(open(p))['overall']
        except Exception: continue
        k += o.get('score', 0) * o.get('n', 0); n += o.get('n', 0)
    return (k / n if n else None), n
rows, tw, ts_new, ts_old = [], 0.0, 0.0, 0.0
for cell, w in W.items():
    old, nold = pooled(glob.glob(f'logs/meta/{ref}_real_s95*/vs_{cell}.json'))
    new, nnew = pooled(glob.glob(f'logs/meta/{ref}_strong_s950000/vs_{cell}.json'))
    rows.append((cell, w, old, nold, new, nnew))
    if old is not None and new is not None:
        tw += w; ts_old += w * old; ts_new += w * new
out = f'docs/口径校准_{ref}.csv'
with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
    wr = csv.writer(fh)
    wr.writerow(['格', '份额%', '端口口径胜率', 'n', '强对手口径胜率', 'n', 'Δpp'])
    for cell, w, old, nold, new, nnew in rows:
        d = '' if (old is None or new is None) else round((new - old) * 100, 1)
        wr.writerow([cell, f'{w:.2f}', '' if old is None else round(old*100, 1), nold,
                     '' if new is None else round(new*100, 1), nnew, d])
print(f'\n→ {out}')
print(f'覆盖权重 {tw:.1f}%：端口口径 {ts_old/tw*100:.1f}% → 强对手口径 {ts_new/tw*100:.1f}%  '
      f'（乐观量 {ts_old/tw*100 - ts_new/tw*100:+.1f}pp）' if tw else '（数据不足，先跑评估）')
PY
