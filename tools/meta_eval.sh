#!/usr/bin/env bash
# **固定测试集评测**：A/B/C/D 四组共用同一套格子，与各组训练分布完全解耦。
#
#   bash tools/meta_eval.sh --cand build/agents/ph3_2617_v2 --tag A_2617
#
# 为什么要固定：各组的训练对手分布不同（A 纯本地 / D 强压力），
# 若各自用训练分布评估，等于"考自己练过的题"，无法比较。
# 这里统一用 decks/testset_meta.tsv 的 12 格（8 本地 + 4 冠军）。
#
# 五项指标：
#   1. meta_w   Meta 加权胜率  —— 按我们 582 局实测占比加权（真实对局强度）
#   2. champ   冠军对位胜率   —— 4 副冠军牌的平均（强度压力是否转成学习收益）
#   3. local   本地卡组平均   —— 8 副本地牌的平均（是否牺牲原有能力）
#   4. worst   最差对位胜率   —— 12 格里最低（防止严重退化）
#   5. seed_var 多种子方差    —— 由 tools/meta_summary.py 跨种子汇总，本脚本只出单种子
#
# 输出：logs/meta/<tag>/vs_<cell>.json + summary.json，可断点续跑。
set -uo pipefail
cd "$(dirname "$0")/.."
CAND=""; TAG=""
while [ $# -gt 0 ]; do
  case "$1" in --cand) CAND="$2"; shift 2;; --tag) TAG="$2"; shift 2;; *) shift;; esac
done
[ -n "$CAND" ] || { echo "✗ 缺 --cand"; exit 2; }
TAG="${TAG:-$(basename "$CAND")}"
TS="${TS:-decks/testset_meta.tsv}"      # 测试集；方案A = decks/testset_meta_A.tsv（对手用各自最佳驾驶员）
N="${N:-200}"; SEED="${SEED:-950000}"
OUT="logs/meta/$TAG"
mkdir -p "$OUT"
echo "候选 $CAND → $OUT（12 格 × $N 场，种子 $SEED）"
while IFS=$'\t' read -r cell dir _deck _grp _wt _drv; do
  case "$cell" in \#*|"") continue;; esac
  [ -n "${dir:-}" ] || continue
  f="$OUT/vs_$cell.json"
  [ -f "$f" ] && { echo "[skip] $cell"; continue; }
  echo "########## $TAG vs $cell（对手驾驶员=${_drv:-init}）##########"
  ./tools/py ptcg_league/eval_pair.py --cand "$CAND" --opp "$cell=$dir" \
      --matches "$N" --workers 0 --seed-base "$SEED" --out "$f" --logdir "$OUT/$cell" 2>&1 | tail -2
done < "$TS"

./tools/py - "$OUT" "$N" "$SEED" <<'PY'
import json, math, os, sys
d, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = []
for ln in open("decks/testset_meta.tsv"):
    if ln.startswith("#") or not ln.strip(): continue
    cell, _dir, _deck, grp, wt = ln.rstrip("\n").split("\t")
    f = os.path.join(d, f"vs_{cell}.json")
    if not os.path.exists(f): continue
    o = json.load(open(f))["overall"]
    rows.append(dict(cell=cell, grp=grp, wt=float(wt), score=o["score"], n=o["n"],
                     lo=o["ci_low"], hi=o["ci_high"]))
if not rows:
    print("⚠️ 没有结果"); raise SystemExit(1)

def wilson(k, nn, z=1.96):
    if nn == 0: return (0, 1)
    p = k / nn; dd = 1 + z*z/nn; c = p + z*z/(2*nn)
    h = z*math.sqrt(p*(1-p)/nn + z*z/(4*nn*nn))
    return max(0, (c-h)/dd), min(1, (c+h)/dd)

print(f"\n{'格':<18}{'组':<7}{'胜率':>7}{'95% CI':>18}{'n':>5}{'权重':>7}")
for r in rows:
    print(f"{r['cell']:<18}{r['grp']:<7}{r['score']:>7.3f}"
          f"{'[%.3f,%.3f]' % (r['lo'], r['hi']):>18}{r['n']:>5}{r['wt']:>7.3f}")

loc = [r for r in rows if r["grp"] == "local"]
ch = [r for r in rows if r["grp"] == "champ"]
wtot = sum(r["wt"] for r in rows) or 1.0
meta = sum(r["wt"] * r["score"] for r in rows) / wtot
# meta 的 CI：按格子独立近似（保守：用权重平方和缩放）
k = sum(r["wt"] * r["score"] * r["n"] for r in rows)
nn = sum(r["wt"] * r["n"] for r in rows)
meta_lo, meta_hi = wilson(k, nn)
local_avg = sum(r["score"] for r in loc) / len(loc) if loc else float("nan")
champ_avg = sum(r["score"] for r in ch) / len(ch) if ch else float("nan")
worst = min(rows, key=lambda r: r["score"])
print(f"\n{'指标':<22}{'值':>8}")
print(f"{'1. meta_w 加权胜率':<22}{meta:>8.4f}   CI[{meta_lo:.3f},{meta_hi:.3f}]")
print(f"{'2. champ 冠军对位':<22}{champ_avg:>8.4f}")
print(f"{'3. local 本地平均':<22}{local_avg:>8.4f}")
print(f"{'4. worst 最差对位':<22}{worst['score']:>8.4f}   ({worst['cell']})")
print(f"{'5. seed_var':<22}{'—':>8}   （需多颗种子后由 tools/meta_summary.py 汇总）")
json.dump(dict(tag=os.path.basename(d), n=n, seed=seed, meta_w=round(meta, 4),
               meta_ci=[round(meta_lo, 4), round(meta_hi, 4)],
               champ=round(champ_avg, 4), local=round(local_avg, 4),
               worst=[worst["cell"], round(worst["score"], 4)],
               cells={r["cell"]: round(r["score"], 4) for r in rows}),
          open(os.path.join(d, "summary.json"), "w"), ensure_ascii=False, indent=1)
print(f"\n→ {os.path.join(d, 'summary.json')}")
PY
