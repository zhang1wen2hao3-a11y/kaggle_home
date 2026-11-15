#!/usr/bin/env bash
# **提交 → 训练指导**：把提交的真实对局变成"下一步训什么"的决策输入。
#
#   bash tools/sub_feedback.sh <ref1> [ref2 ...]        # 抓回放 + 全部分析
#   SKIP_FETCH=1 bash tools/sub_feedback.sh            # 已有 epsoide_sub* 回放，只做分析
#
# 为什么必须有这条流水线（2026-11-16 的教训）：
#   · 12 格面板的权重（b05）来自**自家 582 局历史**，和真实天梯构成偏差极大
#     （7e39 自家口径 35% vs 顶分段实际 1.2%）。分母错了，单格测得再准也会把训练引偏。
#   · 提交后的回放是**唯一**能拿到"真实对手卡组/份额"的渠道；LB 只给一个标量分。
#   · 有了真实份额，"该训哪一格"就不用凭感觉：目标价值 = 真实份额 × (参考线 − 我方胜率)。
#
# 输出（全落到 logs/sub_feedback/）：
#   ① 数据集体检   ② 真实对手构成+胜率   ③ 真实场地权重文件   ④ 决策表 decision.csv
# 决策表里的「可行性」= 该卡组在 data/distill_index.jsonl 里的**真人帧数**：
#   3eaa 只有 1206 帧 ⇒ 训不动（陪练里没有"人怎么打这副牌"的信号）；2617 有 16539 帧 ⇒ 训得动。
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=logs/sub_feedback
mkdir -p "$OUT"
PY=./tools/py

if [ "${SKIP_FETCH:-0}" != 1 ]; then
  for ref in "$@"; do
    case "$ref" in --*) continue;; esac
    echo "== 抓回放 ref=$ref → epsoide_subB_${ref}"
    $PY tools/fetch_episodes.py --submission "$ref" --out "epsoide_subB_${ref}" --workers 6 2>&1 | tail -3
  done
fi

# 默认只分析"本次提交"的回放目录；SUB_DIRS 可指定任意回放目录（自测/复算历史用）
DIRS="${SUB_DIRS:-$(ls -d epsoide_sub* 2>/dev/null | tr '\n' ',' | sed 's/,$//')}"
[ -n "$DIRS" ] || { echo "✗ 没有 epsoide_sub* 目录（先给 submission ref，或先抓回放）"; exit 2; }
echo "== 分析目录：$DIRS"

echo
echo "== ② 数据集体检（我方是否始终同一副牌 / 座位是否可判定）"
for d in ${DIRS//,/ }; do
  label=$(basename "$d")
  $PY tools/verify_sub_dataset.py --dir "$d" --label "$label" 2>&1 | tee "$OUT/verify_$label.log" \
    | grep -E "我方卡组|座位|异常|缺卡表|胜率" | head -12
done

echo
echo "== ③ 真实对手构成（相近配置合并后的份额 + 对每种卡组的胜率）"
$PY tools/analyze_opponents.py --dir "$DIRS" 2>&1 | tee "$OUT/opponents.log" | tail -40

echo
echo "== ④ 真实场地权重（→ decks/weights_field_real.txt；**不改动任何现有权重文件**）"
# ⚠️ 不要在这里调 tools/build_field_pool.py：它会**重命名池内卡组文件并覆盖
#    decks/weights_top.txt 等共享权重**（2026-11-16 自测时踩到，已回滚）。这里只按实测份额
#    生成一个新文件，现有 b05/top 权重一律不动。
$PY - "$DIRS" <<'WEOF'
import collections, glob, hashlib, json, os, sys
sys.path.insert(0, os.getcwd())
from framework.replay import decks_of, our_index
dirs = sys.argv[1].split(',')
TAG, NAME = {}, {}
for f in glob.glob('decks/pool_field/*.csv'):
    try:
        ids = sorted(int(x) for x in open(f) if x.strip())
        h = hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()[:8]
        TAG[h] = os.path.basename(f)[:-4]
    except Exception:
        pass
cnt = collections.Counter()
for d in dirs:
    for f in glob.glob(os.path.join(d, '*.json')):
        try:
            e = json.load(open(f))
        except Exception:
            continue
        dk = decks_of(e.get('steps') or []); me = our_index(e)
        if me is None or len(dk) < 2:
            continue
        opp = next((j for j in dk if j != me), None)
        if opp is None:
            continue
        cnt[hashlib.sha256(','.join(map(str, sorted(dk[opp]))).encode()).hexdigest()[:8]] += 1
if not cnt:
    print('（没有可用回放，跳过权重生成）'); raise SystemExit(0)
tot = sum(cnt.values())
out = ['version 3', 'default 0.0001']
for h, c in cnt.most_common():
    out.append(f'{TAG.get(h, h)} {c / tot:.6f}')
open('decks/weights_field_real.txt', 'w').write('\n'.join(out) + '\n')
print(f'→ decks/weights_field_real.txt（{len(cnt)} 副，实测 {tot} 局对手侧）')
print('  top5:', ', '.join(f'{TAG.get(h, h)} {c/tot*100:.1f}%' for h, c in cnt.most_common(5)))
WEOF

echo
echo "== ⑤ 决策表：真实份额 × 参考线 × 真人样本 ⇒ 下一轮训什么"
$PY - "$DIRS" "$OUT" <<'PYEOF'
import collections, csv, glob, hashlib, json, os, sys
sys.path.insert(0, os.getcwd())
from framework.replay import decks_of, our_index   # 回放解析只走这个入口（见其文件头的错位坑）
dirs, out = sys.argv[1].split(','), sys.argv[2]

TAG = {}
for f in glob.glob('decks/pool_field/*.csv'):
    try:
        ids = sorted(int(x) for x in open(f) if x.strip())
        TAG[hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()[:8]] = os.path.basename(f)[:-4]
    except Exception:
        pass

share, wins = collections.Counter(), collections.Counter()
n_games = n_skip = 0
for d in dirs:
    for f in glob.glob(os.path.join(d, '*.json')):
        try:
            e = json.load(open(f))
        except Exception:
            continue
        steps = e.get('steps') or []
        if len(steps) < 2:
            continue
        dk = decks_of(steps)
        me = our_index(e)
        if me is None or len(dk) < 2:
            n_skip += 1
            continue
        opp = next((j for j in dk if j != me), None)
        if opp is None:
            continue
        n_games += 1
        h = hashlib.sha256(','.join(map(str, sorted(dk[opp]))).encode()).hexdigest()[:8]
        share[h] += 1
        rw = e.get('rewards') or []
        if len(rw) > me and rw[me] is not None and rw[me] > 0:
            wins[h] += 1

tot = sum(share.values()) or 1
frames = collections.Counter()
for ln in open('data/distill_index.jsonl'):
    try:
        r = json.loads(ln)
    except Exception:
        continue
    frames[(r.get('deck') or '')[:4]] += 1

port = {}
for r in csv.DictReader(open('docs/the-pockmon-battle-choice-in-table.csv', encoding='gb18030')):
    if r.get('驾驶员') == 'init' and r.get('对手驾驶员') == 'init':
        port.setdefault(r['挑战者'], {})[r['被挑战者']] = float(r['胜率'])

if not n_games:
    print('⚠️ 没有可用回放（缺 60 张卡表或认不出我方座位）')
    raise SystemExit(0)

print(f"可用回放 {n_games} 局（跳过 {n_skip} 局：缺卡表/认不出座位）")
print()
print(f"{'对手卡组':24s}{'真实份额':>8s}{'我方胜率':>8s}{'样本':>6s}{'真人帧':>8s}{'参考线':>7s}  可行性")
rows = []
for h, c in share.most_common(14):
    name = TAG.get(h, h)
    short = name.split('__')[-1]
    sh = c / tot
    wr = wins[h] / c
    fr = frames.get(short, 0)
    ref = port.get('f5f8', {}).get(short)
    val = max(0.0, ref) * sh * 100 if ref is not None else float('nan')
    feas = '可训' if fr >= 5000 else ('样本薄' if fr >= 2000 else '不可训(<2k帧)')
    refs = 'NA' if ref is None else f'{ref * 100:.1f}%'
    print(f"{name:24s}{sh * 100:7.1f}%{wr * 100:7.1f}%{c:6d}{fr:8d}{refs:>7s}  {feas}")
    rows.append([name, f'{sh * 100:.1f}%', f'{wr * 100:.1f}%', c, fr, refs,
                 f'{val:.2f}' if val == val else '', feas])

with open(os.path.join(out, 'decision.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['对手卡组', '真实份额', '我方胜率', '样本', '真人帧', '端口参考线', '目标价值pp', '可行性'])
    w.writerows(rows)
print()
print(f"→ {out}/decision.csv")
print("判读：目标价值 = 真实份额 × 参考线（该格的上限贡献）；≥0.5pp 且『可训』才开 GPU，否则只登记。")
PYEOF
echo "SUB_FEEDBACK_DONE"
