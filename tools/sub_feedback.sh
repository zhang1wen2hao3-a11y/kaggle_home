#!/usr/bin/env bash
# **提交 → 训练指导**（快路径：并行扫回放 + 缓存，整条几十秒）
#
#   bash tools/sub_feedback.sh <ref1> [ref2 ...]        # 抓回放 + 分析
#   SKIP_FETCH=1 bash tools/sub_feedback.sh            # 已有 epsoide_sub* 回放
#   SUB_DIRS=epsoide_x bash tools/sub_feedback.sh      # 指定目录（自测/复算）
#   STEP_VERIFY=1 STEP_OPP=1 bash tools/sub_feedback.sh  # 额外跑体检/对手合并（串行、慢）
#
# ★ 性能纪律（2026-11-16 用户明确要求）：**这类扫描一律并行**
#   扫描统一走 `tools/replay_scan.py`（multiprocessing + 按 (路径,mtime,size) 缓存）：
#     243 份回放 首次 8.7s、缓存命中 0.1s；1729 份 42.9s（32 进程）。
#   不要写串行的 glob+json.load 循环——之前那版扫全语料直接超时。
#
# 产物（logs/sub_feedback/）：decision.csv（份额/胜率/参考线/可行性）+ 权重文件 + 座位表
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=logs/sub_feedback; mkdir -p "$OUT"
PY=./tools/py
JOBS="${JOBS:-0}"          # 0 = 由 replay_scan 自己按核数决定

if [ "${SKIP_FETCH:-0}" != 1 ]; then
  for ref in "$@"; do
    case "$ref" in --*) continue;; esac
    echo "== 抓回放 ref=$ref → epsoide_subB_${ref}"
    $PY tools/fetch_episodes.py --submission "$ref" --out "epsoide_subB_${ref}" --workers 8 2>&1 | tail -2
  done
fi

DIRS="${SUB_DIRS:-$(ls -d epsoide_sub* 2>/dev/null | tr '\n' ',' | sed 's/,$//')}"
[ -n "$DIRS" ] || { echo "✗ 没有 epsoide_sub* 目录（先给 ref，或 SUB_DIRS=...）"; exit 2; }
echo "== 分析目录：$DIRS"

echo
echo "== ① 并行扫回放（唯一口径：decks_of + firstPlayer）"
$PY tools/replay_scan.py --dirs "$DIRS" --jobs "$JOBS" --out "$OUT/scan.jsonl" 2>&1 | tail -2

if [ "${STEP_VERIFY:-0}" = 1 ]; then
  echo; echo "== ② 数据集体检（串行，慢，按需开启）"
  for d in ${DIRS//,/ }; do
    $PY tools/verify_sub_dataset.py --dir "$d" --label "$(basename "$d")" 2>&1 \
      | tee "$OUT/verify_$(basename "$d").log" | grep -E "我方卡组|座位|异常" | head -6
  done
fi
if [ "${STEP_OPP:-0}" = 1 ]; then
  echo; echo "== ③ 对手配置合并（串行，慢，按需开启）"
  $PY tools/analyze_opponents.py --dir "$DIRS" 2>&1 | tee "$OUT/opponents.log" | tail -30
fi

echo
echo "== ④⑤ 汇总：真实权重 + 决策表 + 座位表（读 scan.jsonl，秒级）"
$PY - "$OUT" <<'PYEOF'
import collections, csv, glob, json, os, sys
out = sys.argv[1]
TAG = {}
for f in glob.glob('decks/pool_field/*.csv'):
    try:
        ids = sorted(int(x) for x in open(f) if x.strip())
        import hashlib
        TAG[hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()[:8]] = os.path.basename(f)[:-4]
    except Exception:
        pass
recs = [json.loads(l) for l in open(os.path.join(out, 'scan.jsonl'))]
games = [r for r in recs if r.get('opp_hash') and r.get('win') is not None]
mirror = [g for g in games if g.get('selfmirror')]
real = [g for g in games if not g.get('selfmirror')]
if not real:
    print('⚠️ 没有可用对局'); raise SystemExit(0)
n = len(real); wins = sum(1 for g in real if g['win'])
print(f"回放 {len(recs)} 份 → 真实对局 {n} 局（自镜像 {len(mirror)} 局已剔除），胜率 {wins/n*100:.1f}%")

# --- 座位/先手（用 firstPlayer，不用 agent 槽位）---
fs = [g for g in real if g.get('first_is_us') is True]
ss = [g for g in real if g.get('first_is_us') is False]
if fs and ss:
    print(f"  先手 {len(fs):3d} 局 {sum(g['win'] for g in fs)/len(fs)*100:.1f}%"
          f" | 后手 {len(ss):3d} 局 {sum(g['win'] for g in ss)/len(ss)*100:.1f}%"
          f"  （注：通用语料里先手总胜率约 50%，所以这不是可训练目标）")

# --- 真实权重 ---
share = collections.Counter(g['opp_hash'] for g in real)
wins_by = collections.Counter(g['opp_hash'] for g in real if g['win'])
lines = ['version 3', 'default 0.0001']
for h, c in share.most_common():
    lines.append(f"{TAG.get(h, h)} {c/n:.6f}")
open('decks/weights_field_real.txt', 'w').write('\n'.join(lines) + '\n')
top = ', '.join(f"{TAG.get(h, h)} {c/n*100:.1f}%" for h, c in share.most_common(5))
print(f"  → decks/weights_field_real.txt（{len(share)} 副）  top5: {top}")

# --- 决策表 ---
frames = collections.Counter()
for ln in open('data/distill_index.jsonl'):
    try:
        r = json.loads(ln); frames[(r.get('deck') or '')[:4]] += 1
    except Exception:
        pass
port = {}
for r in csv.DictReader(open('docs/the-pockmon-battle-choice-in-table.csv', encoding='gb18030')):
    if r.get('驾驶员') == 'init' and r.get('对手驾驶员') == 'init':
        port[r['被挑战者']] = float(r['胜率'])
cells12 = {'0d86', '26c6', 'f5f8', '7e39', '2617', '9d6e', '0bf5', '3eaa', 'db04', '0904', 'e8ee', 'e1d7'}
print()
print(f"{'对手卡组':30s}{'份额':>7s}{'胜率':>7s}{'n':>5s}{'参考线':>7s}{'目标价值':>8s}  可行性")
rows = []
for h, c in share.most_common(14):
    name = TAG.get(h, h); short = name.split('__')[-1]
    sh = c / n; wr = wins_by[h] / c
    ref = port.get(short); fr = frames.get(short, 0)
    val = max(0.0, ref) * sh * 100 if ref is not None else float('nan')
    feas = '可训' if fr >= 5000 else ('样本薄' if fr >= 2000 else '不可训(<2k帧)')
    if short not in cells12:
        feas = '★格子外'

    refs = 'NA' if ref is None else f'{ref*100:.1f}%'
    print(f"{name:30s}{sh*100:6.1f}%{wr*100:6.1f}%{c:5d}{refs:>7s}"
          f"{('NA' if val != val else f'{val:.2f}'):>8s}  {feas}")
    rows.append([name, f'{sh*100:.1f}%', f'{wr*100:.1f}%', c, fr, refs,
                 '' if val != val else f'{val:.2f}', feas])
# 口径盲区：份额全都算一次，不重复计
outside = sum(c for h, c in share.items() if TAG.get(h, h).split('__')[-1] not in cells12)
print(f"\n★ 12 格测试集之外的对手：{outside}/{n} = {outside/n*100:.1f}%  ← 口径盲区")
with open(os.path.join(out, 'decision.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['对手卡组', '真实份额', '我方胜率', '样本', '真人帧', '端口参考线', '目标价值pp', '可行性'])
    w.writerows(rows)
print(f"→ {out}/decision.csv")
PYEOF
echo "SUB_FEEDBACK_DONE"
