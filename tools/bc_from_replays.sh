#!/usr/bin/env bash
# **从新回放目录 → BC 训练包** 的全流程（一条命令跑通四步）。
#
#   ① extract_distill  → 建决策索引（file/step/seat/deck/act）
#   ② bc_materialize   → 物化帧（npz + f32，含 deck/agent/split）
#   ③ bc_kit           → 切子集 + 打包（自带 bundle，GPU 机直接 bash run.sh）
#   ④ 打印验收信息（帧数/玩家数/val 基线）
#
# 用法：
#   bash tools/bc_from_replays.sh --dir epsoide_petit --deck-prefix <8hex> --tag petit
#   bash tools/bc_from_replays.sh --dir epsoide_miwa --deck-prefix <8hex> --tag miwa_og
#
# ⚠️ `--deck-prefix` 是**出手方卡组**的 sha256[:8]；先用 --list-decks 看目录里有哪些。
set -uo pipefail
cd "$(dirname "$0")/.."
PY=./tools/py
DIR=""; DECK=""; TAG=""; MAXG=0; LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --deck-prefix) DECK="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --max-games) MAXG="$2"; shift 2 ;;
    --list-decks) LIST=1; shift ;;
    *) echo "未知参数 $1"; exit 2 ;;
  esac
done
[ -n "$DIR" ] || { echo "✗ 缺 --dir"; exit 2; }
[ -d "$DIR" ] || { echo "✗ 目录不存在：$DIR"; exit 1; }
N=$(ls "$DIR"/*.json 2>/dev/null | wc -l)
echo "目录 $DIR：$N 个回放"

if [ "$LIST" = 1 ]; then
  $PY - "$DIR" <<'PY'
import glob, json, sys, collections, hashlib
d=sys.argv[1]; cnt=collections.Counter()
for p in glob.glob(d+"/*.json"):
    try: j=json.load(open(p))
    except Exception: continue
    st=j.get("steps") or []
    if len(st)<2: continue
    for k in (0,1):
        a=st[1][k].get("action")
        if isinstance(a,list) and len(a)==60:
            cnt[hashlib.sha256(",".join(str(x) for x in sorted(a)).encode()).hexdigest()[:8]]+=1
print("目录里出现的卡表（8hex → 座位数）：")
for k,v in cnt.most_common(10): print(f"  {k}  {v}")
PY
  exit 0
fi
[ -n "$DECK" ] || { echo "✗ 缺 --deck-prefix（先跑 --list-decks）"; exit 2; }
TAG="${TAG:-bc_$DECK}"
IDX="data/distill_${TAG}.jsonl"
NPZ="data/bc_${TAG}.npz"
mkdir -p logs

echo
echo "===== ① 建索引（$DIR）====="
$PY tools/extract_distill.py --dirs "$DIR" --out "$IDX" 2>&1 | tail -4
[ -s "$IDX" ] || { echo "✗ 索引为空"; exit 1; }
echo "  索引行数：$(wc -l < "$IDX")"

echo
echo "===== ② 物化帧（deck-prefix=$DECK）====="
$PY -u tools/bc_materialize.py --materialize --index "$IDX" \
    --deck-prefix "$DECK" --out "$NPZ" ${MAXG:+--max-games "$MAXG"} 2>&1 | tail -6
[ -s "$NPZ" ] || { echo "✗ 物化失败"; exit 1; }

echo
echo "===== ③ 打包 BC kit ====="
$PY tools/bc_kit.py --npz "$NPZ" --only-deck "$DECK" --max-frames 120000 \
    --epochs 12 --tag "$TAG" 2>&1 | tail -5

echo
echo "===== ④ 验收 ====="
$PY - "$NPZ" <<'PY'
import numpy as np, sys, collections
z=np.load(sys.argv[1], allow_pickle=True)
print(f"帧 {len(z['y'])}；玩家 {len(set(z['agent'].astype(str)))} 个；"
      f"局数 {len(set(z['arch'].astype(str)))} 种 arch")
c=collections.Counter(z["agent"].astype(str))
top=c.most_common(1)[0]
print(f"最大贡献玩家 {top[0][:30]}：{top[1]} 帧（{top[1]/len(z['y']):.1%}）")
print(f"⚠️ 判据：玩家<5 或 单玩家>80% ⇒ 只能做专精、不能当天梯分布")
PY
echo
echo "✅ 产物：dist/bc_kit_${TAG}.tar.gz → 传 GPU 机 bash run.sh"
