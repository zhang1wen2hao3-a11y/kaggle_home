#!/usr/bin/env bash
# **db04 卡组线 · 通用运行器**（GPU 机器上跑）：anchor=db04 + 真实野区 + 冻结 bank。
#   G5 = 从端口权重起步（INIT=native/pt_model_field.bin）
#   G6 = 从 G5 权重继续（INIT=init/g5_db04.bin）
#   ⇒ 以后每一代**只换 INIT**，不用再改脚本（这就是这张表唯一的变量）。
#
# | 项 | G4-B（已跑） | G5（本脚本） |
# |---|---|---|
# | 学习者卡组 | `teal_mask_ogerpon_ex_applin__7e39` | **`dunsparce_abra__db04`** |
# | 对手卡组分布 | weights_real（185 副真实野区） | 同 |
# | 冻结 bank | 2 个（league_g0003/bank_0,1） | 同 |
# | p_anchor / pct / lr / AG / seed / T | 1.0 / 0.08 / 2e-4 / 512 / 424242 / 400k | **全部相同** |
#
# 为什么是这个受控设计（下一步.md §26）：卡组线的收益（db04 真实胜率 0.642 vs 7e39 0.568）
# 是驾驶线（同卡组换权重 ≈ +12 Elo）的两倍多，而 G4-B 已经把"配方"有效这一件事证到了
# 0.5337 [0.5093,0.5581]（首次显著强于 port）。所以这里只换牌，不换配方，SEED 也不换。
#
# 用法（云端，$PTCG_REPO 指向训练仓库）：
#   bash tools/db04_line.sh                    # 从 native/pt_model_field.bin 起步（= G5 配方）
#   INIT=init/g5_db04.bin bash tools/db04_line.sh   # 从 G5 权重继续（= G6）
#   bash tools/db04_line.sh --check            # 只预检（不花 GPU）
#   T=1300000 bash tools/db04_line.sh          # 加档
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# 仓库定位：$PTCG_REPO → $HOME/work/... → **当前目录**（cwd 就是 port_repo 时最自然）
REPO="${PTCG_REPO:-$HOME/work/ptcg_train_kit/port_repo}"
if [ ! -f "$REPO/native/train_native.py" ]; then
  if [ -f "./native/train_native.py" ]; then
    REPO="$PWD"      # 已经 cd 进 port_repo 了，别再报"找不到"
  else
    echo "✗ 找不到训练仓库：$REPO（用 PTCG_REPO= 指定，或先 cd 进 port_repo）"; exit 1
  fi
fi
cd "$REPO"

PY="${PYTHON:-python3}"
T="${T:-400000}"
AG="${AG:-512}"
MB="${MB:-512}"
LR="${LR:-2e-4}"
SEED="${SEED:-424242}"           # ★ 与 G4-B 相同：只许"卡组"这一个变量动
THREADS="${THREADS:-8}"
ANCHOR="${ANCHOR:-dunsparce_abra__db04}"
TRACK="${TRACK:-dunsparce_abra__db04,teal_mask_ogerpon_ex_applin__7e39}"
DWF="${DWF:-decks/weights_real.txt}"
POOL="${POOL:-decks/pool_field}"
OUT="${OUT:-/root/g6}"
# ★ 唯一会变的变量：父权重（native blob，18,205,408 B）
INIT="${INIT:-native/pt_model_field.bin}"
BANKS="${BANKS:-/root/league_g0003/bank_0.bin,/root/league_g0003/bank_1.bin}"
LEAGUE_PCT="${LEAGUE_PCT:-0.08}"
# ★ bank 绑牌：与 BANKS 逐位对应；"-" 表示该 bank 不绑定（保留全局权重抽样）。
#   例：BANKS=a.bin,b.bin BANKDECKS=shuppet_dhelmise__0d86,-
BANKDECKS="${BANKDECKS:-}"
TAG="${TAG:-db04}"
RUN="${RUN:-$(basename "${INIT%.bin}" | sed 's/_db04$//')_$TAG}"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

echo "===== $RUN ====="
echo "repo    = $REPO"
echo "anchor  = $ANCHOR   （学习者 100% 这副牌）"
echo "对手卡组 = $DWF"
echo "banks   = ${BANKS:-（无）}  league-pct=$LEAGUE_PCT"
[ -n "$BANKDECKS" ] && echo "bank绑牌 = $BANKDECKS   ← 需要 --bank-decks 补丁（tools/patch_bank_decks.py）"
echo "track   = $TRACK"
echo "INIT    = $INIT   ← 父权重（唯一变量）"
echo "T=$T AG=$AG MB=$MB LR=$LR SEED=$SEED THREADS=$THREADS → 输出 $OUT/$RUN.bin"
echo

# ── ① 预检（一次把能静默失败的地方全查掉）────────────────────────────
mkdir -p "$OUT" logs 2>/dev/null || mkdir -p logs
"$PY" - "$ANCHOR" "$POOL" "$DWF" "native/ptcg_tables_field.bin" <<'EOF' || exit 1
import glob, os, struct, sys

anchor, pool, wf, tables = sys.argv[1:5]

def pool_deck_names(pool):
    names, rows = [], []
    for fn in sorted(glob.glob(os.path.join(pool, "*.csv"))):
        try:
            ids = [int(l) for l in open(fn) if l.strip()]
        except ValueError:
            continue
        if len(ids) == 60:
            names.append(os.path.basename(fn)[:-4]); rows.append(ids)
    return names, rows

def blob_deck_table(path):
    with open(path, "rb") as f:
        assert f.read(8) == b"PTCGTAB1", f"{path}: bad magic"
        nsec = struct.unpack("<15i", f.read(60))[14]
        for _ in range(nsec):
            name = f.read(24).rstrip(b"\0").decode()
            dt, nd = struct.unpack("<2i", f.read(8))
            shape = struct.unpack("<4q", f.read(32))[:nd]
            esz = {0: 4, 1: 4, 2: 1, 3: 1}[dt]
            n = 1
            for s in shape:
                n *= s
            if name == "decks":
                rows = struct.unpack(f"<{n}i", f.read(n * esz))
                return [list(rows[i*60:(i+1)*60]) for i in range(shape[0])]
            f.seek(n * esz, 1)
    raise AssertionError(f"{path}: no decks section")

names, rows = pool_deck_names(pool)
blob = blob_deck_table(tables)
print(f"① 卡组池 {pool}: {len(names)} 副（表格 blob {len(blob)} 行）")
if rows != blob:
    print("✗ 卡组表不一致：pool 的行与 tables blob 不同 ⇒ deck id 会错位，禁止开训")
    sys.exit(1)
print("   ✓ 池与 tables blob 逐行一致")
if anchor not in names:
    print(f"✗ anchor 卡组不在池里：{anchor}")
    sys.exit(1)
i = names.index(anchor)
print(f"② anchor `{anchor}` → deck id {i}")

# 权重文件：必须存在、解析成 version>=3、且给 anchor 非零权重（否则训练器 REJECTED）
ver, dflt, per = 0, 1.0, {}
if not os.path.exists(wf):
    print(f"✗ 权重文件不存在：{wf}"); sys.exit(1)
for ln in open(wf):
    ln = ln.split("#")[0].strip()
    if not ln:
        continue
    k, v = ln.split()
    if k == "version": ver = int(v)
    elif k == "default": dflt = float(v)
    else: per[k] = float(v)
if ver < 3:
    print(f"✗ {wf} version={ver} < 3（会被当成「没有请求」静默跳过）"); sys.exit(1)
w = [dflt] * len(names)
unknown = [k for k in per if k not in names]
if unknown:
    print(f"✗ 权重文件里有池外名字 {unknown[:5]} ⇒ 整份权重会被拒绝"); sys.exit(1)
for k, v in per.items():
    w[names.index(k)] = v
tot = sum(w)
ess = (tot * tot) / sum(x * x for x in w) if tot else 0
nact = sum(1 for x in w if x > 0)
print(f"③ {os.path.basename(wf)}: version={ver} 默认={dflt} "
      f"覆盖 {len(per)} 副 → 生效 {nact}/{len(names)} 副, ESS {ess:.1f}")
if w[i] <= 0:
    print(f"✗ anchor 卡组权重为 0 ⇒ 训练器会 REJECTED（weight of anchor deck must be > 0）")
    sys.exit(1)
print(f"   ✓ anchor 权重 {w[i]:.6f}（占 {100*w[i]/tot:.2f}%）")
print(f"   ✓ 镜像对局率 ≈ {100*w[i]/tot:.1f}%（学习者也打这副，对手抽到同牌的比例）")
top = sorted(zip(names, w), key=lambda kv: -kv[1])[:3]
print("   生效后对手座位最高的 3 副：" +
      "， ".join(f"{n} {100*x/tot:.1f}%" for n, x in top))
for pre in ("shuppet", "dreepy", "dunsparce_abra", "teal_mask_ogerpon"):
    sh = sum(x for n, x in zip(names, w) if n.startswith(pre))
    if sh / tot > 0.05:
        print(f"   · {pre:22s} 合计 {100*sh/tot:.1f}%")
EOF

if [ ! -f "$INIT" ]; then
  echo "✗ INIT 不存在：$INIT（G5 用 native/pt_model_field.bin；G6 用包里的 init/g5_db04.bin）"; exit 1
fi
[ "$(stat -c%s "$INIT")" = "18205408" ] \
  || { echo "✗ $INIT 不是 18,205,408 B（坏包 ⇒ init.bin 会 0 字节）"; exit 1; }
printf "   init %s  %s B\n" "$INIT" "$(stat -c%s "$INIT")"
if [ -n "$BANKS" ]; then
  IFS=, read -ra _b <<< "$BANKS"
  for b in "${_b[@]}"; do
    [ -f "$b" ] || { echo "✗ bank 不存在：$b（G4-B 用过这两个；没有就 BANKS=\"\" 走无 bank 分支）"; exit 1; }
    printf "   bank %-40s %s B\n" "$b" "$(stat -c%s "$b")"
  done
fi
if [ -n "$BANKDECKS" ]; then
  grep -q "bank_deck override" native/ptcg/env.c 2>/dev/null \
    || { echo "✗ 你要绑牌（BANKDECKS 非空）但仓库没打 --bank-decks 补丁"; \
         echo "   在 port_repo 根目录跑：./tools/py tools/patch_bank_decks.py && bash native/build_native.sh"; exit 1; }
  echo "   ✓ 检测到 --bank-decks 补丁（env.c 里有 bank_deck override）"
fi
echo "④ 权重/bank 预检通过"
echo
[ "$CHECK_ONLY" = 1 ] && { echo "（--check 模式，预检通过，未开训）"; exit 0; }

# ── ② 训练 ───────────────────────────────────────────────────────────
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ "${used:-0}" -gt 4000 ]; then
  echo "⚠️ 显存已被占 ${used} MiB，可能有残留进程："
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  echo "   先 kill -9 再跑，否则 CUDA OOM"; exit 1
fi

RD="experiments/native-$RUN"
rm -rf "$RD"; mkdir -p "$RD"
cp -f "$INIT" "$RD/init.bin"
[ -s "$RD/init.bin" ] || { echo "✗ init.bin 拷贝失败（0 字节）"; exit 1; }

PYTHONPATH=data:.:native PTCG_TABLES=native/ptcg_tables_field.bin OMP_WAIT_POLICY=PASSIVE \
"$PY" native/train_native.py --run-name "$RUN" \
  --timesteps "$T" --no-anneal-lr \
  --total-agents "$AG" --minibatch "$MB" --num-threads "$THREADS" \
  --lr "$LR" --seed "$SEED" \
  --deck-pool "$POOL" --deck-weights-file "$DWF" \
  --anchor-decks "$ANCHOR" --track-decks "$TRACK" \
  --p-anchor-learner 1.0 --p-anchor-self-opp 0.0 --p-anchor-league-opp 0.0 \
  ${BANKS:+--league-ckpts "$BANKS" --league-pct "$LEAGUE_PCT"} \
  ${BANKDECKS:+--bank-decks "$BANKDECKS"} \
  2>&1 | tee "logs/$RUN.log" | grep -E "anchor decks resolved|deck weights|REJECTED|epoch 5 |epoch 10 |final" | head -8

# ── ③ 硬门槛（缺证据就不算数）────────────────────────────────────────
echo
echo "===== 硬门槛检查 ====="
if [ -f "$RD/final.bin" ]; then
  cp -f "$RD/final.bin" "$OUT/$RUN.bin"
  echo "  ✅ 模型 → $OUT/$RUN.bin ($(stat -c%s "$OUT/$RUN.bin") B)"
else
  echo "  ❌ 没有 final.bin，看 logs/$RUN.log"
fi
grep -q "anchor decks resolved" "logs/$RUN.log" \
  && echo "  ✅ anchor 解析成功：$(grep -m1 -oE 'anchor decks resolved.*' "logs/$RUN.log")" \
  || echo "  ❌ 没有 anchor 解析行 ⇒ 卡组没换成功"
grep -q "deck weights ->" "logs/$RUN.log" \
  && echo "  ✅ 权重已生效：$(grep -m1 -oE 'deck weights ->.*' "logs/$RUN.log")" \
  || echo "  ❌ 没有 'deck weights ->' 行 ⇒ 权重没被应用（或被 REJECTED）"
if grep -qE "tl[0-9]:[0-9.]+/to[0-9]:[0-9.]+" "logs/$RUN.log"; then
  echo "  ✅ 实际分布证据：$(grep -oE 'tl[0-9]:[0-9.]+/to[0-9]:[0-9.]+' "logs/$RUN.log" | tail -1)"
  echo "     （tl0 应 ≈1.0 = 学习者确实在打 db04；to0 应很小 = 对手很少抽到 db04）"
else
  echo "  ❌ 没有 tl/to 字段 ⇒ 无法证明卡组真的换了，该支不进胜率分析"
fi
echo
echo "拉回本机：scp $OUT/$RUN.bin 与 logs/$RUN.log"
echo "评估协议：与 G5 完全一致（cand@db04 vs port@db04，1600 场配对，ci_low>0.50 才算提升）"
