#!/usr/bin/env python3
"""打包**跨卡组对位评测**包：N 个候选（各自打自己的牌）vs 一个固定对手。

为什么要单独一套：`eval_pair` 的常规 confirm 是**同卡组镜像**（cand@X vs port@X），
它测不出"打某个特定原型"的能力。而真实天梯里最致命的漏洞恰恰是跨卡组的
（例：db04 对 shuppet 0 胜 23 负）。这个包就是量那一格的。

关键约束：**所有候选共用同一批种子**（`--seed-base` 一致）⇒ `per_match` 可逐场配对，
所以能给"候选 X 相对基线提升了多少"的配对差，而不是拿两个置信区间硬比。

用法：
    ./tools/py tools/make_matchup_kit.py \
        --opp shuppet=ptcg_league/pools/league/...或任意 agent 目录 \
        --cand port=/tmp/ag_port_db04 --cand g5=/tmp/ag_g5_db04 \
        --cand g6=/tmp/ag_g6_db04 --cand g7a=/tmp/ag_g7a \
        --matches 400 --seed-base 913000 --tag shuppet_matchup_kit
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK_ITEMS = ("main.py", "cg", "ptcg", "data", "EN_Card_Data.csv", "r2e_manifest.json")

RUN_SH = r"""#!/usr/bin/env bash
# 自动生成（tools/make_matchup_kit.py）：跨卡组对位评测，纯 CPU
#   PYTHON=/path/to/python3 bash run.sh            # 每个候选 {matches} 场 Bo3
#   MATCHES=600 WORKERS=12 bash run.sh             # 加量
# 已有 out/<候选>.json 且跑满、异常 0 的候选会跳过（可续跑）。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
PY="${PYTHON:-python3}"
N="${MATCHES:-@MATCHES@}"
W="${WORKERS:-@WORKERS@}"
SEED="${SEED:-@SEED@}"

echo "===== 环境 ====="
"$PY" - <<'EOF' || { echo "✗ 这个 python 里没有 torch/numpy。用 PYTHON=/path/to/python bash run.sh"; exit 1; }
import sys, torch, numpy
print("python", sys.version.split()[0], "| torch", torch.__version__,
      "| numpy", numpy.__version__, "| cuda_available", torch.cuda.is_available())
EOF
echo "cpu 核数 $(nproc 2>/dev/null || echo '?')  |  每候选 $N 场 Bo3  |  并行 $W（0=auto）  |  种子 $SEED"
echo

bad=0
while read -r sha name; do
  [ -f "$name" ] || { echo "✗ 缺 $name"; bad=1; continue; }
  [ "$sha" = "$(sha256sum "$name" | cut -d' ' -f1)" ] || { echo "✗ $name sha 不符（传坏了）"; bad=1; }
done < <(grep -v '^#' MANIFEST.txt | awk '{print $1, $2}')
[ "$bad" = 0 ] || { echo "包不完整，重传"; exit 1; }
echo "✓ 包完整性通过"
echo

OPP="$(ls -d agents/opp_* 2>/dev/null | head -1)"
[ -n "$OPP" ] || { echo "✗ 找不到固定对手目录 agents/opp_*"; exit 1; }
echo "固定对手 = $OPP"
mkdir -p out logs
for d in agents/cand_*; do
  name=$(basename "$d")
  out="out/$name.json"
  log="logs/$name.stdout.txt"
  if [ -f "$out" ]; then
    ok=$("$PY" - "$out" "$N" <<'PYEOF' 2>/dev/null || echo 0
import json, sys
o = json.load(open(sys.argv[1]))["overall"]
want = int(sys.argv[2])
print(1 if o["n"] >= want * 0.98 and o["errors"] == 0 else 0)
PYEOF
)
    if [ "$ok" = 1 ]; then echo "[skip] $name（已完成，n=$N，异常 0）"; continue; fi
    echo "[重跑] $name（上次没跑满或异常>0）"; rm -f "$out"
  fi
  echo "########## $name ##########"
  "$PY" ptcg_league/eval_pair.py \
      --cand "$HERE/$d" --opp "opp=$HERE/$OPP" \
      --matches "$N" --workers "$W" --seed-base "$SEED" \
      --out "$out" --logdir "$HERE/logs/$name" > "$log" 2>&1
  grep -m1 -E "面板" "$log" || true
  tail -3 "$log" | grep -vE "^$" || true
  got=$("$PY" - "$out" <<'PYEOF' 2>/dev/null || echo 0
import json, sys
try:
    print(json.load(open(sys.argv[1]))["overall"]["n"])
except Exception:
    print(0)
PYEOF
)
  [ "${got:-0}" -lt "$N" ] && echo "   ⚠️ 只收回 $got/$N 场 ⇒ 面板挂了；用 WORKERS=8 重跑"
done

echo
echo "===== 对位汇总（候选胜率 = 它打这个对手赢的比例）====="
"$PY" - <<'EOF'
import glob, json, os, itertools, math, statistics
rows = {}
for f in sorted(glob.glob("out/cand_*.json")):
    d = json.load(open(f))
    o = d["overall"]
    rows[os.path.basename(f)[5:-5]] = o
if not rows:
    raise SystemExit("✗ out/ 里没有结果 —— 上面一定有报错，把日志整段发回")
print("%-14s%7s%18s%6s%5s" % ("候选", "胜率", "95% CI", "场次", "异常"))
for name, o in sorted(rows.items(), key=lambda kv: -kv[1]["score"]):
    ci = "[%.3f,%.3f]" % (o["ci_low"], o["ci_high"])
    print("%-14s%7.3f%18s%6d%5d" % (name, o["score"], ci, o["n"], o["errors"]))

base = "0_port" if "0_port" in rows else sorted(rows)[0]
bpm = rows[base].get("per_match") or {}
print(f"\n===== 配对差（相对基线 {base}，同批种子逐场对减）=====")
print(f"{'候选':<14}{'配对Δ':>8}{'95% CI':>20}{'判定':>10}")
for name, o in sorted(rows.items()):
    if name == base:
        continue
    pm = o.get("per_match") or {}
    keys = sorted(set(bpm) & set(pm))
    if len(keys) < min(4, len(bpm) // 2 or 1):
        print(f"{name:<14}{'—':>8}{'（种子对不上，无法配对）':>20}")
        continue
    diffs = [pm[k] - bpm[k] for k in keys]
    m = statistics.fmean(diffs)
    sd = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
    hw = 1.96 * sd / math.sqrt(len(diffs))
    lo, hi = m - hw, m + hw
    verdict = "显著更好" if lo > 0 else ("显著更差" if hi < 0 else "跨 0（无定论）")
    print(f"{name:<14}{100*m:>+7.1f}pp{f'[{100*lo:+.1f},{100*hi:+.1f}]pp':>20}{verdict:>10}")
EOF
echo
echo "把 out/ 整个目录（几 KB 到几十 KB）发回来即可。"
"""


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def add_agent(stage, bundle_rel, name, src):
    d = os.path.join(stage, "agents", name)
    os.makedirs(d)
    for it in LINK_ITEMS:
        if os.path.exists(os.path.join(stage, "bundle", it)):
            os.symlink(os.path.join("..", "..", "bundle", it), os.path.join(d, it))
    shutil.copyfile(os.path.join(src, "model.pt"), os.path.join(d, "model.pt"))
    shutil.copyfile(os.path.join(src, "deck.csv"), os.path.join(d, "deck.csv"))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", required=True, help="NAME=DIR：固定对手（它打它自己的牌）")
    ap.add_argument("--cand", action="append", required=True, help="NAME=DIR：候选，可重复")
    ap.add_argument("--matches", type=int, default=400)
    ap.add_argument("--workers", type=int, default=0, help="0=auto")
    ap.add_argument("--seed-base", type=int, default=913000)
    ap.add_argument("--bundle", default="build/port_plain")
    ap.add_argument("--tag", default="matchup_kit")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "dist"))
    a = ap.parse_args()

    def split(spec):
        n, _, d = spec.partition("=")
        d = os.path.abspath(d)
        for need in ("model.pt", "deck.csv"):
            if not os.path.exists(os.path.join(d, need)):
                raise SystemExit(f"✗ {n}: {d} 缺 {need}")
        return n.strip(), d

    oname, odir = split(a.opp)
    cands = [split(c) for c in a.cand]

    stage = os.path.join(a.outdir, a.tag)
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(os.path.join(stage, "ptcg_league"))
    os.makedirs(os.path.join(stage, "tools"))
    os.makedirs(os.path.join(stage, "agents"))
    bundle = os.path.join(ROOT, a.bundle)
    subprocess.run(["cp", "-rL", bundle, os.path.join(stage, "bundle")], check=True)
    subprocess.run(["find", os.path.join(stage, "bundle"), "-name", "__pycache__", "-type", "d",
                    "-prune", "-exec", "rm", "-rf", "{}", "+"], check=False)
    # bundle 里的大权重对评测没用（每个 agent 有自己的 model.pt），删掉省 20 MB
    for junk in ("model.pt",):
        p = os.path.join(stage, "bundle", junk)
        if os.path.exists(p):
            os.remove(p)
    os.symlink("bundle/cg", os.path.join(stage, "cg"))
    shutil.copyfile(os.path.join(ROOT, "ptcg_league", "eval_pair.py"),
                    os.path.join(stage, "ptcg_league", "eval_pair.py"))
    shutil.copyfile(os.path.join(ROOT, "tools", "bo3_eval.py"),
                    os.path.join(stage, "tools", "bo3_eval.py"))

    add_agent(stage, "bundle", f"opp_{oname}", odir)
    for i, (n, d) in enumerate(cands):
        add_agent(stage, "bundle", f"cand_{i}_{n}", d)

    run = (RUN_SH.replace("@MATCHES@", str(a.matches))
                .replace("@WORKERS@", str(a.workers)).replace("@SEED@", str(a.seed_base)))
    p = os.path.join(stage, "run.sh")
    open(p, "w", encoding="utf-8").write(run)
    os.chmod(p, 0o755)

    man = []
    for rel in (["tools/bo3_eval.py", "ptcg_league/eval_pair.py"] +
                [f"agents/opp_{oname}/model.pt", f"agents/opp_{oname}/deck.csv"] +
                [f"agents/cand_{i}_{n}/model.pt" for i, (n, _) in enumerate(cands)]):
        f = os.path.join(stage, rel)
        if os.path.exists(f):
            man.append(f"{sha256(f)}  {rel}  {os.path.getsize(f)}")
    open(os.path.join(stage, "MANIFEST.txt"), "w", encoding="utf-8").write(
        "# sha256  path  bytes\n" + "\n".join(man) + "\n")

    readme = f"""跨卡组对位评测包
==================
固定对手：agents/opp_{oname}（{os.path.basename(odir)}）
候选（各自打自己的牌）：
{chr(10).join(f'  cand_{i}_{n}   ← {os.path.basename(d)}' for i, (n, d) in enumerate(cands))}

跑法（纯 CPU，不需要 GPU）：
    tar xzf {a.tag}.tar.gz && cd {a.tag}
    PYTHON=/path/to/python3 bash run.sh                 # 每候选 {a.matches} 场，并行 auto
    MATCHES=600 WORKERS=12 PYTHON=... bash run.sh       # 加量

种子基准 {a.seed_base}（所有候选同一批 ⇒ 可逐场配对，能给"相对基线提升多少 pp"）。
交回：**out/ 整个目录**。中断可直接重跑，跑满且异常 0 的候选会跳过。
"""
    open(os.path.join(stage, "README.txt"), "w", encoding="utf-8").write(readme)

    tar = os.path.join(a.outdir, f"{a.tag}.tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(stage, arcname=a.tag)
    print(f"✅ {tar}  {os.path.getsize(tar) / 1048576:.0f} MB  sha {sha256(tar)[:16]}")
    print(f"   对手 = {oname}（{os.path.basename(odir)}）")
    print(f"   候选 = {', '.join(n for n, _ in cands)}   每候选 {a.matches} 场，种子 {a.seed_base}")


if __name__ == "__main__":
    main()
