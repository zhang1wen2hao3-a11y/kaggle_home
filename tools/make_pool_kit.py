#!/usr/bin/env python3
"""打包**对手池强度阶梯**评测包：给别人在 CPU 机器上跑，产物只有几 KB 的 p_i。

为什么不是把整个联赛搬过去：阶梯只需要"成员 vs 学习者"这一个对局
（`tools/pool_ladder.sh` 的第 29–31 行），不需要 registry、不需要规则、不需要训练。
所以包里有：1 份端口 bundle + N 份权重 + 评测脚本，纯 CPU、无 GPU 依赖。

目录结构（软链全部是**相对路径**，解压到哪都能跑）：
    pool_kit/
      bundle/                     # build/port_plain 的真实副本（cg/ptcg/data/model.pt/deck.csv/main.py）
      cg -> bundle/cg             # 根上再放一个，防某些 cwd 下 import 不到
      ptcg_league/eval_pair.py
      tools/bo3_eval.py
      agents/port_7e39/           # 学习者（端口权重 @ 7e39）—— 只当对手用
      agents/opp_g4b/             # 每个池成员一个目录：model.pt 真实 + 其余软链到 bundle
      run.sh                      # 一键跑完所有成员
      MANIFEST.txt

用法：
    ./tools/py tools/make_pool_kit.py \
        --members opp_g0003=pools/league/opp_g0003,opp_g4a=pools/league/opp_g4a \
        --deck decks/pool_field/teal_mask_ogerpon_ex_applin__7e39.csv
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEAGUE = os.path.join(ROOT, "ptcg_league")

# 每个 agent 目录里除 model.pt / deck.csv 外，其余都软链到 bundle（和 stage_ckpt.py 同一套约定）
LINK_ITEMS = ("main.py", "cg", "ptcg", "data", "EN_Card_Data.csv", "r2e_manifest.json")

RUN_SH = r"""#!/usr/bin/env bash
# 自动生成（tools/make_pool_kit.py）：对手池强度阶梯，纯 CPU
#   PYTHON=/path/to/python bash run.sh          # 默认每个成员 {matches} 场 Bo3
#   MATCHES=400 WORKERS=8 bash run.sh           # 加量
# 中断后重跑会跳过 out/ 里已有结果的成员（可续跑）。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
PY="${PYTHON:-python3}"
N="${MATCHES:-@MATCHES@}"
W="${WORKERS:-@WORKERS@}"      # 0 = auto：按本机内存与核数自动定面板数
SEED="${SEED:-@SEED@}"

echo "===== 环境 ====="
"$PY" - <<'EOF' || { echo "✗ 这个 python 里没有 torch/numpy。用 PYTHON=/path/to/python bash run.sh"; exit 1; }
import sys, torch, numpy
print("python", sys.version.split()[0], "| torch", torch.__version__,
      "| numpy", numpy.__version__, "| cuda_available", torch.cuda.is_available())
EOF
echo "cpu 核数 $(nproc 2>/dev/null || echo '?')  |  每成员 $N 场 Bo3  |  并行 $W（0=auto）  |  种子 $SEED"
echo

# 包完整性：权重/卡组/bundle 少一个都别浪费时间
bad=0
while read -r sha name; do
  [ -f "$name" ] || { echo "✗ 缺 $name"; bad=1; continue; }
  got=$(sha256sum "$name" | cut -d' ' -f1)
  [ "$sha" = "$got" ] || { echo "✗ $name sha 不符（传坏了）"; bad=1; }
done < <(grep -v '^#' MANIFEST.txt | awk '{print $1, $2}')
[ "$bad" = 0 ] || { echo "包不完整，重传"; exit 1; }
echo "✓ 包完整性通过"
echo

LEARNER_NAME="@LEARNER@"
LEARNER="$HERE/agents/$LEARNER_NAME"
mkdir -p out logs
for d in agents/*; do
  name=$(basename "$d")
  [ "$name" = "$LEARNER_NAME" ] && continue        # 学习者只当对手，不测它自己
  out="out/$name.json"
  log="logs/$name.stdout.txt"
  if [ -f "$out" ]; then
    # ⚠️ 只认"跑满且零异常"的结果。半截结果（面板挂了 / OOM 被 kill）必须重跑，
    #    否则下一次会因为"文件已存在"永远跳过一个残缺的 p_i。
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
      --cand "$HERE/$d" --opp "learner=$LEARNER" \
      --matches "$N" --workers "$W" --seed-base "$SEED" \
      --out "$out" --logdir "$HERE/logs/$name" > "$log" 2>&1
  # 面板数那一行是关键诊断（auto 到底选了几个），必须露出来
  grep -m1 -E "面板" "$log" || echo "面板 （没打印，见 logs/$name.stdout.txt）"
  tail -3 "$log" | grep -vE "^$" || true
  # 面板挂了的证据：分片回报数 + agent 日志里的 Traceback
  nfan=$(ls logs/$name/shard-*.jsonl 2>/dev/null | wc -l)
  nrec=$(cat logs/$name/shard-*.jsonl 2>/dev/null | wc -l)
  nerr=$(grep -h "Traceback (most recent call last)" logs/$name/*.log 2>/dev/null | wc -l)
  echo "   分片 $nfan 个，回报 $nrec 场，agent 日志里 Traceback $nerr 条"
  if [ "${nrec:-0}" -lt "$N" ]; then
    echo "   ⚠️ 只收回 $nrec/$N 场 ⇒ 有面板挂了。先看："
    echo "      tail -20 logs/$name/*.log | grep -A5 'Traceback\|load failed\|TimeoutError'"
    echo "      常见原因：内存不够（面板数偏大）或和训练抢 CPU ⇒ 用 WORKERS=8 重跑"
  fi
done

echo
echo "===== 强度阶梯汇总（成员胜率 = p_i = 该成员打赢学习者的比例）====="
"$PY" - <<'EOF'
import glob, json, os
MAP = {}
for line in open("MAP.tsv", encoding="utf-8"):
    if line.startswith("#") or not line.strip():
        continue
    k, v = line.rstrip("\n").split("\t")
    MAP[k] = v
rows = []
for f in sorted(glob.glob("out/*.json")):
    if os.path.basename(f) == "p_i.json":
        continue
    d = json.load(open(f))
    o = d["overall"]
    raw = os.path.basename(f)[:-5]
    aid = MAP.get(raw) or ("opp:" + raw.replace("opp_", "", 1))
    rows.append((aid, o["score"], o["ci_low"], o["ci_high"], o["n"], o["errors"]))
if not rows:
    raise SystemExit("✗ out/ 里没有结果 —— 上面一定有报错，把日志整段发回")
rows.sort(key=lambda r: -r[1])
print(f"{'成员':<30}{'p_i':>7}{'95% CI':>18}{'场次':>6}{'异常':>5}{'学习者胜率':>10}  档位")
res = {}
for aid, s, lo, hi, n, e in rows:
    # ⚠️ 分档必须按**学习者胜率**（= 1 - p_i）判，和 run_league.role_of 同一套阈值。
    #    第一版按成员胜率 p_i 打标签，结果 db04pilot（学习者只赢 30.5%）被标成"C 偏弱"。
    p_l = 1.0 - s
    if p_l < 0.45:   role = "A 强对手"
    elif p_l < 0.60: role = "B 中等"
    elif p_l < 0.75: role = "C 偏弱"
    else:            role = "D 很弱/特殊"
    # ⚠️ 两个方向都要写清楚，血泪教训：
    #    p_i       = 该成员打赢学习者的比例（本脚本量的就是它）
    #    p_learner = 学习者打赢该成员的比例 = 1 - p_i（**池状态 state.pfsp.p 用的就是这个**）
    res[aid] = dict(p_i=round(s, 4), p_learner=round(1.0 - s, 4),
                    ci_p_i=[round(lo, 4), round(hi, 4)], n=n, errors=e, role=role)
    print(f"{aid:<30}{s:>7.3f}{f'[{lo:.3f},{hi:.3f}]':>18}{n:>6}{e:>5}{p_l:>10.3f}  {role}")
json.dump(res, open("out/p_i.json", "w"), ensure_ascii=False, indent=1)
print("\n→ out/p_i.json（只有几 KB，把这个文件发回来即可）")
EOF
echo
echo "把 out/p_i.json 贴回来 / 传回来都行。"
"""


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True,
                    help="逗号分隔 name=dir（dir 相对仓库根，如 pools/league/opp_g4b）")
    ap.add_argument("--bundle", default="build/port_plain")
    ap.add_argument("--deck", default=None,
                    help="学习者的卡组（默认用 bundle/deck.csv 即 7e39）；成员的卡组默认取自己目录里的 deck.csv")
    ap.add_argument("--learner-name", default="port_7e39")
    ap.add_argument("--matches", type=int, default=200)
    ap.add_argument("--workers", type=int, default=0, help="0=auto（跟着本机核数/内存走）")
    ap.add_argument("--seed", type=int, default=760000)
    ap.add_argument("--tag", default="pool_ladder_kit")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "dist"))
    a = ap.parse_args()

    bundle = os.path.join(ROOT, a.bundle) if not os.path.isabs(a.bundle) else a.bundle
    if not os.path.isdir(bundle):
        raise SystemExit(f"✗ bundle 不存在：{bundle}")

    members = []
    for item in [x.strip() for x in a.members.split(",") if x.strip()]:
        name, _, d = item.partition("=")
        aid = None
        if "@" in d:                      # name=dir@opp:xxx —— 目录名猜不出 id 时必须显式给
            d, _, aid = d.partition("@")
        if not aid:
            aid = "opp:" + name[len("opp_"):] if name.startswith("opp_") else \
                  ("sp:" + name[len("sp_"):] if name.startswith("sp_") else name)
        d = d if os.path.isabs(d) else os.path.join(LEAGUE, d) if d.startswith("pools/") \
            else os.path.join(ROOT, d)
        if not os.path.isdir(d):
            raise SystemExit(f"✗ 成员目录不存在：{name} -> {d}")
        for need in ("model.pt", "deck.csv"):
            if not os.path.exists(os.path.join(d, need)):
                raise SystemExit(f"✗ {d} 缺 {need}")
        members.append((name, d, aid))

    stage = os.path.join(a.outdir, a.tag)
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(os.path.join(stage, "agents"))
    os.makedirs(os.path.join(stage, "ptcg_league"))
    os.makedirs(os.path.join(stage, "tools"))

    # ── ① bundle（真实文件）──
    dst_bundle = os.path.join(stage, "bundle")
    subprocess.run(["cp", "-rL", bundle, dst_bundle], check=True)
    subprocess.run(["find", dst_bundle, "-name", "__pycache__", "-type", "d",
                    "-prune", "-exec", "rm", "-rf", "{}", "+"], check=False)
    os.symlink("bundle/cg", os.path.join(stage, "cg"))

    # ── ② 评测脚本（repo 布局要一致：eval_pair 按 parents 找 ROOT/tools）──
    shutil.copyfile(os.path.join(LEAGUE, "eval_pair.py"),
                    os.path.join(stage, "ptcg_league", "eval_pair.py"))
    shutil.copyfile(os.path.join(ROOT, "tools", "bo3_eval.py"),
                    os.path.join(stage, "tools", "bo3_eval.py"))

    # ── ③ agent 目录：model.pt 真实，其余相对软链到 bundle ──
    def make_agent(name, model_src, deck_src):
        d = os.path.join(stage, "agents", name)
        os.makedirs(d)
        for it in LINK_ITEMS:
            src = os.path.join(dst_bundle, it)
            if os.path.exists(src):
                os.symlink(os.path.join("..", "..", "bundle", it), os.path.join(d, it))
        shutil.copyfile(model_src, os.path.join(d, "model.pt"))
        shutil.copyfile(deck_src, os.path.join(d, "deck.csv"))
        return d

    learner_deck = a.deck or os.path.join(bundle, "deck.csv")
    make_agent(a.learner_name, os.path.join(bundle, "model.pt"), learner_deck)
    for name, d, _aid in members:
        make_agent(name, os.path.join(d, "model.pt"), os.path.join(d, "deck.csv"))

    # ── ④ MANIFEST（只校验大件和脚本，几秒内能查完）──
    man = []
    for rel in (["tools/bo3_eval.py", "ptcg_league/eval_pair.py"] +
                [f"agents/{n}/model.pt" for n in [a.learner_name] + [n for n, _, _ in members]] +
                ["bundle/model.pt", "bundle/main.py"]):
        p = os.path.join(stage, rel)
        if os.path.exists(p):
            man.append(f"{sha256(p)}  {rel}  {os.path.getsize(p)}")
    with open(os.path.join(stage, "MAP.tsv"), "w", encoding="utf-8") as f:
        f.write("# dir\tagent_id\n")
        for name, _d, aid in members:
            f.write(f"{name}\t{aid}\n")

    with open(os.path.join(stage, "MANIFEST.txt"), "w", encoding="utf-8") as f:
        f.write("# sha256  path  bytes\n" + "\n".join(man) + "\n")

    with open(os.path.join(stage, "run.sh"), "w", encoding="utf-8") as f:
        run_sh = (RUN_SH.replace("@MATCHES@", str(a.matches))
                     .replace("@WORKERS@", str(a.workers))
                     .replace("@SEED@", str(a.seed)))
        f.write(run_sh.replace("@LEARNER@", a.learner_name))
    os.chmod(os.path.join(stage, "run.sh"), 0o755)

    readme = f"""对手池强度阶梯 · CPU 评测包
================================
用途：测每个池成员**按自己的卡组**打学习者（端口权重 @ 7e39）的胜率 = p_i。
      这个数决定训练里 PFSP 的配额，以及成员该分到 A/B/C/D 哪一档。

成员（{len(members)} 个）：
{chr(10).join('  ' + n + '   (' + aid + ')  ← ' + os.path.basename(d) for n, d, aid in members)}
学习者：agents/{a.learner_name}（端口权重 @ 7e39）

跑法（CPU 即可，不需要 GPU）：
    tar xzf {a.tag}.tar.gz && cd {a.tag}
    PYTHON=/path/to/python3 bash run.sh                  # 默认每成员 {a.matches} 场 Bo3 / 并行 {a.workers}
    MATCHES=400 WORKERS=8 PYTHON=... bash run.sh         # 想更准：400 场，CI 半宽 ~±3.4pp

耗时参考：本机 12 核、每场 Bo3 约 1.5 s ⇒ {a.matches} 场 × {len(members)} 成员 / {a.workers} 并行 ≈ 十几分钟。

要交回来的东西：**out/p_i.json**（几 KB）。其它都不用传。
中断可以直接重跑，已有 out/opp_*.json 的成员会跳过。
"""
    with open(os.path.join(stage, "README.txt"), "w", encoding="utf-8") as f:
        f.write(readme)

    tar = os.path.join(a.outdir, f"{a.tag}.tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(stage, arcname=a.tag)
    mb = os.path.getsize(tar) / 1048576
    print(f"✅ {tar}  {mb:.0f} MB  sha {sha256(tar)[:16]}")
    print(f"   成员 {len(members)}：{', '.join(aid for _, _, aid in members)}")
    print(f"   学习者：{a.learner_name}   每成员 {a.matches} 场 / 并行 {a.workers} / 种子 {a.seed}")
    for line in man:
        print("   " + line.split()[1])


if __name__ == "__main__":
    main()
