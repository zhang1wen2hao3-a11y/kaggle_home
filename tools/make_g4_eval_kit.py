#!/usr/bin/env python3
"""打一个**自包含的评测包**：G4 三支 vs 端口，同种子配对，可直接在任意 Linux 机器跑。

包里有什么（只放必要的东西）：

    g4_eval_kit/
      port/                    端口 agent（对照组，卡组 = 7e39）——含 cg/ 引擎与 ptcg/ 代码
      cand/A|B|C/              三个候选：自己的 model.pt + 软链到 ../../port/* + 自己的 deck.csv
      tools/eval_pair.py       配对评测（复用 tools/bo3_eval.py 的真 Bo3 面板）
      tools/bo3_eval.py
      run.sh                   一条命令跑 A/B(/C) × N 场，并算 Δ_AB / Δ_BC（配对差 + 等效性）
      README.txt               方法、判据、常见失败
      MANIFEST.txt             每个 weight 的 sha256

用法（本机）：
    ./tools/py tools/make_g4_eval_kit.py --banks /tmp/g4/g4_A.bin,/tmp/g4/g4_B.bin,/tmp/g4/g4_C.bin
    # → dist/g4_eval_kit.tar.gz
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

README = """G4 评测包（三支 vs 端口，同种子配对）

## 这是在测什么
G4 是**受控因果实验**：A/B/C 三支训练时除"对手分布"外全部相同。
本包只做一件事——把三支放到**同一套冻结评估**上，回答：
  增加冻结 bank（A→B）和改变训练卡组分布（B→C），有没有可测的胜率差异。

## 环境要求
- Linux + Python3 + **torch（CPU 版即可）**，不需要 GPU、不需要 nvcc、不需要训练仓库
- 内存 ≥ 4 GB（每个面板 2 个 agent 进程）

    python3 -c "import torch, numpy; print(torch.__version__)"

## 跑法
    tar xzf g4_eval_kit.tar.gz && cd g4_eval_kit
    bash run.sh                 # 默认 A/B/C 各 400 场（约 30 分钟）
    N=1600 bash run.sh          # 补到 1600 场（每支约 30 分钟，CI 半宽从 ±6.8pp 缩到 ±3.4pp）
    BRANCHES="A B" N=1600 bash run.sh     # 只跑 A/B（主假设 Δ_AB）

## 判据（预注册，别事后改）
统计单位 = **一场 Bo3 match**；比较用**配对差**（同一个 mid ⇒ 同对手、同座位）。
- Δ_AB = p_B − p_A：加冻结 bank 的效果
- Δ_BC = p_C − p_B：改训练卡组分布的效果（**主假设**）
判读：
- CI 落在 **±3pp** 内            ⇒ 等效（该旋钮无可测收益）
- CI 下界 > 0                     ⇒ 显著更好
- CI 上界 < 0                     ⇒ 显著更差
- 其它                            ⇒ 不显著；**不能**宣称"无效"，只能说"没测出来"
⚠️ 400 场的 CI 半宽约 ±6.8pp ⇒ 对 ±3pp 的等效界**功效不足**，要 1600 场（±3.4pp）。

## 已知正常现象
- `异常 > 0`：某场里 agent 抛异常/随机兜底。**该支数字不可用**，先查内存和超时。
- 单步卡 120 s 超时：内存不够（面板太多），`W=3 bash run.sh` 降面板。
"""

RUNSH = r"""#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PY="${PYTHON:-python3}"
N="${N:-400}"
W="${W:-6}"
SEED="${SEED:-781000}"
BRANCHES="${BRANCHES:-A B C}"

# ⚠️ 必须用带 torch 的解释器：agent 进程要 torch。系统 python3 常常没有，
#    表现是"0 场、异常 0"的静默失败（容易误判成代码 bug）。
if ! $PY -c "import torch, numpy" 2>/dev/null; then
  echo "✗ $PY 缺 torch/numpy。用 PYTHON=/path/to/python3 bash run.sh"
  echo "  找一个：for p in python3 python3.10 python3.11; do \$p -c 'import torch' 2>/dev/null && echo \$p; done"
  exit 1
fi
$PY -c "import torch;print('python', __import__('sys').version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
mkdir -p results/logs

echo "===== G4 评测  N=$N 面板=$W 种子基准=$SEED ====="
echo "分支: $BRANCHES"
echo

for b in $BRANCHES; do
  out="results/g4_$b.json"
  mkdir -p results
  if [ -f "$out" ]; then echo "[skip] G4-$b 已有 $out"; continue; fi
  echo "########## G4-$b vs 端口 ##########"
  $PY tools/eval_pair.py --cand "cand/$b" --opp port=port \
      --matches "$N" --workers "$W" --seed-base "$SEED" \
      --logdir results/logs --out "$out" 2>&1 | tail -3
  echo
done

$PY - <<'EOF'
import glob, json, statistics, os
res = {}
for f in sorted(glob.glob("results/g4_*.json")):
    t = os.path.basename(f)[3]
    o = json.load(open(f))["overall"]
    res[t] = (o, o.get("per_match") or {})
    print(f"G4-{t}: {o['score']:.4f}  CI[{o['ci_low']:.4f},{o['ci_high']:.4f}]  "
          f"n={o['n']}  异常{o['errors']}")
if len(res) < 2:
    raise SystemExit("\n至少需要两支")
print()
def paired(t1, t2, label):
    if t1 not in res or t2 not in res:
        return
    pm1, pm2 = res[t1][1], res[t2][1]
    common = sorted(set(pm1) & set(pm2), key=int)
    if len(common) < 30:
        print(f"{label}: 配对不足（{len(common)}）"); return
    d = [pm2[k] - pm1[k] for k in common]
    m = sum(d) / len(d)
    se = statistics.pstdev(d) / len(d) ** 0.5
    lo, hi = m - 1.96 * se, m + 1.96 * se
    v = ("等效（±3pp 内）" if (lo > -0.03 and hi < 0.03) else
         "显著更好" if lo > 0 else "显著更差" if hi < 0 else "不显著（未落进等效界）")
    print(f"{label}\n  Δ={m:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  半宽±{(hi-lo)/2*100:.1f}pp  "
          f"({sum(1 for x in d if x>0)}胜/{sum(1 for x in d if x<0)}负/"
          f"{sum(1 for x in d if x==0)}平)  ⇒ {v}")
paired("A", "B", "Δ_AB = p_B − p_A（加冻结 bank）")
paired("B", "C", "Δ_BC = p_C − p_B（改训练卡组分布，主假设）")
paired("A", "C", "Δ_AC = p_C − p_A（两者合起来，不可单独归因）")
EOF
"""


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", required=True, help="逗号分隔的 g4_A.bin,g4_B.bin,g4_C.bin")
    ap.add_argument("--port", default="build/port_plain")
    ap.add_argument("--deck", default="decks/pool_field/teal_mask_ogerpon_ex_applin__7e39.csv")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "dist"))
    a = ap.parse_args()

    tag = "g4_eval_kit"
    stage = os.path.join(a.outdir, tag)
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(os.path.join(stage, "tools"), exist_ok=True)

    # 端口 bundle（dereference 软链；去掉 __pycache__）
    port_src = os.path.join(ROOT, a.port)
    port_dst = os.path.join(stage, "port")
    subprocess.run(["cp", "-rL", port_src, port_dst], check=True)
    subprocess.run(["find", port_dst, "-name", "__pycache__", "-type", "d",
                    "-prune", "-exec", "rm", "-rf", "{}", "+"], check=False)

    # ⚠️ 面板进程会 `from cg.game import ...`，而它的 sys.path 是 kit 根
    #    ⇒ kit 根必须有 cg（否则 ModuleNotFoundError，且表现为"0 场、异常 0"的静默失败）
    cg_link = os.path.join(stage, "cg")
    if not os.path.lexists(cg_link):
        os.symlink(os.path.join("port", "cg"), cg_link)

    deck_ids = open(os.path.join(ROOT, a.deck)).read().split()
    deck_txt = "\n".join(deck_ids) + "\n"
    with open(os.path.join(port_dst, "deck.csv"), "w") as f:
        f.write(deck_txt)

    # 候选：真 model.pt + 软链到端口
    man = [f"{sha256(os.path.join(port_dst, 'model.pt'))}  port/model.pt"]
    for spec in a.banks.split(","):
        spec = spec.strip()
        if not spec:
            continue
        b = os.path.basename(spec).replace("g4_", "").replace(".bin", "").upper()
        cd = os.path.join(stage, "cand", b)
        os.makedirs(cd, exist_ok=True)
        for e in os.listdir(port_dst):
            if e in ("model.pt", "deck.csv"):
                continue
            os.symlink(os.path.join("..", "..", "port", e), os.path.join(cd, e))
        # .bin → .pt（需要 torch；本机有）
        pt = os.path.join(cd, "model.pt")
        env = dict(os.environ, PYTHONPATH="data:.:native")
        r = subprocess.run([sys.executable, "native/tools/native_weights.py",
                            "import", os.path.abspath(spec), pt],
                           cwd=os.path.join(ROOT, "port_repo"), env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"✗ {b} 转换失败：\n{r.stdout[-500:]}\n{r.stderr[-800:]}")
        with open(os.path.join(cd, "deck.csv"), "w") as f:
            f.write(deck_txt)
        man.append(f"{sha256(pt)}  cand/{b}/model.pt")
        print(f"  cand/{b}/model.pt  {os.path.getsize(pt):,} B")

    shutil.copyfile(os.path.join(ROOT, "tools", "bo3_eval.py"),
                    os.path.join(stage, "tools", "bo3_eval.py"))
    shutil.copyfile(os.path.join(ROOT, "ptcg_league", "eval_pair.py"),
                    os.path.join(stage, "tools", "eval_pair.py"))
    with open(os.path.join(stage, "run.sh"), "w") as f:
        f.write(RUNSH)
    os.chmod(os.path.join(stage, "run.sh"), 0o755)
    with open(os.path.join(stage, "README.txt"), "w") as f:
        f.write(README)
    with open(os.path.join(stage, "MANIFEST.txt"), "w") as f:
        f.write("# sha256  path\n" + "\n".join(man) + "\n")

    tar = os.path.join(a.outdir, tag + ".tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(stage, arcname=tag)
    print(f"\n✅ {tar}  {os.path.getsize(tar)/1048576:.0f} MB  sha {sha256(tar)[:16]}")


if __name__ == "__main__":
    main()
