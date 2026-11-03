#!/usr/bin/env python3
"""**BC 训练包**：把「某个筛选 + 若干帧」打成一个能上 GPU 机器直接跑的 tar.gz。

为什么需要：本机 RTX 3050（4 GB）+ 磁盘 36 MB/s，跑 BC 慢得离谱
（实测 80k 帧一个 epoch 要 18 分钟，还和评测抢资源）。
而 BC 训练**只依赖「预先算好的行向量」**，不需要引擎、不需要回放、不需要 23 GB 的全量语料
—— 所以可以把「要用到的那几千/几万帧」单独切出来发过去。

生成物结构（与项目内路径一致，`tools/bc_train.py` 无需改动）：

    bc_kit_<tag>/
      tools/bc_train.py            # 原样复制
      build/port_plain/…           # 端口 bundle（policy 起点 + 基线，含 cg/ptcg/data）
      data/slice.npz, data/slice.f32   # **只含这次要用的帧**
      run.sh                       # 训练 → 导出 .bin → 打印回传命令
      MANIFEST.txt, plan.json

用法：
    # 最强玩家（LB 1441，254k 帧里抽 60k）
    ./tools/py tools/bc_kit.py --only-agent "ミワハルキ" --max-frames 60000 --epochs 8

    # 某副卡组
    ./tools/py tools/bc_kit.py --only-deck 2617 --max-frames 12000 --epochs 10 --tag grookey

    # 多个筛选一起打（省一次上传）：--only-deck 2617,ea39,e1f9
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

ROW_W = 10944

RUN_SH = r"""#!/usr/bin/env bash
# 自动生成（tools/bc_kit.py）：BC 训练 + 导出 .bin
#   bash run.sh              # 训练 → 导出 → 打印回传命令
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PY="${{PYTHON:-python3}}"

# ① 包完整性（传坏了别浪费一小时 GPU）
bad=0
while read -r sha name; do
  [ -f "$name" ] || {{ echo "✗ 缺 $name"; bad=1; continue; }}
  got=$(sha256sum "$name" | cut -d' ' -f1)
  [ "$sha" = "$got" ] || {{ echo "✗ $name sha 不符"; bad=1; }}
done < <(grep -v '^#' MANIFEST.txt | awk '{{print $1, $2}}')
[ "$bad" = 0 ] || {{ echo "包不完整，重传"; exit 1; }}
echo "✓ 包完整性通过"

command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.used --format=csv,noheader

# ② 训练
echo
echo "===== 训练 {epochs} epoch / lr {lr} / bs {bs} ====="
$PY -u tools/bc_train.py --npz data/slice.npz --device cuda \
    --epochs {epochs} --lr {lr} --bs {bs} \
    --out "out/{tag}.pt"
rc=$?
[ $rc -eq 0 ] || {{ echo "✗ 训练失败 rc=$rc"; exit $rc; }}

# ③ 导出成训练器能直接吃的原生 blob（.bin）—— 联赛的 bank 只认这个格式
echo
echo "===== 导出 .bin ====="
if [ -f tools/native_weights.py ]; then
  PYTHONPATH=build/port_plain $PY tools/native_weights.py export "out/{tag}.pt" "out/{tag}.bin" 2>&1 | tail -1
else
  echo "⚠️ 没有 native_weights.py，跳过导出（只保留 .pt）"
fi

echo
echo "✅ 产物："
ls -lh out/ 2>/dev/null
echo
echo "传回本机："
echo "  scp -P <端口> <用户>@<主机>:$HERE/out/{tag}.* \\"
echo "      {local_pool}/"
echo "然后本机跑：./tools/py tools/bc_pool_registry.py"
"""


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def subsample_blocks(idx, target, block=512):
    """与 bc_train.py 完全一致的抽样口径（块内连续、块间均匀）。"""
    import numpy as np
    n = len(idx)
    if target <= 0 or target >= n:
        return idx
    nb = max(1, target // block)
    block = max(1, min(block, n // nb))
    starts = np.linspace(0, n - block, nb).astype(np.int64)
    return np.sort(np.concatenate([idx[s:s + block] for s in starts]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(ROOT, "data", "bc_all.npz"))
    ap.add_argument("--only-deck", default="", help="卡组 hash 前缀，逗号分隔可给多个")
    ap.add_argument("--only-agent", default="", help="玩家名子串（正则），逗号分隔可给多个")
    ap.add_argument("--winners-only", action="store_true")
    ap.add_argument("--max-frames", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--bundle", default="build/port_plain")
    a = ap.parse_args()

    import numpy as np
    z = np.load(a.npz, allow_pickle=True)
    n_all = len(z["y"])
    ok = np.ones(n_all, dtype=bool)
    for d in [x.strip() for x in a.only_deck.split(",") if x.strip()]:
        ok &= np.char.startswith(z["deck"].astype(str), d)
    for pat in [x.strip() for x in a.only_agent.split(",") if x.strip()]:
        ok &= np.array([pat in str(s) for s in z["agent"]])
    if a.winners_only:
        ok &= z["won"].astype(bool)
    split = z["split"].astype(str)

    tr = np.where((split == "train") & ok)[0]
    va = np.where((split == "val") & ok)[0]
    tr = subsample_blocks(tr, a.max_frames)
    va = subsample_blocks(va, max(1, int(a.max_frames * 0.25)))
    keep = np.sort(np.concatenate([tr, va]))
    if len(keep) == 0:
        raise SystemExit("✗ 筛选后没有帧")

    tag = a.tag or re.sub(r"[^A-Za-z0-9_.-]", "_",
                          (a.only_agent or a.only_deck or "all"))[:28]
    stage = os.path.join(a.outdir, f"bc_kit_{tag}")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(os.path.join(stage, "data"), exist_ok=True)
    os.makedirs(os.path.join(stage, "out"), exist_ok=True)

    # ── 切数据：只保留 keep 这些行 ──
    xdat = os.path.splitext(a.npz)[0] + ".f32"
    X = np.memmap(xdat, dtype=np.float32, mode="r", shape=(n_all, ROW_W))
    dst_f32 = os.path.join(stage, "data", "slice.f32")
    step = 4096
    with open(dst_f32, "wb") as f:
        for i in range(0, len(keep), step):
            blk = np.ascontiguousarray(X[np.sort(keep[i:i + step])], dtype=np.float32)
            f.write(blk.tobytes())
    # ⚠️ `y` 是**训练目标**（专家动作索引），必须一起切；第一版把它排除了，切片直接缺列
    sub = {k: z[k][keep] for k in z.files}
    np.savez(os.path.join(stage, "data", "slice.npz"), **sub)

    # ── 代码与端口 bundle ──
    os.makedirs(os.path.join(stage, "tools"), exist_ok=True)
    shutil.copyfile(os.path.join(ROOT, "tools", "bc_train.py"),
                    os.path.join(stage, "tools", "bc_train.py"))
    nw = os.path.join(ROOT, "port_repo", "native", "tools", "native_weights.py")
    if os.path.exists(nw):
        shutil.copyfile(nw, os.path.join(stage, "tools", "native_weights.py"))
    os.makedirs(os.path.join(stage, "build"), exist_ok=True)
    dst_bundle = os.path.join(stage, "build", "port_plain")
    subprocess.run(["cp", "-rL", os.path.join(ROOT, a.bundle), dst_bundle], check=True)
    subprocess.run(["find", dst_bundle, "-name", "__pycache__", "-type", "d",
                    "-prune", "-exec", "rm", "-rf", "{}", "+"], check=False)

    # ── MANIFEST / plan / run.sh ──
    man = []
    for rel in ("data/slice.npz", "data/slice.f32", "tools/bc_train.py",
                "build/port_plain/model.pt", "build/port_plain/deck.csv"):
        p = os.path.join(stage, rel)
        if os.path.exists(p):
            man.append(f"{sha256(p)}  {rel}  {os.path.getsize(p)}")
    with open(os.path.join(stage, "MANIFEST.txt"), "w", encoding="utf-8") as f:
        f.write("# sha256  path  bytes\n" + "\n".join(man) + "\n")

    plan = dict(tag=tag, filter=dict(deck=a.only_deck, agent=a.only_agent,
                                     winners=a.winners_only),
                n_frames=int(len(keep)), n_train=int(len(tr)), n_val=int(len(va)),
                max_frames=a.max_frames, epochs=a.epochs, lr=a.lr, bs=a.bs,
                made_by="tools/bc_kit.py")
    with open(os.path.join(stage, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)

    with open(os.path.join(stage, "run.sh"), "w", encoding="utf-8") as f:
        f.write(RUN_SH.format(epochs=a.epochs, lr=a.lr, bs=a.bs, tag=tag,
                              local_pool=os.path.join(ROOT, "bc_pool")))
    os.chmod(os.path.join(stage, "run.sh"), 0o755)

    tar = os.path.join(a.outdir, f"bc_kit_{tag}.tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(stage, arcname=f"bc_kit_{tag}")
    mb = os.path.getsize(tar) / 1048576
    print(f"帧：筛出 {int(ok.sum()):,} → 用 {len(keep):,}（train {len(tr):,} / val {len(va):,}）")
    print(f"✅ {tar}  {mb:.0f} MB  sha {sha256(tar)[:16]}")
    print(f"   slice.f32 {os.path.getsize(dst_f32)/1048576:.0f} MB")
    for line in man:
        print("   " + line.split()[1])


if __name__ == "__main__":
    main()
