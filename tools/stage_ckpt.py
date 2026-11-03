#!/usr/bin/env python3
"""把训练出来的 checkpoint 变成一个**可评测的 agent 目录**。

训练机上跑完出来的东西有两种形态：`experiments/native-<run>/<step>.bin`（原生 blob）
或 `native_weights.py import` 之后的 `.pt`。本地评测（`tools/eval_gate.py` →
`tools/bo3_eval.py`）要的是一个含 `main.py` / `model.pt` / `deck.csv` / `cg/` 的目录。

做法是**软链**：除 `model.pt` 与 `deck.csv` 外全部链到 `--bundle`（线上部署包），
所以每个 ckpt 只多占一份 20 MB 的权重，不像复制整包那样每份 90 MB。
不软链 `model.pt` 是根本目的；不软链 `deck.csv` 是为了能按需换卡组。

⚠️ 常量必须一致：`--bundle` 的 `ptcg/rl/buffers.py` 与训练仓库必须是同一套尺寸
（现在两边都是 1431+17 / 1755+17 / 470+27 = 1448/1772/497）。不一致时载入会形状不符。

用法：
    ./run.sh tools/stage_ckpt.py experiments/native-field/000000000012345.bin \\
        --out /tmp/staged/field@12k --deck decks/pool_field/teal_mask_ogerpon_ex_applin__7e39.csv
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


REPO = None   # 由 main() 按 --repo / 默认 ROOT/port_repo 设置


def to_pt(src, workdir):
    """`.bin`（原生 blob）→ `.pt`；已经是 `.pt` 就原样返回。"""
    if src.endswith(".pt"):
        return os.path.abspath(src)
    if not src.endswith(".bin"):
        raise SystemExit(f"看不懂的后缀：{src}（要 .pt 或 .bin）")
    repo = REPO or os.path.join(ROOT, "port_repo")
    out = os.path.join(workdir, "model.pt")
    cmd = [sys.executable, "native/tools/native_weights.py", "import",
           os.path.abspath(src), out]
    env = dict(os.environ, PYTHONPATH="data:.:native")
    print("转换 blob → pt:", " ".join(cmd[1:]))
    r = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"转换失败：\n{r.stdout[-800:]}\n{r.stderr[-1500:]}")
    print(" ", (r.stdout or "").strip()[-200:])
    return out


def resolve_deck(spec, bundle):
    """卡组来源：池文件名/路径 → 60 张；没给就用 bundle 自带那份。"""
    if not spec:
        p = os.path.join(bundle, "deck.csv")
        return [int(x) for x in open(p).read().split()]
    if os.path.exists(spec):
        return [int(x) for x in open(spec).read().split()]
    for cand in (os.path.join(ROOT, "decks", "pool_field", spec),
                 os.path.join(ROOT, "decks", "pool_field", spec + ".csv")):
        if os.path.exists(cand):
            return [int(x) for x in open(cand).read().split()]
    raise SystemExit(f"找不到卡组：{spec}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle", default="build/port_plain",
                    help="底座包（线上部署包）；除 model.pt/deck.csv 外全部软链过去")
    ap.add_argument("--deck", default=None, help="卡组：池文件名或 csv 路径")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--repo", default=None,
                    help="训练仓库根（含 native/tools/native_weights.py）。"
                         "默认 <本工具父目录>/port_repo；在评测包里跑时可以显式指到训练包")
    a = ap.parse_args()
    global REPO
    if a.repo:
        REPO = os.path.abspath(a.repo)

    bundle = os.path.join(ROOT, a.bundle)
    if not os.path.isdir(bundle):
        raise SystemExit(f"--bundle {bundle} 不存在")
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        if not a.force:
            raise SystemExit(f"{out} 已存在（加 --force 覆盖）")
        shutil.rmtree(out)
    os.makedirs(out)

    ids = resolve_deck(a.deck, bundle)
    if len(ids) != 60:
        raise SystemExit(f"卡组 {len(ids)} 张，不是 60")

    for e in os.listdir(bundle):
        if e in ("model.pt", "deck.csv"):
            continue
        os.symlink(os.path.abspath(os.path.join(bundle, e)), os.path.join(out, e))

    pt = to_pt(os.path.abspath(a.ckpt), out)
    if os.path.dirname(pt) != out:
        os.symlink(pt, os.path.join(out, "model.pt"))
    with open(os.path.join(out, "deck.csv"), "w") as f:
        f.write("\n".join(str(x) for x in ids) + "\n")

    print(f"✅ {out}")
    print(f"   model.pt -> {os.path.realpath(os.path.join(out, 'model.pt'))}")
    print(f"   deck.csv  {len(ids)} 张（前 8：{ids[:8]}）")
    print(f"   其余 {len(os.listdir(out)) - 2} 项软链到 {bundle}")
    print(f"\n评测：./run.sh tools/eval_gate.py --agent {out} --band ours --matches 400")


if __name__ == "__main__":
    main()
