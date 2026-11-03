#!/usr/bin/env python3
"""CPU 上的**卡组对位矩阵**：行 = 候选卡组，列 = 原型代表，每格 N 场真 Bo3。

## 为什么需要它（与 `eval_gate.py` 的分工）

`eval_gate.py` 给的是**单一加权总分**——它为了精度把 53.7% 的场次都给了镜像，
代价是长尾对位几乎没数据（我们实测里 kangaskhan 那格只有 n=16）。

1st place 的方案明确按**"均值 + 最差对位"**选卡组：
"If a deck has only 1 matchup below 45%, it's a candidate"、"worst cross-archetype
matchup exceeded 50%"。加权总分会把"某格崩掉"和"整体平庸"混为一谈，
所以必须有一个**按原型列铺开**的矩阵。

## 口径（与闸门一致，因此可比）

- 两侧都由**我们的端口**开牌（这是我们会提交的飞行员）；
- 列按**原型**取代表（不是按场份额，避免镜像淹没长尾）；
- 每格 N 场真 Bo3（`tools/bo3_eval.py`，进程隔离、每场重置 deck-request、每局满池）；
- 逐格校验 `errors==0`，异常格单独标出。

## 用法

    ./run.sh tools/deck_matrix_cpu.py \
        --rows dunsparce_abra__9d6e,grookey_thwackey__2617 \
        --cols teal_mask_ogerpon_ex_applin__7e39,mega_kangaskhan_ex_dwebble__7e79,... \
        --matches 40 --workers 1 --out logs/matchup_cpu.json
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

BUNDLE = "build/port_plain"      # 默认策略：端口。可用 --agent 换成训练出的 ckpt 目录
STAGE = "/tmp/dcm"
# --agent 指定的策略来源（None = 用 BUNDLE）。见 stage()。
AGENT_DIR = None


def pool_files(pool="decks/pool_field"):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, pool, "*.csv"))):
        out[os.path.basename(f)[:-4]] = f
    return out


def stage(deck_name, files):
    """端口代码 + 指定 deck.csv 的软链目录（幂等）。"""
    src = files.get(deck_name)
    if src is None:
        cand = [k for k in files if k.startswith(deck_name)]
        if len(cand) != 1:
            raise SystemExit(f"卡组 {deck_name!r} 找不到或有多解：{cand[:5]}")
        deck_name = cand[0]
        src = files[deck_name]
    ids = [int(x) for x in open(src).read().split()]
    if len(ids) != 60:
        raise SystemExit(f"{deck_name} 不是 60 张")
    out = os.path.join(STAGE, deck_name)
    if not os.path.exists(os.path.join(out, "model.pt")):
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out, exist_ok=True)
        # --agent 给了就用它的权重与代码；否则用端口 bundle
        b = AGENT_DIR or os.path.join(ROOT, BUNDLE)
        for e in os.listdir(b):
            if e in ("model.pt", "deck.csv"):
                continue
            os.symlink(os.path.abspath(os.path.join(b, e)), os.path.join(out, e))
        os.symlink(os.path.abspath(os.path.join(b, "model.pt")),
                   os.path.join(out, "model.pt"))
        with open(os.path.join(out, "deck.csv"), "w") as f:
            f.write("\n".join(str(x) for x in ids) + "\n")
    return out, deck_name


def wilson(w, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 1.0
    p = w / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / den, (c + r) / den


def score_of(r):
    if r["winner"] is None:
        return 0.0
    return 0.5 if r["winner"] == 2 else float(r["winner"] == r["seat"])


def run_cell(a_dir, b_dir, matches, workers, logdir):
    cmd = [sys.executable, os.path.join(ROOT, "tools", "bo3_eval.py"),
           "--a", a_dir, "--b", b_dir, "--matches", str(matches),
           "--workers", str(workers), "--logdir", logdir]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    rec_p = os.path.join(logdir, "last_run.jsonl")
    if r.returncode != 0 or not os.path.exists(rec_p):
        return None, (r.stderr or "")[-300:]
    recs = [json.loads(x) for x in open(rec_p) if x.strip()]
    if not recs:
        return None, (r.stdout or "")[-300:]
    n = len(recs)
    pts = sum(score_of(x) for x in recs)
    errs = sum(1 for x in recs if x.get("error") or any(x.get("fallback_errors") or []))
    lo, hi = wilson(int(round(pts)), n)
    return dict(n=n, score=pts / n, ci_low=lo, ci_high=hi, errors=errs), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="逗号分隔的卡组名（池内文件名或唯一前缀）")
    ap.add_argument("--cols", required=True, help="逗号分隔的列（原型代表）")
    ap.add_argument("--matches", type=int, default=40, help="每格 Bo3 场数")
    ap.add_argument("--workers", type=int, default=1, help="并行面板（内存紧张就用 1）")
    ap.add_argument("--skip-cols", default="", help="已有结果的列，逗号分隔，跳过")
    ap.add_argument("--out", default="logs/matchup_cpu.json")
    ap.add_argument("--logdir", default="logs/matchup_cpu")
    ap.add_argument("--agent", default=None,
                    help="策略来源目录（含 model.pt/main.py/ptcg 的 agent 目录）。"
                         "默认 build/port_plain（端口）。传训练出的 ckpt 目录即可"
                         "——**两方都用同一策略**，只变卡组，这是验微调的唯一正确口径")
    a = ap.parse_args()

    global AGENT_DIR, STAGE
    if a.agent:
        AGENT_DIR = os.path.abspath(a.agent)
        if not os.path.exists(os.path.join(AGENT_DIR, "model.pt")):
            raise SystemExit(f"--agent {AGENT_DIR} 下没有 model.pt")
        # 换策略必须换 staging 目录，否则会复用端口那份缓存的 model.pt
        STAGE = "/tmp/dcm_agent_" + os.path.basename(AGENT_DIR.rstrip("/"))
        print(f"策略来源: {AGENT_DIR}（非端口）  staging={STAGE}")

    files = pool_files()
    rows = [x.strip() for x in a.rows.split(",") if x.strip()]
    cols = [x.strip() for x in a.cols.split(",") if x.strip()]
    skip = {x.strip() for x in a.skip_cols.split(",") if x.strip()}

    out_path = os.path.join(ROOT, a.out)
    result = {}
    if os.path.exists(out_path):
        try:
            result = json.load(open(out_path))
        except Exception:
            result = {}

    print(f"行 {len(rows)} × 列 {len(cols)}，每格 {a.matches} 场，workers={a.workers}")
    est = len(rows) * len(cols) * a.matches * 3.2 / 60 / max(a.workers, 1)
    print(f"预计约 {est:.0f} 分钟\n")

    for rn in rows:
        a_dir, a_name = stage(rn, files)
        result.setdefault(a_name, {})
        for cn in cols:
            b_dir, b_name = stage(cn, files)
            if b_name in skip or b_name in result[a_name]:
                print(f"  跳过 {a_name} vs {b_name}（已有）")
                continue
            logdir = os.path.join(ROOT, a.logdir, f"{a_name}__vs__{b_name}")
            cell, err = run_cell(a_dir, b_dir, a.matches, a.workers, logdir)
            if cell is None:
                print(f"  ✗ {a_name} vs {b_name}: {err.strip()[:120]}")
                continue
            cell["name"] = b_name
            result[a_name][b_name] = cell
            flag = " ⚠️异常" if cell["errors"] else ""
            print(f"  {a_name[:26]:<27} vs {b_name[:26]:<27} "
                  f"{cell['score']:.3f} ±{(cell['ci_high']-cell['ci_low'])/2:.3f} "
                  f"n={cell['n']}{flag}")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=1)

    # ---- 汇总：按 1st place 的判据 ----
    print("\n" + "=" * 78)
    print("判据（与 1st place 一致）：均值越高越好；**最差跨原型对位 ≥45%**；"
          "低于 45% 的格数 ≤1")
    print(f"{'卡组':<30}{'均值':>8}{'最差':>8}{'<45%格':>8}  最差对位")
    for rn in rows:
        if rn not in result:
            continue
        cells = [c for c in result[rn].values() if c["n"] > 0]
        if not cells:
            continue
        mean = sum(c["score"] for c in cells) / len(cells)
        worst = min(cells, key=lambda c: c["score"])
        n_bad = sum(1 for c in cells if c["score"] < 0.45)
        verdict = "✅ 合格" if (n_bad <= 1 and worst["score"] >= 0.45) else \
                  ("△ 边缘" if n_bad <= 1 else "✗ 不合格")
        print(f"{rn[:28]:<30}{mean:8.3f}{worst['score']:8.3f}{n_bad:8d}  "
              f"{worst['name'][:26]:<27}{verdict}")
    print(f"\n结果已写 {a.out}")


if __name__ == "__main__":
    main()
