#!/usr/bin/env python3
"""**权重位移度量**：给定一个模型，量它与端口权重的距离（全局 + 逐张量分布）。

判读（2026-11-06 实测标定，中位张量相对 L2，剔除 5 张静态表）：
  - 端口自身 = 0.000%
  - 400k 步 RL finetune ≈ 0.32%（v2 0.316 / 9d6e 0.341）
  - g6_db04（天梯 1060，多轮）≈ 0.50%
  - BC 全重训 ≈ 1.9~2.0%
  - 分水岭判据：30M 步的模型若中位位移 > 1%（接近 BC 量级）→ "只是没练够"；
    若仍 ≈0.3% → RL 信号本身没动（对手太弱 / 奖励太稀）。

用法：
    ./tools/py tools/weight_dist.py --model models/weights/pt/xxx.pt
    ./tools/py tools/weight_dist.py --bin models/weights/ph3/xxx.bin   # 自动转 pt
"""
from __future__ import annotations
import argparse, hashlib, os, statistics as st, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PORT_REF = os.path.join(ROOT, "build", "agents", "opp_3eaa", "model.pt")  # 干净端口权重
EXCLUDE = {"card_static", "skill_text", "att_static", "card_attacks", "card_skills"}


def _load(p):
    import torch
    o = torch.load(p, map_location="cpu", weights_only=False)
    return o.get("state_dict", o) if isinstance(o, dict) else o


def _import_bin(binp):
    tmp = tempfile.mkdtemp(prefix="wd_")
    pt = os.path.join(tmp, "m.pt")
    env = dict(os.environ, PYTHONPATH="build/port_plain:.:" + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "port_repo/native/tools/native_weights.py",
                        "import", binp, pt], env=env, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(pt):
        raise SystemExit(f"✗ bin→pt 失败\n{r.stderr[-300:]}")
    return pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--bin", default="")
    ap.add_argument("--ref", default=PORT_REF)
    a = ap.parse_args()
    mp = a.model
    if a.bin:
        mp = _import_bin(a.bin)
    if not mp or not os.path.exists(mp):
        raise SystemExit("✗ 需要 --model 或 --bin 指向存在的文件")
    import torch
    A, B = _load(mp), _load(a.ref)
    num = den = 0.0
    rels = []
    for k in A:
        if not torch.is_tensor(A[k]) or k not in B or A[k].shape != B[k].shape:
            continue
        if k in EXCLUDE:
            continue
        d = A[k].float() - B[k].float()
        num += float((d * d).sum()); den += float((A[k].float() ** 2).sum())
        rels.append(100 * float((d * d).sum() ** 0.5) /
                    max(float((B[k].float() ** 2).sum() ** 0.5), 1e-12))
    print(f"模型 {os.path.basename(mp)}  vs  端口权重 {os.path.basename(a.ref)}")
    print(f"  全局相对 L2  = {100*(num/den)**0.5:.4f}%")
    print(f"  中位张量相对 L2 = {st.median(rels):.3f}%   （最大 {max(rels):.2f}%）")
    print(f"  参数量 = {sum(B[k].numel() for k in B if torch.is_tensor(B[k])):,}")
    print(f"  标定（中位，剔除 5 张静态表）：400k RL≈0.32% / g6_db04≈0.50% / BC 全重训≈1.9~2.0%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
