#!/usr/bin/env python3
"""**权重回拉（KL 的零-CUDA 近似）**：把训练后的权重往初始权重拉回一步。

## 为什么不用真 KL

真 KL-to-init 要改 CUDA 损失（引擎里已有 frozen-bank 前向，理论上可行），但需要在 GPU 机上
编译+调试，迭代慢、风险高。这个脚本用**参数空间回拉**达到同样目的：

    W ← (1-α)·W_trained + α·W_init

α 越大，越钉在初始附近（≈ KL 约束越强）。每跑一小段 PPO 就回拉一次，等价于
"多步小步长 + 锚点"，能防止策略在 PPO 中漂成只在某个对位有效的怪招。

## 用法

    # .pt 之间回拉
    ./tools/py tools/kl_blend.py --new out/r1.pt --init bc_26c6.pt --alpha 0.15 --out out/r1_blend.pt

    # 直接用 .bin（内部走 pt）
    ./tools/py tools/kl_blend.py --new-bin out/r1.bin --init-bin init/bc_26c6.bin \
        --alpha 0.15 --out-bin out/r1_blend.bin

判据（配合评测）：α 太小 → 决策漂移（与初始一致率从 ~85% 掉到 60%）；
α 太大 → 不涨。目标区间：与初始一致率 80~90% 且胜率小幅上行。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_nw():
    """native_weights.py 的位置：仓库里在 port_repo/native/tools/，但训练机上仓库本身就是
    port_repo ⇒ 两种布局都要认，再给 PTCG_NW 兜底。"""
    cands = [os.environ.get("PTCG_NW", ""),
             os.path.join(ROOT, "port_repo", "native", "tools", "native_weights.py"),
             os.path.join(ROOT, "native", "tools", "native_weights.py")]
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise SystemExit("✗ 找不到 native_weights.py（用 PTCG_NW=<路径> 指定）")


NW = _find_nw()
PY = os.path.join(ROOT, "tools", "py")


def _import(binp, ptp):
    bundle = os.environ.get("PTCG_BUNDLE", os.path.join(ROOT, "build", "port_plain"))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [bundle, ROOT, os.environ.get("PYTHONPATH", "")]).strip(os.pathsep))
    r = subprocess.run([sys.executable, NW, "import", binp, ptp], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(ptp):
        raise SystemExit(f"✗ bin→pt 失败 {binp}\n{r.stderr[-400:]}")


def _export(ptp, binp):
    r = subprocess.run([sys.executable, NW, "export", ptp, binp],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(binp):
        raise SystemExit(f"✗ pt→bin 失败 {ptp}\n{r.stderr[-400:]}")


def blend_pt(new_pt, init_pt, alpha, out_pt):
    """W = (1-α)·new + α·init（逐张量；只混可训练参数，缓冲/统计量沿用 new）。"""
    import torch
    a = torch.load(new_pt, map_location="cpu", weights_only=False)
    b = torch.load(init_pt, map_location="cpu", weights_only=False)
    a = a.get("state_dict", a) if isinstance(a, dict) else a
    b = b.get("state_dict", b) if isinstance(b, dict) else b
    out, n_blend, n_keep = {}, 0, 0
    for k, v in a.items():
        w = b.get(k)
        if torch.is_tensor(v) and torch.is_tensor(w) and v.shape == w.shape \
                and v.is_floating_point():
            out[k] = (1.0 - alpha) * v.float() + alpha * w.float()
            n_blend += 1
        else:
            out[k] = v
            n_keep += 1
    torch.save(out, out_pt)
    print(f"回拉 α={alpha}: 混合 {n_blend} 个张量，沿用 {n_keep} 个（缓冲/标量）→ {out_pt}")
    return out_pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="", help="训练后的 .pt")
    ap.add_argument("--init", default="", help="初始（锚点）.pt")
    ap.add_argument("--new-bin", default="", help="训练后的 .bin（会自动转 pt）")
    ap.add_argument("--init-bin", default="", help="初始 .bin（会自动转 pt）")
    ap.add_argument("--alpha", type=float, required=True, help="回拉强度：0=不回拉，1=完全回到初始")
    ap.add_argument("--out", default="", help="输出 .pt")
    ap.add_argument("--out-bin", default="", help="输出 .bin")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="klblend_")
    new_pt, init_pt = a.new, a.init
    if a.new_bin:
        new_pt = os.path.join(tmp, "new.pt"); _import(a.new_bin, new_pt)
    if a.init_bin:
        init_pt = os.path.join(tmp, "init.pt"); _import(a.init_bin, init_pt)
    if not (new_pt and init_pt) or not os.path.exists(new_pt) or not os.path.exists(init_pt):
        raise SystemExit("✗ 需要 --new/--new-bin 与 --init/--init-bin 各给一个，且文件存在")
    out_pt = a.out or os.path.join(tmp, "blend.pt")
    blend_pt(new_pt, init_pt, a.alpha, out_pt)
    if a.out_bin:
        _export(out_pt, a.out_bin)
        print(f"→ {a.out_bin} ({os.path.getsize(a.out_bin)} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
