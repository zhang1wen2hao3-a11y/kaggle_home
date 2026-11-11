#!/usr/bin/env python3
"""**A/B 阶段动态切换触发器**：跟读训练日志，算覆盖率加权的对位胜率，达标就杀训练。

为什么需要它：参考工程的 two-net 是「A 的 WR 涨 2pp 就冻结换 B」，不是固定步数。
固定步数会浪费（A 早到顶还在跑、B 早该切却没切）。

## 数据来源（训练器每 epoch 都打印，免费）
    epoch 50 step 1638400 ... lgfrac 0.486 lgw 0.612 [1:0.58/0.05][2:0.71/0.05]...
    ↑ 每个 bank 的胜率/出现率  ← bank 钉死了卡组，所以这是"对某副牌的胜率"

## 触发规则
    baseline = 前 WARM 个 epoch 的加权胜率均值（默认 20~50）
    ema      = 最近 EMA_N 个 epoch 的加权胜率
    当 ema - baseline >= THRESH_PP（默认 2.0pp，A 阶段）/ ema >= baseline - MARGIN（B 阶段）⇒ 触发

## 用法
    # A 阶段：涨 2pp 就触发（触发后杀掉 run-name 含 r1A 的训练进程）
    ./tools/py tools/ab_watch.py --log logs/r1A.log --mode A --weights 0.35,0.12,... \
        --thresh-pp 2.0 --kill-pattern "run-name r1A"

    # B 阶段：B 的胜率回到 baseline 附近（"追回失分"）就触发
    ./tools/py tools/ab_watch.py --log logs/r1B.log --mode B --baseline-ref 0.50 --thresh-pp 1.0
"""
import argparse, os, re, subprocess, sys, time

EPOCH_RE = re.compile(r"^epoch (\d+) step (\d+).*?lgw (-?[0-9.]+)")
BANK_RE = re.compile(r"\[(\d+):(-?[0-9.]+)/([0-9.]+)\]")


def parse_epoch(line):
    m = EPOCH_RE.search(line)
    if not m:
        return None
    ep, step = int(m.group(1)), int(m.group(2))
    banks = [(int(b), float(w), float(n)) for b, w, n in BANK_RE.findall(line)]
    lgw_m = re.search(r"lgw (-?[0-9.]+)", line)
    lgw = float(lgw_m.group(1)) if lgw_m else None
    return ep, step, banks, lgw


def weighted(banks, weights):
    """覆盖率加权的对位胜率（只算出现过的 bank）。"""
    num = den = 0.0
    for b, w, n in banks:
        if n <= 0 or w < -0.5:            # -1 = 该 bank 没样本
            continue
        cw = weights[b] if b < len(weights) else 1.0
        num += cw * w
        den += cw
    return (num / den) if den > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--mode", choices=["A", "B"], default="A")
    ap.add_argument("--weights", default="", help="逗号分隔，每个 bank 对应卡组的覆盖率")
    ap.add_argument("--thresh-pp", type=float, default=2.0)
    ap.add_argument("--warm", type=int, default=30, help="前 N 个 epoch 作 baseline")
    ap.add_argument("--ema", type=int, default=20, help="最近 N 个 epoch 做 EMA")
    ap.add_argument("--baseline-ref", type=float, default=None, help="B 模式：baseline 直接用这个值")
    ap.add_argument("--kill-pattern", default="", help="触发后 pkill -f 这个模式")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--max-wait", type=float, default=0.0, help="秒；0=不限")
    ap.add_argument("--pid", type=int, default=0, help="训练进程 pid；它退出就结束（防止无限等待）")
    a = ap.parse_args()

    weights = [float(x) for x in a.weights.split(",") if x.strip()] or [1.0]
    seen = []          # [(ep, step, wr)]
    t0 = time.time()
    def alive(pid):
        if pid <= 0: return True
        try:
            os.kill(pid, 0); return True
        except OSError:
            return False
    while True:
        if not alive(a.pid):
            print(f"[watch] 训练进程 {a.pid} 已退出且未触发阈值（看了 {len(seen)} 个 epoch）", flush=True)
            return 3
        if a.max_wait > 0 and time.time() - t0 > a.max_wait:
            print(f"[watch] 超时未触发，放弃（已看 {len(seen)} 个 epoch）"); return 2
        if not os.path.exists(a.log):
            time.sleep(a.poll); continue
        with open(a.log, errors="ignore") as f:
            for line in f:
                if not line.startswith("epoch "):
                    continue
                r = parse_epoch(line)
                if not r:
                    continue
                ep, step, banks, lgw = r
                if seen and seen[-1][0] >= ep:
                    continue
                wr = weighted(banks, weights) if a.mode == "A" else lgw
                if wr is None:
                    continue
                seen.append((ep, step, wr))
                # 判定
                if a.mode == "B" and a.baseline_ref is not None:
                    base = a.baseline_ref
                else:
                    if len(seen) < max(a.warm, a.ema + 1):
                        print(f"[watch] ep{ep} wr={wr:.3f} (攒 baseline: {len(seen)}/{a.warm})", flush=True)
                        continue
                    base = sum(w for _, _, w in seen[:a.warm]) / a.warm
                tail = [w for _, _, w in seen[-a.ema:]]
                ema = sum(tail) / len(tail)
                d = (ema - base) * 100
                if a.mode == "A":
                    ok = d >= a.thresh_pp
                else:
                    ok = d >= -abs(a.thresh_pp) if a.baseline_ref is not None else d >= a.thresh_pp
                print(f"[watch] ep{ep} step{step} wr={wr:.3f} ema={ema:.3f} base={base:.3f} Δ={d:+.1f}pp"
                      + ("  ★触发" if ok else ""), flush=True)
                if ok:
                    if a.kill_pattern:
                        subprocess.run(["pkill", "-f", a.kill_pattern], check=False)
                        print(f"[watch] 已 pkill -f '{a.kill_pattern}'", flush=True)
                    print(f"[watch] TRIGGER ep={ep} step={step} ema={ema:.4f} base={base:.4f}", flush=True)
                    return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
