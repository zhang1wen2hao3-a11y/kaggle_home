#!/usr/bin/env python3
"""**A/B 阶段动态切换触发器**：跟读训练日志，算覆盖率加权的对位胜率，达标就杀训练。

为什么需要它：参考工程的 two-net 是「A 的 WR 涨 2pp 就冻结换 B」，不是固定步数。
固定步数会浪费（A 早到顶还在跑、B 早该切却没切）。

## 数据来源（训练器每 epoch 都打印，免费）
    epoch 50 step 1638400 ... lgfrac 0.486 lgw 0.612 [1:0.58/0.05][2:0.71/0.05]...
    ↑ 每个 bank 的胜率/出现率  ← bank 钉死了卡组，所以这是"对某副牌的胜率"

## 指标口径（★ 2025-11 修正：A 阶段必须换算到学习者视角）
    A 阶段：日志里的 [b:wr] 是**冻结对手**（bank 驾驶员）的胜率，对手胜率**下降** = 我们变强。
            所以本脚本用 learner_wr = 1 − Σ(coverage·bank_wr) 做指标（单 bank 时就等于 lgw）。
            旧版直接盯对手胜率、要求"上涨 2pp"，方向反了 ⇒ 10M 步一次都没触发（血泪）。
    B 阶段：lgw 就是学习者自己的胜率，直接用。

## 换边规则（★ 修正：参考工程里"换边"和"晋升"是两件事）
    参考工程 `--cross-gate-delta 0.03` 是**晋升门**（candidate vs teacher 镜像 h2h，n=400，CI ±2.5pp），
    而换边靠 `--ft-phase-timebox`（固定预算到点就换）。我们把两者混成一个阈值后，A 阶段对冻结端口的
    真实增益只有 +1pp/2M 步 ⇒ 3pp 永远到不了，只能跑满上限（实测 12M cap，白烧 1.5h）。

    ⇒ 现在规则 =「timebox 优先 + 增益兜底」：
      · 本阶段步数 >= MAX_STEPS（默认 4M，`AB_MAX_STEPS` 可改，0=关）⇒ 立刻换边
      · 或 ema − baseline >= THRESH_PP ⇒ 提前换边（谁先到算谁）

## 触发规则
    baseline = 前 WARM 个 epoch 的 learner_wr 均值（默认 20~50）
    ema      = 最近 EMA_N 个 epoch 的 learner_wr
    当 ema - baseline >= THRESH_PP（默认 3.0pp，对齐参考工程 cross-gate-delta 0.03）⇒ 杀掉本阶段

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


def kill_matching(pattern, a_kill_requires="train_native.py"):
    """按 cmdline 模式杀进程，但**排除自己**。

    为什么不能用 `pkill -f`：watcher 自己的命令行里就带着 `--kill-pattern <pattern>`，
    pkill -f 会匹配到自己 ⇒ 触发时 watcher 先把自己杀掉，TRIGGER 行永远打不出来
    （实测四个阶段的日志里都缺这一行；换成括号写法只是绕过，脚本里没法这么写）。
    """
    me = os.getpid()
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    killed = []
    for pid_s in out.stdout.split():
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid == me:
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "ab_watch.py" in cmd:          # 另一个 watcher（比如热换下来的旧实例），别误杀
            continue
        # 只杀真正的训练进程：模式串经常也会出现在启动它的 shell（`bash -c "..."`）命令行里，
        # 误杀 shell 会把整条链路带走（实测把自己杀掉两次）。
        if a_kill_requires and a_kill_requires not in cmd:
            continue
        try:
            os.kill(pid, 15)
            killed.append(pid)
        except OSError:
            pass
    return killed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--mode", choices=["A", "B"], default="A")
    ap.add_argument("--weights", default="", help="逗号分隔，每个 bank 对应卡组的覆盖率")
    ap.add_argument("--thresh-pp", type=float, default=2.0)
    ap.add_argument("--warm", type=int, default=30, help="前 N 个 epoch 作 baseline")
    ap.add_argument("--skip", type=int, default=5,
                    help="跳过最前 N 个 epoch（样本极少，lgw 抖动到 0.9，会把基线抬高）")
    ap.add_argument("--ema", type=int, default=20, help="最近 N 个 epoch 做 EMA")
    ap.add_argument("--baseline-ref", type=float, default=None, help="B 模式：baseline 直接用这个值")
    ap.add_argument("--kill-pattern", default="", help="触发后杀掉 cmdline 匹配这个模式的进程")
    ap.add_argument("--kill-requires", default="train_native.py",
                    help="额外护栏：只杀 cmdline 里同时含这个串的进程（防误杀启动它的 shell；默认 train_native.py）")
    ap.add_argument("--max-steps", type=int,
                    default=int(os.environ.get("AB_MAX_STEPS", "4000000") or 0),
                    help="timebox：本阶段跑满这么多步就换边（0=关闭，默认 4M）")
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
                if a.mode == "A":
                    wb = weighted(banks, weights)
                    wr = None if wb is None else 1.0 - wb    # 学习者视角：对手胜率↓ = 我们↑
                else:
                    wr = lgw
                if wr is None:
                    continue
                seen.append((ep, step, wr))
                # 判定
                if a.mode == "B" and a.baseline_ref is not None:
                    base = a.baseline_ref
                else:
                    # ★ baseline 从 --skip 之后开始算（前几个 epoch 样本极少，lgw 会抖到 0.9）
                    valid = seen[a.skip:]
                    if len(valid) < max(a.warm, a.ema + 1):
                        print(f"[watch] ep{ep} wr={wr:.3f} (攒 baseline: {len(valid)}/{a.warm}, 跳过前{a.skip})", flush=True)
                        continue
                    base = sum(w for _, _, w in valid[:a.warm]) / a.warm
                tail = [w for _, _, w in seen[-a.ema:]]
                ema = sum(tail) / len(tail)
                d = (ema - base) * 100
                if a.mode == "A":
                    ok = d >= a.thresh_pp
                else:
                    ok = d >= -abs(a.thresh_pp) if a.baseline_ref is not None else d >= a.thresh_pp
                tb = a.max_steps > 0 and step >= a.max_steps     # timebox 到点
                if tb:
                    tag = f"  ★触发(timebox {step}>={a.max_steps})"
                elif ok:
                    tag = "  ★触发(增益)"
                else:
                    tag = ""
                print(f"[watch] ep{ep} step{step} wr={wr:.3f} ema={ema:.3f} base={base:.3f} Δ={d:+.1f}pp{tag}",
                      flush=True)
                ok = ok or tb
                if ok:
                    why = f"timebox(step={step}>={a.max_steps})" if tb else f"gain(Δ={d:+.1f}pp)"
                    print(f"[watch] TRIGGER ep={ep} step={step} ema={ema:.4f} base={base:.4f} reason={why}", flush=True)
                    if a.kill_pattern:
                        killed = kill_matching(a.kill_pattern, a.kill_requires)
                        print(f"[watch] 已杀 {len(killed)} 个进程 {killed}（模式 '{a.kill_pattern}'）", flush=True)
                    return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
