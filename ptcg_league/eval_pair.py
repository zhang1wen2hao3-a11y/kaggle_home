#!/usr/bin/env python3
"""**单项任务：配对评测**。候选 vs 一组固定对手，固定种子、固定卡组、座位交替。

调度器只调这一个脚本做评测，不自己拼引擎。
底层完全复用 `tools/bo3_eval.py`（真 Bo3、真进程隔离、600 s 池、兜底计数），
所以"联赛里量出来的分"和"上线前量出来的分"是**同一把尺**。

## 协议（可复现的三件套）

1. **种子固定**：第 i 场的 deck-request 用 `seed_base + i`；
   引擎动作是确定性的、agent 是 temp=0 argmax ⇒ 同一 `(cand, opp, seed_base, n)`
   重复跑结果完全一致。
2. **卡组固定**：对手目录里的 `deck.csv` 就是它的卡组（`stage_ckpt.py` 写好的）。
3. **座位交替**：`seat = mid % 2`，且每场内第 2 局 `reverse_player=True`。

## 用法

    # 筛查：候选对当前 best，400 场 Bo3
    ./tools/py ptcg_league/eval_pair.py --cand pools/evo/b1/cand --opp best=pools/league/best \
        --matches 400 --workers 6 --seed-base 90000 --out /tmp/screen.json

    # 固定评测集：候选对冻结 roster（多对手，场次平均分）
    ./tools/py ptcg_league/eval_pair.py --cand X --opp bc0=D0 --opp bc1=D1 \
        --matches 500 --workers 6 --seed-base 70000 --out /tmp/anchor.json

输出 json：
    {"cand":.., "protocol":{...}, "per_opp":{name:{score,ci_low,ci_high,n,errors,elapsed_s}},
     "overall":{...}, "elapsed_s":..}
"""
import argparse
import json
import os
import signal
import sys
import time
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from bo3_eval import AgentProcess, play_match, resolve, summarize    # noqa: E402


def shard(job):
    """一个面板：(mid, opp_dir) 任务列表串行跑，**同时最多 2 个 agent 进程**。

    ⚠️ 这里必须"换对手就关旧进程"：早先按对手名缓存 pane，锚点评测有 8 个对手时
       一个面板会攒下 16 个 agent 进程（每个 300–400 MB），在 7 GB 机器上直接
       吃满 swap —— 表现是**单步卡 120 s 超时**，而不是任何显式报错。
    """
    k, tasks, dir_a, logdir = job
    panes = {}
    cur = None
    recs = []
    try:
        for mid, name, dir_b in tasks:
            if name != cur:                     # 换对手：关掉上一对，内存恒定
                for pair in panes.values():
                    for ag in pair:
                        ag.close()
                panes.clear()
                cur = name
            if name not in panes:
                panes[name] = (AgentProcess(f"{k}A{name}", dir_a, logdir),
                               AgentProcess(f"{k}B{name}", dir_b, logdir))
            pa, pb = panes[name]
            seat = mid % 2
            left, right = (pa, pb) if seat == 0 else (pb, pa)
            r = play_match(left, right, mid)
            r.update(seat=seat, opp=name)
            recs.append(r)
        return recs
    finally:
        for pair in panes.values():
            for ag in pair:
                ag.close()


def cgroup_limit_gb(paths=(
        "/sys/fs/cgroup/memory.max",                    # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
        "/sys/fs/cgroup/memory.high",                   # v2 软限（兜底）
)):
    """容器内存限额（GB）；读不到/无限额时返回 None。

    容器里 `/proc/meminfo` 报的是**宿主机**内存（AutoDL 就是这样），而真正会 OOM 的是
    cgroup 限额。只读 /proc/meminfo 会在 32 核容器里算出"能开 32 个面板"，
    然后被内核 OOM kill —— 表现是**面板莫名消失、回收场次不足、异常数飙高**，
    而不是任何一条像样的报错（2026-11-01 实测）。
    """
    for p in paths:
        try:
            with open(p) as f:
                v = f.read().strip()
        except OSError:
            continue
        if v in ("max", ""):
            continue
        try:
            gb = int(v) / 1073741824.0
        except ValueError:
            continue
        if 0.1 < gb < 1e6:          # 忽略 1<<63-1 这类"无限额"哨兵值
            return gb
    return None


def shard_to_file(job):
    # SIGTERM 默认会直接杀进程、**不跑 finally**，于是它 spawn 的 agent 子进程会被过继成
    # 孤儿继续占内存（实测：杀掉主进程后残留 17 个 ~600 MB 的 worker，把 7 GB 机器吃到 98%）。
    # 转成 SystemExit 才能让 shard() 的 finally 逐个 close() agent。
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))
    recs = shard(job)
    with open(os.path.join(job[3], f"shard-{job[0]}.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, help="候选 agent 目录（含 main.py/model.pt/deck.csv）")
    ap.add_argument("--opp", action="append", required=True, metavar="NAME=DIR",
                    help="对手，可重复。NAME 进结果 json 的 per_opp 键")
    ap.add_argument("--matches", type=int, default=400, help="总 Bo3 场数（多对手时平均分，向上补偶数）")
    ap.add_argument("--workers", type=int, default=0,
                    help="面板数；0=auto（取 min(内存预算/0.7GB, 核数)）")
    ap.add_argument("--seed-base", type=int, default=90000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logdir", default=None)
    a = ap.parse_args()

    opps = []
    for spec in a.opp:
        if "=" not in spec:
            raise SystemExit(f"--opp 要 NAME=DIR：{spec}")
        name, d = spec.split("=", 1)
        opps.append((name, resolve(d)))
    dir_a = resolve(a.cand)

    n = a.matches + (a.matches % 2)
    per = n // len(opps)
    if per * len(opps) != n:                     # 不能整除就轮流多给一场
        per = [n // len(opps)] * len(opps)
        for i in range(n - sum(per)):
            per[i] += 1
    else:
        per = [per] * len(opps)
    per = [p + (p % 2) for p in per]             # 每个对手内部也要偶数，座位才平衡

    # 与 bc_train 共用 `eval` 锁：评测在跑时别的重任务不该插进来抢内存
    if os.path.exists(os.path.join(ROOT, "tools", "onejob.sh")) and not os.environ.get("PTCG_NO_LOCK"):
        pass  # 锁由调用方（run_league / onejob.sh）持，这里不重复加，避免自锁

    logdir = os.path.abspath(a.logdir or os.path.join(ROOT, "ptcg_league", "logs", "eval"))
    os.makedirs(logdir, exist_ok=True)
    for f in os.listdir(logdir):
        if f.startswith("shard-"):
            os.remove(os.path.join(logdir, f))

    # 任务表：每个 (对手, 场次) 一个 mid，mid 全局唯一 ⇒ 种子唯一
    tasks = []
    mid = a.seed_base
    for (name, d), p in zip(opps, per):
        for _ in range(p):
            tasks.append((mid, name, d))
            mid += 2                              # ★ 步长 2：play_match 用 mid*2+i 做种子
    # 任务按对手**连续**排列：shard() 靠"对手变了就关进程"来把内存钉住
    # 面板数按**整体内存天花板**算（默认 90%），不是"能塞多少塞多少"。
    # 实测（2026-10-30）：8 面板把已用从 34% 顶到 94% ⇒ 8 面板 ≈ 4.5 GB ⇒ **0.7 GB/面板**
    # （一个面板 2 个 agent 进程，各 300–400 MB）。早先用 0.45 估得太乐观，会越过天花板。
    # 越过天花板会 swap 抖动，表现是单步 120 s 超时（不是报错），比慢更糟。
    ceil_pct = float(os.environ.get("CEIL_PCT", "90"))
    # ⚠️ 容器里 `/proc/meminfo` 报的是**宿主机**内存（AutoDL 就是这样），
    #    而真正会 OOM 的是 cgroup 限额。只读 /proc/meminfo 会在 32 核容器里
    #    算出"能开 32 个面板"，然后被内核 OOM kill —— 表现是**面板莫名消失、
    #    回收场次不足、异常数飙高**，而不是任何一条像样的报错（2026-11-01 实测）。
    try:
        with open("/proc/meminfo") as f:
            mi = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}
        host_total = mi["MemTotal"] / 1048576
        host_avail = mi["MemAvailable"] / 1048576
    except Exception:
        host_total = host_avail = 8.0
    cg = cgroup_limit_gb()
    if cg is not None and cg < host_total:
        # 容器：总预算按 cgroup 限额算，已用按"宿主机已用比例"折算到限额上
        total_gb = cg
        used_gb = max(0.0, (1.0 - host_avail / max(host_total, 1e-9))) * cg
        src = f"cgroup 限额 {cg:.1f} GB（宿主机 {host_total:.0f} GB）"
    else:
        total_gb, used_gb = host_total, max(0.0, host_total - host_avail)
        src = f"宿主机 {host_total:.1f} GB"
    per_panel = float(os.environ.get("PTCG_MEM_GB_PER_PANEL", "0.7"))
    # 每个面板实际要吃掉几个 vCPU：1 个引擎进程 + 2 个 agent 进程，而 agent 的
    # main.py 里 `torch.set_num_threads(2)` 是写死的 ⇒ 面板多了就是**线程超卖**，
    # 表现是单步 120 s 超时 / agent 侧 fallback Traceback（2026-11-01 实测：
    # 32 vCPU 机器上 auto 选了 32 面板 ≈ 160 线程，异常率到 ~24%）。
    cores_per_panel = float(os.environ.get("PTCG_PANEL_CORES", "3"))
    budget = total_gb * ceil_pct / 100.0 - used_gb - 0.3      # 0.3 GB 余量
    safe = max(1, int(budget / per_panel))
    ncore = max(1, int((os.cpu_count() or 4) / cores_per_panel))
    safe = min(safe, ncore)                                   # 也受核数约束
    # 面板数自适应：`--workers 0`（或负数）= auto，直接吃满"内存预算 ∩ 核数"的较小者；
    # 给了正数就尊重它，只在会破内存天花板时下调。
    # ⚠️ 旧代码的 else 分支也在写 `a.workers = safe`，等于**把用户显式给的数悄悄抬上去**
    #    （7.5 GB 的机器上永远走 > safe 分支，所以一直没暴露）。
    if a.workers <= 0:
        print(f"面板 auto → {safe}（{src}，预算 {budget:.1f} GB @ {per_panel} GB/面板；"
              f"核数 {os.cpu_count()} @ {cores_per_panel:.1f} 核/面板 → 上限 {ncore}）")
        a.workers = safe
    elif a.workers > safe:
        print(f"⚠️ 内存预算 {budget:.1f} GB（{src}，上限 {ceil_pct:.0f}%，已用 {used_gb:.1f} GB）"
              f" → 面板数从 {a.workers} 降到 {safe}")
        a.workers = safe
    else:
        print(f"面板 {a.workers} 个（内存预算 {budget:.1f} GB 够放 {safe} 个，"
              f"上限 {ceil_pct:.0f}%）")

    shards = [[] for _ in range(max(1, a.workers))]
    for i, t in enumerate(tasks):
        shards[i % len(shards)].append(t)

    t0 = time.perf_counter()
    jobs = [(k, ids, dir_a, logdir) for k, ids in enumerate(shards) if ids]
    if len(jobs) <= 1:
        records = shard(jobs[0]) if jobs else []
    else:
        ctx = mp.get_context("spawn")     # 非 daemon；引擎是进程内单例，必须跨进程并行
        procs = [ctx.Process(target=shard_to_file, args=(j,)) for j in jobs]

        def _bail(*_a):
            """SIGINT/SIGTERM：先请子进程体面退出，超时再强杀。**不能只杀自己**。"""
            print("收到中断，正在清理子进程…", flush=True)
            for pr in procs:
                if pr.is_alive():
                    pr.terminate()
            for pr in procs:
                pr.join(10)
            for pr in procs:
                if pr.is_alive():
                    pr.kill()
            sys.exit(130)

        old = {sig: signal.signal(sig, _bail) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for pr in procs:
                pr.start()
            for pr in procs:
                pr.join()
        finally:
            for pr in procs:
                if pr.is_alive():
                    pr.kill()
                    pr.join(5)
            for sig, h in old.items():
                signal.signal(sig, h)
        records = []
        for j in jobs:
            fp = os.path.join(logdir, f"shard-{j[0]}.jsonl")
            if os.path.exists(fp):
                with open(fp, encoding="utf-8") as f:
                    records += [json.loads(x) for x in f if x.strip()]

    out = dict(cand=a.cand, protocol=dict(unit="bo3_match", matches=sum(per),
                                          seed_base=a.seed_base, per_opp=dict(zip([o[0] for o in opps], per)),
                                          workers=a.workers),
               per_opp={}, elapsed_s=round(time.perf_counter() - t0, 1))

    def pack(recs):
        s = summarize(recs)
        # per_match：mid → 得分(1/0.5/0)。有了它，两次运行（候选 vs 端口）在同一批
        # 种子上就能做**配对**比较（同一 mid 的两局对手相同、座位相同），
        # 比"两个独立比例相减"精确得多 —— 专精判定要用这个。
        pm = {str(r["match_id"]): (0.5 if r["winner"] == 2 else float(r["winner"] == r["seat"]))
              for r in recs}
        return dict(score=round(s["score"], 4), ci_low=round(s["ci_low"], 4),
                    ci_high=round(s["ci_high"], 4), n=s["matches"], wins=s["wins"],
                    draws=s["draws"], errors=s["errors"], per_match=pm)

    for name, _d in opps:
        rs = [r for r in records if r.get("opp") == name]
        if rs:
            out["per_opp"][name] = pack(rs)
    out["overall"] = pack(records) if records else dict(score=0.0, ci_low=0.0, ci_high=1.0,
                                                        n=0, errors=0)
    if len(records) != sum(per):
        out["warning"] = f"回收 {len(records)} 场，期望 {sum(per)} 场（有面板挂了）"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)

    o = out["overall"]
    print(f"候选 {a.cand}  总 {o['n']} 场 Bo3  得分 {o['score']:.4f} "
          f"CI [{o['ci_low']:.4f},{o['ci_high']:.4f}]  异常 {o['errors']}  "
          f"{out['elapsed_s']:.0f}s")
    for name, s in out["per_opp"].items():
        print(f"  vs {name:<24} {s['n']:>4} 场  {s['score']:.4f}  "
              f"[{s['ci_low']:.4f},{s['ci_high']:.4f}]  异常 {s['errors']}")
    if out.get("warning"):
        print("  ⚠️", out["warning"])
    print("→", a.out)


if __name__ == "__main__":
    main()
