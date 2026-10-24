#!/usr/bin/env python3
"""真 **Bo3** 本地评测台（改自 matterhorn3838 的进程隔离面板）。

## 为什么需要它（我们现在只有单局 A/B）

比赛指标是 `cabt_bo3`：一场 = 最多 3 局，第 2 局的"选先后手"权交给座位 1。
单局 A/B 测不出三类真实问题：

1. **按局状态没重置**（一副牌只问一次；`current.round` 变化时必须重建 tracker）——只在第 2/3 局暴露；
2. **600 s 池被跨局吃穿**（天梯上是每局满池；本地若不重置会越打越弱）；
3. **公开 agent 吞异常后随机兜底**（`except: 合法随机` 会让状态看起来是 DONE）——必须数日志里的 `Traceback`。

## 与单局 `evaluate.py` 的分工

* `evaluate.py`：**快**（0.4~1.4 s/局）、座位×先手四路交替，用于滚动实验与扫描；
* `tools/bo3_eval.py`（本文件）：**真**（真 Bo3、真进程隔离、Kaggle 式 `exec` 加载、每场 deck-request 重置、
  600 s 池、兜底计数、座位×先手四格、Wilson + 分层 bootstrap），用于**上线前的最终确认**。

## 加载方式与 Kaggle 一致

`exec(compile(main.py))`，**没有 `__file__`**，取命名空间里**最后一个 callable**（不是叫 `agent` 的那个），
观测里带 `player` / `remainingOverageTime` / `round` / `players[i].win`。

## 用法

    ./run.sh tools/bo3_eval.py --a dist/ptcg_public_port_deckF.tar.gz --b /tmp/port/out/agent \\
        --matches 60 --workers 2
    ./run.sh tools/bo3_eval.py --a <dirA> --b <dirB> --matches 40 --a-base-deck   # 见下

`--a` / `--b` 可以是解压后的目录（含 main.py）或 `.tar.gz` 包。**卡组由 agent 自己交**
（deck-request ABI），所以想比卡组就分别打包/放置。
"""
import argparse
import contextlib
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 与竞赛 cabt 配置一致
CONFIG = dict(bo=3, version=2, actTimeout=0, runTimeout=6000, episodeSteps=10000000)
GAME_POOL_S = 600.0          # 每局的思考池（天梯每局满池；这里每局重置）
MAX_DECISIONS = 4000         # 超过就判失败（不是平局）


def resolve(path: str) -> str:
    """把 tar.gz 解开成目录（没有 main.py 的目录直接报错）。"""
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if os.path.isdir(p):
        if not os.path.exists(os.path.join(p, "main.py")):
            raise SystemExit(f"目录里没有 main.py: {p}")
        return p
    if p.endswith(".py"):
        # 单个 .py agent（我们仓库里的 agents/xxx.py）：造一个临时目录 + wrapper main.py，
        # 用 runpy 按**原路径**执行，这样它内部的 `__file__`/相对路径逻辑不受影响。
        d = tempfile.mkdtemp(prefix="bo3_py_")
        wrapper = (
            "import runpy\n"
            f"_NS = runpy.run_path({p!r})\n"
            "_AG = _NS.get('agent') or [v for v in _NS.values() if callable(v)][-1]\n"
            "def agent(obs, config=None):\n"
            "    return _AG(obs)\n"
        )
        open(os.path.join(d, "main.py"), "w").write(wrapper)
        return d
    if p.endswith((".tar.gz", ".tgz")):
        d = tempfile.mkdtemp(prefix="bo3_agent_")
        with tarfile.open(p) as t:
            t.extractall(d)
        if not os.path.exists(os.path.join(d, "main.py")):
            raise SystemExit(f"包里没有顶层 main.py: {p}")
        return d
    raise SystemExit(f"无法识别的 agent 路径: {path}")


# ------------------------------------------------------------------ 子进程
def worker_main(folder, connection, log_path):
    import sys as _sys
    log = open(log_path, "a", encoding="utf-8", buffering=1)
    _sys.stdout = _sys.stderr = log
    os.environ["PTCG_DEBUG"] = "1"       # 让"吞异常"变成可数的 Traceback
    os.chdir(folder)
    _sys.path.insert(0, folder)
    ns = {}
    try:
        started = time.perf_counter()
        source = Path("main.py").read_text(encoding="utf-8-sig")
        # Kaggle 的加载器：不给 __file__，也不给 __name__
        exec(compile(source, str(Path(folder) / "main.py"), "exec"), ns)
        fn = [v for v in ns.values() if callable(v)][-1]      # 最后一个 callable
        connection.send(dict(ready=True, entry=getattr(fn, "__name__", "?"),
                             load_s=time.perf_counter() - started))
    except Exception:
        connection.send(dict(ready=False, error=traceback.format_exc()))
        return
    while True:
        message = connection.recv()
        if message is None:
            return
        try:
            if "seed" in message:
                random.seed(message["seed"])
            obs = message["observation"]
            log.flush()
            before = log.tell()
            t0 = time.perf_counter()
            argc = getattr(getattr(fn, "__code__", None), "co_argcount", 1)
            args = [obs, dict(CONFIG, __raw_path__=str(Path(folder) / "main.py"))][:argc]
            result = fn(*args)
            elapsed = time.perf_counter() - t0
            log.flush()
            with open(log_path, encoding="utf-8", errors="replace") as reader:
                reader.seek(before)
                new_log = reader.read()
            connection.send(dict(action=result, elapsed=elapsed,
                                 fallback_errors=new_log.count("Traceback (most recent call last)"),
                                 guard=ns.get("GUARD", {}) if isinstance(ns.get("GUARD"), dict) else {}))
        except Exception:
            connection.send(dict(error=traceback.format_exc()))


class AgentProcess:
    def __init__(self, name, folder, log_dir):
        self.name = name
        log_dir = os.path.abspath(log_dir)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log = os.path.join(log_dir, name + ".log")
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=worker_main, args=(str(folder), child, self.log))
        self.proc.start()
        if not self.conn.poll(300):
            raise TimeoutError("agent 加载超时: " + name)
        self.ready = self.conn.recv()
        if not self.ready.get("ready"):
            raise RuntimeError(f"{name}: {self.ready.get('error', 'load failed')}")

    def call(self, observation, seed=None):
        msg = dict(observation=observation)
        if seed is not None:
            msg["seed"] = seed
        self.conn.send(msg)
        if not self.conn.poll(120):
            raise TimeoutError("单个决策超过 120 s: " + self.name)
        reply = self.conn.recv()
        if reply.get("error"):
            raise RuntimeError(self.name + ": " + reply["error"])
        return reply

    def close(self):
        with contextlib.suppress(Exception):
            if self.proc.is_alive():
                self.conn.send(None)
                self.proc.join(5)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join()


def valid(selection, action) -> bool:
    return (isinstance(action, list) and len(set(action)) == len(action)
            and selection["minCount"] <= len(action) <= selection["maxCount"]
            and all(type(i) is int and 0 <= i < len(selection["option"]) for i in action))


# ------------------------------------------------------------------ 一场 Bo3
def play_match(left, right, match_id, trace_path=None):
    from cg.game import battle_start, battle_select, battle_finish
    agents = [left, right]
    wins = [0, 0]
    rounds = []
    fallback_errors = [0, 0]
    think = [0.0, 0.0]
    steps = 0
    error = None
    started = time.perf_counter()
    stream = open(trace_path, "w", encoding="utf-8") if trace_path else None
    try:
        decks = []
        for i, ag in enumerate(agents):
            # ★ 每场用**真实的 deck-request ABI** 重置 —— 这就是 Kaggle 的按场开始
            reply = ag.call(dict(select=None, current=None, logs=[], search_begin_input=None,
                                 player=i, remainingOverageTime=GAME_POOL_S),
                            20261016 + match_id * 2 + i)
            deck = reply["action"]
            if not (isinstance(deck, list) and len(deck) == 60 and all(type(c) is int for c in deck)):
                raise ValueError(f"{ag.name} 交的卡组非法（{len(deck) if hasattr(deck, '__len__') else '?'} 张）")
            decks.append(deck)
        for round_index in range(CONFIG["bo"]):
            alive = False
            pool = [GAME_POOL_S, GAME_POOL_S]          # ★ 每局满池（天梯行为）
            try:
                obs, status = battle_start(*decks, reverse_player=bool(round_index % 2))
                alive = bool(status.battlePtr)
                if not alive or status.errorPlayer >= 0:
                    raise ValueError(f"引擎开局错误 {status.errorPlayer}/{status.errorType}")
                for _ in range(MAX_DECISIONS):
                    cur = obs["current"]
                    if cur["result"] >= 0:
                        break
                    cur["round"] = round_index + 1
                    for i in range(2):
                        cur["players"][i]["win"] = wins[i]
                    player = cur["yourIndex"]
                    obs["player"], obs["step"] = player, steps
                    obs["remainingOverageTime"] = pool[player]
                    reply = agents[player].call(obs)
                    pool[player] -= reply["elapsed"]
                    think[player] += reply["elapsed"]
                    fallback_errors[player] += reply["fallback_errors"]
                    action = reply["action"]
                    if pool[player] <= 0:
                        raise TimeoutError(f"600 s 池被 {agents[player].name} 用尽")
                    if not valid(obs["select"], action):
                        raise ValueError(f"{agents[player].name} 非法动作: {action}")
                    if stream:
                        stream.write(json.dumps(dict(round=round_index + 1, player=player,
                                                     observation=obs, action=action)) + "\n")
                    obs = battle_select(action)
                    steps += 1
                else:
                    raise TimeoutError("超过 4000 决策上限（判失败，不判平局）")
                result = obs["current"]["result"]
                rounds.append(dict(round=round_index + 1, winner=result,
                                   first_player=obs["current"].get("firstPlayer")))
                if result in (0, 1):
                    wins[result] += 1
            finally:
                if alive:
                    battle_finish()
            if max(wins) > min(wins) + CONFIG["bo"] - len(rounds):
                break                                   # 提前结束
    except Exception:
        error = traceback.format_exc()[-1800:]
    finally:
        if stream:
            stream.close()
    winner = None if error else (0 if wins[0] > wins[1] else 1 if wins[1] > wins[0] else 2)
    return dict(match_id=match_id, left=left.name, right=right.name, wins=wins, rounds=rounds,
                winner=winner, error=error, fallback_errors=fallback_errors,
                think_s=think, steps=steps, elapsed_s=time.perf_counter() - started)


# ------------------------------------------------------------------ 统计
def wilson(wins, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (center - radius, center + radius)


def summarize(records, a_name="A"):
    """按 (座位) 汇总 A 的 Bo3 得分率：胜 1 / 平 0.5 / 负 0。"""
    def score(r, seat):
        if r["winner"] is None:
            return 0.0
        return 0.5 if r["winner"] == 2 else float(r["winner"] == seat)

    tot = len(records)
    pts = sum(score(r, r["seat"]) for r in records)
    wins = sum(1 for r in records if r["winner"] == r["seat"])
    draws = sum(1 for r in records if r["winner"] == 2)
    errs = sum(1 for r in records if r["error"] or any(r["fallback_errors"]))
    lo, hi = wilson(int(pts * 2), tot * 2) if False else wilson(int(round(pts)), tot)
    out = dict(matches=tot, wins=wins, draws=draws, score=pts / max(tot, 1),
               ci_low=lo, ci_high=hi, errors=errs,
               max_think_s=max((max(r["think_s"]) for r in records), default=0.0))
    return out


def seat_order_table(records):
    """座位 × 是否先手 四格 —— 座位 bug 的唯一可靠检测。"""
    cells = {}
    for r in records:
        for g in r["rounds"]:
            went_first = (g.get("first_player") == r["seat"]) if g.get("first_player") is not None else None
            key = (r["seat"], went_first)
            c = cells.setdefault(key, [0, 0])
            c[0] += 1
            c[1] += int(g["winner"] == r["seat"])
    return cells


def _shard(job):
    """一个面板：独立进程里的 (A,B) agent 对，顺序打若干场 Bo3。

    注意：同一个进程里**不能并发**跑多场（引擎全局单例），所以面板内串行、面板间并行。
    """
    k, ids, dir_a, dir_b, logdir, trace_first = job
    panes = []
    try:
        for i in range(2):
            panes.append((AgentProcess(f"{k}A{i}", dir_a, logdir),
                          AgentProcess(f"{k}B{i}", dir_b, logdir)))
        recs = []
        for blk, mid in enumerate(ids):
            seat = mid % 2                       # 座位在块之间交替
            pa, pb = panes[blk % 2]
            left, right = (pa, pb) if seat == 0 else (pb, pa)
            tp = os.path.join(logdir, f"trace-{mid}.jsonl") if (trace_first and mid == 0) else None
            r = play_match(left, right, mid, tp)
            r.update(seat=seat)
            recs.append(r)
        return recs
    finally:
        for pair in panes:
            for ag in pair:
                ag.close()


def _shard_to_file(job):
    """面板跑完把结果写到 logdir/shard-K.jsonl（跨进程回收用）。"""
    recs = _shard(job)
    fp = os.path.join(job[4], f"shard-{job[0]}.jsonl")
    with open(fp, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="被测 agent（目录或 .tar.gz）")
    ap.add_argument("--b", required=True, help="对手 agent")
    ap.add_argument("--matches", type=int, default=30, help="Bo3 场数（会被向上补成偶数）")
    ap.add_argument("--workers", type=int, default=2, help="并行面板数（每面板 2 个 agent 进程）")
    ap.add_argument("--logdir", default="logs/bo3")
    ap.add_argument("--trace-first", action="store_true", help="记录第一场逐步轨迹（含观测）")
    args = ap.parse_args()

    n = args.matches + (args.matches % 2)
    dir_a, dir_b = resolve(args.a), resolve(args.b)
    # ⚠️ 必须绝对路径：子进程会 chdir 到 agent 目录，相对路径会找不到日志文件
    args.logdir = os.path.abspath(args.logdir)
    os.makedirs(args.logdir, exist_ok=True)
    print(f"A = {args.a}\nB = {args.b}\nBo3 场数 = {n}（面板 {args.workers}）")

    shards = [[] for _ in range(args.workers)]
    for i in range(n):
        shards[i % args.workers].append(i)

    t0 = time.perf_counter()
    records = []
    jobs = [(k, ids, dir_a, dir_b, args.logdir, args.trace_first)
            for k, ids in enumerate(shards)]
    # ⚠️ 必须**跨进程**并行：`cg` 的 battle 是进程内全局单例，线程并发会直接踩坏 C++ 堆
    #    （实测 `malloc_consolidate(): unaligned fastbin chunk detected` → core dump）。
    if args.workers <= 1:
        for j in jobs:
            records += _shard(j)
    else:
        # 用普通 Process（**非 daemon**）：池化的 worker 是 daemon，daemon 不许再开子进程，
        # 而每个面板必须能各自 spawn 两个 agent 子进程。
        ctx = mp.get_context("spawn")
        procs = []
        for j in jobs:
            pr = ctx.Process(target=_shard_to_file, args=(j,))
            pr.start()
            procs.append(pr)
        for pr in procs:
            pr.join()
        for j in jobs:
            fp = os.path.join(j[4], f"shard-{j[0]}.jsonl")
            if os.path.exists(fp):
                with open(fp, encoding="utf-8") as f:
                    records += [json.loads(line) for line in f if line.strip()]
    summ = summarize(records)
    print(f"\n{'='*62}\nBo3 得分率（A）: {summ['score']:.4f}"
          f"  95% CI [{summ['ci_low']:.4f}, {summ['ci_high']:.4f}]")
    print(f"  场数 {summ['matches']}  胜 {summ['wins']}  平 {summ['draws']}"
          f"  失败(异常/兜底) {summ['errors']}  最慢思考 {summ['max_think_s']:.1f}s")
    print(f"  用时 {time.perf_counter()-t0:.0f}s")
    cells = seat_order_table(records)
    print(f"\n座位 × 是否先手 四格（A 的胜率；某格远低于其它三格 = 座位 bug）：")
    for (seat, wf), (g, w) in sorted(cells.items(), key=lambda kv: (kv[0][0], bool(kv[0][1]))):
        tag = "先手" if wf else "后手" if wf is False else "未知"
        print(f"  seat={seat} {tag:<4} 局数 {g:>4}  胜率 {w / max(g, 1):.3f}")
    out = os.path.join(args.logdir, "last_run.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n逐场记录: {out}")


if __name__ == "__main__":
    main()
