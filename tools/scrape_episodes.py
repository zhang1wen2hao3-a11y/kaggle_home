#!/usr/bin/env python3
"""抓取某个提交的近期天梯对局，统计【真实野区分布】与【我方逐原型胜率】。

这是 1st place 那个 "last day metagame read" 的来源：每个 replay 里都有
对手队名 + 双方完整 60 张卡表 + rewards（胜负）。不需要本地评测。

用法：
    python3 tools/scrape_episodes.py --submission 57586502 --limit 40
    python3 tools/scrape_episodes.py --submission 57586502 --limit 40 --csv data/episodes.csv --reuse

要点：
  * 每局下载 8–16 MB，解析完立刻删除，磁盘占用恒定
  * 已有 CSV 中的 episode 默认跳过（--no-reuse 强制重抓）
  * 不在 185 池里的卡组会单独列出（说明池子过期了）
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
KAGGLE = os.environ.get("KAGGLE_BIN", "/home/zhang/miniforge3/bin/kaggle")
CARD_CSV = ROOT / "data" / "EN_Card_Data_R2_full.csv"
POOL = ROOT / "decks" / "pool_field"


def run(cmd: list[str], timeout: int = 300) -> str:
    env = dict(os.environ, KAGGLE_CONFIG_DIR=os.path.expanduser("~/.kaggle"))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} 失败: {p.stderr.strip()[:300]}")
    return p.stdout


def my_username() -> str:
    cfg = pathlib.Path(os.path.expanduser("~/.kaggle/kaggle.json"))
    return json.loads(cfg.read_text())["username"]


def load_card_names() -> dict[int, str]:
    out: dict[int, str] = {}
    with open(CARD_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                out[int(r["Card ID"])] = r["Card Name"]
            except (TypeError, ValueError):
                pass
    return out


def load_pool() -> dict[tuple[int, ...], str]:
    """sorted 60-id tuple -> pool stem"""
    pool: dict[tuple[int, ...], str] = {}
    for f in POOL.glob("*.csv"):
        ids = tuple(sorted(int(x) for x in f.read_text().split() if x.strip()))
        if len(ids) == 60:
            pool[ids] = f.stem
    return pool


def list_episodes(submission: int, limit: int) -> list[int]:
    out = run([KAGGLE, "competitions", "episodes", str(submission), "-v"])
    rows = list(csv.DictReader(out.splitlines()[1:] if out.startswith("Warning") else out.splitlines()))
    ids: list[int] = []
    for r in rows:
        try:
            ids.append(int(r["id"]))
        except (KeyError, ValueError):
            continue
    return ids[:limit]


def parse_replay(path: pathlib.Path, me: str, pool, names) -> dict | None:
    d = json.loads(path.read_text())
    agents = [a["Name"] for a in d["info"]["Agents"]]
    if me not in agents:
        return None
    i = agents.index(me)
    j = 1 - i
    steps = d["steps"]
    # 卡组在 steps[1][*].action（决策记录滞后一步）
    decks = []
    for k in (0, 1):
        a = steps[1][k].get("action") if len(steps) > 1 else None
        decks.append(a if isinstance(a, list) and len(a) == 60 else None)
    my_deck, op_deck = decks[i], decks[j]
    r = d["rewards"][i]
    return dict(
        episode=int(d["info"]["EpisodeId"]),
        opponent=agents[j],
        won=int(r > 0),
        loss=int(r < 0),
        steps=len(steps),
        my_deck=pool.get(tuple(sorted(my_deck)), "?") if my_deck else "?",
        op_deck=pool.get(tuple(sorted(op_deck)), "?") if op_deck else "?",
        op_top=" / ".join(
            f"{c}x {names.get(c, c)}" for c, c in collections.Counter(op_deck).most_common(4)
        ) if op_deck else "",
        seed=d["configuration"].get("seed"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=int, required=True)
    ap.add_argument("--limit", type=int, default=40, help="最多抓多少局")
    ap.add_argument("--csv", default="data/episodes.csv")
    ap.add_argument("--reuse", action="store_true", help="复用 CSV 里已有的 episode")
    ap.add_argument("--no-fetch", action="store_true", help="只统计已有 CSV")
    ap.add_argument("--workers", type=int, default=4,
                    help="并发下载数；Kaggle 按连接限速，实测 3 并发≈2.5x")
    args = ap.parse_args()

    me = my_username()
    out_csv = pathlib.Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    known: set[int] = set()
    rows: list[dict] = []
    fields = ["episode", "opponent", "won", "loss", "steps", "my_deck", "op_deck", "op_top", "seed"]
    if out_csv.exists() and args.reuse:
        with open(out_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        known = {int(r["episode"]) for r in rows}
        print(f"复用 {len(rows)} 局")

    if not args.no_fetch:
        import concurrent.futures as cf
        import threading
        import time

        pool, names = load_pool(), load_card_names()
        eps = list_episodes(args.submission, args.limit)
        todo = [e for e in eps if e not in known]
        print(f"提交 {args.submission} 共列出 {len(eps)} 局，待抓 {len(todo)} 局"
              f"（我方={me}，workers={args.workers}）", flush=True)
        # ⚠️ 必须 RLock：save() 会在已持锁的代码块里被再次调用（每局增量落盘）。
        #    用普通 Lock 会在**第一局完成时**永久死锁 —— 进程卡 futex、无子进程、无输出，
        #    表象极像"网络被限速"，实际是代码 bug（2026-10-26 曾因此白等 8 小时）。
        lock = threading.RLock()
        t0 = time.time()
        done = [0]

        def save() -> None:
            with lock:
                with open(out_csv, "w", newline="", encoding="utf-8") as f:
                    wr = csv.DictWriter(f, fieldnames=fields)
                    wr.writeheader()
                    wr.writerows(rows)

        def one(eid: int) -> dict | None:
            with tempfile.TemporaryDirectory() as td:
                try:
                    run([KAGGLE, "competitions", "replay", str(eid), "-p", td, "-q"])
                except Exception as exc:  # noqa: BLE001
                    print(f"  {eid} 跳过：{str(exc)[:120]}", flush=True)
                    return None
                p = next(pathlib.Path(td).glob(f"*{eid}*.json"), None)
                if p is None:
                    return None
                try:
                    return parse_replay(p, me, pool, names)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {eid} 解析失败：{str(exc)[:120]}", flush=True)
                    return None

        with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(one, e): e for e in todo}
            for fu in cf.as_completed(futs):
                rec = fu.result()
                with lock:
                    done[0] += 1
                    n = done[0]
                    if rec:
                        rows.append(rec)
                        save()          # 每局都落盘，中断不丢
                        el = time.time() - t0
                        eta = el / n * (len(todo) - n)
                        print(f"  [{n}/{len(todo)}] {rec['episode']} vs "
                              f"{rec['opponent'][:20]:20s} {rec['op_deck'][:32]:32s} "
                              f"{'胜' if rec['won'] else '负'}  剩余~{eta/60:.0f}分", flush=True)
                    else:
                        print(f"  [{n}/{len(todo)}] 跳过 1 局", flush=True)
        print(f"\n写入 {out_csv}（{len(rows)} 局）")

    if not rows:
        return 0
    # ---------------- 统计
    def arch(deck: str) -> str:
        return deck.split("__")[0] if deck and deck != "?" else "（不在 185 池）"

    W = sum(int(r["won"]) for r in rows)
    L = sum(int(r["loss"]) for r in rows)
    print(f"\n总战绩 {W}胜 {L}负  胜率 {W / max(W + L, 1):.3f}")
    print(f"\n{'对手原型':40s}{'场次':>5s}{'胜':>4s}{'负':>4s}{'胜率':>7s}")
    agg = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        a = agg[arch(r["op_deck"])]
        a[0] += int(r["won"])
        a[1] += int(r["won"]) + int(r["loss"])
    for a, (w, n) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"{a[:40]:40s}{n:5d}{w:4d}{n - w:4d}{w / max(n, 1):7.3f}")
    unknown = [r for r in rows if r["op_deck"] == "?"]
    if unknown:
        print(f"\n⚠️ {len(unknown)} 局的对手卡组不在池中（池子过期）：")
        for r in unknown[:6]:
            print(f"   {r['opponent'][:20]:20s} {r['op_top']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
