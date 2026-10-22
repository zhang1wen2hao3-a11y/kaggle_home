#!/usr/bin/env python3
"""并发下载仿真赛的 episode 回放（`kaggle competitions replay`）。

为什么要自己写：CLI 一次只能下一个 episode，而且每个 episode 5~19 秒。
300+ 个串行要 30~45 分钟，并发后能压到几分钟。

⚠️ 必须用 **新 CLI**（`Kaggle CLI 2.x`，本机装在 base 环境：
`/home/zhang/miniforge3/bin/kaggle`）。旧的 1.7.4.5 没有 `replay` 子命令。
⚠️ 已知 Kaggle 对 `GetEpisodeReplay` 有 429 限流，所以默认并发只有 6，
并带指数退避重试；如果大量失败，把 `--workers` 降到 3。

用法：
    # 按 submission 自动列 episode 再下
    ./run.sh tools/fetch_episodes.py --submission 57396725 --out epsoide --workers 6
    # 手给 id 列表（一行一个）
    ./run.sh tools/fetch_episodes.py --ids-file /tmp/ep_list.txt --out epsoide
    # 用 py310 之外的解释器也能跑（只用标准库）
    python3 tools/fetch_episodes.py --submission 57396725 --out epsoide

已存在的文件自动跳过，所以可以随时中断再重跑。
"""
import argparse
import concurrent.futures as cf
import glob
import os
import random
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_KAGGLE = "/home/zhang/miniforge3/bin/kaggle"

_lock = threading.Lock()
_done = 0
_fail = []


def list_episodes(kaggle: str, submission: int, timeout: int = 300) -> list:
    """`kaggle competitions episodes <sid>` → episode id 列表。"""
    out = subprocess.run([kaggle, "competitions", "episodes", str(submission)],
                         capture_output=True, text=True, timeout=timeout)
    ids = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if parts and parts[0].isdigit() and len(parts[0]) >= 6:
            ids.append(parts[0])
    return ids


def already(out_dir: str, eid: str) -> bool:
    return bool(glob.glob(os.path.join(out_dir, f"*{eid}*.json")))


def fetch(kaggle: str, eid: str, out_dir: str, retries: int, timeout: int) -> tuple:
    """下一个 episode，带退避重试。返回 (eid, ok, 说明)。"""
    global _done
    if already(out_dir, eid):
        with _lock:
            _done += 1
        return eid, True, "已存在"
    delay = 2.0
    last = ""
    for attempt in range(retries):
        try:
            r = subprocess.run([kaggle, "competitions", "replay", eid, "-p", out_dir],
                               capture_output=True, text=True, timeout=timeout)
            if already(out_dir, eid):
                with _lock:
                    _done += 1
                return eid, True, f"{attempt + 1} 次成功"
            last = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or [""]
            last = last[0][:120]
        except subprocess.TimeoutExpired:
            last = f"超时 {timeout}s"
        time.sleep(delay + random.random())
        delay = min(delay * 2, 30)
    with _lock:
        _fail.append((eid, last))
    return eid, False, last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=int, default=None, help="submission id（自动列 episode）")
    ap.add_argument("--ids-file", default=None, help="episode id 列表文件（一行一个）")
    ap.add_argument("--out", default="epsoide", help="下载目录（相对仓库根）")
    ap.add_argument("--workers", type=int, default=6, help="并发数；遇 429 就降到 3")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=300, help="单个 episode 超时秒数")
    ap.add_argument("--kaggle", default=NEW_KAGGLE)
    ap.add_argument("--limit", type=int, default=None, help="只下前 N 个（调试用）")
    args = ap.parse_args()

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(args.kaggle):
        raise SystemExit(f"找不到新 CLI: {args.kaggle}")

    if args.submission:
        print(f"列出 submission {args.submission} 的 episode …", flush=True)
        ids = list_episodes(args.kaggle, args.submission)
    elif args.ids_file:
        ids = []
        for line in open(args.ids_file):
            p = line.split()
            if p and p[0].isdigit() and len(p[0]) >= 6:
                ids.append(p[0])
    else:
        raise SystemExit("要么给 --submission，要么给 --ids-file")
    if args.limit:
        ids = ids[:args.limit]
    if not ids:
        raise SystemExit("没拿到 episode id（CLI 输出格式变了？）")

    have = len(glob.glob(os.path.join(out_dir, "*.json")))
    print(f"共 {len(ids)} 个 episode，目录里已有 {have} 个 json；"
          f"并发 {args.workers}，目标 {out_dir}", flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch, args.kaggle, e, out_dir, args.retries, args.timeout)
                for e in ids]
        for f in cf.as_completed(futs):
            eid, ok, note = f.result()
            n_total = len(ids)
            with _lock:
                cur = _done + len(_fail)
            if not ok:
                print(f"  ✗ {eid}  {note}", flush=True)
            if cur % 20 == 0 or cur == n_total:
                el = time.time() - t0
                print(f"  {cur}/{n_total}  已用 {el:.0f}s  "
                      f"（{el / max(cur, 1):.1f}s/个）  失败 {len(_fail)}", flush=True)

    el = time.time() - t0
    total = len(glob.glob(os.path.join(out_dir, "*.json")))
    print(f"\n完成：目录里现在 {total} 个 json，用时 {el:.0f}s，失败 {len(_fail)}")
    if _fail:
        print("  失败的（可重跑本命令，已存在的会跳过）：")
        for eid, why in _fail[:10]:
            print(f"    {eid}  {why}")


if __name__ == "__main__":
    main()
