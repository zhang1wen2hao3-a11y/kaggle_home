#!/usr/bin/env python3
"""从**官方每日全量天梯回放 dump** 里抽卡组 meta（流式读 zip，不解压 21 GB）。

## 数据

`kaggle/the-pokemon-company-ptcg-ai-battle-challenge-playground-episodes-YYYY-MM-DD`
（每天一份 ~712~746 MB zip，内含数千个 ~5 MB 的回放 JSON；许可 CC0-1.0）。
这份 dump 是**全池、全队伍、按天**的天梯样本 —— 比我们自己的 2,169 局（我们 + 前两名队伍）
宽得多，是做卡组选择该用的语料。

## 关键事实（已在本仓库独立验证，见 `framework/replay.py`）

卡表在 `steps[1][j].action`，长度恰为 60；胜者在 `rewards`（末尾）。
同一个决策的 observation 会连续两步出现，action 在**后者**。

## 用法

    ./run.sh tools/deck_meta_from_dump.py --zip /tmp/epdump/*.zip --out data/deck_meta.csv
    ./run.sh tools/deck_meta_from_dump.py --zip X.zip --workers 8 --limit 2000
"""
import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def deck_hash(ids) -> str:
    return hashlib.sha256(",".join(str(x) for x in sorted(ids)).encode()).hexdigest()[:8]


def parse_one(payload: bytes) -> list:
    """一个回放 → 两行（每个座位一行）；解析失败返回 []。"""
    try:
        d = json.loads(payload)
    except Exception:
        return []
    steps = d.get("steps") or []
    if len(steps) < 2:
        return []
    info = d.get("info") or {}
    names = [a.get("Name") or "" for a in (info.get("Agents") or [])]
    teams = info.get("TeamNames") or names
    rew = d.get("rewards") or []
    decks = {}
    for j, ent in enumerate(steps[1] or []):
        act = (ent or {}).get("action")
        if act and len(act) == 60 and all(isinstance(x, int) for x in act):
            decks[j] = act
    if len(decks) < 2:
        return []
    rows = []
    for j, ids in decks.items():
        won = ""
        if len(rew) > j and rew[j] is not None:
            others = [r for k, r in enumerate(rew) if k != j and r is not None]
            if others:
                won = int(rew[j] > max(others))
        rows.append({
            "agent": names[j] if j < len(names) else "",
            "team": teams[j] if j < len(teams) else (names[j] if j < len(names) else ""),
            "deck_hash": deck_hash(ids),
            "won": won,
            "steps": len(steps),
            "deck": " ".join(str(x) for x in ids),
        })
    return rows


def _work(args):
    zip_path, name = args
    try:
        with zipfile.ZipFile(zip_path) as z:
            return parse_one(z.read(name))
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="一个或多个 zip（支持通配）")
    ap.add_argument("--out", default="data/deck_meta.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个回放（调试用）")
    args = ap.parse_args()

    paths = []
    for p in args.zip.split(","):
        paths += glob.glob(p)
    if not paths:
        raise SystemExit(f"找不到 zip：{args.zip}")
    tasks = []
    for zp in paths:
        with zipfile.ZipFile(zp) as z:
            names = [n for n in z.namelist() if n.endswith(".json")]
        print(f"  {os.path.basename(zp)}: {len(names)} 个回放")
        tasks += [(zp, n) for n in names]
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"共 {len(tasks)} 个回放待解析（workers={args.workers}）")

    rows = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(_work, tasks, chunksize=8)):
                rows += r
                if (k + 1) % 500 == 0:
                    print(f"    {k + 1}/{len(tasks)}  已得 {len(rows)} 行", flush=True)
    else:
        for t in tasks:
            rows += _work(t)

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["agent", "team", "deck_hash", "won", "steps", "deck"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n写出 {out}：{len(rows)} 行（{len(rows) // 2} 局左右）")

    share = collections.Counter(r["deck_hash"] for r in rows)
    print(f"不同卡组 {len(share)} 副；最常见 12 副：")
    for h, n in share.most_common(12):
        print(f"  {h}  {n:>6}  {n / max(len(rows), 1):>6.1%}")
    teams = {r["team"] for r in rows}
    print(f"队伍 {len(teams)} 支")


if __name__ == "__main__":
    main()
