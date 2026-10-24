#!/usr/bin/env python3
"""卡组 gauntlet：**同一个策略网络**分别驾驶不同卡组互相对打。

## 为什么这是最高性价比的实验

公开界（kellydannimao）用同一个策略网络换卡组、对同一个对手，Bo3 得分率从
**0.09 到 0.88**；而所有搜索/规则改进都在噪声内。也就是说：在这个赛题里，
**卡组的匹配场效应比任何微操都大一个量级**。

我们的处境：移植包现在用的是 Ogerpon ex（草）那副，在本地语料的 Bradley-Terry 里
deck 效应 +0.61±0.38（中游）。候选是语料里效应最高的几副。

## 做法

对每副候选卡组，在 `/tmp/deck_gauntlet/<hash>/` 下建一个 agent 目录
（软链移植包的代码与权重，只把 `deck.csv` 换成该卡组），然后用同一个策略
两两对打（`evaluate.py` 自带座位×先后手四路交替）。

判据（跑之前定）：
    * 每对 n=200，Wilson SE ≈ 3.5pp；只有区间不重叠才算真差异
    * 最终按语料份额加权，**相对当前卡组**的提升 ≥ +5pp 才值得换

用法：
    ./run.sh tools/deck_gauntlet.py --decks 26c6e843,e8eef8d8,0d86f3a5,f5f8201b --games 200
"""
import argparse
import collections
import csv
import itertools
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = "/tmp/port/out/agent"
WORK = "/tmp/deck_gauntlet"


def load_decks(corpus, hashes, current_csv):
    """从语料里取指定 hash 的卡表；另可加一副"当前"卡组。"""
    want = set(hashes)
    decks = {}
    with open(corpus) as f:
        for r in csv.DictReader(f):
            if r["deck_hash"] in want and r["deck_hash"] not in decks:
                decks[r["deck_hash"]] = [int(x) for x in r["deck"].split()]
    if current_csv:
        ids = [int(x) for x in open(current_csv).read().split() if x.strip()][:60]
        import hashlib
        h = hashlib.sha256(",".join(str(x) for x in sorted(ids)).encode()).hexdigest()[:8]
        decks[h] = ids
    return decks


def stage(name, ids):
    """建 agent 目录：软链代码/权重，只替换 deck.csv。"""
    d = os.path.join(WORK, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    for entry in os.listdir(BUNDLE):
        if entry in ("deck.csv", "port_agent.py"):
            continue          # wrapper 单独复制（软链会导致 copy2 撞同一文件）
        src, dst = os.path.join(BUNDLE, entry), os.path.join(d, entry)
        if os.path.isdir(src):
            os.symlink(src, dst)
        else:
            os.symlink(src, dst)
    # 移植包的 wrapper 在包目录里；为每副卡组各建一份（它按 __file__ 定位数据目录）
    wrap = os.path.join(BUNDLE, "port_agent.py")
    if os.path.exists(wrap):
        shutil.copy2(wrap, os.path.join(d, "port_agent.py"))
    with open(os.path.join(d, "deck.csv"), "w") as f:
        f.write("\n".join(str(x) for x in ids) + "\n")
    return os.path.join(d, "port_agent.py")


def one_match(a_file, b_file, a_deck, b_deck, games, workers):
    cmd = [sys.executable, "-u", os.path.join(ROOT, "evaluate.py"),
           "--agent", a_file, "--opponent", b_file,
           "--deck", a_deck, "--deck-opponent", b_deck,
           "--games", str(games), "--workers", str(workers), "--quiet"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    m = re.search(r"胜率\s*:\s*([0-9.]+)", out.stdout)
    if not m:
        return None
    return float(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decks", required=True, help="逗号分隔的 deck_hash")
    ap.add_argument("--corpus", default="data/deck_corpus.csv")
    ap.add_argument("--current", default="/tmp/port/out/agent/deck.csv",
                    help="当前卡组 csv（会一起进 gauntlet 当基准）")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="logs/deck_gauntlet.tsv")
    args = ap.parse_args()

    hashes = [h.strip() for h in args.decks.split(",") if h.strip()]
    decks = load_decks(os.path.join(ROOT, args.corpus), hashes, args.current)
    missing = [h for h in hashes if h not in decks]
    if missing:
        print(f"⚠️ 语料里找不到这些卡组，跳过：{missing}")
    print(f"参赛卡组 {len(decks)} 副：")
    files, csvs = {}, {}
    for h, ids in decks.items():
        if len(ids) != 60:
            print(f"  ⚠️ {h} 张数 {len(ids)} ≠ 60，跳过"); continue
        files[h] = stage(h, ids)
        csvs[h] = os.path.join(WORK, h, "deck.csv")
        print(f"  {h}  ({len(ids)} 张)")

    names = sorted(files)
    wins = collections.Counter(); n = collections.Counter()
    rows = []
    for a, b in itertools.combinations(names, 2):
        w = one_match(files[a], files[b], csvs[a], csvs[b], args.games, args.workers)
        if w is None:
            print(f"  {a} vs {b}: 解析失败"); continue
        rows.append((a, b, w))
        wins[a] += w * args.games; n[a] += args.games
        wins[b] += (1 - w) * args.games; n[b] += args.games
        print(f"  {a} vs {b}: {w:.3f}（n={args.games}）", flush=True)

    print(f"\n{'deck':<10}{'总胜率':>8}{'局数':>7}")
    ranked = sorted(names, key=lambda h: -(wins[h] / max(n[h], 1)))
    for h in ranked:
        if n[h]:
            print(f"{h:<10}{wins[h] / n[h]:>7.3f}{n[h]:>7}")
    if len(ranked) >= 2:
        print(f"\n最强 {ranked[0]} vs 最弱 {ranked[-1]}："
              f"{wins[ranked[0]] / max(n[ranked[0]], 1):.3f} vs "
              f"{wins[ranked[-1]] / max(n[ranked[-1]], 1):.3f}")
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)) or ".", exist_ok=True)
    with open(os.path.join(ROOT, args.out), "w") as f:
        f.write("a\tb\twin_rate_a\n")
        for a, b, w in rows:
            f.write(f"{a}\t{b}\t{w:.4f}\n")


if __name__ == "__main__":
    main()
