#!/usr/bin/env python3
"""从**我们输掉的对局**里提取对手卡组，按"配置基本相同就合并"去重，导出代表样本。

## 为什么这么做

真实场里我们被谁打、用什么构筑打——这是唯一"免费"得到的对手情报。有了它就能：
1. 看清**哪些卡组在打我们**（份额 + 对我们的胜率 + Wilson 区间）；
2. 把 90 个 hash 合并成少数几种**配置**（很多只是相差 1~2 张），只留代表；
3. 用这些代表卡组 + **强策略驾驶员**搭本地对手池（以前做不了：我们自己的 agent 开不动复杂卡组，
   现在有移植包可以开）。

## 去重规则（"配置基本相同就合并"）

* 先按 `deck_hash`（排序后卡 id 的 sha256 前 8 位）合并**完全相同**的卡表；
* 再做**近邻聚类**：两副牌按 multiset 的 Jaccard ≥ `--jaccard`（默认 0.90）合并成一簇，
  代表取簇内出现次数最多的那副。0.90 ≈ 允许交换 3 张（60 张里 57 张相同）。

## 用法

    ./run.sh tools/analyze_opponents.py                      # 只看统计
    ./run.sh tools/analyze_opponents.py --export decks/opponents --top 8
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from framework.replay import decks_of, our_index  # noqa: E402

OUR_NAME = "zhang1wen2hao3"


def load_cards():
    """返回 (id→名字, 基本能量 id 集合)。

    ⚠️ 聚类必须**剔掉基本能量**：35 张水能量 vs 33 张水能量是同一套构筑，
    按 multiset Jaccard 会被拆成两个簇（实测 76 个簇里一大半是这种噪声）。
    """
    path = os.path.join(ROOT, "data", "EN_Card_Data_R2_full.csv")
    name, basics = {}, set()
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                cid = int(r["Card ID"])
            except Exception:
                continue
            nm = (r.get("Card Name") or r.get("Name") or "").strip()
            name[cid] = nm
            if nm.startswith("Basic {") and nm.endswith("Energy"):
                basics.add(cid)
    return name, basics


def wilson(w, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 1.0)
    p = w / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - r, c + r)


def jaccard(a, b) -> float:
    ca, cb = collections.Counter(a), collections.Counter(b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 0.0


def cluster(decks: dict, thresh: float):
    """把 hash→卡表 按 Jaccard 阈值聚成簇（贪心，按簇大小降序）。"""
    order = sorted(decks, key=lambda h: -len(decks[h]))          # 实际按次数在外面排
    used, clusters = set(), []
    for h in order:
        if h in used:
            continue
        grp = [h]
        used.add(h)
        for h2 in order:
            if h2 in used:
                continue
            if jaccard(decks[h], decks[h2]) >= thresh:
                grp.append(h2)
                used.add(h2)
        clusters.append(grp)
    return clusters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="epsoide", help="我们自己的回放目录")
    ap.add_argument("--jaccard", type=float, default=0.90)
    ap.add_argument("--top", type=int, default=0, help="导出前 N 簇的代表卡组")
    ap.add_argument("--min-games", type=int, default=5, help="打印阈值")
    ap.add_argument("--export", default="", help="导出目录（留空 = 不导出）")
    args = ap.parse_args()

    names, BASIC = load_cards()
    files = sorted(glob.glob(os.path.join(ROOT, args.dir, "*.json")))
    print(f"扫描我们自己的回放 {len(files)} 局（{args.dir}）")

    # ---- 逐局：我们的卡组 / 对手卡组 / 结果
    recs = []
    our_decks = collections.Counter()
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        steps = d.get("steps") or []
        me = decks = None
        me = our_index(d, OUR_NAME)
        if me is None:
            continue
        decks = decks_of(steps)
        if len(decks) < 2:
            continue
        rew = d.get("rewards") or []
        won = None
        if len(rew) > me and rew[me] is not None:
            others = [r for k, r in enumerate(rew) if k != me and r is not None]
            if others:
                won = rew[me] > max(others)
        opp = 1 - me if me in (0, 1) else 0
        if opp not in decks:
            continue
        our_decks[decks[me] and tuple(sorted(decks[me]))] += 1
        recs.append(dict(file=os.path.basename(f), our=tuple(sorted(decks[me])),
                         opp=tuple(sorted(decks[opp])), won=won))
    print(f"可用对局 {len(recs)}；我们的卡表种类 {len({r['our'] for r in recs})}"
          f"（{len(our_decks)} 种）")
    if not recs:
        raise SystemExit("没有可用对局")
    base_w = sum(1 for r in recs if r["won"])
    print(f"总体胜率 {base_w}/{len(recs)} = {base_w/len(recs):.3f}")

    import hashlib

    def h(ids):
        return hashlib.sha256(",".join(str(x) for x in sorted(ids)).encode()).hexdigest()[:8]

    cnt = collections.Counter(h(r["opp"]) for r in recs)
    wins = collections.Counter(h(r["opp"]) for r in recs if r["won"])
    decks = {}
    for r in recs:
        decks.setdefault(h(r["opp"]), r["opp"])

    def sig(ids):
        """构筑签名：去掉基本能量后的卡牌集合（= 真正的"配置"）。"""
        return tuple(sorted(c for c in ids if c not in BASIC))

    sigs = {h: sig(v) for h, v in decks.items()}
    clusters = cluster(sigs, args.jaccard)
    clusters.sort(key=lambda g: -sum(cnt[h] for h in g))

    def label(ids):
        c = collections.Counter(ids)
        n_energy = sum(n for i, n in c.items() if i in BASIC)
        top = sorted(((i, n) for i, n in c.items() if i not in BASIC), key=lambda x: -x[1])[:4]
        txt = " + ".join(f"{names.get(i, i)}×{n}" for i, n in top)
        return f"[{n_energy} 能量] {txt}"

    print(f"\n=== 对手卡组：{len(cnt)} 个 hash → 合并成 {len(clusters)} 簇"
          f"（Jaccard ≥ {args.jaccard}，即允许交换 ~{round(60*(1-args.jaccard))} 张）===")
    print(f"{'簇':<4}{'局数':>5}{'我们胜':>7}{'我们的胜率':>10}{'95% CI':>16}  代表构筑")
    out_rows = []
    for k, grp in enumerate(clusters, 1):
        n = sum(cnt[h] for h in grp)
        w = sum(wins[h] for h in grp)
        lo, hi = wilson(w, n)
        rep = max(grp, key=lambda x: cnt[x])
        out_rows.append((k, grp, rep, n, w))
        if n < args.min_games:
            continue
        print(f"{k:<4}{n:>5}{w:>7}{w/max(n,1):>10.2f}    [{lo:.2f},{hi:.2f}]  "
              f"{rep} · {label(decks[rep])}")
    print(f"  （只显示 ≥{args.min_games} 局的簇；共 {len(clusters)} 簇）")

    # ---- 只看"我们输掉的对局"里对手的构成
    loss = [r for r in recs if r["won"] is False]
    lc = collections.Counter(h(r["opp"]) for r in loss)
    print(f"\n=== 我们输掉的 {len(loss)} 局里，对手卡组 Top10（占败局比例）===")
    for hh, n in lc.most_common(10):
        print(f"  {hh}  {n:>3} 局  {n/max(len(loss),1):>5.1%}  {label(decks[hh])}")

    if args.export and args.top:
        outdir = os.path.join(ROOT, args.export)
        os.makedirs(outdir, exist_ok=True)
        manifest = []
        for k, grp, rep, n, w in out_rows[: args.top]:
            safe = "".join(ch for ch in label(decks[rep])[:40] if ch.isalnum() or ch in " +×")
            fn = f"{rep}__{safe.replace(' ', '_')}.csv"
            path = os.path.join(outdir, fn)
            with open(path, "w") as f:
                f.write("\n".join(str(x) for x in decks[rep]) + "\n")
            manifest.append(dict(cluster=k, rep=rep, hashes=grp, matches=n, our_wins=w,
                                 our_win_rate=(w / max(n, 1)), file=fn, label=label(decks[rep])))
            print(f"  ✅ 导出 {fn}（{n} 局，我们胜率 {w/max(n,1):.2f}）")
        with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"\n清单: {os.path.join(outdir, 'manifest.json')}")


if __name__ == "__main__":
    main()
