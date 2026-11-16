#!/usr/bin/env python3
"""**真实份额口径的面板分**：老 12 格复用方案 A 面板的实测，新格用 real_panel_run 的结果，
长尾用回放观测胜率，按 `decks/testset_meta_real.tsv` 的权重加权。

    ./tools/py tools/real_panel_agg.py candf5f8G candf5f8S
"""
from __future__ import annotations
import glob, json, math, os, sys, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = os.path.join(ROOT, "decks", "testset_meta_real.tsv")
# 老 12 格：结果来自方案 A 面板（同一批对手 agent、同一套格子名）
OLD_TAG = {"candf5f8G": "candf5f8G_A", "candf5f8S": "candf5f8S_A",
           "cand26c6M": "cand26c6M_A", "cand7e39S": "cand7e39S_A", "r1A": "r1A_A"}
OLD_CELLS = {"ogerpon_7e39", "grookey_2617", "abra_e8ee", "abra_f5f8", "champ_0d86",
             "dunsparce_9d6e", "dunsparce_0bf5", "dreepy_3eaa", "dunsparce_db04",
             "champ_0904", "champ_26c6", "champ_e1d7"}


def wilson(k, n, z=1.96):
    if not n: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0, (c - h) / d), min(1, (c + h) / d)


def pooled(paths):
    k = n = 0
    for p in paths:
        try: o = json.load(open(p))["overall"]
        except Exception: continue
        k += o["score"] * o["n"]; n += o["n"]
    return (k / n if n else None), n


def main():
    cands = sys.argv[1:] or ["candf5f8G"]
    cells = []
    tail_w, tail_n = 0.0, 0
    for ln in open(TS):
        if ln.startswith("#") or not ln.strip(): continue
        f = ln.rstrip("\n").split("\t")
        if f[0] == "tail_mix":
            tail_w = float(f[4]); tail_n = 0
            continue
        cells.append(dict(cell=f[0], grp=f[3], wt=float(f[4])))
    # 长尾的观测胜率：来自 scan.jsonl 里不属于任何格的对局
    import hashlib
    TAG = {}
    for p in glob.glob(os.path.join(ROOT, "decks", "pool_field", "*.csv")):
        ids = sorted(int(x) for x in open(p) if x.strip())
        TAG[hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:8]] = os.path.basename(p)[:-4]
    known = {c["cell"].split("_")[-1] for c in cells}
    tk = tn = 0
    for ln in open(os.path.join(ROOT, "logs", "sub_feedback", "scan.jsonl")):
        r = json.loads(ln)
        if not r.get("opp_hash") or r.get("win") is None or r.get("selfmirror"): continue
        short = TAG.get(r["opp_hash"], r["opp_hash"]).split("__")[-1]
        if short in known: continue
        tn += 1; tk += 1 if r["win"] else 0
    tail_wr = (tk / tn) if tn else float("nan")
    print(f"长尾：{tn} 局，观测胜率 {tail_wr*100:.1f}%")
    for cand in cands:
        tot_w = 0.0; tot_s = 0.0
        print(f"\n=== {cand}（真实份额口径）===")
        print(f"{'格':22s}{'权重':>7s}{'来源':>6s}{'胜率':>7s}{'n':>5s}")
        for c in cells:
            tag = OLD_TAG.get(cand)
            paths = []
            if c["cell"] in OLD_CELLS and tag:
                paths += glob.glob(os.path.join(ROOT, "logs", "meta", f"{tag}_s95*", f"vs_{c['cell']}.json"))
            paths += glob.glob(os.path.join(ROOT, "logs", "meta", f"{cand}_real_s95*", f"vs_{c['cell']}.json"))
            wr, n = pooled(paths)
            src = "A面板" if (c["cell"] in OLD_CELLS and wr is not None and n and len(paths) and 'real' not in paths[0]) else "本次"
            if wr is None:
                print(f"{c['cell']:22s}{c['wt']:6.1f}%{'-':>6s}{'NA':>7s}{0:5d}  ← 缺")
                continue
            lo, hi = wilson(round(wr * n), n)
            print(f"{c['cell']:22s}{c['wt']:6.1f}%{src:>6s}{wr*100:6.1f}%{n:5d}  CI[{lo*100:.0f},{hi*100:.0f}]")
            tot_w += c["wt"]; tot_s += c["wt"] * wr
        if tail_n or tail_w:
            tot_w += tail_w; tot_s += tail_w * tail_wr
        print(f"{'合计':22s}{tot_w:6.1f}%{'':>6s}{tot_s/tot_w*100:6.1f}%")
        print("（b05 旧口径参考：candf5f8S 61.3% / candf5f8G 62.7%，n=450）")


if __name__ == "__main__":
    main()
