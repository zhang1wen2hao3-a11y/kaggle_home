#!/usr/bin/env python3
"""**逐格差距诊断**：把"我们 vs 高分玩家"在同一对手卡组上的胜率差算出来，
并判断这个差是"我们偏弱"还是"我们的测试对手偏弱（需要精修对手）"。

为什么要这张表（2026-11-17 用户要求）：
  · 面板里的对手是**端口权重**开的牌——它可能远弱于天梯上的真人驾驶员；
  · 若我方胜率**明显高于**真人高手，说明对手太弱 ⇒ 面板高估了我们的实力，该**精修这个对手**；
  · 若我方胜率**明显低于**真人高手，说明是我们自己弱 ⇒ 该训我们的驾驶员。
两者要分开，否则会把"测量偏差"当成"我们很强"。

数据源：
  · 我方：`logs/meta/<cand>_real_s95*/vs_<cell>.json`（面板结果，自动池化）
  · 高手：`data/ladder_games.tsv`（1756 局真人档案，按"对手卡组"聚合）
  · 份额：`decks/testset_meta_real.tsv`
  · 对手驾驶员：`decks/testset_meta_real.tsv` 的第 6 列（init=端口 / 其它=训练件）

    ./tools/py tools/cell_gap_diagnosis.py --cand cand0bf5S
    ./tools/py tools/cell_gap_diagnosis.py --cand candf5f8E --min-hi 30
"""
from __future__ import annotations
import argparse, collections, csv, glob, hashlib, json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = os.path.join(ROOT, "decks", "pool_field")
TS = os.path.join(ROOT, "decks", "testset_meta_real.tsv")


def pool_short():
    m = {}
    for p in glob.glob(os.path.join(POOL, "*.csv")):
        ids = sorted(int(x) for x in open(p) if x.strip())
        h8 = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:8]
        m[h8] = os.path.basename(p)[:-4]
        m.setdefault(h8[:4], os.path.basename(p)[:-4])
    return m


def wilson(k, n, z=1.96):
    if not n: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0, (c - h) / d), min(1, (c + h) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--min-hi", type=int, default=20, help="高手样本低于此值不判定")
    ap.add_argument("--thresh", type=float, default=10.0, help="|差| 超过此 pp 才判定")
    a = ap.parse_args()
    m = pool_short()

    # ① 测试集：份额 + 对手驾驶员
    share, oppdrv = {}, {}
    for ln in open(TS):
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.rstrip("\n").split("\t")
        if f[0] == "tail_mix":
            continue
        share[f[0]] = float(f[4])
        oppdrv[f[0]] = (f[5] if len(f) > 5 else "init").strip() or "init"

    # ② 我方面板
    ours = {}
    for cell in share:
        k = n = 0
        for p in glob.glob(os.path.join(ROOT, "logs", "meta", f"{a.cand}_real_s95*", f"vs_{cell}.json")):
            o = json.load(open(p))["overall"]
            k += o["score"] * o["n"]; n += o["n"]
        if n:
            ours[cell] = (k / n, n)

    # ③ 高手：按对手卡组聚合
    hi = collections.Counter(); hiw = collections.Counter()
    for r in csv.DictReader(open(os.path.join(ROOT, "data", "ladder_games.tsv")), delimiter="\t"):
        try: win = int(r["winner"])
        except (TypeError, ValueError): continue
        for i in (0, 1):
            theirs = r.get(f"seat{1-i}_deck") or ""
            if not theirs:
                continue
            key = (theirs or "")[:4]        # 档案里是 8 位 hash，取前 4 位与测试集格名对齐
            hi[key] += 1
            if win == (1 - i):          # 对手（我方视角）获胜 = 该卡组赢
                hiw[key] += 1

    # ④ 投资分（补强优先级）：读 docs/卡组天花板与投资分.csv，按 4 位后缀索引
    inv = {}
    invp = os.path.join(ROOT, "docs", "卡组天花板与投资分.csv")
    if os.path.exists(invp):
        for r in csv.DictReader(open(invp, encoding="utf-8-sig")):
            try:
                inv[(r["卡组"] or "").strip()] = float(r["投资分"])
            except (TypeError, ValueError):
                pass

    rows = []
    for cell, (wr, n) in ours.items():
        short = cell.split("_")[-1]
        # 高手胜率：按 4 位后缀匹配（ladder 档案里的 deck tag 前 4 位）
        hk = hn = 0
        for k4, cnt in hi.items():
            if (k4 or "")[:4] == short:
                hk += hiw[k4]; hn += cnt
        hwr = (hk / hn) if hn else None
        drv = oppdrv.get(cell, "init")
        verdict, prio = "", 0.0
        if hwr is not None and hn >= a.min_hi and hwr > 0:
            delta = (wr - (1 - hwr)) * 100   # ↑ 说明"我们比真人打得还好" = 对手偏弱
            better = (wr > 1 - hwr)
            if delta >= a.thresh:
                verdict = "对手偏弱（该精修对手）"
                prio = share[cell] * delta / 100.0
            elif delta <= -a.thresh:
                verdict = "我方偏弱（该训我们）"
                prio = share[cell] * (-delta) / 100.0
            else:
                verdict = "接近"
        else:
            delta = None
            verdict = "高手样本不足"
        inval = inv.get(short)
        # 综合优先级 = 份额 × 差 × (投资分/100)；没投资分的格按 1.0 处理
        prio2 = prio * ((inval / 100.0) if inval else 1.0)
        rows.append(dict(cell=cell, share=share[cell], wr=wr, n=n,
                         hwr=(1 - hwr) if hwr is not None else None, hn=hn,
                         delta=delta, drv=drv, verdict=verdict, prio=prio,
                         inv=inval, prio2=prio2))
    rows.sort(key=lambda r: -r["share"])
    out = os.path.join(ROOT, "docs", f"格子差距诊断_{a.cand}.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["格", "份额%", "我方胜率%", "我方n", "对手胜率%(高手侧)", "高手n",
                    "差pp(我方-对手)", "测试对手驾驶员", "判定", "精修优先级(份额×差)",
                    "该卡组投资分", "综合优先级(×投资分)"])
        for r in rows:
            w.writerow([r["cell"], round(r["share"], 2),
                        round(r["wr"] * 100, 1), r["n"],
                        "" if r["hwr"] is None else round(r["hwr"] * 100, 1), r["hn"],
                        "" if r["delta"] is None else round(r["delta"], 1),
                        r["drv"], r["verdict"], round(r["prio"], 3),
                        "" if r["inv"] is None else round(r["inv"], 1), round(r["prio2"], 3)])
    print(f"→ {os.path.relpath(out, ROOT)}\n")
    print(f"{'格':18s}{'份额':>6s}{'我方':>7s}{'n':>4s}{'高手对手胜率':>11s}{'n':>5s}{'差pp':>7s}  判定")
    for r in rows:
        print(f"{r['cell']:18s}{r['share']:5.1f}%{r['wr']*100:6.1f}%{r['n']:4d}"
              f"{('' if r['hwr'] is None else f'{r[chr(104)+chr(119)+chr(114)]*100:.1f}%'):>11s}{r['hn']:5d}"
              f"{('' if r['delta'] is None else f'{r[chr(100)+chr(101)+chr(108)+chr(116)+chr(97)]:+.1f}'):>7s}  {r['verdict']}")
    print("\n需要精修对手的格子（按 份额×差 排序）：")
    ref = [r for r in rows if r["verdict"].startswith("对手偏弱")]
    ref.sort(key=lambda r: -r["prio"])
    for r in sorted(ref, key=lambda x: -x["prio2"])[:6]:
        iv = "—" if r["inv"] is None else f"{r['inv']:.1f}"
        print(f"  {r['cell']:18s} 份额 {r['share']:.1f}%  我们 {r['wr']*100:.1f}% vs 高手 {r['hwr']*100:.1f}%"
              f"  差 {r['delta']:+.1f}pp  投资分 {iv}  综合 {r['prio2']:.3f}  现用对手={r['drv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
