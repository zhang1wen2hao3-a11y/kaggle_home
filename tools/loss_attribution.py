#!/usr/bin/env python3
"""**输在哪：是我们的强度问题，还是卡组结构性克制？**

三条参照线，同一格三方并列：
  ① 我们（实测回放）：我方胜率
  ② 端口对端口（结构性基线，母表 init-vs-init）：**同牌同对手、只换驾驶员** ⇒ 如果它也赢不了，就是结构问题
  ③ 天梯高手（data/ladder_games.tsv 按对手卡组聚合）：真人/强 agent 打这副牌时，对手的胜率

判定规则：
  · 我们 ≥ ②            → 比端口基线还强，不算洞
  · 我们 < ②，且 ③ 也 < ② − 3pp  → **结构性克制**（谁来都输）：该换牌/改构筑，不是训驾驶员
  · 我们 < ②，但 ③ ≥ ② − 3pp    → **我们的驾驶员弱于标准水平**：该训我们
  · 无 ② 参照（新格）→ 用 ③ 对照：我们 ≪ ③ ⇒ 我方偏弱；≈ ⇒ 无法判断

    ./tools/py tools/loss_attribution.py --scan logs/scan_arms.jsonl --our-deck f5f8
    ./tools/py tools/loss_attribution.py --scan logs/scan_arms.jsonl --our-deck 9d6e
"""
from __future__ import annotations
import argparse, collections, csv, glob, hashlib, json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-in-table.csv")
POOL = os.path.join(ROOT, "decks", "pool_field")


def pool_map():
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
    ap.add_argument("--scan", default="logs/scan_arms.jsonl")
    ap.add_argument("--our-deck", required=True, help="我方卡组 4 位后缀（f5f8 / 9d6e / 0bf5 …）")
    ap.add_argument("--min-n", type=int, default=3)
    a = ap.parse_args()
    m = pool_map()

    # ① 我方回放
    ours = collections.Counter(); ow = collections.Counter()
    for ln in open(os.path.join(ROOT, a.scan)):
        try: r = json.loads(ln)
        except Exception: continue
        if not r.get("opp_hash") or r.get("win") is None or r.get("selfmirror"): continue
        # ★ 只统计"我们开这副牌"的对局（不同提交臂的卡组不同，混在一起会算错）
        if (m.get(r.get("our_hash") or "") or "").split("__")[-1] not in (a.our_deck, ""):
            continue
        short = (m.get(r["opp_hash"]) or r["opp_hash"]).split("__")[-1]
        ours[short] += 1; ow[short] += 1 if r["win"] else 0

    # ② 端口对端口（结构性）
    struct = {}
    txt = open(MASTER, "rb").read().decode("gb18030", errors="replace")
    for r in csv.DictReader(txt.splitlines()):
        if r["挑战者"] == a.our_deck and r["驾驶员"] == "init" and (r.get("对手驾驶员") or "init") == "init":
            struct[r["被挑战者"]] = (float(r["胜率"]), int(float(r["样本场次"])))

    # ③ 天梯高手：对手胜率 = 1 - 该卡的胜率
    hi = collections.Counter(); hiw = collections.Counter()
    for r in csv.DictReader(open(os.path.join(ROOT, "data", "ladder_games.tsv")), delimiter="\t"):
        try: win = int(r["winner"])
        except (TypeError, ValueError): continue
        for i in (0, 1):
            d = (r.get(f"seat{i}_deck") or "")[:4]
            if not d: continue
            hi[d] += 1
            if win == i: hiw[d] += 1

    rows = []
    for short, n in ours.items():
        if n < a.min_n: continue
        wr = ow[short] / n
        s = struct.get(short)
        s_wr = s[0] if s else None
        hk, hn = 0, 0
        for k4 in hi:
            if (k4 or "")[:4] == short:
                hk += hiw[k4]; hn += hi[k4]
        hi_opp = (1 - hk / hn) if hn else None      # 高手视角：打这副牌的胜率
        verdict = ""
        if s_wr is not None:
            if wr >= s_wr:
                verdict = "不比端口差"
            elif hi_opp is not None and hi_opp < s_wr - 0.03:
                verdict = "**结构性克制**（换牌/构筑，别训）"
            else:
                verdict = "我方偏弱（该训我们）"
        elif hi_opp is not None and hn >= 15:
            verdict = "我方偏弱（无端口参照）" if wr < hi_opp - 0.08 else "接近高手（无端口参照）"
        loss = n * (1 - wr)
        rows.append(dict(opp=short, n=n, wr=wr, struct=s_wr, hi=hi_opp, hn=hn,
                         verdict=verdict, loss=loss))
    rows.sort(key=lambda r: -r["loss"])
    print(f"== 我方卡组 {a.our_deck}：共 {sum(r['n'] for r in rows)} 局（n≥{a.min_n} 的格）")
    print(f"{'对手卡组':12s}{'局数':>5s}{'我方':>7s}{'端口基线':>9s}{'高手对手胜率':>12s}{'n':>5s}  判定")
    for r in rows:
        print(f"{r['opp']:12s}{r['n']:5d}{r['wr']*100:6.1f}%"
              f"{('NA' if r['struct'] is None else f'{r[chr(115)+chr(116)+chr(114)+chr(117)+chr(99)+chr(116)]*100:.1f}%'):>9s}"
              f"{('NA' if r['hi'] is None else f'{r[chr(104)+chr(105)]*100:.1f}%'):>12s}{r['hn']:5d}  {r['verdict']}")
    out = os.path.join(ROOT, "docs", f"输在哪_{a.our_deck}.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["对手卡组", "局数", "我方胜率", "端口基线", "高手对手胜率", "高手n", "判定", "丢局数"])
        for r in rows:
            w.writerow([r["opp"], r["n"], f"{r['wr']*100:.1f}%",
                        "" if r["struct"] is None else f"{r['struct']*100:.1f}%",
                        "" if r["hi"] is None else f"{r['hi']*100:.1f}%",
                        r["hn"], r["verdict"], round(r["loss"], 1)])
    print(f"\n→ {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
