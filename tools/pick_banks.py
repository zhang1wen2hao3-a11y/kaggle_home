#!/usr/bin/env python3
"""**陪练槽位分配器**：按「天梯场权重 + 梯度实测」自动生成 BANKS / BANKDECKS / LEAGUE_PCT。

为什么要有它
------------
2026-11-15 那次 f5f8 特攻，陪练池是我手工拼的，结果第 4 个槽位塞了一个被 learner 打 **100%**
的 BC 克隆（等于浪费 12.5% 的训练行）。问题不在"哪个模型最强"，而在**槽位分配没有规则**：
  · 槽位上限 8 个（显存），对手牌有 11 副 ⇒ 必须按天梯权重取舍
  · 陪练强度必须落在"有梯度"区间：learner 胜率 **>95% 无梯度**（白给），**<55% 又太难**
  · 同一副牌最多给 2 个槽位（要的是风格多样性，不是同一个人的复读）

规则（写死，跑之前不用再商量）
------------------------------
1. 候选 = `tools/driver_registry.py` 里登记的驾驶员（有 deck 的），排除 learner 自己的卡组
   （镜像交给自对弈，不占槽位）
2. 每个候选的评分 = 天梯场权重（默认顶分段 `decks/weights_top.txt`）× 梯度系数
   梯度系数：胜率 55%~92% → 1.0；92%~95% → 0.5；>95% → **0（直接排除）**；<55% → 0.8（难关，保留）
3. 贪心填槽：按评分从高到低，同卡组最多 2 个；填满 `--max-banks` 或候选用完为止
4. `LEAGUE_PCT = LEAGUE_TOTAL / 槽位数`（默认 LEAGUE_TOTAL=0.5；>0.5 会变成"打固定 bot"）

用法
----
    # 1) 先量梯度（learner vs 每个候选陪练，200 场 Bo3 就够）
    ./tools/py tools/bank_ladder_probe.py --anchor-deck slowpoke_slowking__26c6 ...   # 或手工填
    # 2) 生成槽位
    ./tools/py tools/pick_banks.py --anchor slowpoke_slowking__26c6 --ladder logs/bank_ladder.tsv
    ./tools/py tools/pick_banks.py --anchor slowpoke_slowking__26c6 --emit   # 只打 export 行
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import driver_registry as REG          # noqa: E402

TOPW = os.path.join(ROOT, "decks", "weights_top.txt")


def load_weights(path):
    w = {}
    for ln in open(path):
        t = ln.split()
        if len(t) == 2 and "__" in t[0]:
            w[t[0]] = float(t[1])
    tot = sum(w.values()) or 1.0
    return {k: v / tot for k, v in w.items()}


def load_ladder(path):
    """ladder 文件：`名字<TAB>learner胜率[<TAB>n]`，也可写成 csv。"""
    out = {}
    if not path or not os.path.exists(path):
        return out
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.replace(",", "\t").split("\t")
        if len(parts) < 2:
            continue
        try:
            out[parts[0].strip()] = float(parts[1])
        except ValueError:
            continue
    return out


def grad_factor(winrate):
    if winrate is None:
        return 0.6, "未测（按中等估）"
    if winrate > 0.95:
        return 0.0, "无梯度(>95%)"
    if winrate > 0.92:
        return 0.5, "偏弱(92~95%)"
    if winrate >= 0.55:
        return 1.0, "理想(55~92%)"
    return 0.8, f"难关(<55%)"


def threat_of(path, anchor_short):
    """从 ladder_games.tsv 算：各对手牌打 anchor 的胜率（= 威胁度）与样本数。"""
    import csv
    out = {}
    if not path or not os.path.exists(path):
        return out
    agg = collections.defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open(path), delimiter="\t"):
        d0 = (r.get("seat0_deck") or "")[:4]
        d1 = (r.get("seat1_deck") or "")[:4]
        w = r.get("winner")
        if w not in ("0", "1"):
            continue
        if d0 == anchor_short and d1 and d1 != anchor_short:
            agg[d1][0] += 1 if w == "1" else 0
            agg[d1][1] += 1
        elif d1 == anchor_short and d0 and d0 != anchor_short:
            agg[d0][0] += 1 if w == "0" else 0
            agg[d0][1] += 1
    for k, (win, n) in agg.items():
        if n >= 8:
            out[k] = win / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True, help="learner 的卡组（如 slowpoke_slowking__26c6）")
    ap.add_argument("--ladder", default="logs/bank_ladder.tsv", help="梯度实测表")
    ap.add_argument("--weights", default=TOPW, help="场权重文件（默认顶分段）")
    ap.add_argument("--max-banks", type=int, default=4)
    ap.add_argument("--league-total", type=float, default=0.5)
    ap.add_argument("--emit", action="store_true", help="只输出 BANKS/BANKDECKS/LEAGUE_PCT 三行")
    ap.add_argument("--per-deck-max", type=int, default=2,
                    help="同一副牌最多占几个槽（默认 2；想多拿几份风格不同的强对手时调到 3）")
    ap.add_argument("--min-score-ratio", type=float, default=0.02,
                    help="评分低于最高分的这个比例就不占槽（宁可用少一点 bank，也不塞没用的）")
    ap.add_argument("--must", default="",
                    help="必进池的卡组短名（逗号分隔）：已知的克星牌，哪怕场份额低也要占槽")
    ap.add_argument("--ladder-games", default="data/ladder_games.tsv",
                    help="天梯对局档案：用来算「哪些牌是 anchor 的克星」（威胁度）")
    a = ap.parse_args()

    W = load_weights(a.weights)
    lad = load_ladder(a.ladder)
    anchor_short = a.anchor.split("__")[-1][:4]
    threat = threat_of(a.ladder_games, anchor_short)

    # 1) 候选：登记的驾驶员，且不是 learner 自己的卡组
    cands = []
    for name, d in REG.DRIVERS.items():
        if name == "init" or not d.get("agent") or not d.get("deck"):
            continue
        if d["deck"] == anchor_short:            # 镜像交给自对弈
            continue
        deck_full = next((k for k in W if k.endswith("__" + d["deck"])), None)
        w = W.get(deck_full, 0.0) if deck_full else 0.0
        wr = lad.get(name)
        gf, why = grad_factor(wr)
        # 威胁度：这副牌在天梯上打 anchor 的胜率（>0.5 说明是克星，必须进池）
        thr = threat.get(d["deck"])
        thr_f = 1.0 if thr is None else (0.5 + max(0.0, thr - 0.5) * 2.0)   # 0.5→1.0, 0.84→1.68
        cands.append(dict(name=name, deck=d["deck"], deck_full=deck_full or d["deck"],
                          weight=w, wr=wr, gf=gf, why=why, threat=thr, thr_f=thr_f,
                          score=w * gf * thr_f + (0.02 if (thr or 0) >= 0.6 else 0.0),
                          bin=d.get("remote") or "-"))
    must = [x.strip() for x in a.must.split(",") if x.strip()]
    for c in cands:                      # 必进池的牌，评分抬到最高一档
        if c["deck"] in must and c["gf"] > 0:
            c["score"] = 1e6 + c["score"]
            c["why"] += " ·必进"
    cands.sort(key=lambda c: -c["score"])

    # 2) 贪心填槽（同卡组 ≤2）
    per_deck = collections.Counter()
    picked = []
    for c in cands:
        if len(picked) >= a.max_banks:
            break
        if c["gf"] <= 0:
            continue
        if per_deck[c["deck"]] >= a.per_deck_max:
            continue
        per_deck[c["deck"]] += 1
        picked.append(c)

    # 太弱的候选不占槽（否则等于把冻结对手的比例浪费掉）
    if picked:
        top = picked[0]["score"] if picked[0]["score"] < 1e5 else max(c["score"] for c in cands if c["score"] < 1e5)
        picked = [c for c in picked if c["score"] >= top * a.min_score_ratio or c["score"] >= 1e5]
    if not picked:
        sys.exit("✗ 没有任何候选通过梯度筛选（先跑梯度实测，或放宽 --ladder）")

    pct = round(a.league_total / len(picked), 4)
    bankdecks = [c["deck_full"] for c in picked]
    banks = [c["bin"] for c in picked]

    if a.emit:
        print("BANKS=" + ",".join(banks))
        print("BANKDECKS=" + ",".join(bankdecks))
        print(f"LEAGUE_PCT={pct}")
        return 0

    print(f"A 的卡组 = {a.anchor}   槽位 {len(picked)}/{a.max_banks}   冻结对手占 {a.league_total:.0%} 行"
          f"（每槽 {pct:.3f}）\n")
    print(f"{'驾驶员':20}{'卡组':30}{'场权重':>8}{'克星度':>7}{'learner胜率':>11}  判定")
    for c in picked:
        wr = f"{c['wr']*100:.1f}%" if c["wr"] is not None else "—"
        th = f"{c['threat']*100:.0f}%" if c["threat"] is not None else "—"
        print(f"{c['name']:20}{c['deck_full']:30}{c['weight']*100:7.1f}%{th:>7}{wr:>11}  {c['why']}")
    skipped = [c for c in cands if c["gf"] <= 0]
    if skipped:
        print("\n被排除（无梯度）：" + ", ".join(f"{c['name']}({c['wr']*100:.0f}%)" for c in skipped
                                          if c["wr"] is not None))
    unknown = [c for c in cands if c["wr"] is None]
    if unknown:
        print(f"未测梯度的候选 {len(unknown)} 个（按中等估）："
              + ", ".join(c["name"] for c in unknown[:8]) + " …")
    print("\n" + "=" * 70)
    print("BANKS=" + ",".join(banks))
    print("BANKDECKS=" + ",".join(bankdecks))
    print(f"LEAGUE_PCT={pct}")
    by_deck = collections.Counter(c["deck"] for c in picked)
    print(f"⇒ 冻结对手占 {a.league_total:.0%}： "
          + " / ".join(f"{k} {v} 槽（{v*pct*100:.1f}% 行）" for k, v in by_deck.items()))
    hdr = ""
    try:
        hdr = open(a.ladder, encoding="utf-8").readline().strip("# \n")
    except OSError:
        pass
    if hdr:
        print(f"（梯度表来源：{hdr} —— 换了 learner 必须重量，否则梯度系数是搬来的）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
