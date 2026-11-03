#!/usr/bin/env python3
"""场地加权期望胜率：给定「对手卡组分布」，算某副牌在真实天梯的**期望胜率**。

数据来源（严格标注每格来源与样本量，不混用不标注）：
  * P(opp)          ← 我们自己 640 局的对手分布（= 我们当前评分段实际遇到的场地）
  * W(cand vs opp)  ← 优先用**我们自己的真实对局**；没有时退回**真人 vs 真人档案**（1755 局）

用法：./tools/py tools/field_ev.py
"""
import collections, csv, glob, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELD_ORDER = ["teal_mask_ogerpon_ex_applin", "abra_kadabra", "grookey_thwackey",
               "dunsparce_abra", "dreepy_drakloak", "shuppet_dhelmise",
               "slowpoke_slowking", "beldum_metang"]


def arche(name):
    return (name or "?").split("__")[0] or "?"


def load_our():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "our_games.tsv"), encoding="utf-8"),
                              delimiter="\t"))
    field = collections.Counter()
    w = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        o = arche(r["opp_deck"])
        if o == "?":
            continue
        field[o] += 1
        k = (arche(r["our_deck"]), o)
        w[k][0] += 1
        w[k][1] += int(r["result"])
    return field, w


def load_archive():
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "data", "ladder_games.tsv"), encoding="utf-8"),
                                      delimiter="\t") if r["result_known"] == "1"]
    fam = {}
    for f in glob.glob(os.path.join(ROOT, "decks", "pool_field", "*.csv")):
        n = os.path.basename(f)[:-4]
        fam[n.split("__")[-1]] = n.split("__")[0]
    w = collections.defaultdict(lambda: [0, 0])
    field = collections.Counter()
    for r in rows:
        a = fam.get(r["seat0_deck"][:4], "?")
        b = fam.get(r["seat1_deck"][:4], "?")
        win0 = 1 if r["winner"] == "0" else 0
        field[b] += 1
        w[(a, b)][0] += 1
        w[(a, b)][1] += win0
        w[(b, a)][0] += 1
        w[(b, a)][1] += (1 - win0)
    return field, w


def main():
    field_ours, w_ours = load_our()
    field_arch, w_arch = load_archive()
    tot = sum(field_ours.values())
    print(f"场地分布（来自我们 {tot} 场真实对局）：")
    for a, n in field_ours.most_common():
        if a in FIELD_ORDER or n >= 8:
            print(f"   {a[:30]:32s} {n:4d}  {n/tot:6.1%}")
    print()

    cands = ["dunsparce_abra", "abra_kadabra", "slowpoke_slowking",
             "grookey_thwackey", "teal_mask_ogerpon_ex_applin", "dreepy_drakloak"]
    print("期望胜率 = Σ P(对手) × W(我这副 vs 那个对手)")
    print("每格标注来源：o=我们自己的实地对局，a=真人vs真人档案；n 为样本量\n")
    print(f"{'候选卡组':28s}{'期望胜率':>9}  明细")
    results = []
    for c in cands:
        ev, parts, missing = 0.0, [], 0.0
        for o, n in field_ours.most_common():
            p = n / tot
            cell_o = w_ours.get((c, o), [0, 0])
            cell_a = w_arch.get((c, o), [0, 0])
            if cell_o[0] >= 10:
                rate, src = cell_o[1] / cell_o[0], f"o{cell_o[0]}"
            elif cell_a[0] >= 20:
                rate, src = cell_a[1] / cell_a[0], f"a{cell_a[0]}"
            else:
                missing += p
                continue
            ev += p * rate
            parts.append(f"{o[:12]}={rate:.2f}[{src}]({p:.0%})")
        results.append((c, ev, missing, parts))
    for c, ev, miss, parts in sorted(results, key=lambda x: -x[1]):
        print(f"{c[:27]:28s}{ev:>9.3f}  {' '.join(parts[:5])}")
        if miss > 0.01:
            print(f"{'':28s}{'':>9}  ⚠️ {miss:.0%} 的场地没有可靠对位数据（未计入）")
    print("\n注：db04(1071.4) 与 9d6e(1048.5) 是 dunsparce_abra 的两个不同构筑，此处按原型合并；")
    print("    构筑级差异需另行拆分（实测 db04 打 ogerpon 30/30，9d6e 32/37）。")


if __name__ == "__main__":
    main()
