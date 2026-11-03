#!/usr/bin/env python3
"""给 BC 帧打标签：**语义类别 + 该帧出牌玩家的天梯战绩**，供 bc_kit 做"按玩家过滤 / 按类别加权"。

两个标签都**不需要重放**：

1. **语义类别** ← 直接从前向行里读：
   `opt_int[j, 0] = option_type`（encoder.py:325），
   offset 见 `ptcg.rl.encoder.OFFSETS["opt_int"]`（当前 (9760, 10144)，OPT_INT=6）
   ⇒ `row[9760 + y*6 + 0]` 就是专家动作的类型。已验证分布正常
   （f5f8: play 32.2% / card 24.2% / evolve 10.9% / attach 8.1% / ability 8.0% / attack 5.1% / end 3.0%）

2. **玩家战绩** ← 从 `data/ladder_games.tsv`（1755 局真人 vs 真人）算每个 agent 的胜率，
   用 Wilson 收缩（向 0.5 收敛）避免小样本玩家被高估/低估。

输出 `data/bc_frame_labels.tsv`：frame, deck, agent, category, n_opt, player_wr, player_n

用法：
    ./tools/py tools/bc_frame_labels.py                      # 全部 529,188 帧
    ./tools/py tools/bc_frame_labels.py --deck f5f8 --min-opt 2
"""
import argparse
import collections
import csv
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "build", "port_plain"))
import numpy as np                                   # noqa: E402
import ptcg.rl.encoder as E                          # noqa: E402

CAT = {7: "play", 8: "attach", 9: "evolve", 10: "ability", 11: "discard",
       12: "retreat", 13: "attack", 14: "end", 15: "skill", 3: "card",
       4: "tool", 5: "energy_card", 6: "energy", 16: "special", 17: "stop",
       0: "number", 1: "yes", 2: "no"}


def player_ratings(prior=0.5, k=20.0):
    """按 agent 统计真人天梯战绩（Wilson 收缩）。"""
    st = collections.defaultdict(lambda: [0, 0])
    path = os.path.join(ROOT, "data", "ladder_games.tsv")
    if not os.path.exists(path):
        return {}
    for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"):
        if r.get("result_known") != "1":
            continue
        w = r.get("winner")
        for seat, ag in ((0, r.get("seat0_agent")), (1, r.get("seat1_agent"))):
            if not ag:
                continue
            st[ag][0] += 1
            st[ag][1] += 1 if w == str(seat) else 0
    out = {}
    for ag, (n, w) in st.items():
        out[ag] = ((w + k * prior) / (n + k), n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(ROOT, "data", "bc_all.npz"))
    ap.add_argument("--deck", default="", help="只看某副牌（前4位），空=全部")
    ap.add_argument("--min-opt", type=int, default=2, help="只要 n_opt>=该值的帧（有真实选择）")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "bc_frame_labels.tsv"))
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    deck = z["deck"].astype(str)
    y = z["y"].astype(int)
    nopt = z["nopt"].astype(int)
    agent = z["agent"].astype(str)
    off0, _ = E.OFFSETS["opt_int"]
    W = E.OPT_INT
    X = np.memmap(os.path.splitext(a.npz)[0] + ".f32", dtype=np.float32, mode="r",
                  shape=(len(y), 10944))
    pr = player_ratings()
    print(f"玩家战绩表 {len(pr)} 个 agent（来自 ladder_games.tsv）")

    sel = np.ones(len(y), dtype=bool)
    if a.deck:
        sel &= np.char.startswith(deck, a.deck)
    if a.min_opt > 1:
        sel &= nopt >= a.min_opt
    idx = np.where(sel)[0]
    cats = collections.Counter()
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["frame", "deck", "agent", "category", "n_opt", "player_wr", "player_n"])
        for i in idx:
            t = int(X[i][off0 + y[i] * W + 0])
            c = CAT.get(t, f"type{t}")
            cats[c] += 1
            wr, n = pr.get(agent[i], (0.5, 0))
            w.writerow([int(i), deck[i], agent[i], c, int(nopt[i]), f"{wr:.4f}", n])
    print(f"→ {a.out}（{len(idx)} 帧）")
    tot = sum(cats.values()) or 1
    for c, v in cats.most_common():
        print(f"   {c:12s}{v:8d}  {v/tot:6.1%}")


if __name__ == "__main__":
    main()
