#!/usr/bin/env python3
"""从 episode 回放**重建真实场上的卡组**（近似）。

为什么能做：回放不揭示隐藏牌，但**弃牌堆是公开区**，场面/附着也可见。
所以一局里能看到对手 18~24 种、20~40 张牌。按"单步内某张牌出现的最多次数"
估计它的张数（下限，封顶 4），余下的位置用该队主属性基本能量补齐。

产出的卡组是**近似**，用途是当**评估对手池**，不是用来提交。
（真实卡表拿不到 —— 这是数据授权与回放机制共同决定的。）

用法：
    ./run.sh tools/reconstruct_field_decks.py --dir epsoide_field --out decks/field
    ./run.sh tools/reconstruct_field_decks.py --dir epsoide --out decks/field --min-games 1
"""
import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUR_NAME = "zhang1wen2hao3"
ENERGY_BY_TYPE = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}  # 草火水雷超斗恶钢


def card_ids(seq):
    return [c.get("id") for c in (seq or []) if isinstance(c, dict) and c.get("id") is not None]


def step_counts(p):
    """单步内某张牌的**出现次数**：弃牌堆 + 场面 + 附着（这些都是公开信息）。"""
    cnt = collections.Counter()
    for c in card_ids(p.get("discard")):
        cnt[c] += 1
    for pk in list(p.get("active") or []) + list(p.get("bench") or []):
        if not pk:
            continue
        cnt[pk.get("id")] += 1
        for k in ("energyCards", "tools", "preEvolution"):
            for c in card_ids(pk.get(k)):
                cnt[c] += 1
    return cnt


def deck_of(d, player):
    """跨所有步取每张牌的**最大出现次数**，作为张数下限。"""
    best = collections.Counter()
    for st in d.get("steps") or []:
        for a in st:
            cur = (a.get("observation") or {}).get("current") or {}
            pls = cur.get("players") or []
            if len(pls) < 2 or not pls[player]:
                continue
            for cid, n in step_counts(pls[player]).items():
                if n > best[cid]:
                    best[cid] = n
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="epsoide_field")
    ap.add_argument("--out", default="decks/field")
    ap.add_argument("--skip-ours", action="store_true", default=True,
                    help="跳过我们自己（zhang1wen2hao3）的卡组")
    ap.add_argument("--top", type=int, default=12, help="只输出出现最多的前 N 套")
    ap.add_argument("--min-cards", type=int, default=18,
                    help="可见牌少于这个数就不重建（信息太少）")
    args = ap.parse_args()

    from framework.deck import CardPool, is_legal
    pool = CardPool()

    def type_of(cid):
        try:
            c = pool.by_id.get(int(cid))
            return int(getattr(c, "energyType", 0)) if c is not None else 0
        except Exception:
            return 0

    files = sorted(glob.glob(os.path.join(ROOT, args.dir, "*.json")))
    if not files:
        raise SystemExit(f"{args.dir} 下没有 json")
    print(f"共 {len(files)} 个回放，开始重建 …")

    decks = collections.Counter()
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        names = [a.get("Name") or "" for a in ((d.get("info") or {}).get("Agents") or [])]
        if len(names) < 2:
            continue
        for pi, nm in enumerate(names):
            if args.skip_ours and OUR_NAME in nm:
                continue
            cnt = deck_of(d, pi)
            n_cards = sum(min(n, 4) if not pool.is_basic_energy(c) else n
                          for c, n in cnt.items())
            if n_cards < args.min_cards:
                continue
            key = tuple(sorted(cnt.items()))
            decks[key] += 1

    print(f"  去重后 {len(decks)} 种对手卡组\n")
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    made = 0
    for (key, freq) in decks.most_common(args.top):
        cnt = dict(key)
        # 张数封顶 4（基本能量不限）
        cards = []
        for cid, n in sorted(cnt.items(), key=lambda t: -t[1]):
            k = n if pool.is_basic_energy(cid) else min(n, 4)
            cards += [cid] * k
        # 主属性基本能量补齐
        types = collections.Counter(type_of(c) for c in cards
                                    if not pool.is_basic_energy(c) and type_of(c) in ENERGY_BY_TYPE)
        etype = types.most_common(1)[0][0] if types else 3
        energy_id = ENERGY_BY_TYPE[etype]
        if len(cards) < 60:
            cards += [energy_id] * (60 - len(cards))
        cards = cards[:60]
        ok, why = is_legal(pool, cards)
        mark = "✅" if ok else f"❌ {why}"
        print(f"  freq={freq:<3} 可见 {len(cnt)} 种 / {len(cards)} 张  "
              f"能量用 id={energy_id}  {mark}")
        if ok:
            made += 1
            p = os.path.join(out_dir, f"field_{made:02d}.csv")
            with open(p, "w") as fh:
                fh.write("\n".join(map(str, cards)) + "\n")
    print(f"\n写出 {made} 套到 {os.path.relpath(out_dir, ROOT)}/")


if __name__ == "__main__":
    main()
