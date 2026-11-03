#!/usr/bin/env python3
"""从 `weights_real.txt` 生成**定向野区权重**：把某类卡组的对手座位概率抬上去。

用途：专精训练。我们这副牌（9d6e）的 `Powerful Hand` 是"放置伤害指示物"=招式效果，
被 `Hide 'n' Sneak`（shuppet 族）全额免疫，真实天梯对 shuppet **0/16**。
常规采样里 shuppet 只占 8.4% ⇒ 训练里它几乎不值得学。把它的采样概率抬到 ~65%，
学习器就被迫反复进入那个必输的对位。

    ./tools/py tools/make_deck_weights.py --boost-prefix shuppet --boost 20 \\
        --out decks/weights_shuppet.txt

输出格式与训练器一致（`version 3` / `default` / `名字 权重`），并对全部条目重新归一化。
"""
import argparse
import collections
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_weights(path):
    head, items = {}, collections.OrderedDict()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] in ("version", "default"):
            head[parts[0]] = parts[1]
            continue
        items[parts[0]] = float(parts[1])
    return head, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=os.path.join(ROOT, "port_repo/decks/weights_real.txt"))
    ap.add_argument("--boost-prefix", default="shuppet",
                    help="要抬权重的卡组名前缀（逗号分隔可给多个）")
    ap.add_argument("--boost", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    head, items = read_weights(a.src)
    prefs = [p.strip() for p in a.boost_prefix.split(",") if p.strip()]
    boosted, tot, tot_b = [], 0.0, 0.0
    for name, w in items.items():
        hit = any(name.startswith(p) for p in prefs)
        w2 = w * a.boost if hit else w
        items[name] = w2
        tot += w2
        if hit:
            boosted.append(name)
            tot_b += w2
    if not boosted:
        raise SystemExit(f"✗ 没有卡组匹配前缀 {prefs}")
    for name in items:
        items[name] /= tot
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(f"version {head.get('version', 3)}\n")
        f.write(f"default {head.get('default', 0.0001)}\n")
        for name, w in items.items():
            f.write(f"{name} {w:.6f}\n")
    share = tot_b / tot
    print(f"→ {a.out}")
    print(f"  抬高 {len(boosted)} 副（前缀 {prefs}, ×{a.boost}）→ 对手座位占比 **{share:.1%}**")
    for n in boosted:
        print(f"    {n:<40} {items[n]:.4f}")
    top = sorted(items.items(), key=lambda kv: -kv[1])[:5]
    names = "， ".join("{} {:.3f}".format(n.split("__")[0], w) for n, w in top)
    print("  其余最高：" + names)


if __name__ == "__main__":
    main()
