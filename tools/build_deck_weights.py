#!/usr/bin/env python3
"""**按「实测场地 + 冠军保底」生成对手分布**（训练用的 decks/weights_*.txt）。

## 为什么需要它

原来的 `weights_2617.txt` 是手拍的：7e39 0.34、0904 0.10、e1d7 0.05、26c6 0.04 ……
既没有可复现规则，也和实测场地对不上（我们 582 局里 7e39 是 34.7%、0bf5 只有 3.8%）。
而且冠军卡组只拿到一组随手给的权重。

## 公式

    w'(d) = (1 − K·b) · f(d)  +  b · 1[d ∈ 冠军卡组]

* `f(d)`  实测场地分布（`data/our_games.tsv` 的 deck 级计数，剔除未识别）
* `K`     冠军卡组数量
* `b`     每副冠军卡组的**保底比例**（默认 0.05 = 每副 +5pp）

直观理解：先给每副冠军牌固定 b 的份额，剩下的 `1 − K·b` 按实测场地比例分。
⇒ 冠军比例被"抬"到至少 b，其余卡组按比例被**压缩**，总强度上升但场地形状不变。

## 冠军卡组的判定（数据驱动，不靠感觉）

`--champ` 默认取「帧数 ≥ 100k 且教师 LB ≥ 1500」的四副：
`0904`（Petit Canard 1510，209,735 帧）、`26c6`（ミワハルキ 1580，167,053）、
`e1d7`（ミワハルキ 1580，137,602）、`shuppet_0d86`（ミワハルキ 1580，130,236）。
7e39 **不算**冠军牌——它场次最多（34.7%），但最好教师只有 Dipam 1478 / 1,768 帧。
（见 `下一步.md` §56.4 的语料表）

## 用法

    ./tools/py tools/build_deck_weights.py --b 0.05 --out decks/weights_field_b05.txt
    ./tools/py tools/build_deck_weights.py --b 0     --out decks/weights_field_b00.txt
    ./tools/py tools/build_deck_weights.py --b 0.10 --out decks/weights_field_b10.txt
    ./tools/py tools/build_deck_weights.py --b 0.05 --dry-run     # 只打印不写文件
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES = os.path.join(ROOT, "data", "our_games.tsv")
POOL = os.path.join(ROOT, "models", "weights", "conf", "pool_field")

# 冠军卡组：帧数 ≥100k 且教师 LB ≥1500（§56.4）
CHAMPS = [
    "z_dragapult_dreepy_petit__0904",     # Petit Canard 1510 / 209,735 帧
    "slowpoke_slowking__26c6",            # ミワハルキ 1580 / 167,053 帧
    "z_ogerpon_toolbox_miwa__e1d7",       # ミワハルキ 1580 / 137,602 帧
    "shuppet_dhelmise__0d86",             # ミワハルキ 1580 / 130,236 帧
]
DEFAULT_TAIL = 0.0001     # 池里其余卡组的兜底权重（与旧文件一致）


def field_distribution(games_path: str) -> tuple[dict[str, float], int, int]:
    rows = list(csv.DictReader(open(games_path, encoding="utf-8"), delimiter="\t"))
    c = collections.Counter(r["opp_deck"] for r in rows if r.get("opp_deck") not in (None, "", "?"))
    tot = sum(c.values())
    return {k: v / tot for k, v in c.items()}, tot, len(rows)


def build(field: dict[str, float], champs: list[str], b: float) -> dict[str, float]:
    k = len(champs)
    if k * b >= 1.0:
        raise SystemExit(f"✗ K·b = {k}×{b} = {k*b:.2f} ≥ 1，保底比例太大")
    rest = 1.0 - k * b
    out = {d: rest * w for d, w in field.items()}
    for d in champs:
        out[d] = out.get(d, 0.0) + b
    return out


def ess(w: dict[str, float]) -> float:
    """有效对手数（1/Σw²，按总权重 1 归一后的口径）。"""
    s = sum(v * v for v in w.values())
    return 1.0 / s if s else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=GAMES)
    ap.add_argument("--b", type=float, default=0.05, help="每副冠军卡组的保底比例（0.05 = +5pp）")
    ap.add_argument("--champ", nargs="*", default=CHAMPS)
    ap.add_argument("--tail", type=float, default=DEFAULT_TAIL)
    ap.add_argument("--min-share", type=float, default=0.002,
                    help="实测占比低于此值的卡组不写进文件（走 tail）")
    ap.add_argument("--out", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    field, n_games, n_rows = field_distribution(a.games)
    w = build(field, a.champ, a.b)
    # 只显式写"占比够大或属于冠军"的卡组，其余走 tail
    keep = {d: v for d, v in w.items() if v >= a.min_share or d in a.champ}
    keep = dict(sorted(keep.items(), key=lambda t: -t[1]))
    # 被 min-share 剔掉的份额不能让权重总和 < 1（否则实际生效比例整体偏小），
    # 按比例回摊到保留的卡组上，落选的走 tail。
    s = sum(keep.values())
    keep = {d: v / s for d, v in keep.items()}
    tail_decks = len(field) - len([d for d in field if d in keep])
    total = sum(keep.values())

    print(f"实测场地：{n_games} 局（{n_rows} 条记录，已剔除未识别）  唯一对手卡组 {len(field)} 副")
    print(f"冠军卡组 {len(a.champ)} 副 × 保底 b={a.b:.2f} ⇒ 占用 {len(a.champ)*a.b*100:.0f}% 权重，"
          f"其余 {(1-len(a.champ)*a.b)*100:.0f}% 按实测比例压缩\n")
    print(f"{'卡组':<42}{'实测':>8}{'新权重':>9}{'冠军':>6}")
    for d, v in keep.items():
        star = "★" if d in a.champ else ""
        print(f"{d:<42}{field.get(d,0)*100:>7.1f}%{v*100:>8.1f}%{star:>6}")
    print(f"\n显式覆盖 {total*100:.2f}% 权重，{len(keep)} 副；其余 {tail_decks} 副走 tail={a.tail}")
    print(f"有效对手数 ESS = {ess({**keep, **{}}):.1f}（旧 weights_2617.txt 只有 5.8 —— 极度集中）")

    if a.dry_run or not a.out:
        print("\n（dry-run：未写文件）")
        return 0
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    with open(out, "w") as f:
        f.write("version 3\n")
        f.write(f"default {a.tail}\n")
        for d, v in keep.items():
            f.write(f"{d} {v:.6f}\n")
    print(f"\n✓ 写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
