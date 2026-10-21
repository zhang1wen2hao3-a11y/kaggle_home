#!/usr/bin/env python3
"""生成「进化线主力」卡组（第一批）。

背景（`下一步.md` §2）：把整个卡牌库按「最便宜招式的伤害/能量」排序，
前 16 名几乎全是**进化怪**，而我们现有 24 套卡组一张都没用过
（只用了 733 张可用宝可梦里的 17 张）。现有 24 套全是「两只独立基础怪」，
结构上够不到这批牌。

所以这批卡组**只改主力结构**：固定训练家包与能量数量都与现有 24 套一致，
把「4× Mega X ex(基础怪) + 4× 副攻基础怪」换成「4× 基础怪(HP70~90) + 4× 一进 ex(HP330~340)」。

## 自检（不通过就不写文件）

1. **构建逻辑自检**：用同一套组装函数生成「35 能量 + Snover/Abomasnow 线」，
   必须与 `decks/sample_official.csv` 的**多重集完全相同**。
   这条最重要 —— 它证明我的组装逻辑和官方样例一致，否则后面所有对比都没意义。
2. **每套 60 张**（这条抓过一次：`下一步.md` 里 `line_abomasnow_37` 的算式是 62 张）。
3. **合法性**：走 `framework.deck.is_legal`（引擎按**卡名**判合法，不是 cardId）。

用法：
    ./run.sh make_line_decks.py --check      # 只自检，不写文件
    ./run.sh make_line_decks.py             # 自检通过后写入 decks/
"""
import argparse
import collections
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ---- 卡牌 id 常量（来自 data/EN_Card_Data_R2_full.csv）
ENERGY = {   # 基本能量
    "G": 1, "R": 2, "W": 3, "L": 4, "P": 5, "F": 6, "D": 7, "M": 8,
}
MEGA_SIGNAL = 1145
LILLIES = 1227
WAITRESS = 1235
CYRANO = 1205
MAX_BELT = 1158

TYPE_CN = {"G": "草", "R": "火", "W": "水", "L": "雷",
           "P": "超", "F": "斗", "D": "恶", "M": "钢"}

TRAINERS = [  # 固定训练家包：与现有 24 套 / sample_official 完全一致
    (MEGA_SIGNAL, 4), (LILLIES, 4), (WAITRESS, 4), (CYRANO, 2), (MAX_BELT, 1),
]             # 合计 15 张


def build(energy_type: str, energy_n: int, line: list[tuple[int, int]],
          extras: list[tuple[int, int]] | None = None) -> list[int]:
    """组装一套卡组：能量 + 进化线 + 额外牌 + 固定训练家包。"""
    deck = [ENERGY[energy_type]] * energy_n
    for cid, n in line:
        deck += [cid] * n
    for cid, n in (extras or []):
        deck += [cid] * n
    for cid, n in TRAINERS:
        deck += [cid] * n
    return deck


# ------------------------------------------------------------------ 候选卡组
# 第一批只换主力结构：37 能量（62%）+ 4 基础 + 4 一进 + 固定训练家 = 60
DECKS = {
    # Palafin ex：250 伤 / 1 能（HP 340），代价是下回合不能攻击
    "line_palafin": dict(energy_type="W", energy_n=37,
                         line=[(105, 4), (107, 4)]),          # Finizen → Palafin ex
    # 同一条线但 35 能量 + 2× Kyogre —— 复刻 sample_official 的「58% 能量 + 副攻」
    "line_palafin_35": dict(energy_type="W", energy_n=35,
                            line=[(105, 4), (107, 4)], extras=[(721, 2)]),
    # Mega Lucario ex：130 伤 / 1 能，攒 2 能可打 270（HP 340）
    "line_lucario": dict(energy_type="F", energy_n=37,
                         line=[(333, 4), (678, 4)]),          # Riolu → Mega Lucario ex
    # Mega Starmie ex：120 伤 / 1 能 + 50 溅射（HP 330）
    "line_starmie": dict(energy_type="W", energy_n=37,
                         line=[(1030, 4), (1031, 4)]),        # Staryu → Mega Starmie ex

    # ---- 第二批：修正「基础怪太少」
    # 第一批每套只有 4 张基础怪（line_palafin 甚至只赢 1/300），而
    # `sample_official` 有 6 张（Snover×4 + Kyogre×2）。空场直接判负，
    # 4 张基础怪 = 场上最多只能撑住一两次被击倒。
    # 修正：能量 37 → 33，腾出 4 个位置加一只**同属性副攻基础怪**，
    # 于是有 8 张基础怪 + 4 张一进 ex = 12 只宝可梦，能量仍占 55%。
    "line_palafin_v2": dict(energy_type="W", energy_n=33,
                            line=[(105, 4), (107, 4)], extras=[(721, 4)]),
    "line_lucario_v2": dict(energy_type="F", energy_n=33,
                            line=[(333, 4), (678, 4)], extras=[(979, 4)]),
    "line_starmie_v2": dict(energy_type="W", energy_n=33,
                            line=[(1030, 4), (1031, 4)], extras=[(721, 4)]),
}

# 构建逻辑自检用：必须与 decks/sample_official.csv 多重集相同
SELF_CHECK = dict(energy_type="W", energy_n=35,
                  line=[(722, 4), (723, 4)],                  # Snover → Mega Abomasnow ex
                  extras=[(721, 2)])


def as_counter(ids):
    return collections.Counter(ids)


def read_deck(path):
    with open(path) as f:
        return [int(x) for x in f.read().split() if x.strip()][:60]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只自检，不写文件")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "decks"))
    args = ap.parse_args()

    from framework.deck import CardPool, is_legal     # 需要引擎来判合法性

    pool = CardPool()
    ref = os.path.join(ROOT, "decks", "sample_official.csv")
    ok = True

    # ---- 自检 1：构建逻辑与官方样例一致
    print("=" * 72)
    print("自检 1：组装逻辑 vs decks/sample_official.csv")
    mine = build(**SELF_CHECK)
    if not os.path.exists(ref):
        print("  ⚠️ 找不到 sample_official.csv，跳过（但在做对比前必须补上）")
    else:
        theirs = read_deck(ref)
        if as_counter(mine) == as_counter(theirs):
            print(f"  ✅ 多重集完全一致（{len(mine)} 张，{len(as_counter(mine))} 种）")
        else:
            ok = False
            print("  ❌ 不一致！")
            print(f"     我生成 - 参考: {dict(as_counter(mine) - as_counter(theirs))}")
            print(f"     参考 - 我生成: {dict(as_counter(theirs) - as_counter(mine))}")

    # ---- 自检 2 + 3：每套 60 张 + 合法
    print("\n" + "=" * 72)
    print("自检 2/3：张数与合法性")
    built = {}
    for name, spec in DECKS.items():
        deck = build(**spec)
        built[name] = deck
        if len(deck) != 60:
            ok = False
            print(f"  ❌ {name:<20} {len(deck)} 张（必须 60）")
            continue
        legal, why = is_legal(pool, deck)
        mark = "✅" if legal else "❌"
        if not legal:
            ok = False
        etype = TYPE_CN.get(spec["energy_type"], spec["energy_type"])
        detail = "" if legal else why
        print(f"  {mark} {name:<20} {len(deck)} 张  "
              f"能量 {spec['energy_n']}×{etype}  "
              f"线 {spec['line']}  {detail}")

    print("=" * 72)
    if not ok:
        print("❌ 自检未通过，不写文件。")
        sys.exit(1)
    if args.check:
        print("✅ 自检通过（--check 模式，未写文件）")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for name, deck in built.items():
        path = os.path.join(args.out_dir, f"{name}.csv")
        with open(path, "w") as f:
            f.write("\n".join(str(x) for x in deck) + "\n")
        print(f"  写入 {os.path.relpath(path, ROOT)}")
    print(f"\n✅ 已写入 {len(built)} 套到 {os.path.relpath(args.out_dir, ROOT)}/")


if __name__ == "__main__":
    main()
