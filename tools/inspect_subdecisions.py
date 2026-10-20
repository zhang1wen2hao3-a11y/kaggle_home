#!/usr/bin/env python3
"""子决策盘点：哪些决策不是 MAIN、每类有多少、有多少真的在做选择、我们怎么处理的。

背景：子决策占全部决策的 **37.5%**（MAIN 占 62.5%），但线上的 agent 对它们
基本是"**取引擎枚举的第一个选项**"。这个工具用来回答三个问题：

  1. 有哪些 context？各占多少？
  2. 每个 context 平均几个选项？**多少比例是 ≥2 个选项**（有选择空间）？
  3. **选项是否退化** —— 所有选项解析出来是同一张牌时，其实没什么可选。

第 3 点特别重要：我们卡组 60 张里 35 张是同一种基本能量，于是
`DISCARD_ENERGY`（15.5%）和 `ATTACH_TO`（贴哪张牌，4.2%）**完全退化**。
不查这一点就会得出"子决策有 37% 的空间"这种错误估计。

顺带还能看到：每个选项的 `area/index` 可以解析成具体牌 —— 这是写卡价值规则的前提。
`current.looking[i].id` 这个区是可见的（`PlayerState.looking`），
虽然 `select.deck` 对 LOOKING 区是 None，仍然能反查。

用法：
    ./run.sh tools/inspect_subdecisions.py                      # 默认 6 局
    ./run.sh tools/inspect_subdecisions.py --games 12 --deck decks/grass_heracross.csv
    ./run.sh tools/inspect_subdecisions.py --agent agents/greedy.py --samples 3
"""
import argparse
import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluate import load_agent                       # noqa: E402
from local_match import play_match, load_deck          # noqa: E402
from cg.api import SelectType, SelectContext           # noqa: E402

TN = {v.value: k for k, v in SelectType.__members__.items()}
CN = {v.value: k for k, v in SelectContext.__members__.items()}
from cg.api import OptionType                          # noqa: E402
OT = {v.value: k for k, v in OptionType.__members__.items()}

# AreaType: DECK=1 HAND=2 DISCARD=3 ACTIVE=4 BENCH=5 PRIZE=6 STADIUM=7
#           ENERGY=8 TOOL=9 PRE_EVOLUTION=10 PLAYER=11 LOOKING=12


def load_card_table():
    """从官方卡牌 CSV 读 名字 / 阶段。只用于本地分析，不进提交。"""
    path = os.path.join(ROOT, "data", "EN_Card_Data_R2_full.csv")
    if not os.path.exists(path):
        return {}, {}
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    tycol = next((c for c in rows[0] if c.startswith("Stage")), None)
    name, stage = {}, {}
    for r in rows:
        try:
            cid = int(r["Card ID"])
        except (TypeError, ValueError):
            continue
        name.setdefault(cid, r["Card Name"])
        stage.setdefault(cid, r[tycol] if tycol else "")
    return name, stage


def stage_code(s: str) -> str:
    if "Basic Energy" in s:
        return "E"       # 基本能量
    if "Basic Pok" in s:
        return "0"       # 基础怪（非进化）
    if "Stage 1" in s:
        return "1"
    if "Stage 2" in s:
        return "2"
    if "Pok" in s:
        return "P"       # 其它宝可梦
    return "T"           # 训练家/道具/场地


def resolve_card_id(obs: dict, opt: dict, sel: dict | None = None):
    """把 option 的 (area, index) 解析成具体 card id。解析不了返回 None。

    各区的可见性（`cg/api.py` 的 `PlayerState` / `SelectData`）：
        HAND/DISCARD/ACTIVE/BENCH/PRIZE  -> players[i] 下直接可取
        LOOKING(12)                      -> current.looking[i].id
        DECK(1)                          -> **只能**从 select.deck 取
                                            （"An array of cards; None unless
                                              selecting cards from the deck"）
    对手的手牌是 None，解析不出来属于正常情况。
    """
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if len(players) < 2:
        return None
    if opt.get("area") == 1:                      # DECK：走 select.deck
        arr = (sel or {}).get("deck") or []
        i = opt.get("index")
        if i is None or i < 0 or i >= len(arr) or not arr[i]:
            return None
        return arr[i].get("id")
    owner = opt.get("playerIndex")
    if owner not in (0, 1):
        owner = int(cur.get("yourIndex", 0))
    p = players[owner]
    arr = {2: p.get("hand"), 3: p.get("discard"), 4: p.get("active"),
           5: p.get("bench"), 6: p.get("prize"), 12: cur.get("looking")}.get(
               opt.get("area"))
    if not arr:
        return None
    i = opt.get("index")
    if i is None or i < 0 or i >= len(arr) or not arr[i]:
        return None
    return arr[i].get("id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="agents/ro_attack_w10.py")
    ap.add_argument("--opponent", default="agents/greedy.py")
    ap.add_argument("--deck", default="decks/sample_official.csv")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--samples", type=int, default=3, help="每个 context 打印几个例子")
    args = ap.parse_args()

    name_of, stage_of = load_card_table()
    A = load_agent(args.agent)
    B = load_agent(args.opponent)
    deck = load_deck(args.deck)

    cnt = collections.Counter()                 # (type, context) -> 次数
    nopt = collections.defaultdict(list)        # (type, context) -> 选项数列表
    same = collections.Counter()                # 选项牌全同的次数
    total_decisions = [0]
    samples = collections.defaultdict(list)

    def spy(obs):
        sel = obs.get("select")
        if sel is not None:
            total_decisions[0] += 1
            st = TN.get(int(sel.get("type", -1)), "?")
            ctx = CN.get(int(sel.get("context", -1)), "?")
            key = (st, ctx)
            opts = sel.get("option") or []
            cnt[key] += 1
            nopt[key].append(len(opts))
            if opts:
                ids = [resolve_card_id(obs, o, sel) for o in opts]
                known = [i for i in ids if i is not None]
                if len(known) >= 2 and len(set(known)) == 1:
                    same[key] += 1
                if len(samples[key]) < args.samples:
                    shown_ids = []
                    for o, i in zip(opts, ids):
                        if i is not None:
                            shown_ids.append(
                                f"{stage_code(stage_of.get(i, ''))}:{name_of.get(i, i)}")
                        else:
                            # 解析不出具体牌时退化显示选项类型（YES/NO/NUMBER…）
                            shown_ids.append(f"[{OT.get(int(o.get('type', -1)), '?')}]")
                    samples[key].append(shown_ids)
        return A(obs)

    for i in range(args.games):
        play_match(spy if i % 2 == 0 else B, B if i % 2 == 0 else spy,
                   deck, deck, reverse=bool(i % 2))

    N = max(total_decisions[0], 1)
    rows = []
    for key, n in cnt.items():
        a = nopt[key]
        multi = sum(1 for x in a if x >= 2) / len(a)
        rows.append((n / N, key, n, sum(a) / len(a), multi, same[key] / n))
    rows.sort(reverse=True)

    print("=" * 92)
    print(f"子决策盘点：{args.games} 局、{N} 个决策  |  agent={args.agent}  deck={args.deck}")
    print("=" * 92)
    print(f"{'占比':>7}  {'type':<12}{'context':<24}{'次数':>7}{'平均选项':>9}"
          f"{'≥2选项':>8}{'选项全同':>9}")
    for p, (st, ctx), n, m, multi, deg in rows:
        flag = "  ← 退化" if deg > 0.5 else ""
        print(f"{p:>7.1%}  {st:<12}{ctx:<24}{n:>7}{m:>9.2f}{multi:>8.0%}{deg:>9.0%}{flag}")

    main_p = sum(r[0] for r in rows if r[1][0] == "MAIN")
    sub = [r for r in rows if r[1][0] != "MAIN"]
    sub_p = sum(r[0] for r in sub)
    degenerate = sum(r[0] for r in sub if r[5] > 0.5)
    multi_p = sum(r[0] * r[4] for r in sub)
    print("-" * 92)
    print(f"  MAIN 占 {main_p:.1%}；非 MAIN 占 {sub_p:.1%}")
    print(f"  非 MAIN 里，有 ≥2 个选项（真实选择空间）的占 **{multi_p:.1%}**（占全部决策）")
    print(f"  非 MAIN 里，**选项全是同一张牌（退化）**的占 **{degenerate:.1%}**（占全部决策）")
    print(f"  ⇒ 真正可动的子决策 ≈ {multi_p - degenerate:.1%}")

    if args.samples:
        print("\n" + "=" * 92)
        print("样本（选项按**引擎枚举顺序**列出；第 0 个就是当前「取第一个」会选的）")
        print("=" * 92)
        shown = 0
        for p, key, n, m, multi, deg in rows:
            if key[0] == "MAIN" or not samples.get(key):
                continue
            print(f"\n[{key[0]} / {key[1]}]  占比 {p:.1%}  平均 {m:.1f} 项")
            for s in samples[key][:2]:
                print(f"    选项: {s}")
            shown += 1
            if shown >= 8:
                break


if __name__ == "__main__":
    main()
