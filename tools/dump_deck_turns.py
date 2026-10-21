#!/usr/bin/env python3
"""逐决策 dump 对局，统计「每回合做了什么、有没有推进」——用来查"空转"。

背景（`下一步.md` §2.8）：官方卡组用我们的 agent 打，`ATTACK 可选` 比例和
`sample_official` 差不多（33.7% vs 39.9%），但**每 12 局的决策总数差 5 倍**
（721 vs 138）。说明不是因为"打不出攻击"，而是做了大量动作却转化不成胜利。

本工具把"空转"操作化为一个可测的定义：

    一回合里动作数 >= --min-actions，但
    **能量净增 == 0 且 没有进化 且 没有造成伤害**（对手奖赏卡数没减少）
    → 判为「空转回合」

用法：
    ./run.sh tools/dump_deck_turns.py --deck decks/official_dragapult.csv --games 12
    ./run.sh tools/dump_deck_turns.py --deck decks/sample_official.csv --games 12
    ./run.sh tools/dump_deck_turns.py --deck decks/official_dragapult.csv --games 3 --verbose
"""
import argparse
import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluate import load_agent                        # noqa: E402
from local_match import play_match, load_deck           # noqa: E402
from cg.api import OptionType                           # noqa: E402

OT = {v.value: k for k, v in OptionType.__members__.items()}


def load_cards():
    path = os.path.join(ROOT, "data", "EN_Card_Data_R2_full.csv")
    if not os.path.exists(path):
        return {}, {}
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    name, cat = {}, {}
    for r in rows:
        try:
            cid = int(r["Card ID"])
        except (TypeError, ValueError):
            continue
        name.setdefault(cid, r["Card Name"])
        cat.setdefault(cid, r.get("Stage (Pokémon)/Type (Energy and Trainer)", ""))
    return name, cat


def snapshot(obs, me_idx):
    cur = obs.get("current") or {}
    p = (cur.get("players") or [{}, {}])[me_idx] or {}
    op = (cur.get("players") or [{}, {}])[1 - me_idx] or {}
    mons = [x for x in (list(p.get("active") or []) + list(p.get("bench") or [])) if x]
    return {
        "turn": int(cur.get("turn", 0)),
        "hand": len(p.get("hand") or []),
        "bench": len([x for x in (p.get("bench") or []) if x]),
        "mons": len(mons),
        "energy": sum(len(x.get("energyCards") or []) for x in mons),
        "evolved": sum(len(x.get("preEvolution") or []) for x in mons),
        "op_prize": len(op.get("prize") or []),
    }


def resolve_cid(obs, opt, sel):
    """把选项解析成 cardId（牌库区在 select.deck 里）。"""
    area, idx = opt.get("area"), opt.get("index")
    if idx is None:
        return None
    cur = obs.get("current") or {}
    me = int(cur.get("yourIndex", 0))
    players = cur.get("players") or [{}, {}]
    if area == 1:                      # DECK
        arr = sel.get("deck") or []
    else:
        arr = {2: players[me].get("hand"), 3: players[me].get("discard"),
               4: players[me].get("active"), 5: players[me].get("bench"),
               6: players[me].get("prize"), 12: cur.get("looking")}.get(area) or []
    if 0 <= idx < len(arr) and arr[idx]:
        c = arr[idx]
        return c.get("id") if isinstance(c, dict) else getattr(c, "id", None)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="decks/official_dragapult.csv")
    ap.add_argument("--agent", default="agents/ro_deck_base.py")
    ap.add_argument("--opponent", default="agents/ro_attack_w10.py")
    ap.add_argument("--deck-opponent", default="decks/sample_official.csv")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--min-actions", type=int, default=3, help="空转判定：该回合至少几个动作")
    ap.add_argument("--verbose", action="store_true", help="打印每局每回合明细")
    args = ap.parse_args()

    os.environ["PTCG_RO_DECK"] = args.deck      # ro_deck_base 读这个
    name_of, cat_of = load_cards()
    A = load_agent(args.agent)
    B = load_agent(args.opponent)
    deck = load_deck(args.deck)
    opp = load_deck(args.deck_opponent) if args.deck_opponent else deck

    # 当前局/当前回合的累积容器
    state = {"turns": {}, "order": []}

    def on_decision(obs):
        sel = obs.get("select")
        if sel is None:
            return A(obs)
        cur = obs.get("current") or {}
        me = int(cur.get("yourIndex", 0))
        snap = snapshot(obs, me)
        t = snap["turn"]
        if t not in state["turns"]:
            state["turns"][t] = {"first": snap, "last": snap, "acts": []}
            state["order"].append(t)
        rec = state["turns"][t]
        rec["last"] = snap
        # 我们这一步选了什么（**只调用一次** —— 调两次会让 rollout 多跑一遍，
        # 时间预算提前触发，测出来的行为就不是部署行为了）
        try:
            chosen = A(obs)
        except Exception:
            chosen = []
        if chosen:
            i = chosen[0]
            opts = sel.get("option") or []
            if 0 <= i < len(opts):
                o = opts[i]
                tname = OT.get(int(o.get("type", -1)), "?")
                cid = resolve_cid(obs, o, sel)
                label = tname
                if cid is not None:
                    label += f":{(name_of.get(cid, cid) or str(cid))[:16]}"
                    c = cat_of.get(cid, "")
                    if "Supporter" in c:
                        label += " [支援者]"
                    elif "Item" in c:
                        label += " [道具]"
                    elif "Stadium" in c:
                        label += " [场地]"
                rec["acts"].append(label)
        return chosen

    tot = collections.Counter()
    per_game = []
    for g in range(args.games):
        state["turns"] = {}
        state["order"] = []
        # 与 evaluate.py 一致的四路交替：座位 × reverse 各自独立
        # （只交替座位会让"我方总能选先后手"，实测能把胜率抬高几十个百分点）
        seat = g % 2
        rev = (g // 2) % 2
        if seat == 0:
            r = play_match(on_decision, B, deck, opp, reverse=bool(rev))
            won = r.get("result") == 0
        else:
            r = play_match(B, on_decision, opp, deck, reverse=bool(rev))
            won = r.get("result") == 1
        n_turns = len(state["order"])
        acts = sum(len(v["acts"]) for v in state["turns"].values())
        per_game.append((g, won, n_turns, acts, r.get("steps")))
        # 判定每个回合是否空转
        for t in state["order"]:
            rec = state["turns"][t]
            f, l = rec["first"], rec["last"]
            n_act = len(rec["acts"])
            d_energy = l["energy"] - f["energy"]
            d_evo = l["evolved"] - f["evolved"]
            d_dmg = f["op_prize"] - l["op_prize"]      # 对手奖赏被拿走 = 我们造成击杀
            tot["回合数"] += 1
            tot["动作数"] += n_act
            if n_act >= args.min_actions:
                tot["多动作回合"] += 1
                if d_energy <= 0 and d_evo <= 0 and d_dmg <= 0:
                    tot["空转回合"] += 1
                if d_energy <= 0:
                    tot["多动作且能量没涨"] += 1
                if d_dmg <= 0:
                    tot["多动作且没造成伤害"] += 1
            for a in rec["acts"]:
                tot["类型:" + a.split(":")[0]] += 1
                if "[支援者]" in a:
                    tot["支援者动作"] += 1
                if "[道具]" in a:
                    tot["道具动作"] += 1
            if args.verbose:
                print(f"  game{g} t{t:>2} acts={n_act} "
                      f"手牌{f['hand']}→{l['hand']} 能量{f['energy']}→{l['energy']} "
                      f"怪{f['mons']}→{l['mons']} 击杀{f['op_prize']}→{l['op_prize']}"
                      f"{'  ← 空转' if (n_act >= args.min_actions and d_energy <= 0 and d_evo <= 0 and d_dmg <= 0) else ''}"
                      f"  {rec['acts'][:6]}")

    print("=" * 78)
    print(f"卡组 {args.deck}  vs  {args.deck_opponent}   共 {args.games} 局")
    print("=" * 78)
    wins = sum(1 for x in per_game if x[1])
    print(f"  胜 {wins}/{args.games}")
    print(f"  每局回合数   {sum(x[2] for x in per_game)/len(per_game):.1f}")
    print(f"  每局动作数   {sum(x[3] for x in per_game)/len(per_game):.1f}")
    print(f"  每回合动作数 {tot['动作数']/max(tot['回合数'],1):.1f}")
    print()
    n_turn = max(tot["回合数"], 1)
    print(f"  动作类型分布: " + ", ".join(
        f"{k.split(':')[1]}={v}" for k, v in sorted(tot.items(), key=lambda t: -t[1])
        if k.startswith("类型:")))
    print(f"  支援者动作 {tot['支援者动作']}（{tot['支援者动作']/max(tot['动作数'],1):.0%}）"
          f"   道具动作 {tot['道具动作']}（{tot['道具动作']/max(tot['动作数'],1):.0%}）")
    print()
    print(f"  多动作回合（动作≥{args.min_actions}）: {tot['多动作回合']}/{n_turn}"
          f"（{tot['多动作回合']/n_turn:.0%}）")
    print(f"    └ 其中**空转**（能量没涨 且 没进化 且 没击杀）: "
          f"{tot['空转回合']}（{tot['空转回合']/max(tot['多动作回合'],1):.0%} 的多动作回合）")
    print(f"    └ 能量没涨: {tot['多动作且能量没涨']}   没造成伤害: {tot['多动作且没造成伤害']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
