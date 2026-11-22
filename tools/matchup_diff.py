#!/usr/bin/env python3
"""**同一对位下，高手怎么赢、我们怎么输**：把两组回放的"打法指纹"摆在一起比。

用户 2026-11-22 的原话：
    「我觉得最靠谱的是看同样的对手，高分卡组怎么战胜的，而我们是怎么输的，通过这个区别，
     找一下共性，是他们的卡组里有我们没有的卡，还是他们的打法我们没使用」

为此把每局折成两类可比的量：
  A. **卡组维度**：双方 60 张（multiset）—— 直接给出"张数差"
  B. **打法维度**（只看主阶段 MAIN 的选择，按回合分桶统计）：
     · 每回合做了几件事（PLAY 出牌 / ATTACH 贴能 / EVOLVE 进化 / ABILITY 特性 / RETREAT 撤退 / ATTACK 攻击 / END 结束）
     · 前 2 回合出了哪些牌（哪些卡他们出、我们不出）
     · 第一次攻击发生在第几回合、bench 到场人数、能量贴附节奏
     · 用的招式（attackId）

用法
----
    # 我们这批（自己提交的回放）
    ./tools/py tools/matchup_diff.py --deck z_ogerpon_toolbox_miwa__e1d7 \
        --opp teal_mask_ogerpon_ex_applin__7e39 --dir epsoide_e1d7r4 --label 我们
    # 高手那批（同一副牌、同一对手）
    ./tools/py tools/matchup_diff.py --deck z_ogerpon_toolbox_miwa__e1d7 \
        --opp teal_mask_ogerpon_ex_applin__7e39 --dir epsoide_miwa --label 高手 --player "ミワハルキ"
    # 两边一起比
    ./tools/py tools/matchup_diff.py --compare "epsoide_e1d7r4:我们" "epsoide_miwa:高手" \
        --deck z_ogerpon_toolbox_miwa__e1d7 --opp teal_mask_ogerpon_ex_applin__7e39 --player "ミワハルキ"
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OT = {0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL_CARD", 5: "ENERGY_CARD", 6: "ENERGY",
      7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY", 11: "DISCARD", 12: "RETREAT",
      13: "ATTACK", 14: "END", 15: "SKILL", 16: "SPECIAL_CONDITION"}


def deck_hash(ids) -> str:
    """★ 与流水线同口径（framework/replay.py / POOL_ID_MAP）：卡 id **按数值**排序后拼接。
    2026-11-22 修：以前这里按字符串排序，算出的 hash（e40f33bb）和仓库里写的 4hex（e1d754a7）对不上。"""
    return hashlib.sha256(",".join(str(x) for x in sorted(map(int, ids))).encode()).hexdigest()[:8]


def load_pool() -> dict:
    out = {}
    for p in glob.glob(os.path.join(ROOT, "decks", "pool_field", "*.csv")):
        ids = [l.strip() for l in open(p) if l.strip()]
        out[deck_hash(ids)] = (os.path.basename(p)[:-4], ids)
    return out


def card_names() -> dict:
    out = {}
    p = os.path.join(ROOT, "build", "port_plain", "EN_Card_Data.csv")
    try:
        import csv
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            out[str(r["Card ID"])] = r["Card Name"]
    except OSError:
        pass
    return out


def pick_card(obs, opt):
    """把 option 指向的卡翻成 card id（尽量取，取不到返回 None）。"""
    try:
        cur = obs.get("current") or {}
        players = cur.get("players") or []
        pi = opt.get("playerIndex", 0)
        area = opt.get("area")
        idx = opt.get("index")
        # 区域：1=deck 2=hand 3=discard?（本仓约定见 cg/api.py AreaType）；这里按玩家对象里常见的名键试
        pl = players[pi] if pi < len(players) else {}
        for key, arr in (("hand", pl.get("hand")), ("active", pl.get("active")), ("bench", pl.get("bench"))):
            pass
        if area == 2 and pl.get("hand") is not None and idx is not None and idx < len(pl["hand"]):
            return str(pl["hand"][idx].get("id"))
        if area == 4 and pl.get("bench") is not None and idx is not None and idx < len(pl["bench"]):
            return str(pl["bench"][idx].get("id"))
        if area == 3 and pl.get("active") is not None and idx is not None and idx < len(pl["active"]):
            return str(pl["active"][idx].get("id"))
    except Exception:
        return None
    return None


def scan_game(path, my_hash, opp_hash, player=None):
    """返回一局的指纹；不符合 (我方牌/对手牌/玩家) 就返回 None。"""
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return None
    try:
        st = d["steps"][1]
        decks = [[str(x) for x in (s.get("action") or [])] for s in st]
    except (KeyError, TypeError):
        return None
    if len(decks) != 2 or any(len(x) != 60 for x in decks):
        return None
    hs = [deck_hash(x) for x in decks]
    if hs[0] == my_hash and hs[1] == opp_hash:
        me = 0
    elif hs[1] == my_hash and hs[0] == opp_hash:
        me = 1
    else:
        return None
    agents = [a.get("Name", "") for a in (d.get("info", {}).get("Agents") or [])]
    if player and player not in (agents[me] or ""):
        return None
    rw = d.get("rewards") or [0, 0]
    winner = 0 if rw[0] > rw[1] else 1          # ★ 2026-11-22 修 bug：以前少了这个取反，
    win = winner == me                          #   高手那批 21/22 胜被算成 1/22

    fp = dict(turns=0, own_turns=set(), acts=collections.Counter(),
              per_turn=collections.defaultdict(collections.Counter),
              plays_early=[], attacks=[], first_attack=None, bench_at=collections.defaultdict(int),
              attaches=0, ended_with=0)
    steps = d["steps"]
    for t, row in enumerate(steps):
        if me >= len(row):
            break
        seat = row[me]
        obs = seat.get("observation") or {}
        sel = obs.get("select") or {}
        act = seat.get("action") or []
        cur = obs.get("current") or {}
        turn = cur.get("turn")
        if turn is not None:
            fp["turns"] = max(fp["turns"], int(turn) + 1)
            try:
                bench = len((cur.get("players") or [{}])[me].get("bench") or [])
                if turn < 6:
                    fp["bench_at"][int(turn)] = max(fp["bench_at"][int(turn)], bench)
            except (IndexError, TypeError):
                pass
        if sel.get("type") != 0:            # 只看主阶段
            continue
        if turn is not None:
            fp["own_turns"].add(int(turn))
        opts = sel.get("option") or []
        for a in (act if isinstance(act, list) else [act]):
            try:
                o = opts[int(a)]
            except (IndexError, ValueError, TypeError):
                continue
            ty = OT.get(int(o.get("type", -1)), "?")
            fp["acts"][ty] += 1
            if turn is not None:
                fp["per_turn"][min(int(turn), 9)][ty] += 1
            if ty == "PLAY":
                cid = pick_card(obs, o)
                if cid and turn is not None and int(turn) <= 1:
                    fp["plays_early"].append(cid)
            elif ty == "ATTACH":
                fp["attaches"] += 1
            elif ty == "ATTACK":
                aid = int(o.get("attackId") or 0)
                # 记录"谁在用这招"：攻击方是当前 active
                act_id = None
                try:
                    actives = (cur.get("players") or [{}])[me].get("active") or []
                    if actives:
                        act_id = str(actives[0].get("id"))
                except (IndexError, TypeError):
                    pass
                fp["attacks"].append(aid)
                fp.setdefault("attack_by", collections.Counter())[(aid, act_id)] += 1
                if fp["first_attack"] is None and turn is not None:
                    fp["first_attack"] = int(turn) + 1
            elif ty == "END":
                fp["ended_with"] += 1
    fp["win"] = win
    fp["agents"] = agents
    return fp


def summarize(fps, names, top=12):
    n = len(fps)
    if not n:
        return None
    tot = collections.Counter()
    for f in fps:
        for k, v in f["acts"].items():
            tot[k] += v
    games = n
    out = dict(
        n=n, wins=sum(1 for f in fps if f["win"]),
        acts_per_game={k: v / games for k, v in tot.most_common()},
        avg_turns=sum(f["turns"] for f in fps) / games,
        avg_attaches=sum(f["attaches"] for f in fps) / games,
        first_attack=[f["first_attack"] for f in fps if f["first_attack"]],
        avg_bench_t2=sum(f["bench_at"].get(1, 0) for f in fps) / games,
        avg_bench_t3=sum(f["bench_at"].get(2, 0) for f in fps) / games,
        own_turns=sum(len(f["own_turns"]) for f in fps) / games,
        per_own_turn={k: (v / max(1e-9, sum(len(f["own_turns"]) for f in fps)))
                      for k, v in tot.most_common()},
        early_plays=collections.Counter(c for f in fps for c in f["plays_early"]),
        attacks=collections.Counter(a for f in fps for a in f["attacks"]),
        attack_by=collections.Counter({k: v for f in fps for k, v in (f.get("attack_by") or {}).items()}),
        ended=sum(f["ended_with"] for f in fps) / games,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True, help="我方牌（pool 名或 4hex 后缀）")
    ap.add_argument("--opp", required=True, help="对手牌（pool 名或 4hex 后缀）")
    ap.add_argument("--dir", default="", help="单一回放目录")
    ap.add_argument("--label", default="我方")
    ap.add_argument("--compare", nargs="+", default=[], help='多组："目录:标签[:玩家]" ...')
    ap.add_argument("--player", default="", help="只统计该玩家（高手那批用）")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    pool = load_pool()
    names = card_names()

    def resolve(spec):
        for h, (nm, ids) in pool.items():
            if spec in (nm, h) or nm.endswith("__" + spec) or nm.endswith("-" + spec):
                return h, nm, ids
        raise SystemExit(f"✗ 找不到牌：{spec}")

    my_h, my_name, my_ids = resolve(a.deck)
    op_h, op_name, op_ids = resolve(a.opp)
    groups = a.compare or ([f"{a.dir}:{a.label}"] if a.dir else [])
    if not groups:
        raise SystemExit("✗ 需要 --dir 或 --compare")

    # 卡组差异一眼可见
    cm = collections.Counter(my_ids)
    co = collections.Counter(op_ids)
    print(f"我方牌 {my_name}；对手牌 {op_name}\n")
    res = {}
    for g in groups:
        parts = g.split(":")
        d = parts[0]
        label = parts[1] if len(parts) > 1 and parts[1] else d
        who = parts[2] if len(parts) > 2 else (a.player or "")
        fps = []
        for p in glob.glob(os.path.join(ROOT, d, "*.json")):
            f = scan_game(p, my_h, op_h, who or None)
            if f:
                fps.append(f)
        s = summarize(fps, names, a.top)
        res[label] = s
        if not s:
            print(f"== {label}: 没有符合条件的对局（{d}）")
            continue
        print(f"== {label}（{d}）：{s['n']} 局，胜 {s['wins']}，平均 {s['avg_turns']:.1f} 回合，"
              f"平均贴能 {s['avg_attaches']:.1f}，首次攻击回合中位 "
              f"{sorted(s['first_attack'])[len(s['first_attack'])//2] if s['first_attack'] else '—'}，"
              f"T2/T3 bench {s['avg_bench_t2']:.1f}/{s['avg_bench_t3']:.1f}")
        print("   每局动作数：" + "  ".join(f"{k} {v:.2f}" for k, v in list(s["acts_per_game"].items())[:9]))
        print("   每回合（我方回合）动作数：" + "  ".join(
            f"{k} {v:.2f}" for k, v in list(s["per_own_turn"].items())[:9]))
        if s["early_plays"]:
            print("   前 2 回合出的牌：" + "  ".join(
                f"{names.get(c, c)}×{v/groups.__len__() if False else v/s['n']:.1f}"
                for c, v in s["early_plays"].most_common(8)))
        print()

    # 逐项对比
    labels = [k for k, v in res.items() if v]
    if len(labels) >= 2:
        A, B = labels[0], labels[1]
        print(f"=== 对比：{A} vs {B}（同一副牌 {my_name} 打 {op_name}）")
        keys = set(res[A]["acts_per_game"]) | set(res[B]["acts_per_game"])
        print(f"{'动作':10s}{A:>10s}{B:>10s}{'差':>8s}")
        for k in sorted(keys, key=lambda x: -max(res[A]["acts_per_game"].get(x, 0), res[B]["acts_per_game"].get(x, 0))):
            x, y = res[A]["acts_per_game"].get(k, 0), res[B]["acts_per_game"].get(k, 0)
            print(f"{k:10s}{x:10.2f}{y:10.2f}{x-y:+8.2f}")
        print()
        ea = res[A]["early_plays"]; eb = res[B]["early_plays"]
        rows = [(c, ea.get(c, 0) / res[A]["n"], eb.get(c, 0) / res[B]["n"])
                for c in set(ea) | set(eb)]
        rows.sort(key=lambda r: -abs(r[1] - r[2]))
        print(f"{'前2回合出牌率':24s}{A:>10s}{B:>10s}{'差':>8s}")
        for c, x, y in rows[:12]:
            print(f"{(names.get(c, c) or c)[:23]:24s}{x:10.0%}{y:10.0%}{x-y:+8.0%}")
        print()
        ta, tb = res[A]["attacks"], res[B]["attacks"]
        rows = [(k, ta.get(k, 0) / res[A]["n"], tb.get(k, 0) / res[B]["n"]) for k in set(ta) | set(tb)]
        rows.sort(key=lambda r: -abs(r[1] - r[2]))
        print(f"{'招式（攻击方卡）':34s}{A:>10s}{B:>10s}{'差':>8s}")
        for k, x, y in rows[:8]:
            print(f"{('atk#%d' % k):20s}{x:10.0%}{y:10.0%}{x-y:+8.0%}")
        # 招式 × 攻击方（才知道是哪只宝可梦的招）
        ca, cb = res[A].get("attack_by") or {}, res[B].get("attack_by") or {}
        keys = set()
        for (aid, cid) in list(ca.keys()) + list(cb.keys()):
            keys.add((aid, cid))
        rows2 = [((aid, cid), ca.get((aid, cid), 0) / res[A]["n"], cb.get((aid, cid), 0) / res[B]["n"])
                 for (aid, cid) in keys]
        rows2.sort(key=lambda r: -abs(r[1] - r[2]))
        print()
        print(f"{'招式@攻击方（每局次数）':40s}{A:>8s}{B:>8s}")
        for (aid, cid), x, y in rows2[:8]:
            nm = names.get(cid or "", cid or "?")
            print(f"{f'atk#{aid} @ {nm}'[:39]:40s}{x:8.2f}{y:8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
