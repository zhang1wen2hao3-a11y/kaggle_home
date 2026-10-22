#!/usr/bin/env python3
"""解析 Kaggle 仿真赛的 episode 回放（kaggle_environments 格式的 JSON）。

**这个工具要回答的第一个问题**：回放里到底有没有对手的隐藏牌？

Kaggle env 的 episode JSON 结构（见 `logs/` 里那个历史样本）：

    {"name": "cabt", "steps": [[{agent0…}, {agent1…}], …], "rewards": [...], "statuses": [...]}
    每个 agent 的 {"observation": {"current": {...}, "select": {...}, "logs": [...]}, "action": [...]}

关键看 `current.players[i]`：
  · `deck` / `hand` / `prize` 里是**真的 cardId（整数）** → 回放揭示了隐藏信息
    ⇒ 能还原**对手的完整卡表**（这是本地评估最缺的东西：对手池只有我们自己）
  · 是 `None` / 空数组 → 回放只是"该 agent 当时看到的东西"，只能看到对手打出/弃掉的牌
    ⇒ 仍然有用（可推 archetype、可复盘自己的败因），但价值降一档

**实测结论（2026-10-22，episode 125203460，对手 Ryosuke Yoshikawa，299 步，我方胜）**：
回放**不揭示隐藏牌**（对手 `hand=None`、`deck=[]`）。但**弃牌堆是公开区**
（对手 34 张可见），加上场面/附着，能还原对手卡表的一部分（该局 18 种），
足够判定 archetype —— 该局对手在打**官方四套里的 Dragapult**（自改 Munkidori/
Judge/Dark 能量）。
⇒ 「还原完整卡表」做不到，但「识别对手 archetype + 统计对每种卡组的胜负」做得到。

用法：
    ./run.sh tools/parse_episodes.py --dir data/episodes
    ./run.sh tools/parse_episodes.py --dir data/episodes --dump-decks decks/field
"""
import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _cards(seq):
    """从 Card / dict / None 的混合序列里取 cardId。"""
    out = []
    for c in (seq or []):
        if isinstance(c, dict):
            out.append(c.get("id"))
        elif c is not None:
            out.append(getattr(c, "id", None))
    return out


def _player_cards(p):
    """一个 player 的**全部**可见牌：牌库+手牌+奖赏+弃牌+场上（含附着）。"""
    if not p:
        return [], {}
    detail = {
        "deck": _cards(p.get("deck")),
        "hand": _cards(p.get("hand")),
        "prize": _cards(p.get("prize")),
        "discard": _cards(p.get("discard")),
        "field": [],
    }
    for pk in list(p.get("active") or []) + list(p.get("bench") or []):
        if not pk:
            continue
        detail["field"].append(pk.get("id"))
        for k in ("energyCards", "tools", "preEvolution"):
            detail["field"] += _cards(pk.get(k))
    allc = detail["deck"] + detail["hand"] + detail["prize"] + detail["discard"] + detail["field"]
    return [c for c in allc if c is not None], detail


# 官方四套样例卡组的签名牌（用于识别对手在打哪套）
ARCHETYPES = {
    "dragapult":  {119, 120, 121},
    "palafin":    {105, 107},
    "iono":       {265, 268, 269},
    "lucario":    {673, 674, 677, 678},
    "abomasnow":  {721, 722, 723},
    "heracross":  {25, 781},
    "zeraora":    {952, 1369},
}


def classify(card_ids) -> str:
    """按签名牌命中数判断 archetype。"""
    got = set(c for c in card_ids if c)
    best, best_n = "other", 0
    for name, sig in ARCHETYPES.items():
        n = len(got & sig)
        if n > best_n:
            best, best_n = name, n
    return best if best_n > 0 else "other"


def load_episode(path):
    with open(path) as f:
        return json.load(f)


def analyse(path, verbose=True):
    d = load_episode(path)
    steps = d.get("steps") or []
    info = d.get("info") or {}
    rec = {
        "file": os.path.basename(path),
        "episode_id": info.get("EpisodeId"),
        "steps": len(steps),
        "statuses": d.get("statuses"),
        "rewards": d.get("rewards"),
        "agent_names": [a.get("Name") for a in (info.get("Agents") or [])],
        "hidden_revealed": None,      # 回放是否揭示了隐藏牌
        "opp_deck": None,             # 还原出来的对手卡表
        "our_trace": [],
        "winner": None,
    }
    if not steps:
        return rec

    # 找"信息最全"的一步：current 里 players 的牌最多
    best, best_n = None, -1
    for st in steps:
        for a in st:
            cur = (a.get("observation") or {}).get("current") or {}
            pls = cur.get("players") or []
            n = sum(len(_player_cards(p)[0]) for p in pls)
            if n > best_n:
                best_n, best = n, (cur, pls)
    if best:
        cur, pls = best
        me = int(cur.get("yourIndex", 0)) if cur else 0
        if len(pls) >= 2:
            opp = pls[1 - me] or {}
            opp_cards, detail = _player_cards(opp)
            rec["hidden_revealed"] = bool(detail.get("deck")) and bool(detail.get("hand"))
            rec["opp_detail"] = {k: len(v) for k, v in detail.items()}
            rec["opp_deck"] = sorted(c for c in opp_cards if c)
            rec["opp_deck_size"] = len(opp_cards)

    # 我们的动作轨迹（用每步 observation.select + action 还原）
    for st in steps:
        for a in st:
            obs = a.get("observation") or {}
            sel, act = obs.get("select"), a.get("action")
            if not sel or act is None:
                continue
            opts = sel.get("option") or []
            if act and 0 <= act[0] < len(opts):
                rec["our_trace"].append({
                    "turn": (obs.get("current") or {}).get("turn"),
                    "type": opts[act[0]].get("type"),
                    "ctx": sel.get("context"),
                })

    rec["archetype"] = classify(rec.get("opp_deck") or [])

    rew = d.get("rewards") or []
    if len(rew) >= 2 and rew[0] is not None and rew[1] is not None:
        rec["winner"] = 0 if rew[0] > rew[1] else (1 if rew[1] > rew[0] else 2)
    if verbose:
        print(f"  {rec['file']}: steps={rec['steps']} statuses={rec['statuses']} "
              f"rewards={rec['rewards']}")
        if rec["hidden_revealed"] is not None:
            mark = "✅ 揭示了隐藏牌" if rec["hidden_revealed"] else "❌ 未揭示（只有可见牌）"
            print(f"      对手牌: {rec.get('opp_deck_size')} 张 {mark}  细分={rec.get('opp_detail')}")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "episodes"))
    ap.add_argument("--dump-decks", default=None,
                    help="把还原出来的对手卡表写成 decks/<prefix>_<n>.csv（仅当回放揭示了隐藏牌）")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    if not files:
        raise SystemExit(f"{args.dir} 下没有 *.json。把 episode 文件放进去再跑。")

    print(f"共 {len(files)} 个 episode\n" + "=" * 78)
    recs = [analyse(f) for f in files]

    ok = [r for r in recs if r["statuses"] and all(s == "DONE" for s in r["statuses"])]
    revealed = [r for r in recs if r["hidden_revealed"]]
    print("=" * 78)
    print(f"  完成的对局: {len(ok)}/{len(recs)}")
    print(f"  回放揭示隐藏牌: {len(revealed)}/{len(recs)}"
          f"{'   ✅ 可以还原对手卡表' if revealed else '   ❌ 只能看到对手可见牌'}")

    # 对手卡牌频次（无论是否揭示，都统计"出现过的牌"）
    cnt = collections.Counter()
    for r in recs:
        for cid in (r.get("opp_deck") or []):
            cnt[cid] += 1
    if cnt:
        print(f"\n  对手出现过的牌（前 25 种，覆盖 {len(cnt)} 种）：")
        for cid, n in cnt.most_common(25):
            print(f"    {cid:<6} 出现 {n:>3} 次")

    # ---- archetype 分布 + 在该 archetype 上的胜负
    by_arch = collections.defaultdict(lambda: [0, 0])   # name -> [局数, 我方胜]
    for r in recs:
        a = r.get("archetype") or "other"
        by_arch[a][0] += 1
        if r.get("winner") == 0:          # player0 = 我们
            by_arch[a][1] += 1
    print("\n  对手 archetype 分布（我方在 player0 时的胜负）：")
    for a, (n, w) in sorted(by_arch.items(), key=lambda t: -t[1][0]):
        print(f"    {a:<12} {n:>3} 局   我方胜 {w}（{w/max(n,1):.0%}）")

    if args.dump_decks and revealed:
        os.makedirs(os.path.join(ROOT, args.dump_decks), exist_ok=True)
        seen = set()
        k = 0
        for r in revealed:
            deck = r.get("opp_deck") or []
            key = tuple(sorted(deck))
            if len(deck) != 60 or key in seen:
                continue
            seen.add(key)
            k += 1
            p = os.path.join(ROOT, args.dump_decks, f"field_{k:02d}.csv")
            with open(p, "w") as f:
                f.write("\n".join(map(str, deck)) + "\n")
        print(f"\n  已写出 {k} 套去重后的对手卡表到 {args.dump_decks}/")
    elif args.dump_decks:
        print("\n  ⚠️ 回放没有揭示隐藏牌，无法还原完整卡表（--dump-decks 跳过）")


if __name__ == "__main__":
    main()
