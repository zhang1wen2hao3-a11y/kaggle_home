#!/usr/bin/env python3
"""在指定回放里逐决策对比"某座位实际的选择" vs "我们的端口在同一位置会怎么选"。

用途：把策略差距量化。`check_port_fidelity.py` 只对比我们自己的座位（验线上等价），
这里可以指定任意座位——例如榜首 Dipam 的座位，从而回答
"他的策略和我们差多少、差在哪些决策上"。

用法：
    ./run.sh tools/replay_policy_diff.py \
        --replay epsoide_dipam/episode-125308777-replay.json \
        --agent-name "Dipam Chakraborty" --agent-dir /tmp/s/d3eaa
    ./run.sh tools/replay_policy_diff.py \
        --replay epsoide_dipam/episode-125308777-replay.json \
        --agent-name "zhang1wen2hao3" --agent-dir build/port_ala      # 自检应 ~100%
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from check_port_fidelity import load_agent, prepare          # noqa: E402
from framework.replay import iter_decisions, decks_of        # noqa: E402


def tname(o):
    """选项类型 + 关键字段，用来读懂'他打了什么、我们想打什么'。"""
    if not o:
        return "?"
    t = o.get("type")
    extra = []
    if o.get("cardId") is not None:
        extra.append(f"card{o['cardId']}")
    if o.get("attackId") is not None:
        extra.append(f"atk{o['attackId']}")
    return f"t{t}" + ("(" + ",".join(extra) + ")" if extra else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--agent-name", required=True)
    ap.add_argument("--agent-dir", default=None,
                    help="我们的端口 staging 目录；给了 --auto-deck 时可省略")
    ap.add_argument("--auto-deck", action="store_true",
                    help="按回放里该座位实际使用的卡组自动 staging（跨多局/多卡组批处理用）")
    ap.add_argument("--pool", default="decks/pool_field")
    a = ap.parse_args()

    d = json.load(open(a.replay))
    names = [x.get("Name", "") for x in ((d.get("info") or {}).get("Agents") or [])]
    if a.agent_name not in names:
        sys.exit(f"回放里没有 {a.agent_name}，实际是 {names}")
    seat = names.index(a.agent_name)
    decks = decks_of(d.get("steps") or [])
    print(f"回放 {os.path.basename(a.replay)}")
    print(f"  {a.agent_name} = 座位 {seat}，卡组 {sorted(decks.get(seat, []))[:3]}…")


    def _auto_agent_dir(deck_ids):
        """按卡组哈希找一个 staging 目录（缓存在 /tmp/autoagent/<hash>）。"""
        import hashlib
        import shutil
        hh = hashlib.sha256(",".join(str(x) for x in sorted(deck_ids)).encode()).hexdigest()
        out = os.path.join("/tmp/autoagent", hh[:12])
        if not os.path.exists(os.path.join(out, "model.pt")):
            shutil.rmtree(out, ignore_errors=True)
            os.makedirs(out, exist_ok=True)
            bundle = os.path.join(ROOT, "build", "port_plain")
            for e in os.listdir(bundle):
                if e in ("model.pt", "deck.csv"):
                    continue
                os.symlink(os.path.abspath(os.path.join(bundle, e)), os.path.join(out, e))
            os.symlink(os.path.join(bundle, "model.pt"), os.path.join(out, "model.pt"))
            with open(os.path.join(out, "deck.csv"), "w") as f:
                f.write("\n".join(str(x) for x in deck_ids) + "\n")
        return out

    agent_dir = a.agent_dir
    if a.auto_deck:
        seat_deck = decks.get(seat) or []
        if len(seat_deck) < 60:
            sys.exit("回放里取不到该座位的 60 张卡组")
        agent_dir = _auto_agent_dir(seat_deck)
    if not agent_dir:
        sys.exit("要么给 --agent-dir，要么加 --auto-deck")

    print(f"  用我们的端口（{agent_dir}）在该座位复演\n", flush=True)
    fn = load_agent(prepare(agent_dir))
    stat = collections.Counter()
    diffs = []
    for t, j, obs, act in iter_decisions(d.get("steps") or []):
        if j != seat:
            continue
        call = dict(obs)
        call.setdefault("logs", [])
        call.setdefault("remainingOverageTime", 600.0)
        call["player"] = seat
        call["step"] = t
        try:
            got = list(fn(dict(call)) or [])
        except Exception as e:
            stat["异常"] += 1
            continue
        stat["决策"] += 1
        if got == act:
            stat["一致"] += 1
            continue
        opts = (obs.get("select") or {}).get("option") or []
        sa = opts[act[0]] if act and 0 <= act[0] < len(opts) else None
        sb = opts[got[0]] if got and 0 <= got[0] < len(opts) else None
        # 语义一致（同类型同目标，只是挑了不同副本）
        if sa and sb and (sa.get("type"), sa.get("cardId"), sa.get("attackId")) == \
                         (sb.get("type"), sb.get("cardId"), sb.get("attackId")):
            stat["语义一致(索引不同)"] += 1
            continue
        stat["实质分歧"] += 1
        diffs.append((t, len(opts), tname(sa), tname(sb)))

    tot = max(stat["决策"], 1)
    print(f"决策帧数 {stat['决策']}（回合 {d.get('steps') and len(d['steps'])} 步）")
    for k in ("一致", "语义一致(索引不同)", "实质分歧", "异常"):
        if stat[k]:
            print(f"  {k:<18} {stat[k]:5d}  ({stat[k] / tot:6.2%})")
    print(f"\n我们的端口与该座位的一致率 = "
          f"{(stat['一致'] + stat['语义一致(索引不同)']) / tot:.2%}")

    if diffs:
        print(f"\n前 20 个实质分歧（帧, 选项数, 他选了, 我们选了）:")
        for t, n, x, y in diffs[:20]:
            print(f"  t={t:4d}  n={n:2d}   他 {x:>14s}   我们 {y:>14s}")
        t7_away = sum(n for (x, y), n in
                      collections.Counter((x.split("(")[0], y.split("(")[0])
                                          for _, _, x, y in diffs).items()
                      if x == "t7" and y != "t7")
        to_t7 = sum(n for (x, y), n in
                    collections.Counter((x.split("(")[0], y.split("(")[0])
                                        for _, _, x, y in diffs).items()
                    if y == "t7" and x != "t7")
        print(f"\n  他出牌(t7)而我们没出: {t7_away} 次 | 我们出牌而他没出: {to_t7} 次")
        print(f"\n  分歧集中在:")
        agg = collections.Counter((x.split("(")[0], y.split("(")[0]) for _, _, x, y in diffs)
        for (x, y), n in agg.most_common(6):
            print(f"    {x:>6s} → {y:<6s}  {n:3d} 次")


def _unused():
    pass


if __name__ == "__main__":
    main()
