#!/usr/bin/env python3
"""诊断指定 archetype 的真实败局：把 replay 拆成"我们做了什么 / 怎么输的"。

动机：`shuppet` 在我们提交 57586502 的真实战绩里是 **0 胜 9 负**，而本地闸门说是 0.95。
本地闸门是"我们的弱策略开 shuppet"当对手，真实对手是强玩家 —— 所以本地必然虚高。
本工具不猜，直接把那 9 局的决策分布和终局状态摆出来。

用法：
    python3 tools/diagnose_losses.py --csv data/episodes_ala.csv --archetype shuppet
    python3 tools/diagnose_losses.py --csv data/episodes_ala.csv --archetype ogerpon --control
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
KAGGLE = os.environ.get("KAGGLE_BIN", "/home/zhang/miniforge3/bin/kaggle")
CARD_CSV = ROOT / "data" / "EN_Card_Data_R2_full.csv"

# select.option.type 语义（来自引擎日志实证）
OPT_TYPE = {
    7: "打出手牌",
    8: "区域卡",
    9: "区域卡",
    10: "进化/场内操作",
    13: "攻击",
    14: "结束回合",
}


def my_name() -> str:
    cfg = pathlib.Path(os.path.expanduser("~/.kaggle/kaggle.json"))
    return json.loads(cfg.read_text())["username"]


def fetch(eid: int, dest: pathlib.Path) -> pathlib.Path | None:
    env = dict(os.environ, KAGGLE_CONFIG_DIR=os.path.expanduser("~/.kaggle"))
    p = subprocess.run([KAGGLE, "competitions", "replay", str(eid), "-p", str(dest), "-q"],
                       capture_output=True, text=True, timeout=300, env=env)
    if p.returncode != 0:
        return None
    return next(dest.glob(f"*{eid}*.json"), None)


def cid_of(x):
    """手牌元素可能是 int，也可能是 dict（含 cardId/id）。"""
    if isinstance(x, int):
        return x
    if isinstance(x, dict):
        for k in ("cardId", "id", "card_id"):
            if k in x:
                return x[k]
    return None


def analyse(path: pathlib.Path, me: str, names: dict[int, str]) -> dict | None:
    d = json.loads(path.read_text())
    agents = [a["Name"] for a in d["info"]["Agents"]]
    if me not in agents:
        return None
    i = agents.index(me)
    j = 1 - i
    steps = d["steps"]
    types = collections.Counter()
    played = collections.Counter()
    attacks = 0
    for t in range(1, len(steps) - 1):
        obs = steps[t][i].get("observation")
        if not obs or not obs.get("current"):
            continue
        if obs["current"].get("yourIndex") != i:
            continue
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        act = steps[t + 1][i].get("action")
        if not isinstance(act, list) or not act:
            continue
        k = act[0]
        if not isinstance(k, int) or k < 0 or k >= len(opts):
            continue
        ty = opts[k].get("type")
        types[ty] += 1
        if ty == 13:
            attacks += 1
        if ty == 7:
            idx = opts[k].get("index")
            hand = (obs["current"]["players"][i].get("hand") or [])
            if isinstance(idx, int) and 0 <= idx < len(hand):
                cid = cid_of(hand[idx])
                if cid is not None:
                    played[names.get(cid, str(cid))] += 1
    # 终局
    last = None
    for t in range(len(steps) - 1, -1, -1):
        o = steps[t][i].get("observation")
        if o and o.get("current"):
            last = o["current"]
            break
    fin = {}
    if last:
        me_p, op_p = last["players"][i], last["players"][j]
        fin = dict(my_prize=len(me_p.get("prize") or []),
                   op_prize=len(op_p.get("prize") or []),
                   my_deck=me_p.get("deckCount"), op_deck=op_p.get("deckCount"),
                   my_bench=len(me_p.get("bench") or []),
                   op_bench=len(op_p.get("bench") or []),
                   round=last.get("round"))
    return dict(episode=d["info"]["EpisodeId"], opponent=agents[j],
                won=int(d["rewards"][i] > 0), steps=len(steps),
                attacks=attacks,
                types=dict(types),
                top_played=played.most_common(6), **fin)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--archetype", required=True, help="对手原型前缀，如 shuppet")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--control", action="store_true", help="反过来挑我们赢的局做对照")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.csv, encoding="utf-8"))]
    pick = [r for r in rows
            if r["op_deck"].startswith(a.archetype)
            and (r["won"] == "1") == bool(a.control)]
    pick = pick[:a.limit]
    if not pick:
        raise SystemExit("没有匹配的对局")
    me = my_name()
    names = {}
    with open(CARD_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                names[int(r["Card ID"])] = r["Card Name"]
            except (TypeError, ValueError):
                pass
    out = []
    with tempfile.TemporaryDirectory() as td:
        for n, r in enumerate(pick, 1):
            eid = int(r["episode"])
            p = fetch(eid, pathlib.Path(td))
            if not p:
                print(f"  [{n}] {eid} 取不到，跳过")
                continue
            rec = analyse(p, me, names)
            p.unlink()
            if rec:
                out.append(rec)
                print(f"  [{n}] {eid} vs {rec['opponent'][:18]:18s} "
                      f"{'胜' if rec['won'] else '负'} steps={rec['steps']:3d} "
                      f"攻击{rec['attacks']:3d}次 奖品 我{rec.get('my_prize')}/敌{rec.get('op_prize')} "
                      f"牌库 我{rec.get('my_deck')}/敌{rec.get('op_deck')}")
    if not out:
        return 1
    print(f"\n=== 汇总（{len(out)} 局，{'赢' if a.control else '输'}）===")
    tot = collections.Counter()
    for rec in out:
        for k, v in rec["types"].items():
            tot[k] += v
    n = sum(tot.values())
    print(f"我方决策总数 {n}，动作类型分布：")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"  type {k:>2} {OPT_TYPE.get(k, '?'):12s} {v:5d}  ({100*v/n:4.1f}%)")
    avg_atk = sum(r["attacks"] for r in out) / len(out)
    print(f"\n平均每局我方攻击次数 {avg_atk:.1f}")
    print(f"平均步数 {sum(r['steps'] for r in out)/len(out):.0f}")
    print(f"输/赢时我方剩余奖品均值 {sum((r.get('my_prize') or 0) for r in out)/len(out):.1f}")
    print(f"我方剩余牌库均值 {sum((r.get('my_deck') or 0) for r in out)/len(out):.1f}")
    print("\n我方出牌 Top（跨局合计）：")
    agg = collections.Counter()
    for rec in out:
        for nm, c in rec["top_played"]:
            agg[nm] += c
    for nm, c in agg.most_common(10):
        print(f"  {nm:32s} {c}")
    pathlib.Path(ROOT / "logs" / f"diagnose_{a.archetype}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细已写 logs/diagnose_{a.archetype}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
