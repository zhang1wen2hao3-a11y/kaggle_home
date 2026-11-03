#!/usr/bin/env python3
"""量「攻击时的手牌数」——验证 Powerful Hand 类引擎是否被手牌管理拖垮。

Powerful Hand 伤害 = 20 × 手牌数。若策略习惯把牌打光再攻击，伤害就会塌掉。
本工具对指定两组对手取样，统计每次「我方选择攻击」那一刻的手牌数。

用法：
    python3 tools/handsize_at_attack.py --csv data/episodes_ala.csv \
        --a shuppet --b ogerpon --n 5
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pathlib
import statistics as st
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
KAGGLE = os.environ.get("KAGGLE_BIN", "/home/zhang/miniforge3/bin/kaggle")


def my_name() -> str:
    return json.loads(pathlib.Path(os.path.expanduser("~/.kaggle/kaggle.json")).read_text())["username"]


def fetch(eid: int, dest: pathlib.Path) -> pathlib.Path | None:
    env = dict(os.environ, KAGGLE_CONFIG_DIR=os.path.expanduser("~/.kaggle"))
    p = subprocess.run([KAGGLE, "competitions", "replay", str(eid), "-p", str(dest), "-q"],
                       capture_output=True, text=True, timeout=300, env=env)
    if p.returncode != 0:
        return None
    return next(dest.glob(f"*{eid}*.json"), None)


def sample(path: pathlib.Path, me: str) -> dict | None:
    """返回：攻击次数、攻击时手牌数列表、非攻击决策时手牌数列表、终局"""
    d = json.loads(path.read_text())
    agents = [a["Name"] for a in d["info"]["Agents"]]
    if me not in agents:
        return None
    i = agents.index(me)
    steps = d["steps"]
    at_hand, other_hand, attacks = [], [], 0
    for t in range(1, len(steps) - 1):
        obs = steps[t][i].get("observation")
        if not obs or not obs.get("current") or obs["current"].get("yourIndex") != i:
            continue
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        act = steps[t + 1][i].get("action")
        if not isinstance(act, list) or not act or not isinstance(act[0], int):
            continue
        k = act[0]
        if not (0 <= k < len(opts)):
            continue
        hand = obs["current"]["players"][i].get("hand") or []
        if opts[k].get("type") == 13:
            attacks += 1
            at_hand.append(len(hand))
        else:
            other_hand.append(len(hand))
    return dict(episode=d["info"]["EpisodeId"], won=int(d["rewards"][i] > 0),
                attacks=attacks, at_hand=at_hand, other_hand=other_hand,
                steps=len(steps))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--a", required=True, help="目标原型前缀（如 shuppet）")
    ap.add_argument("--b", default=None, help="对照原型前缀（如 ogerpon）")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    me = my_name()
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for tag, pref in (("A", a.a), ("B", a.b)):
            if not pref:
                continue
            pick = [r for r in rows if r["op_deck"].startswith(pref)][:a.n]
            recs = []
            for r in pick:
                p = fetch(int(r["episode"]), pathlib.Path(td))
                if not p:
                    continue
                rec = sample(p, me)
                p.unlink()
                if rec:
                    recs.append(rec)
                    ah = f"{st.mean(rec['at_hand']):.1f}" if rec["at_hand"] else "-"
                    oh = f"{st.mean(rec['other_hand']):.1f}" if rec["other_hand"] else "-"
                    print(f"  [{tag}] ep {rec['episode']} {'胜' if rec['won'] else '负'} "
                          f"攻击 {rec['attacks']:3d} 次  攻击时手牌均值 {ah:>5s}  "
                          f"其它决策手牌均值 {oh:>5s}")
            out[tag] = recs
    print()
    for tag, recs in out.items():
        if not recs:
            continue
        ath = [x for r in recs for x in r["at_hand"]]
        oth = [x for r in recs for x in r["other_hand"]]
        wins = sum(r["won"] for r in recs)
        print(f"=== {tag}（{tag == 'A' and a.a or a.b}）：{len(recs)} 局 {wins} 胜 ===")
        if ath:
            print(f"  攻击时手牌：均值 {st.mean(ath):.1f}  中位 {st.median(ath):.0f}  "
                  f"P25 {sorted(ath)[len(ath)//4]}  P75 {sorted(ath)[3*len(ath)//4]}  样本 {len(ath)}")
            print(f"  ⇒ 期望伤害 ≈ {20*st.mean(ath):.0f} 点/次")
        if oth:
            print(f"  其它决策时手牌均值 {st.mean(oth):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
