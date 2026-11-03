#!/usr/bin/env python3
"""量化"衰减型资源的使用时机"假设。

## 假设

对手那套牌（Dragapult+Blaziken）的核心是一次性/衰减型 setup 卡：
Buddy-Buddy Poffin（要空板凳位）、Poké Pad、Ultra Ball（要先有弃牌料）、
Rare Candy（只能用在未进化基础上）、Lillie's Determination（手牌越少抽越多）。
这些卡**越晚用越贬值**。

若对手在"有 setup 卡可打"时几乎总是**立刻打**，而我们的策略倾向先做别的
（进化/贴能量/攻击），那就是分歧的真实内容。

## 口径

对每一帧：从 `select.option` 里解析出**手牌卡**（type==7 的 index → obs 手牌），
按卡名判断是否属于 SETUP_CARDS；只在「≥1 张 setup 卡可打 AND ≥1 个非 setup 选项」的
帧上统计，比较他/我们选择 setup 卡的比例。

用法：
    ./run.sh tools/measure_setup_timing.py --dir epsoide_dipam \
        --agent-name "Dipam Chakraborty" --games 12
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from framework.replay import iter_decisions, decks_of   # noqa: E402
from agentpool import get_agent                         # noqa: E402

SETUP_CARDS = {
    "Buddy-Buddy Poffin", "Poké Pad", "Poke Pad", "Ultra Ball", "Rare Candy",
    "Nest Ball", "Buddy Poffin", "Earthen Vessel", "Super Rod",
    "Night Stretcher", "Lillie's Determination", "Professor's Research",
}


def load_sheet():
    sheet = {}
    p = os.path.join(ROOT, "data", "EN_Card_Data_R2_full.csv")
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                sheet[int(r["Card ID"])] = r["Card Name"]
            except Exception:
                pass
    return sheet


def hand_of(obs, seat):
    cur = obs.get("current") or {}
    pls = cur.get("players") or []
    if seat < len(pls):
        return pls[seat].get("hand") or []
    return []


def opt_is_setup(obs, seat, opt, sheet):
    """type==7 的选项带 index → 手牌第 index 张。"""
    if not opt or opt.get("type") != 7:
        return False
    idx = opt.get("index")
    hand = hand_of(obs, seat)
    if not isinstance(idx, int) or not (0 <= idx < len(hand)):
        return False
    h = hand[idx]
    cid = h.get("id") if isinstance(h, dict) else None
    nm = sheet.get(cid)
    return bool(nm and nm in SETUP_CARDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="epsoide_dipam")
    ap.add_argument("--agent-name", required=True)
    ap.add_argument("--games", type=int, default=12)
    a = ap.parse_args()
    sheet = load_sheet()

    files = sorted(glob.glob(os.path.join(ROOT, a.dir, "*.json")))
    st = collections.Counter()
    done = 0
    for fp in files:
        if done >= a.games:
            break
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        steps = d.get("steps") or []
        names = [x.get("Name", "") for x in ((d.get("info") or {}).get("Agents") or [])]
        if a.agent_name not in names:
            continue
        seat = names.index(a.agent_name)
        decks = decks_of(steps)
        if seat not in decks or len(decks[seat]) != 60:
            continue
        fn = get_agent(decks[seat])
        done += 1
        for t, j, obs, act in iter_decisions(steps):
            if j != seat:
                continue
            opts = (obs.get("select") or {}).get("option") or []
            if not act or act[0] >= len(opts):
                continue
            flags = [opt_is_setup(obs, seat, o, sheet) for o in opts]
            if not any(flags):
                continue
            n_setup = sum(flags)
            if n_setup == len(opts):
                st["只有 setup 可选"] += 1
                continue
            st["可判定帧"] += 1
            his_setup = flags[act[0]]
            st["他打setup" if his_setup else "他做别的"] += 1
            call = dict(obs)
            call.setdefault("logs", [])
            call.setdefault("remainingOverageTime", 600.0)
            call["player"] = seat
            call["step"] = t
            try:
                got = list(fn(dict(call)) or [])
            except Exception:
                continue
            if not got or got[0] >= len(opts):
                continue
            our_setup = flags[got[0]]
            st["我们打setup" if our_setup else "我们做别的"] += 1
            if his_setup and not our_setup:
                st["他打_我们没打"] += 1
            if our_setup and not his_setup:
                st["我们打_他没打"] += 1

    n = st["可判定帧"]
    if not n:
        sys.exit("没有可判定帧")
    print(f"局数 {done}；「有 setup 卡可打且还有其他选项」的帧 {n}\n")
    for who in ("他", "我们"):
        s, o = st[f"{who}打setup"], st[f"{who}做别的"]
        t = s + o
        print(f"  {who:<4} 打 setup {s:5d} ({s / max(t, 1):6.2%})   "
              f"做别的 {o:5d}")
    print()
    print(f"  他打 / 我们没打 : {st['他打_我们没打']:5d} 帧")
    print(f"  我们打 / 他没打 : {st['我们打_他没打']:5d} 帧")
    print(f"  （另有 {st['只有 setup 可选']} 帧只有 setup 可选，未计入）")


if __name__ == "__main__":
    main()
