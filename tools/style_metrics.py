#!/usr/bin/env python3
"""**打法体检表**（2026-11-22 定的五项指标 + 闸门）

为什么需要：今天把"同一对位、同一副牌"拆开比，发现我们的驾驶员在**做事效率**上系统性落后
（e1d7 打 7e39：贴能 −6.5/局、攻击 −2/局、撤退 +2.2/局、首攻晚 1 回合；0904 同方向）。
面板胜率会抖动，这五项不会 —— 它们是**策略指纹**。

指标（每局 / 每我方回合）：
  ① attach  贴能     ② attack  攻击     ③ retreat 撤退
  ④ evolve  进化（对进化链牌才有意义）    ⑤ first_attack 首次攻击回合
辅助：own_turns、bench T2/T3、play 出牌数

用法
----
    # 单个回放目录
    ./tools/py tools/style_metrics.py --dir epsoide_e1d7r4 --deck z_ogerpon_toolbox_miwa__e1d7
    # 与人类基线一起出一张表
    ./tools/py tools/style_metrics.py --compare "epsoide_e1d7r4:我们@e1d7" "epsoide_miwa:高手ミワハルキ@e1d7:ミワハルキ" \
        --deck z_ogerpon_toolbox_miwa__e1d7 --opp teal_mask_ogerpon_ex_applin__7e39 --csv docs/打法体检表.csv
    # 闸门：候选的五项 vs 人类基线（不超过 RETREAT×1.3、不低于 ATTACH×0.85/ATTACK×0.90）
    ./tools/py tools/style_metrics.py --gate --dir <候选回放> --baseline <人类回放> --deck X --opp Y
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OT = {7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY", 12: "RETREAT", 13: "ATTACK", 14: "END"}
# 闸门（我们相对人类基线的容忍度）
GATE = dict(attach=0.85, attack=0.90, retreat=1.30, evolve=0.80)


def dh(ids) -> str:
    """★ 与流水线同口径（数值排序，见 framework/replay.py）"""
    return hashlib.sha256(",".join(str(x) for x in sorted(map(int, ids))).encode()).hexdigest()[:8]


def pool() -> dict:
    out = {}
    for p in glob.glob(os.path.join(ROOT, "decks", "pool_field", "*.csv")):
        ids = [l.strip() for l in open(p) if l.strip()]
        out[dh(ids)] = os.path.basename(p)[:-4]
    return out


def scan(path, my_h, opp_h, player=None):
    try:
        d = json.load(open(path))
        decks = [[str(x) for x in (s.get("action") or [])] for s in d["steps"][1]]
    except Exception:
        return None
    if len(decks) != 2 or any(len(x) != 60 for x in decks):
        return None
    hs = [dh(x) for x in decks]
    me = 0 if (hs[0] == my_h and hs[1] == opp_h) else (1 if (hs[1] == my_h and hs[0] == opp_h) else -1)
    if me < 0:
        return None
    ag = [a.get("Name", "") for a in (d.get("info", {}).get("Agents") or [])]
    if player and player not in (ag[me] or ""):
        return None
    rw = d.get("rewards") or [0, 0]
    win = (0 if rw[0] > rw[1] else 1) == me
    cnt = collections.Counter()
    own_turns, first_attack, bench = set(), None, {}
    for row in d["steps"]:
        if me >= len(row):
            break
        seat = row[me]
        obs = seat.get("observation") or {}
        sel = obs.get("select") or {}
        cur = obs.get("current") or {}
        turn = cur.get("turn")
        if sel.get("type") != 0:
            continue
        if turn is not None:
            own_turns.add(int(turn))
            try:
                bench[int(turn)] = max(bench.get(int(turn), 0),
                                       len((cur.get("players") or [{}])[me].get("bench") or []))
            except (IndexError, TypeError):
                pass
        opts = sel.get("option") or []
        for a in (seat.get("action") or []):
            try:
                o = opts[int(a)]
            except (IndexError, ValueError, TypeError):
                continue
            ty = OT.get(int(o.get("type", -1)))
            if not ty:
                continue
            cnt[ty] += 1
            if ty == "ATTACK" and first_attack is None and turn is not None:
                first_attack = int(turn) + 1
    return dict(win=win, cnt=cnt, turns=len(own_turns), first_attack=first_attack,
                bench2=bench.get(1, 0), bench3=bench.get(2, 0))


def agg(fps):
    n = len(fps)
    if not n:
        return None
    c = collections.Counter()
    for f in fps:
        c.update(f["cnt"])
    turns = sum(f["turns"] for f in fps) or 1
    fa = [f["first_attack"] for f in fps if f["first_attack"]]
    return dict(n=n, wins=sum(f["win"] for f in fps),
                attach=c["ATTACH"] / n, attack=c["ATTACK"] / n, retreat=c["RETREAT"] / n,
                evolve=c["EVOLVE"] / n, play=c["PLAY"] / n, end=c["END"] / n,
                own_turns=turns / n,
                first_attack=(sorted(fa)[len(fa) // 2] if fa else None),
                bench2=sum(f["bench2"] for f in fps) / n, bench3=sum(f["bench3"] for f in fps) / n)


_CACHE = {}


def collect(dirp, my_h, opp_h, player=""):
    """带缓存：同一目录重复调用（闸门里会调用两次）不再重扫 600+ 局回放。"""
    key = (dirp, my_h, opp_h, player)
    if key in _CACHE:
        return _CACHE[key]
    fps = []
    for p in glob.glob(os.path.join(ROOT, dirp, "*.json")):
        f = scan(p, my_h, opp_h, player or None)
        if f:
            fps.append(f)
    _CACHE[key] = agg(fps)
    return _CACHE[key]


def line(label, s):
    if not s:
        return f"  {label:28s} 无对局"
    return (f"  {label:28s} {s['n']:3d}局 胜{s['wins']:3d} | 贴能 {s['attach']:5.2f} 攻击 {s['attack']:5.2f} "
            f"撤退 {s['retreat']:5.2f} 进化 {s['evolve']:5.2f} | 首攻 T{s['first_attack']} | "
            f"回合 {s['own_turns']:.1f} bench {s['bench2']:.1f}/{s['bench3']:.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True)
    ap.add_argument("--opp", default="", help="只统计该对手（留空=全部对手）")
    ap.add_argument("--dir", default="")
    ap.add_argument("--compare", nargs="+", default=[], help='"目录:标签[:玩家]" ...')
    ap.add_argument("--gate", action="store_true", help="拿 --dir 的五项对比 --baseline（人类）并判过/不过")
    ap.add_argument("--baseline", default="", help="人类基线回放目录")
    ap.add_argument("--baseline-player", default="")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    poolmap = pool()
    my_h = next((h for h, nm in poolmap.items() if nm == a.deck or nm.endswith("__" + a.deck)), None)
    if not my_h:
        raise SystemExit(f"✗ 找不到牌 {a.deck}")
    opp_h = None
    if a.opp:
        opp_h = next((h for h, nm in poolmap.items() if nm == a.opp or nm.endswith("__" + a.opp)), None)
        if not opp_h:
            raise SystemExit(f"✗ 找不到对手牌 {a.opp}")
    if opp_h is None:
        # 不指定对手：用"该牌打任意对手"（把 opp_h 换成一个通配位）
        pass

    def run(dirp, player="", any_opp=False):
        if not any_opp:
            return collect(dirp, my_h, opp_h, player)
        allf = []
        for oh in set(poolmap):
            if oh == my_h:
                continue
            s = collect(dirp, my_h, oh, player)
            if s:
                allf.append(s)
        return None

    groups = a.compare or ([f"{a.dir}:{a.dir}"] if a.dir else [])
    if not groups:
        raise SystemExit("✗ 需要 --dir 或 --compare")
    rows = []
    print(f"我方牌 {poolmap.get(my_h)}" + (f"；对手 {poolmap.get(opp_h)}" if opp_h else "（全部对手）"))
    base = None
    for g in groups:
        parts = g.split(":")
        dirp = parts[0]
        label = parts[1] if len(parts) > 1 and parts[1] else dirp
        who = parts[2] if len(parts) > 2 else ""
        s = collect(dirp, my_h, opp_h, who) if opp_h else None
        print(line(f"{label}（{dirp}）", s))
        if s:
            rows.append((label, dirp, who, s))
    if a.gate and a.baseline:
        bs = collect(a.baseline, my_h, opp_h, a.baseline_player)
        me = collect(a.dir or groups[0].split(":")[0], my_h, opp_h)
        print(f"\n闸门（我方 vs 人类基线）")
        print(line("人类基线", bs))
        print(line("我方", me))
        ok = True
        for k, thr in GATE.items():
            if not bs or not me:
                print("  （缺一侧数据，无法判定）")
                ok = False
                break
            b, m = bs[k], me[k]
            if b <= 0:
                continue
            if k == "retreat":
                good = m <= b * thr
            else:
                good = m >= b * thr
            print(f"  {k:8s} 我们 {m:5.2f} / 人类 {b:5.2f} = {m/b*100:5.0f}%  阈值 {'≤'+format(thr,'.2f') if k=='retreat' else '≥'+format(thr,'.2f')}"
                  f"  ⇒ {'✅' if good else '❌'}")
            ok = ok and good
        print("  ⇒ " + ("✅ 通过" if ok else "❌ 不过（面板涨了也别信）"))
        return 0 if ok else 3
    if a.csv and rows:
        out = os.path.join(ROOT, a.csv)
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["组", "回放目录", "玩家", "局数", "胜", "贴能/局", "攻击/局", "撤退/局", "进化/局",
                        "出牌/局", "首攻回合", "我方回合数", "bench_T2", "bench_T3"])
            for label, dirp, who, s in rows:
                w.writerow([label, dirp, who, s["n"], s["wins"], f"{s['attach']:.2f}", f"{s['attack']:.2f}",
                            f"{s['retreat']:.2f}", f"{s['evolve']:.2f}", f"{s['play']:.2f}",
                            s["first_attack"], f"{s['own_turns']:.1f}", f"{s['bench2']:.2f}", f"{s['bench3']:.2f}"])
        print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
