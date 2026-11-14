#!/usr/bin/env python3
"""**生成 a→b 胜率矩阵**：docs/the-pockmon-battle-choices-win-rate.csv

口径（与 choice / choice-detail 两张表一致，便于对照）：
    行 = 我方卡组 a 的**最强驾驶员**（choice 表选出来的那个：db04→g6_db04、2617→ph3_2617_v1、
         26c6→r1A、0d86→shuppet_specialist，其余端口权重）
    列 = 对手用**端口权重 + 该卡组**（choice 表"只差卡组"的口径）
    格 = 我方 a 打对手 b 的胜率（0~1，4 位小数）
    对角 a→a（自己打自己）填 0.5（镜像期望值）

数据来源（优先级从高到低）：
    1) logs/winrate/<a>_<b>.json      —— tools/fill_winrate.sh 现跑的（n=200, seed 970000）
    2) docs/the-pockmon-battle-choice-detail.csv 里 状态=实测 的格子（老 eval，n=178~300）

用法：
    ./tools/py tools/build_winrate_table.py            # 写 csv
    ./tools/py tools/build_winrate_table.py --dry      # 只看覆盖情况
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETAIL = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-detail.csv")
OUTCSV = os.path.join(ROOT, "docs", "the-pockmon-battle-choices-win-rate.csv")
NEWDIR = os.path.join(ROOT, "logs", "winrate")
DECKS = ["0904", "0bf5", "0d86", "2617", "26c6", "3eaa", "7e39",
         "9d6e", "db04", "e1d7", "e8ee", "f5f8"]


def load_detail():
    """老 eval 的实测结果：{(a,b): (胜率, n, 驾驶员)}"""
    txt = open(DETAIL, "rb").read().decode("gb18030", errors="replace")
    out = {}
    for r in csv.DictReader(io.StringIO(txt)):
        if r.get("状态") != "实测" or not r.get("胜率"):
            continue
        try:
            out[(r["挑战者"], r["被挑战者"])] = (float(r["胜率"]), r.get("样本场次", ""), r.get("驾驶员", ""))
        except ValueError:
            continue
    return out


SIDECAR = os.path.join(ROOT, "docs", "the-pockmon-battle-choices-win-rate-pilot.csv")


def load_pilots():
    """sidecar：{我方卡组: (驾驶员, [面板 tag...])}"""
    out = {}
    if not os.path.exists(SIDECAR):
        return out
    txt = open(SIDECAR, "rb").read().decode("gb18030", errors="replace")
    for r in csv.DictReader(io.StringIO(txt)):
        out[r["我方卡组"]] = (r["驾驶员"], [t for t in r["面板tag"].split(";") if t])
    return out


def pooled_panel(tags):
    """把同协议多个种子的面板结果按格子池化：{对手卡组: (胜率, n)}"""
    acc = {}
    for t in tags:
        for f in glob.glob(os.path.join(ROOT, "logs", "meta", t, "vs_*.json")):
            try:
                o = json.load(open(f))["overall"]
            except Exception:
                continue
            # 面板格子名是长的（ogerpon_7e39 / champ_0d86 / dunsparce_9d6e），
            # 表头用的是短名（7e39 / 0d86 / 9d6e）⇒ 统一取最后一段
            c = os.path.basename(f)[3:-5].split("_")[-1]
            k, n = acc.get(c, (0.0, 0))
            acc[c] = (k + float(o["score"]) * int(o["n"]), n + int(o["n"]))
    return {c: (k / n, n) for c, (k, n) in acc.items()}


def load_new():
    """tools/fill_winrate.sh 的新结果：{(a,b): (胜率, n)}"""
    out = {}
    for p in glob.glob(os.path.join(NEWDIR, "*_*.json")):
        a, b = os.path.basename(p)[:-5].split("_")
        try:
            o = json.load(open(p))["overall"]
            out[(a, b)] = (float(o["score"]), int(o["n"]))
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    old, new = load_detail(), load_new()
    pilots = load_pilots()                    # {我方卡组: (驾驶员, tags)}
    pilot_cells = {k: pooled_panel(v[1]) for k, v in pilots.items()}
    val, src, miss = {}, {}, []
    for x in DECKS:
        for y in DECKS:
            if x == y:
                val[(x, y)] = "0.5000"                  # 镜像：期望胜率 0.5
                src[(x, y)] = "对角"
                continue
            if x in pilot_cells and y in pilot_cells[x]:
                sc, n = pilot_cells[x][y]
                val[(x, y)] = f"{sc:.4f}"
                src[(x, y)] = f"驾驶员{pilots[x][0]}(n={n})"
            elif (x, y) in new:
                val[(x, y)] = f"{new[(x, y)][0]:.4f}"
                src[(x, y)] = f"新跑(n={new[(x, y)][1]})"
            elif (x, y) in old:
                val[(x, y)] = f"{old[(x, y)][0]:.4f}"
                src[(x, y)] = f"老eval(n={old[(x, y)][1]},{old[(x, y)][2]})"
            else:
                val[(x, y)] = ""
                src[(x, y)] = "缺"
                miss.append(f"{x}→{y}")

    n_fill = sum(1 for k, v in val.items() if v not in ("", "/") and k[0] != k[1])
    print(f"覆盖 {n_fill}/{len(DECKS) * (len(DECKS) - 1)} 个非对角格"
          f"（新跑 {len(new)}，老 eval {len(old)}，对角 {len(DECKS)} 填 /）")
    for k, (pname, tags) in pilots.items():
        cnt = len(pilot_cells.get(k, {}))
        print(f"  指定驾驶员：{k} 行 → {pname}（{cnt} 格来自 {len(tags)} 个面板 tag）")
    if miss:
        print(f"仍缺 {len(miss)} 格：" + " ".join(miss))
    if a.dry:
        return 0

    with open(OUTCSV, "w", encoding="gb18030", newline="") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(["挑战者\\被挑战者"] + DECKS)
        for x in DECKS:
            w.writerow([x] + [val[(x, y)] for y in DECKS])
    print(f"→ {OUTCSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
