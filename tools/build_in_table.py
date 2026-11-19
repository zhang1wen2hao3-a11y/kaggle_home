#!/usr/bin/env python3
"""**生成 docs/the-pockmon-battle-choice-in-table.csv**：我们自家"最佳驾驶员"之间的内部对局表。

    ./tools/py tools/build_in_table.py            # 生成/刷新
    ./tools/py tools/build_in_table.py --dry      # 只看还缺哪些对局

## 表是什么

`the-pockmon-battle-choices-win-rate.csv` 的每一行 = 一副我方卡组 + 它的最佳驾驶员（见下 PILOT_OF）。
本表记录**这些"最佳驾驶员"两两之间**的对局，用于后续挑"最泛用的 agent"：

| 列 | 含义 |
|---|---|
| 挑战者 | 该对局里**胜率高的一方**的卡组（规则：以获胜多的一方作为挑战者） |
| 被挑战者 | 输的一方的卡组 |
| 驾驶员 | **挑战者**用的驾驶员（被挑战者用的见 `来源` 列） |
| 胜率 / 样本场次 | 挑战者的胜率与总场次（两边合并） |
| 来源 | `derived`（由矩阵双向推出）或 `run <A>@<a> vs <B>@<b>` |

**每个无序对只记一条**（不是两个方向各一条），按挑战者名字的字典序排序。

## 数据从哪来

1. **两侧都是 `init` 的对局**：直接由 win-rate 矩阵推出 ——
   矩阵 `(a,b)` = `init@a vs 端口@b`，`(b,a)` = `init@b vs 端口@a`，是同一个对局的两次独立测量，
   把 `cell(a,b)` 与 `1 − cell(b,a)` 合并（胜场与场次相加）即可。
2. **任一侧不是 `init`**：必须实跑（`tools/run_in_table.sh` 产出 `logs/in_table/<a>_<b>.json`）。

`PILOT_OF` 与 `the-pockmon-battle-choices-win-rate.csv` 的 README 口径一致；
`9d6e` 行按 sidecar（`…win-rate-pilot.csv`）走。
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINRATE = os.path.join(ROOT, "docs", "the-pockmon-battle-choices-win-rate.csv")
OUTCSV = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-in-table.csv")
RUNDIR = os.path.join(ROOT, "logs", "in_table")

_LEGACY_DECKS = ["0904", "0bf5", "0d86", "2617", "26c6", "3eaa", "7e39",
         "9d6e", "db04", "e1d7", "e8ee", "f5f8"]

def _load_decks():
    """维护表牌序的唯一来源：decks/deck_order.txt（一行一副）；文件不存在时退回旧的 12 副。"""
    import os as _os
    _p = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "decks", "deck_order.txt")
    try:
        ids = [ln.strip() for ln in open(_p) if ln.strip() and not ln.startswith("#")]
        if ids:
            return ids
    except OSError:
        pass
    return _LEGACY_DECKS


DECKS = _load_decks()

# 每副卡组的最佳驾驶员（= win-rate 矩阵每行的口径）
PILOT_OF = {"db04": "g6_db04", "2617": "ph3_2617_v1", "26c6": "r1A",
            "0d86": "g8_bc_parent"}
DEFAULT_PILOT = "init"

# (驾驶员, 卡组) → agent 目录（本地与评估机上同名）
AGENT_OF = {
    ("init", "0904"): "port_0904", ("init", "0bf5"): "opp_dunsparce_0bf5",
    ("init", "0d86"): "port_0d86", ("init", "2617"): "c_grookey",
    ("init", "26c6"): "port_26c6", ("init", "3eaa"): "opp_3eaa",
    ("init", "7e39"): "port_7e39", ("init", "9d6e"): "port_on_9d6e",
    ("init", "db04"): "port_db04", ("init", "e1d7"): "port_e1d7",
    ("init", "e8ee"): "port_e8ee", ("init", "f5f8"): "c_abra",
    ("g6_db04", "db04"): "g6_db04", ("ph3_2617_v1", "2617"): "ph3_2617_v1",
    ("r1A", "26c6"): "r1A_26c6", ("g8_bc_parent", "0d86"): "g8_bc_parent",
}


def pilot_of(deck):
    return PILOT_OF.get(deck, DEFAULT_PILOT)


def matrix():
    txt = open(WINRATE, "rb").read().decode("gb18030", errors="replace")
    rows = list(csv.reader(io.StringIO(txt)))
    hdr = rows[0][1:]
    return {r[0]: {hdr[i]: r[i + 1] for i in range(len(hdr))} for r in rows[1:]}


def load_runs():
    """logs/in_table/<a>_<b>[__s2].json → {(a,b): (a 的胜率, n)}（多种子按胜场/场次合并）"""
    acc = {}
    for p in glob.glob(os.path.join(RUNDIR, "*.json")):
        base = os.path.basename(p)[:-5].split("__s")[0]
        if "_" not in base:
            continue
        a, b = base.split("_", 1)
        try:
            o = json.load(open(p))["overall"]
            sc, nn = float(o["score"]), int(o["n"])
        except Exception:
            continue
        k, n0 = acc.get((a, b), (0.0, 0))
        acc[(a, b)] = (k + sc * nn, n0 + nn)
    return {k: (w / n, n) for k, (w, n) in acc.items() if n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    M, runs = matrix(), load_runs()

    pairs = [(x, y) for i, x in enumerate(DECKS) for y in DECKS[i + 1:]]
    rows, missing = [], []
    for x, y in pairs:
        px, py = pilot_of(x), pilot_of(y)
        wins = n = 0.0
        src = ""
        # ① 实跑优先（任一方向）
        for (ca, cb) in ((x, y), (y, x)):
            if (ca, cb) in runs:
                score, nn = runs[(ca, cb)]
                wins += (score if ca == x else 1 - score) * nn
                n += nn
        if n:
            src = f"run {px}@{x} vs {py}@{y}"
        # ② 两侧都是 init ⇒ 由矩阵双向推出
        elif px == "init" and py == "init":
            vx, vy = M[x][y], M[y][x]
            for v, flip in ((vx, False), (vy, True)):
                if v in ("", "/"):
                    continue
                f = float(v)
                wins += ((1 - f) if flip else f) * 200
                n += 200
            src = f"derived {x}→{y}/{y}→{x}" if n else ""
        if n == 0:
            missing.append((x, y, px, py))
            continue
        rate = wins / n
        if rate >= 0.5:
            rows.append((x, y, px, rate, int(n), src))
        else:
            rows.append((y, x, py, 1 - rate, int(n), src))

    rows.sort(key=lambda r: (r[0], r[1]))
    print(f"共 {len(pairs)} 个无序对：已填 {len(rows)}，缺 {len(missing)}")
    if missing:
        by_pilot = collections.Counter((m[2], m[3]) for m in missing)
        print("  缺的对局（驾驶员组合）：")
        for k, v in by_pilot.most_common():
            print(f"    {k[0]:<18} vs {k[1]:<18} {v} 对")
        with open(os.path.join(ROOT, "logs", "in_table_missing.txt"), "w") as f:
            for x, y, px, py in missing:
                f.write(f"{x} {y} {px} {py} {AGENT_OF[(px, x)]} {AGENT_OF[(py, y)]}\n")
        print(f"  → 清单写到 logs/in_table_missing.txt（tools/run_in_table.sh 会读它）")
    if a.dry:
        return 0

    os.makedirs(os.path.dirname(OUTCSV), exist_ok=True)
    with open(OUTCSV, "w", newline="", encoding="gb18030") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(["挑战者", "被挑战者", "驾驶员", "胜率", "样本场次", "来源"])
        for ch, op, pl, r, n, src in rows:
            w.writerow([ch, op, pl, f"{r:.4f}", n, src])
    print(f"→ {OUTCSV}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
