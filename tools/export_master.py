#!/usr/bin/env python3
"""**导出数据母表 docs/the-pockmon-battle-choice-in-table.csv**

    ./tools/py tools/export_master.py            # 导出/刷新
    ./tools/py tools/export_master.py --dry      # 只看统计

母表是**唯一证据源**：`the-pockmon-battle-choice.csv`（该格用谁）与
`the-pockmon-battle-choices-win-rate.csv`（该驾驶员的胜率）都由它重新算出来，
所以这两张表被改坏时可以用母表复原。

## 列

`挑战者, 被挑战者, 驾驶员, 胜率, 样本场次, 来源`
- 挑战者 = 我方卡组短名，被挑战者 = 对手卡组短名
- 驾驶员 = **我方**派谁开这张牌
- 来源 = 数据来自哪个面板/tag（可追溯）

## 证据源（按优先级）

1. `logs/meta/<tag>/vs_<cell>.json`：12 格面板，多种子（`_s95000x`）先按格池化
2. `data/opp_matrix/panel_*.json`：7 格历史面板（cand = 我方 agent）
3. `data/scheduler/matrix_full.csv`：49 格远程矩阵（双方都是端口权重）
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
OUT = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-in-table.csv")

# meta tag → (我方卡组, 驾驶员)；多种子 tag 用 _s95000x 后缀，池化时归到基名
TAG2CELL = {
    "meta_port_9d6e": ("9d6e", "init"),
    "meta_port_2617": ("2617", "init"), "meta_port_26c6": ("26c6", "init"),
    "meta_g6_db04": ("db04", "g6_db04"),
    "meta_ph3_2617_v1": ("2617", "ph3_2617_v1"), "meta_ph3_2617_v2": ("2617", "ph3_2617_v2"),
    "meta_ph3_9d6e_v2": ("9d6e", "ph3_9d6e_v2"), "meta_r1A": ("26c6", "r1A"),
    "ab9d6e_r1A": ("9d6e", "ab9d6e_r1A"), "ab9d6e_r1B": ("9d6e", "ab9d6e_r1B"),
    "ab9d6e_r2A": ("9d6e", "ab9d6e_r2A"), "ab9d6e_r2B": ("9d6e", "ab9d6e_r2B"),
    "cand7e39S": ("7e39", "cand7e39S"), "candE1D7S": ("e1d7", "candE1D7S"),
    "candf5f8S": ("f5f8", "candf5f8S"), "candf5f8G": ("f5f8", "candf5f8G"),
    "cand26c6S": ("26c6", "cand26c6S"),
    "cand26c6M": ("26c6", "cand26c6M"),
    "r1A": ("26c6", "r1A"), "r1A_26c6": ("26c6", "r1A"), "r1A_9d6e": ("9d6e", "r1A"),
    "f5f8s_final": ("f5f8", "candf5f8S"),   # 同一权重的旧临时名（3.xM 步快照）
    "g8_bc_parent": ("0d86", "g8_bc_parent"), "g8_shuppet": ("0d86", "g8_shuppet"),
    "bc_26c6": ("26c6", "bc_26c6"),
    # init 基线（新补）
    "port_7e39": ("7e39", "init"), "port_0d86": ("0d86", "init"), "port_e1d7": ("e1d7", "init"),
    "port_db04": ("db04", "init"), "port_0904": ("0904", "init"), "port_0bf5": ("0bf5", "init"),
    "port_3eaa": ("3eaa", "init"), "port_e8ee": ("e8ee", "init"), "port_f5f8": ("f5f8", "init"),
}

# `data/opp_matrix/<key>.json`：我们 vs **强驾驶员**（对手驾驶员 ≠ init）的单格测量。
# 与"对手=端口"的格是**不同记录**（对手驾驶员列不同），互不覆盖。
OPP2CELL = {
    # key: (我方卡组, 我方驾驶员, 对手卡组, 对手驾驶员)
    #   文件名 `data/opp_matrix/<key>.json` 或 `<key>_s95000x.json`（多 seed 会按基名池化）
    "shppo_vs_9d6e_r2A": ("9d6e", "ab9d6e_r2A", "0d86", "g8_shuppet"),
    "shppo_vs_f5f8port": ("f5f8", "init", "0d86", "g8_shuppet"),
    # slowking 特攻 vs 旧专才（对手 = 强克隆 / 克星牌）
    "cand26c6S_vs_shbc": ("26c6", "cand26c6S", "0d86", "g8_bc_parent"),
    "cand26c6S_vs_f5f8": ("26c6", "cand26c6S", "f5f8", "deck_f5f8_"),
    "r1A_vs_shbc": ("26c6", "r1A", "0d86", "g8_bc_parent"),
    "r1A_vs_f5f8": ("26c6", "r1A", "f5f8", "deck_f5f8_"),
    # f5f8 特攻 vs 强克隆（收尾期统一改成 `f5f8S_*` 前缀；旧文件靠 OPP_GLOBS 一起池化）
    "f5f8S_vs_shbc": ("f5f8", "candf5f8S", "0d86", "g8_bc_parent"),
    "f5f8S_vs_d0d86": ("f5f8", "candf5f8S", "0d86", "deck_0d86_"),
    "f5f8S_mirror": ("f5f8", "candf5f8S", "f5f8", "deck_f5f8_"),
    # 7e39 特攻 vs 强驾驶员
    "cand7e39S_vs_shbc": ("7e39", "cand7e39S", "0d86", "g8_bc_parent"),
    "cand7e39S_vs_f5f8": ("7e39", "cand7e39S", "f5f8", "deck_f5f8_"),
    "cand7e39S_vs_slowking": ("7e39", "cand7e39S", "26c6", "r1A"),
    # f5f8 特攻 vs 最强 slowking 驾驶员（r1A 与 cand26c6M 是不同对手驾驶员 ⇒ 不同记录）
    "f5f8S_vs_slowking": ("f5f8", "candf5f8S", "26c6", "r1A"),
    "f5f8S_vs_cand26c6M": ("f5f8", "candf5f8S", "26c6", "cand26c6M"),
    # 镜像强化版 slowking（cand26c6M）：镜像 & 对旧专才
    "cand26c6M_mirror": ("26c6", "cand26c6M", "26c6", "r1A"),
    "cand26c6M_vs_shbc": ("26c6", "cand26c6M", "0d86", "g8_bc_parent"),
    "cand26c6M_vs_f5f8": ("26c6", "cand26c6M", "f5f8", "deck_f5f8_"),
}

# 同一格的历史文件名：协议相同（200 场/seed-base 95000x）、只是命名不同 ⇒ 池化时一起算，别丢数据
#   典型：早期随手写的 `f5f8S_vs_r1A`（对手就是 r1A_26c6）和后来的 `candf5f8S_vs_slowking` 是**同一格**。
OPP_GLOBS = {
    "f5f8S_vs_shbc": ["f5f8S_vs_shbc*.json", "rm_f5f8_candf5f8S_vs_shuppet*.json"],
    "f5f8S_mirror": ["f5f8S_mirror*.json", "f5f8s_final_mirror*.json"],
    "f5f8S_vs_slowking": ["f5f8S_vs_slowking*.json", "candf5f8S_vs_slowking*.json",
                          "f5f8S_vs_r1A*.json"],
    "f5f8S_vs_cand26c6M": ["f5f8S_vs_cand26c6M*.json", "candf5f8S_vs_cand26c6M*.json"],
    "cand26c6M_mirror": ["cand26c6M_mirror*.json"],
}

# 7 格历史面板：data/opp_matrix/panel_<x>.json → (我方卡组, 驾驶员)
PANEL2CELL = {
    "panel_d0904": ("0904", "init"), "panel_d3eaa": ("3eaa", "init"),
    "panel_db04": ("db04", "g6_db04"), "panel_ph3": ("db04", "ph3_db04_v1"),
    "panel_ph3_2617": ("2617", "ph3_2617_v1"), "panel_ph3_9d6e": ("9d6e", "ph3_9d6e_v1"),
    "panel_shbc": ("0d86", "g8_bc_parent"), "port_9d6e": ("9d6e", "init"),
}

PRI = {"meta": 3, "panel7": 2, "matrix": 1}


def cell_short(name):
    return name.split("_")[-1]


def from_meta():
    """多种子先按 (base tag, 格) 池化"""
    acc = collections.defaultdict(lambda: [0.0, 0])   # (tag_base, cell) -> [wins, n]
    for tag_dir in glob.glob(os.path.join(ROOT, "logs", "meta", "*")):
        tag = os.path.basename(tag_dir)
        base = tag.split("_s95000")[0] if "_s95000" in tag else tag
        # ★ 方案 A 面板的 tag 是 `<驾驶员>_A_s95000x`：先按基名找，找不到再脱掉 `_A`。
        #   漏了这一步会让**全部方案 A 证据**进不了母表（2026-11-15 发现：收尾期所有面板都跑在
        #   方案 A 上，母表里却一条都没有，行级最佳驾驶员一直停在 init）。
        if base not in TAG2CELL and base.endswith("_A"):
            base = base[:-2]
        if base not in TAG2CELL:
            continue
        for f in glob.glob(os.path.join(tag_dir, "vs_*.json")):
            o = json.load(open(f))["overall"]
            c = cell_short(os.path.basename(f)[3:-5])
            k = (base, c)
            acc[k][0] += float(o["score"]) * int(o["n"])
            acc[k][1] += int(o["n"])
    out = []
    for (base, c), (w, n) in acc.items():
        our, drv = TAG2CELL[base]
        out.append((our, c, drv, w / n, n, f"meta:{base}"))
    return out


def from_panel7():
    out = []
    for p in glob.glob(os.path.join(ROOT, "data", "opp_matrix", "panel_*.json")):
        key = os.path.basename(p)[:-5]
        if key not in PANEL2CELL:
            continue
        d = json.load(open(p))
        our, drv = PANEL2CELL[key]
        for opp, o in (d.get("per_opp") or {}).items():
            if isinstance(o, dict) and "score" in o:
                out.append((our, cell_short(opp), drv, o["score"], o["n"], f"panel7:{key}"))
    return out


def from_matrix():
    p = os.path.join(ROOT, "data", "scheduler", "matrix_full.csv")
    out = []
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            out.append((r["cand_deck"], r["opp_deck"], "init",
                        float(r["cand_win"]), int(float(r["n_games"])), "matrix_full"))
    return out


def from_strong_opp():
    """对手是**强驾驶员**的单格测量（对手驾驶员≠init），每个 json 只有一个格子。"""
    out = []
    for key, (our, drv, opp_deck, opp_drv) in OPP2CELL.items():
        wins = n = 0.0
        pats = OPP_GLOBS.get(key) or [f"{key}*.json"]
        for p in [q for pat in pats
                  for q in glob.glob(os.path.join(ROOT, "data", "opp_matrix", pat))]:
            o = (json.load(open(p)) or {}).get("overall") or {}
            if "score" in o and o.get("n"):
                wins += o["score"] * o["n"]
                n += o["n"]
        if n:
            out.append((our, opp_deck, drv, wins / n, int(n), f"vsopp:{key}", opp_drv))
    return out


def _reg_ok(REG, name):
    try:
        REG.info(name)
        return True
    except REG.UnknownDriver:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    recs = [r + ("init",) for r in from_meta() + from_panel7() + from_matrix()] + from_strong_opp()

    best = {}
    for our, opp, drv, wr, n, src, opp_drv in recs:
        pri = PRI["meta"] if src.startswith("meta:") else (PRI["panel7"] if src.startswith("panel7:") else PRI["matrix"])
        key = (our, opp, drv, opp_drv)
        cur = best.get(key)
        cand = (pri, n, wr, src)
        if cur is None or (cand[0], cand[1]) > (cur[0], cur[1]):
            best[key] = cand
    rows = sorted(((o, p, d, v[2], v[1], v[3], od) for (o, p, d, od), v in best.items()))
    print(f"母表记录 {len(rows)} 条；覆盖 {len({(r[0],r[1]) for r in rows})} 格，{len({r[2] for r in rows})} 个驾驶员")
    cnt = collections.Counter(r[5].split(":")[0] for r in rows)
    print("来源分布:", dict(cnt))
    if a.dry:
        return 0
    with open(OUT, "w", newline="", encoding="gb18030") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(["挑战者", "被挑战者", "驾驶员", "对手驾驶员", "胜率", "样本场次", "来源"])
        for o, p, d, wr, n, src, od in rows:
            # ★ 面板协议里"对手"一律是端口权重（build/agents/port_*）⇒ 对手驾驶员记 init。
            #   用强驾驶员当对手重测时，同一格会多出一条 对手驾驶员=<model> 的记录，
            #   用于区分"评测基线版本"，并判断谁的结果已经过期（见 tools/check_stale.py）。
            w.writerow([o, p, d, od, f"{wr:.4f}", n, src])
    # ★ 母表里的驾驶员名字必须都能解析（表要给 choice_to_banks / make_testset_A 用）
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import driver_registry as REG
    bad = sorted({r[2] for r in rows if r[2] != "init"
                  and not _reg_ok(REG, r[2])})
    if bad:
        print(f"⚠️ 母表里有未登记的驾驶员：{bad}（训练生成对手时会报错，请登记 tools/driver_registry.py）")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
