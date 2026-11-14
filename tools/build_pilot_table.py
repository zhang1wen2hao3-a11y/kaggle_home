#!/usr/bin/env python3
"""**生成 docs/the-pockmon-battle-choice-self.csv：每副牌"自己的最佳驾驶员"**

    ./tools/py tools/build_pilot_table.py            # 生成/刷新
    ./tools/py tools/build_pilot_table.py --dry      # 只看结论

## 这张表回答什么

> **这副牌交给谁驾驶最强？** → 训练时**对手池**就按这个结论给每副对手牌配驾驶员。

⚠️ 与 `the-pockmon-battle-choice.csv` 的区别（2026-11-14 修正）：
- `choice.csv` 的格子 =「我方用牌 a 打对手牌 b 时**我方**派谁」→ 回答"这一格我方用谁"；
- 本表 =「牌 b **自己**交给谁」→ 回答"训练时对手的牌 b 该由谁驾驶"。
  两者不能互相替代：`9d6e` 行里 `ab9d6e_r2A` 只在 `→e8ee`/`→f5f8` 两格更强，
  但**整副牌**的场地加权最高者仍是 `init`（0.7248 vs r1A 0.7211 vs r2A 0.7202）。

## 判据

该牌的 **12 格面板 × 多种子池化**后的场地加权胜率（Σ 场地权重 × 胜率）。
非 init 候选需比 init 高 **≥ THRESH**（默认 1.0pp）才替换，否则记 init；无面板数据记「待测」。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-self.csv")
THRESH = 1.0   # pp

# 牌 → [(驾驶员, [面板 tag；多个 = 多种子池化])]，tag 目录在 logs/meta/<tag>/vs_*.json
FAMILY = {
    "9d6e": [("init", ["meta_port_9d6e", "meta_port_9d6e_s950001", "meta_port_9d6e_s950002"]),
             ("ab9d6e_r1A", ["ab9d6e_r1A", "ab9d6e_r1A_s950001", "ab9d6e_r1A_s950002"]),
             ("ab9d6e_r2A", ["ab9d6e_r2A", "ab9d6e_r2A_s950001", "ab9d6e_r2A_s950002"]),
             ("ab9d6e_r1B", ["ab9d6e_r1B"]),
             ("ab9d6e_r2B", ["ab9d6e_r2B"]),
             ("ph3_9d6e_v2", ["meta_ph3_9d6e_v2"])],
    "26c6": [("init", ["meta_port_26c6"]), ("r1A", ["meta_r1A"])],
    "2617": [("init", ["meta_port_2617"]), ("ph3_2617_v1", ["meta_ph3_2617_v1"]),
             ("ph3_2617_v2", ["meta_ph3_2617_v2"])],
    "db04": [("g6_db04", ["meta_g6_db04"])],
    "7e39": [("init", ["port_7e39_s950000", "port_7e39_s950001", "port_7e39_s950002"]),
             ("cand7e39S", ["cand7e39S_s950000", "cand7e39S_s950001", "cand7e39S_s950002"])],
    "0d86": [("init", ["port_0d86_s950000", "port_0d86_s950001", "port_0d86_s950002"]),
             ("g8_bc_parent", ["g8_bc_parent_s950000", "g8_bc_parent_s950001", "g8_bc_parent_s950002"]),
             ("g8_shuppet", ["g8_shuppet_s950000", "g8_shuppet_s950001", "g8_shuppet_s950002"])],
    "e1d7": [("init", ["port_e1d7_s950000", "port_e1d7_s950001", "port_e1d7_s950002"]),
             ("candE1D7S", ["candE1D7S_s950000", "candE1D7S_s950001", "candE1D7S_s950002"])],
}


WEIGHTS_FILE = "decks/weights_field_b05.txt"


def field_weights():
    w = {}
    for line in open(os.path.join(ROOT, WEIGHTS_FILE)):
        p = line.split()
        if len(p) == 2 and "__" in p[0]:
            w[p[0].split("__")[-1][:4]] = float(p[1])
    tot = sum(w.values()) or 1.0
    return {k: v / tot for k, v in w.items()}


def pooled_wr(tags, W):
    cells, used = {}, 0
    for t in tags:
        files = glob.glob(os.path.join(ROOT, "logs", "meta", t, "vs_*.json"))
        if not files:
            continue
        used += 1
        for f in files:
            o = json.load(open(f))["overall"]
            c = os.path.basename(f)[3:-5].split("_")[-1]
            k, n = cells.get(c, (0.0, 0))
            cells[c] = (k + float(o["score"]) * int(o["n"]), n + int(o["n"]))
    if not cells:
        return None, 0
    tot = sum(W.get(c, 0.0) for c in cells)
    if tot <= 0:
        return None, used
    return sum(W.get(c, 0.0) * (k / n) for c, (k, n) in cells.items()) / tot, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--weights", default="decks/weights_field_b05.txt",
                    help="场地权重文件；顶分段用 decks/weights_top.txt")
    ap.add_argument("--suffix", default="", help="输出文件名后缀，如 _top")
    a = ap.parse_args()
    global WEIGHTS_FILE, OUT
    WEIGHTS_FILE = a.weights
    if a.suffix:
        OUT = OUT[:-4] + a.suffix + ".csv"
    W = field_weights()
    rows = []
    for deck in sorted(FAMILY):
        cand = []
        for pilot, tags in FAMILY[deck]:
            wr, seedn = pooled_wr(tags, W)
            if wr is not None:
                cand.append((pilot, wr, seedn))
        if not cand:
            rows.append([deck, "待测", "NA", "0", "NA", "还没有 12 格面板数据"])
            print(f"{deck:<8}待测")
            continue
        base = next((wr for p, wr, _ in cand if p == "init"), None)
        if base is None:      # 没有任何 init 基线 ⇒ 没法比较，标待测（别急着选候选）
            rows.append([deck, "待测", "NA", str(max(t[2] for t in cand)), "NA",
                         "缺 init 基线：" + " > ".join(f"{p} {wr:.4f}" for p, wr, _ in cand)])
            print(f"{deck:<8}待测（缺 init 基线）：" + " > ".join(f"{p} {wr:.4f}" for p, wr, _ in cand))
            continue
        best = max(cand, key=lambda t: t[1])
        if best[0] != "init" and base is not None and (best[1] - base) * 100 >= THRESH:
            pick, gain = best, (best[1] - base) * 100
        else:
            pick = next((t for t in cand if t[0] == "init"), best)
            gain = 0.0
        detail = " > ".join(f"{p} {wr:.4f}" for p, wr, _ in sorted(cand, key=lambda t: -t[1]))
        rows.append([deck, pick[0], f"{pick[1]:.4f}", str(pick[2]),
                     f"{gain:+.1f}pp" if gain else "0.0pp",
                     f"面板场地加权排序：{detail}"])
        print(f"{deck:<8}{pick[0]:<14}{pick[1]:.4f}  {detail}")
    # ★ 写盘前校验：表里出现的驾驶员必须能在注册表里解析（否则训练时生成对手会认错人）
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import driver_registry as REG
    for r in rows:
        if r[1] in ("init", "待测"):
            continue
        try:
            REG.info(r[1])
        except REG.UnknownDriver as e:
            sys.exit(f"✗ {r[0]} 的最佳驾驶员 {r[1]!r} 没在 tools/driver_registry.py 登记：{e}")

    n_other = sum(1 for r in rows if r[1] not in ("init", "待测"))
    print(f"\n{len(rows)} 副牌：{n_other} 副由非 init 驾驶员驾驶")
    if a.dry:
        return 0
    with open(OUT, "w", newline="", encoding="gb18030") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(["卡组", "最佳驾驶员", "面板场地加权", "池化种子数", "相对 init 提升", "依据"])
        w.writerows(rows)
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
