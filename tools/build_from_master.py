#!/usr/bin/env python3
"""**从数据母表重新生成两张交付表**（母表 = the-pockmon-battle-choice-in-table.csv）

    ./tools/py tools/build_from_master.py            # 重建 choice / choice-detail / win-rate / pilot
    ./tools/py tools/build_from_master.py --dry

产出：
1. `the-pockmon-battle-choice.csv`      每格（我方牌 a × 对手牌 b）该用哪个**我方**驾驶员
2. `the-pockmon-battle-choice-detail.csv` 同上 + 胜率/样本/状态/依据
3. `the-pockmon-battle-choices-win-rate.csv` 每行（我方牌 a）的**最佳驾驶员**打 12 副对手牌的胜率
4. `the-pockmon-battle-choices-win-rate-pilot.csv` 每行用的驾驶员（可追溯）

## 规则（与既有文档一致）

- **格级选人**：非 init 驾驶员需比同格 init 高 **≥ THRESH(5pp)**，且必须有**多种子（池化 n≥400）**证据；
  否则该格用 init。
- **行级选人**（win-rate 矩阵每行）：按该牌全部 12 格做**场地加权**，取最高者；同样要求比 init 高 ≥1pp。
- 对角（a→a）在矩阵里留白填 `/`（自己打自己不算对位）。
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-in-table.csv")
CH = os.path.join(ROOT, "docs", "the-pockmon-battle-choice.csv")
DET = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-detail.csv")
WR = os.path.join(ROOT, "docs", "the-pockmon-battle-choices-win-rate.csv")
PILOT = os.path.join(ROOT, "docs", "the-pockmon-battle-choices-win-rate-pilot.csv")

_LEGACY_DECKS = ["0904", "0bf5", "0d86", "2617", "26c6", "3eaa", "7e39", "9d6e", "db04", "e1d7", "e8ee", "f5f8"]

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
THRESH_CELL = 5.0     # pp：格级替换门槛
THRESH_ROW = 1.0      # pp：行级替换门槛
VERIFIED_N = 400      # 池化样本 ≥ 这个数才算"多种子验证"


WEIGHTS_FILE = "decks/weights_field_b05.txt"   # 由 --weights 覆盖


def field_weights():
    w = {}
    for line in open(os.path.join(ROOT, WEIGHTS_FILE)):
        p = line.split()
        if len(p) == 2 and "__" in p[0]:
            w[p[0].split("__")[-1][:4]] = float(p[1])
    tot = sum(w.values()) or 1.0
    return {k: v / tot for k, v in w.items()}


def load_master():
    txt = open(MASTER, "rb").read().decode("gb18030", errors="replace")
    rec = collections.defaultdict(dict)     # (our, opp) -> {driver: (wr, n, src)}
    for r in csv.DictReader(io.StringIO(txt)):
        try:
            wr, n = float(r["胜率"]), int(float(r["样本场次"]))
        except (TypeError, ValueError):
            continue
        opp_drv = r.get("对手驾驶员") or "init"
        # 只把"对手=预期基线(init)"的记录用于选表；强对手版本另算（见 check_stale.py）
        if opp_drv != "init":
            continue
        rec[(r["挑战者"], r["被挑战者"])][r["驾驶员"]] = (wr, n, r["来源"])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--weights", default="decks/weights_field_b05.txt",
                    help="场地权重文件；顶分段用 decks/weights_top.txt")
    ap.add_argument("--suffix", default="", help="输出文件名后缀，如 _top")
    a = ap.parse_args()
    global WEIGHTS_FILE
    WEIGHTS_FILE = a.weights
    rec, W = load_master(), field_weights()
    # 行 = 我方（已合并同构变体的 15 副）；列 = 数据里出现的全部对手牌（含 0bf5/db04/e8ee 这些
    # "我们不再当主牌、但场上仍然会遇到"的牌），否则会丢掉 0bf5 这种占 12.9% 的对手格。
    global OPP_DECKS
    OPP_DECKS = sorted({b for (_, b) in rec} | set(DECKS))
    if a.suffix:
        global CH, DET, WR, PILOT
        f = lambda p: p[:-4] + a.suffix + ".csv"
        CH, DET, WR, PILOT = f(CH), f(DET), f(WR), f(PILOT)

    # ── 1/2. 格级选人 ──
    choice, detail = {}, []
    for a_ in DECKS:
        for b_ in OPP_DECKS:
            cell = rec.get((a_, b_), {})
            if a_ == b_:
                # ★ 对角 = 内战格：写"当前内战最强的驾驶员"（我方驾驶员 vs 对手端口），
                #   胜率在矩阵里统一记 0.5（镜像期望值），这里只决定**派谁**。
                #   仍然要求 n ≥ VERIFIED_N：单种子 n=200 的 53% 只是噪声（±6.9pp）。
                mirror = {d: v for d, v in cell.items() if v[1] >= VERIFIED_N}
                if mirror:
                    mb = max(mirror.items(), key=lambda t: (t[1][0], t[1][1]))
                else:
                    mb = None
                if mb is not None:
                    choice[(a_, b_)] = mb[0]
                    detail.append([a_, b_, mb[0], "0.5000", mb[1][1], "内战",
                                   f"内战最强：{mb[0]} {mb[1][0]:.4f}（n={mb[1][1]}，vs 对手端口）；"
                                   f"矩阵记 0.5；候选 {sorted(mirror)}"])
                else:
                    choice[(a_, b_)] = "init"
                    detail.append([a_, b_, "init", "0.5000", 0, "内战",
                                   f"无 n≥{VERIFIED_N} 的内战数据（该格候选全部样本不足）"])
                continue
            ini = cell.get("init")
            cands = {d: v for d, v in cell.items()
                     if d != "init" and ini is not None and v[1] >= VERIFIED_N
                     and (v[0] - ini[0]) * 100 >= THRESH_CELL}
            if cands:
                best = max(cands.items(), key=lambda t: (t[1][0], t[1][1]))
                choice[(a_, b_)] = best[0]
                detail.append([a_, b_, best[0], f"{best[1][0]:.4f}", best[1][1], "实测",
                               f"训练件 {best[1][0]:.4f} vs init {ini[0]:.4f}"
                               f"（+{(best[1][0]-ini[0])*100:.1f}pp ≥{THRESH_CELL:.0f}pp；n={best[1][1]}；{best[1][2]}）"])
            elif ini is not None:
                choice[(a_, b_)] = "init"
                detail.append([a_, b_, "init", f"{ini[0]:.4f}", ini[1], "实测",
                               "init（无已验证训练件领先 ≥5pp）"])
            else:
                choice[(a_, b_)] = "init"
                detail.append([a_, b_, "init", "", "", "默认", "无数据，默认 init"])

    # ── 3. 行级选人（win-rate 矩阵每行用一个驾驶员）──
    row_pilot, row_wr, pilot_rows = {}, {}, []
    for a_ in DECKS:
        best_by_drv = collections.defaultdict(lambda: [0.0, 0.0, 0])   # driver -> [num, weight, n]
        for b_ in DECKS:
            if a_ == b_:
                continue
            for d, (wr, n, _src) in rec.get((a_, b_), {}).items():
                w = W.get(b_, 0.0)
                acc = best_by_drv[d]
                acc[0] += w * wr
                acc[1] += w
                acc[2] = max(acc[2], n)
        scored = {d: v[0] / v[1] for d, v in best_by_drv.items() if v[1] > 0}
        base = scored.get("init")
        pick = "init"
        if scored:
            top = max(scored.items(), key=lambda t: t[1])
            if top[0] != "init" and base is not None and (top[1] - base) * 100 >= THRESH_ROW:
                pick = top[0]
        row_pilot[a_] = pick
        pilot_rows.append([a_, pick, f"{scored.get(pick, 0):.4f}" if pick in scored else "NA",
                           str(max((rec.get((a_, b_), {}).get(pick, (0, 0, ""))[1] for b_ in DECKS), default=0)),
                           "面板场地加权选行级驾驶员"])
        for b_ in DECKS:
            if a_ == b_:
                row_wr[(a_, b_)] = "0.5000"     # 镜像格：期望胜率 0.5（派谁由 choice 表给）
                continue
            v = rec.get((a_, b_), {}).get(pick) or rec.get((a_, b_), {}).get("init")
            row_wr[(a_, b_)] = f"{v[0]:.4f}" if v else ""

    n_cell_other = sum(1 for v in choice.values() if v != "init")
    n_row_other = sum(1 for v in row_pilot.values() if v != "init")
    print(f"格级：{len(choice)} 格，其中 {n_cell_other} 格用非 init 驾驶员")
    print(f"行级：{n_row_other} 副牌由非 init 驾驶员驾驶 → {[k for k,v in row_pilot.items() if v!='init']}")
    if a.dry:
        return 0

    with open(CH, "w", newline="", encoding="gb18030") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(["挑战者\\被挑战者"] + OPP_DECKS)
        for x in DECKS:
            w.writerow([x] + [choice.get((x, y), "init") for y in OPP_DECKS])
    with open(DET, "w", newline="", encoding="gb18030") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(["挑战者", "被挑战者", "驾驶员", "胜率", "样本场次", "状态", "依据"])
        w.writerows(detail)
    with open(WR, "w", newline="", encoding="gb18030") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(["挑战者\\被挑战者"] + OPP_DECKS)
        for x in DECKS:
            w.writerow([x] + [row_wr.get((x, y), "") for y in OPP_DECKS])
    with open(PILOT, "w", newline="", encoding="gb18030") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(["卡组", "最佳驾驶员", "面板场地加权", "每格样本", "依据"])
        w.writerows(pilot_rows)
    print(f"→ {CH}\n→ {DET}\n→ {WR}\n→ {PILOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
