#!/usr/bin/env python3
"""填 docs/the-pockmon-battle-choice.csv：**挑战者卡组 × 被挑战卡组 → 该用哪个驾驶员**。

## 表结构
  行 = 挑战者卡组（我们上场的牌）
  列 = 被挑战卡组（对手的牌）
  格 = 该用哪个驾驶员（agent 名）打这一格胜率最高

## 选择规则（带噪声阈值，不裸取 max）
  对每一格：
    1. 若某个**训练过的驾驶员**比 init（端口权重）高 **≥5pp** ⇒ 用训练驾驶员；
    2. 否则用 **init**（端口）。
    3. 只有 init 没数据、其它驾驶员有数据时，才退回"取最高"并标注。
  为什么加 5pp 阈值：单格 n=200 的 95% CI ≈ ±6.5pp，"取最高"等于**系统性挑中幸运值**
  （赢家诅咒）。实测 78 个非 init 格里只有 7 个真的领先 ≥5pp，其余 71 个是噪声。

## 数据源（全部是已跑过的实测，不新增对局）
  data/scheduler/matrix_full.csv          49 格，端口 vs 端口（n=300）
  logs/meta/<臂>/vs_*.json                12 格面板（n=200）
  logs/panel/<臂>/vs_*.json               7 格面板（n=200）
  data/opp_matrix/panel_*.json            历史面板（含 shuppet specialist，n≈180）

用法：./tools/py tools/build_choice_table.py
"""
import csv, json, glob, os, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESH = 0.05     # 训练驾驶员至少要领先 init 多少才算数

CELL2HASH = {
    "ogerpon_7e39": "7e39", "dunsparce_0bf5": "0bf5", "grookey_2617": "2617",
    "shuppet_0d86": "0d86", "dunsparce_9d6e": "9d6e", "abra_f5f8": "f5f8",
    "dreepy_3eaa": "3eaa", "abra_e8ee": "e8ee", "dunsparce_db04": "db04",
    "champ_0904": "0904", "champ_26c6": "26c6", "champ_e1d7": "e1d7",
}


def cell2hash(c):
    return CELL2HASH.get(c, c.split("_")[-1])


def collect():
    recs = []
    # ① 49 格矩阵（端口 vs 端口）
    for r in csv.DictReader(open(os.path.join(ROOT, "data/scheduler/matrix_full.csv"))):
        recs.append((r["cand_deck"], r["opp_deck"], "init",
                     float(r["cand_win"]), int(float(r["n_games"])), "matrix_full"))
    # ② 12 格 meta 面板
    META = {"meta_port_2617": ("2617", "init"), "meta_port_9d6e": ("9d6e", "init"),
            "meta_port_26c6": ("26c6", "init"),          # slowking 基线
            "meta_r1A": ("26c6", "r1A"),                 # 10M 步 slowking 专才
            "meta_g6_db04": ("db04", "g6_db04"), "meta_ph3_2617_v2": ("2617", "ph3_2617_v2"),
            "meta_ph3_2617_v1": ("2617", "ph3_2617_v1"), "meta_ph3_9d6e_v2": ("9d6e", "ph3_9d6e_v2"),
            # 2026-11-13 A/B 四轮（舰牌 9d6e）。★ 名字带 ab9d6e_ 前缀，和上一轮 slowking 专才的
            # r1A/r1B 区分开（那个不带前缀，见 data/ab_run1/）。
            "ab9d6e_r1A": ("9d6e", "ab9d6e_r1A"), "ab9d6e_r1B": ("9d6e", "ab9d6e_r1B"),
            "ab9d6e_r2A": ("9d6e", "ab9d6e_r2A"), "ab9d6e_r2B": ("9d6e", "ab9d6e_r2B"),
            # ab9d6e_r1A 另外两个种子：同名驾驶员，下面 POOL 会合并成 n≈600
            "ab9d6e_r1A_s950001": ("9d6e", "ab9d6e_r1A"),
            "ab9d6e_r1A_s950002": ("9d6e", "ab9d6e_r1A"),
            # ab9d6e_r2A 的另外两个种子（同样池化成 n≈600 后才允许改表）
            "ab9d6e_r2A_s950001": ("9d6e", "ab9d6e_r2A"),
            "ab9d6e_r2A_s950002": ("9d6e", "ab9d6e_r2A"),
            # 基线（端口）也补两个种子，保证"候选 vs 基线"两边都是 n≈600
            "meta_port_9d6e_s950001": ("9d6e", "init"),
            "meta_port_9d6e_s950002": ("9d6e", "init")}
    for arm, (ch, pilot) in META.items():
        for f in glob.glob(os.path.join(ROOT, "logs/meta", arm, "vs_*.json")):
            cell = os.path.basename(f)[3:-5]
            o = json.load(open(f))["overall"]
            recs.append((ch, cell2hash(cell), pilot, o["score"], o["n"], arm))
    # ③ 7 格面板
    PANEL = {"c_grookey_port": ("2617", "init"), "port_9d6e": ("9d6e", "init"),
             "port_db04": ("db04", "init"), "g6_db04": ("db04", "g6_db04"),
             "g6_db04_on_2617": ("2617", "g6_db04_on_2617"), "bc_grookey3": ("2617", "bc_grookey3"),
             "ph3_2617_v2": ("2617", "ph3_2617_v2"), "ph3_2617_v1": ("2617", "ph3_2617_v1"),
             "ph3_9d6e_v2": ("9d6e", "ph3_9d6e_v2")}
    for arm, (ch, pilot) in PANEL.items():
        for f in glob.glob(os.path.join(ROOT, "logs/panel", arm, "vs_*.json")):
            cell = os.path.basename(f)[3:-5]
            o = json.load(open(f))["overall"]
            recs.append((ch, cell2hash(cell), pilot, o["score"], o["n"], "panel/" + arm))
    # ④ 历史面板（含 shuppet specialist / ph3 v1 / g6）
    OPP = {"panel_d0904": ("0904", "init"), "panel_d3eaa": ("3eaa", "init"),
           "panel_db04": ("db04", "g6_db04"), "panel_ph3": ("db04", "ph3_db04_v1"),
           "panel_ph3_2617": ("2617", "ph3_2617_v1"), "panel_ph3_9d6e": ("9d6e", "ph3_9d6e_v1"),
           "panel_shbc": ("0d86", "shuppet_specialist")}
    for arm, (ch, pilot) in OPP.items():
        p = os.path.join(ROOT, "data/opp_matrix", arm + ".json")
        if not os.path.exists(p):
            continue
        for cell, v in json.load(open(p)).get("per_opp", {}).items():
            if v.get("score") is None:
                continue
            recs.append((ch, cell2hash(cell), pilot, float(v["score"]), int(v.get("n") or 0), arm))
    return recs


SRC_PRI = {"matrix_full": 1}


def pri_of(src):
    """同协议优先：多种子池化 > n=200 的 meta/panel 面板 > opp_matrix > 49 格矩阵（n=300 remote）。"""
    if src.startswith("meta_pooled"):
        return 4
    if src.startswith("meta_") or src.startswith("panel/"):
        return 3
    if src.startswith("panel_") or src.startswith("panel"):
        return 2
    return SRC_PRI.get(src, 2)


# ★ 只有**多种子验证过**的新驾驶员才有资格改表：
#   单种子 n=200 的 ±6.5pp 噪声会让"取最高"系统性地挑中幸运值（赢家诅咒），
#   ab9d6e_r1B / r2A / r2B 目前只有单种子数据，因此只入明细、不参与选表。
UNVERIFIED = {"ab9d6e_r1B", "ab9d6e_r2B"}   # r2A 已补 2 个种子，进入验证集合
# 需要把多个种子的面板合并的驾驶员（同 (格, 驾驶员) 的胜场与场次相加）
# ★ 池化必须限定"同协议来源"：早先不加限定把 init 的所有来源（49 格矩阵 n=300 remote、
#   7 格面板、多种子 meta）混在一起求和，导致 9d6e 行出现 +44pp 的假结论（init 基线被污染）。
POOL_SPEC = {
    "ab9d6e_r1A": {"ab9d6e_r1A", "ab9d6e_r1A_s950001", "ab9d6e_r1A_s950002"},
    "ab9d6e_r2A": {"ab9d6e_r2A", "ab9d6e_r2A_s950001", "ab9d6e_r2A_s950002"},
    "init": {"meta_port_9d6e", "meta_port_9d6e_s950001", "meta_port_9d6e_s950002"},
}


def main():
    recs = collect()
    # 池化：把同一驾驶员多个种子臂的记录按格合并
    if POOL_SPEC:
        acc = {}
        keep = []
        for r in recs:
            ch, de, pilot, w, n, src = r
            allowed = POOL_SPEC.get(pilot)
            if allowed and src in allowed:
                k = (ch, de, pilot)
                pw, pn = acc.get(k, (0.0, 0))
                acc[k] = (pw + w * n, pn + n)
            else:
                keep.append(r)
        for (ch, de, pilot), (pw, pn) in acc.items():
            keep.append((ch, de, pilot, pw / pn, pn, "meta_pooled/" + pilot))
        recs = keep
    # 每个 (格, 驾驶员) 保留：先比来源优先级，再比胜率、样本数
    by = collections.defaultdict(dict)
    for ch, de, pilot, w, n, src in recs:
        key = (ch, de)
        cur = by[key].get(pilot)
        cand = (pri_of(src), w, n, src)
        if cur is None or cand > cur:
            by[key][pilot] = cand

    decks = sorted({h for k in by for h in k})
    choice, why = {}, {}
    # 先对所有 (行,列) 组合给默认值，再用实测数据覆盖
    for ch in decks:
        for de in decks:
            choice[(ch, de)] = "init"
            why[(ch, de)] = "无数据，默认 init"
    for k, pilots in by.items():
        ini = pilots.get("init")
        trained = {p: v for p, v in pilots.items() if p != "init"}
        if ini is not None:
            cands = {p: v for p, v in trained.items()
                     if v[1] - ini[1] >= THRESH and p not in UNVERIFIED}
            if cands:
                best = max(cands.items(), key=lambda t: (t[1][1], t[1][2]))
                choice[k] = best[0]
                is_pooled = str(best[1][3]).startswith("meta_pooled")
                tag = ("多种子验证 n=%d" if is_pooled else "单种子 n=%d") % best[1][2]
                why[k] = (f"训练件 {best[1][1]:.3f} vs init {ini[1]:.3f}"
                          f"（+{(best[1][1]-ini[1])*100:.1f}pp ≥5pp；{tag}）")
            else:
                choice[k] = "init"
                why[k] = "init（无训练件领先 ≥5pp）"
        elif trained:
            best = max(trained.items(), key=lambda t: (t[1][1], t[1][2]))
            choice[k] = best[0]
            why[k] = f"无 init 数据，退回取最高 {best[1][1]:.3f}"
        else:
            choice[k] = "init"          # 无数据：默认 init（假定未知就用端口），后续训出更强的再覆盖
            why[k] = "无数据，默认 init"

    # 写三张表
    with open(os.path.join(ROOT, "docs/the-pockmon-battle-choice.csv"), "w", newline="", encoding="gb18030", errors="replace") as f:
        w = csv.writer(f)
        w.writerow(["挑战者\\被挑战者"] + decks)
        for ch in decks:
            w.writerow([ch] + [choice.get((ch, de), "") for de in decks])

    with open(os.path.join(ROOT, "docs/the-pockmon-battle-choice-detail.csv"), "w", newline="", encoding="gb18030", errors="replace") as f:
        w = csv.writer(f)
        w.writerow(["挑战者", "被挑战者", "驾驶员", "胜率", "样本场次", "状态", "依据"])
        for ch in decks:
            for de in decks:
                k = (ch, de)
                if k not in choice or not choice[k]:
                    continue
                why_k = why.get(k, "")
                if "无数据" in why_k:
                    w.writerow([ch, de, choice[k], "", "", "默认", why_k])
                else:
                    v = by[k][choice[k]]
                    w.writerow([ch, de, choice[k], round(v[1], 4), v[2], "实测", why_k])

    with open(os.path.join(ROOT, "docs/the-pockmon-battle-choice-all.csv"), "w", newline="", encoding="gb18030", errors="replace") as f:
        w = csv.writer(f)
        w.writerow(["挑战者", "被挑战者", "驾驶员", "胜率", "样本场次", "来源"])
        for ch, de, pilot, wr, n, src in sorted(recs):
            w.writerow([ch, de, pilot, round(wr, 4), n, src])

    cnt = collections.Counter(choice.values())
    n_default = sum(1 for k, v in why.items() if "无数据" in v)
    print(f"覆盖 {len(decks)} 副卡组；实测 {len(decks)**2 - n_default} 格，默认 init {n_default} 格")
    for k, v in cnt.most_common():
        print(f"  {v:>3} 格  {k}")
    print("\n非 init 的格子（全部是 ≥5pp 的强信号）：")
    for (ch, de), c in sorted(choice.items()):
        if c and c != "init":
            v = by[(ch, de)][c]
            print(f"  {ch:<6}→{de:<6} {c:<20} {v[1]:.3f}  {why[(ch, de)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
