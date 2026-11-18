#!/usr/bin/env python3
"""**投资分表（2026-11-19 重写）：向"高分卡组 + 高覆盖率"倾斜**

用户要求（原话）：「训练资源要向高分卡组和高覆盖率两个方向倾斜，所以我们的投资分必须考虑我们本地的分
和远端高分的差距，以及卡组在我们对手中的覆盖率，进行选择。」

旧的算法只有 `面板补后水平 × K(天花板/中位)`：它偏向"我们自己现在打得好的牌"，**不看覆盖面**，
也不看"我们和高手差多远"，于是 `9d6e` 这种场内只占 0.8% 的牌能排第一，而占 34.9% 的 `7e39` 排到后面。

新口径（三项相乘后开三次方，任一项为 0 就整体为 0，便于解释也便于审计）：

    投资分 = 100 × ( G × C × H ) ^ (1/3)

  · 缺口 G = clip((天花板 − 我方水平分) / 400, 0, 1)
      "本地的分和远端高分的差距"。没数据时：我们**从没开过这副牌** ⇒ G = 1.0（全是可赚的）；
      有外部天花板但没有我们的分 ⇒ 同样 G = 1.0。
  · 覆盖 C = sqrt(份额 / 全场最大份额)
      "卡组在我们对手中的覆盖率"（份额来自我们自己的对局扫描 `logs/sub_feedback/scan.jsonl`，
      即这副牌作为**对手**出现的比例）。开方是为了不把 34.9% 的 7e39 压成唯一目标。
  · 高峰 H = clip((天花板 − 600) / 1000, 0.2, 1.0)
      "高分卡组"方向：天花板（外部高手在**这副牌本身**打到 ≥50% 胜率时的分，规则见 lb_maps）
      越高越值得投；没有天花板数据时用牌天梯胜率折算一个代理值。

必须卡组覆盖面：**表的行不再局限于母表**，而是 `docs/必须卡组清单.csv`
（17 格真实份额格 ∪ 母表 ∪ 榜前 N 队用过的牌）。缺驾驶员的行也会出现，
`驾驶员类型` 列会标 `BC / init`，并在 `下一步动作` 里提示"先补训练驾驶员"。

    ./tools/py tools/build_investment_table.py
"""
from __future__ import annotations
import collections, csv, glob, hashlib, json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import build_investment_table as LEGACY  # 复用口径：真实份额权重、母表、天花板规则、17 格面板分
import required_decks as REQ

REQ_LIST = os.path.join(ROOT, "docs", "必须卡组清单.csv")
OUT = os.path.join(ROOT, "docs", "卡组天花板与投资分.csv")
GAP_SCALE = 400.0        # 分差到这个量级 ⇒ G=1
CEIL_FLOOR = 600.0       # 天花板归一化的下界（600 是"什么都没赢"的默认分）


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def main():
    W = LEGACY.real_weights()
    LEGACY.REALW = W
    D = LEGACY.load_master()
    PANEL = LEGACY.real_panel_levels()
    CD = LEGACY.cand_deck_map()
    ext, extwho, deckwr, ours = LEGACY.lb_maps()

    port_md5 = hashlib.md5(open(os.path.join(ROOT, "build", "port_plain", "model.pt"), "rb").read()).hexdigest()
    REQ.pool_index()                      # 填好 DECK_IDS
    req_rows = list(csv.DictReader(open(REQ_LIST, encoding="utf-8-sig")))

    rows = []
    for r in req_rows:
        h4 = r["4hex"]
        all_d, trained = REQ.driver_for(h4, port_md5)
        # ① 我方水平：优先用真实 17 格面板（哪个驾驶员有实测就用哪个）
        level, src, best = None, "", ""
        cand = [(d, PANEL[d][0]) for d in all_d if d in PANEL]
        if cand:
            best, level = max(cand, key=lambda t: t[1])
            src = f"真实17格({best})"
        elif D:
            # ② 退到母表：该牌最佳驾驶员的覆盖加权胜率
            deck_key = next((k for k in D if k.lower()[:4] == h4), None)
            if deck_key:
                cells = D[deck_key]

                def sc(drv):
                    num = wn = 0.0
                    for c, v in cells.items():
                        if drv in v and W.get(c):
                            num += W[c] * v[drv][0]
                            wn += W[c]
                    return (num / wn * 100.0) if wn else None

                opts = [(d, sc(d)) for d in {d for v in cells.values() for d in v}]
                opts = [(d, s) for d, s in opts if s is not None]
                if opts:
                    best, level = max(opts, key=lambda t: t[1])
                    src = "母表12格"
        # 远端天花板：回放口径（榜前队真实对局 + 最新榜单分数）优先，母表语料口径兜底
        try:
            ext_r = float(r.get("回放口径天花板") or 0) or None
        except ValueError:
            ext_r = None
        ext_who = r.get("回放口径最高分玩家") or ""
        if ext_r and (not ext.get(h4) or ext_r >= ext[h4]):
            ext[h4], extwho[h4] = ext_r, ext_who
        ceil = max([x for x in (ext.get(h4), ours.get(h4)) if x] or [0]) or None
        own = ours.get(h4)

        # 缺口 G：**只用远端（外部）天花板**，不掺我们自己的历史最高分
        #   —— 自己持有记录 = "这副牌我们已经打到头了"，没有外部参照就没法说还有多少空间
        if ext.get(h4):
            if own:
                G = clip((ext[h4] - own) / GAP_SCALE, 0.0, 1.0)
                gap_note = ""
            else:
                G = 1.0                                  # 外部有分、我们从没打到过 ⇒ 全是可赚的
                gap_note = "我们无分"
        else:
            G = 0.5                                      # 没有外部天花板 ⇒ 中性（不奖励也不惩罚）
            gap_note = "无外部天花板"
        # 覆盖 C：线性归一到全场最常出现的牌（不再开方 —— 用户要的就是"高覆盖率"倾斜）
        cov = float((r["对手覆盖率"] or "0%").rstrip("%") or 0) / 100.0
        cov_max = max([float((x["对手覆盖率"] or "0%").rstrip("%") or 0) / 100.0 for x in req_rows] or [0.01])
        C = (cov / cov_max) if cov_max else 0.0
        # 高峰 H
        if ext.get(h4):
            H = clip((ext[h4] - CEIL_FLOOR) / 1000.0, 0.2, 1.0)
        elif deckwr.get(h4):
            H = clip(deckwr[h4][0] / 0.60, 0.2, 1.0)   # 牌本身在天梯能赢多少当代理
        elif (r.get("榜前N队胜率") or "").endswith("%"):
            H = clip(float(r["榜前N队胜率"].rstrip("%")) / 100.0 / 0.60, 0.2, 1.0)
        else:
            H = 0.35
        inv = 100.0 * (max(G, 0) * max(C, 0) * max(H, 0)) ** (1.0 / 3.0)

        todo = ""
        if not trained and ext.get(h4) and G >= 0.9:
            todo = "最高优先：远端有分、我们从没练过"
        elif not trained:
            todo = "补训练驾驶员（现只有 BC/init）" if all_d else "补 init 驾驶员"
        elif G >= 0.9 and cov >= 0.02:
            todo = "训练优先（覆盖高且差距大）"
        elif G >= 0.9:
            todo = "训练（差距大，覆盖面小）"
        rows.append(dict(h4=h4, name=r["卡组文件"], arch=r["原型"], ts=r["17格"],
                         team_games=int(r["榜前N队局数"] or 0), team_wr=r["榜前N队胜率"],
                         cov=cov, owncov=float((r.get("覆盖率(我方对局)") or "0%").rstrip("%") or 0)/100.0, level=level, src=src, best=best, ceil=ceil,
                         ext=ext.get(h4), extwho=extwho.get(h4, ""), own=own,
                         deckwr=deckwr.get(h4), G=G, C=C, H=H, inv=inv,
                         drv_trained=r["已训练驾驶员"], drv_all=r["现有驾驶员"], todo=todo, note=gap_note))
    rows.sort(key=lambda r: -r["inv"])
    with open(OUT, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["投资分", "卡组", "原型", "对手覆盖率", "覆盖率(我方对局)", "榜前N队局数", "榜前N队胜率", "17格",
                    "最佳驾驶员", "我方水平%", "口径", "外部天花板", "外部最高分玩家", "我们最高分",
                    "天花板", "牌天梯胜率", "牌天梯局数", "缺口G", "覆盖C", "高峰H", "下一步动作", "备注"])
        for r in rows:
            w.writerow([round(r["inv"], 1), r["h4"], r["arch"], f"{r['cov']*100:.1f}%", f"{r['owncov']*100:.1f}%", r["team_games"], r["team_wr"], r["ts"],
                        r["best"], ("" if r["level"] is None else round(r["level"], 1)), r["src"],
                        r["ext"] or "", r["extwho"], r["own"] or "", r["ceil"] or "",
                        ("" if not r["deckwr"] else f"{r['deckwr'][0]*100:.1f}%"),
                        ("" if not r["deckwr"] else r["deckwr"][1]),
                        round(r["G"], 3), round(r["C"], 3), round(r["H"], 3), r["todo"], r["note"]])
    print(f"→ {os.path.relpath(OUT, ROOT)}（{len(rows)} 副必须卡组）")
    print(f"{'#':<3}{'卡组':6s}{'覆盖率':>7s}{'我方水平':>9s}{'天花板':>8s}{'G':>6s}{'C':>6s}{'H':>6s}{'投资分':>8s}  下一步")
    for i, r in enumerate(rows[:12], 1):
        lv = "—" if r["level"] is None else f"{r['level']:.1f}"
        print(f"{i:<3}{r['h4']:6s}{r['cov']*100:6.1f}%{lv:>9s}{(r['ceil'] or 0):8.0f}"
              f"{r['G']:6.2f}{r['C']:6.2f}{r['H']:6.2f}{r['inv']:8.1f}  {r['todo']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
