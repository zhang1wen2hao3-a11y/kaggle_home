#!/usr/bin/env python3
"""**新卡组接入 + 相似度检查 + 补种子清单**

用户 2026-11-19 要求（原话）：
> 新的 7 个高分卡组，这类确认上线较高的、且环境中已经存在的，也应该加入到这个表格里来，
> 当然你要做卡组相似度检查，如果和某个现有已知卡组有重合（卡组仅有不超过 4 张不同），
> 就应该仅对卡牌做覆盖（用高分的牌组），其他的复用之前的，高分也做合并；
> 用脚本同步刷新 docs 下的表格，种子不够的告诉我，回头开 CPU 刷种子。

规则（写死）：
1. **相似度**：`diff(a,b)` = 把 a 改成 b 最少要换几张牌 = `Σ|计数差| / 2`（60 张的多重集差）。
2. `diff ≤ MAX_DIFF(4)` ⇒ 与表内某牌同构 ⇒ **不新增表行**：视为该牌的变体，
   差异卡记进卡牌覆盖清单，天花板取两者最大（"高分也做合并"），驾驶员/结构全部复用原牌。
3. `diff > MAX_DIFF` ⇒ 作为**新卡组**接入 `decks/deck_order.txt`（维护表的唯一牌序来源），
   并算出还缺哪些种子（对每一副现有牌、双向各要一格）。
4. 表本身由 `build_from_master.py` 从母表重算，本脚本只维护牌序 + 清单，不手改矩阵。

输出：
  · `decks/deck_order.txt`（表内牌序，一行一副）
  · `docs/新卡组接入与补种子清单.csv`
  · `docs/新卡组相似度.csv`（每副新牌 vs 全部现有牌的最小差异与对手）

    ./tools/py tools/add_highscore_decks.py --dry
    ./tools/py tools/add_highscore_decks.py
"""
from __future__ import annotations
import argparse, collections, csv, glob, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = os.path.join(ROOT, "decks", "pool_field")
ORDER = os.path.join(ROOT, "decks", "deck_order.txt")
INTABLE = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-in-table.csv")
REQ = os.path.join(ROOT, "docs", "必须卡组清单.csv")
INV = os.path.join(ROOT, "docs", "卡组天花板与投资分.csv")
OUT_LIST = os.path.join(ROOT, "docs", "新卡组接入与补种子清单.csv")
OUT_SIM = os.path.join(ROOT, "docs", "新卡组相似度.csv")

MAX_DIFF = 4          # ≤4 张不同 ⇒ 同构变体，合并
MIN_N = 200           # 一个"格"至少这么多局才算有种子
DEFAULT_NEW = ["dreepy-drakloak__8fe3", "dreepy-drakloak__5b8f", "dreepy-drakloak__049c",
               "dreepy-drakloak__1917", "tealmaskogerponex-forestofvitality__0b34",
               "slowpoke-academyatnight__71b2", "beldum-metang__cb25"]
LEGACY_ORDER = ["0904", "0bf5", "0d86", "2617", "26c6", "3eaa", "7e39", "9d6e", "db04", "e1d7", "e8ee", "f5f8"]


def pool_decks():
    out = {}
    for p in glob.glob(os.path.join(POOL, "*.csv")):
        ids = [int(x) for x in open(p) if x.strip()]
        if ids:
            out[os.path.basename(p)[:-4]] = collections.Counter(ids)
    return out


def diff(a, b):
    return sum(abs(a[c] - b.get(c, 0)) for c in set(a) | set(b)) // 2


def load_order():
    if os.path.exists(ORDER):
        ids = [ln.strip() for ln in open(ORDER) if ln.strip() and not ln.startswith("#")]
        if ids:
            return ids
    return list(LEGACY_ORDER)


def master_n():
    """母表里的(挑战者, 被挑战者) → 最大样本量。"""
    n = collections.defaultdict(int)
    if not os.path.exists(INTABLE):
        return n
    txt = open(INTABLE, "rb").read().decode("gb18030", errors="replace")
    for r in csv.DictReader(txt.splitlines()):
        try:
            v = int(float(r.get("样本场次") or 0))
        except ValueError:
            v = 0
        k = ((r.get("挑战者") or "").strip(), (r.get("被挑战者") or "").strip())
        n[k] = max(n[k], v)
    return n


def load_pct(path, key, field):
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        v = (r.get(field) or "").rstrip("%") or "0"
        try:
            out[(r.get(key) or "").strip()] = float(v) / (100.0 if "%" in (r.get(field) or "") else 1.0)
        except ValueError:
            pass
    return out


def load_num(path, key, field):
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        try:
            out[(r.get(key) or "").strip()] = float(r.get(field) or 0) or None
        except ValueError:
            out[(r.get(key) or "").strip()] = None
    return out


def card_names():
    """卡号 → 卡名（`build/port_plain/EN_Card_Data.csv`，第一列 Card ID）。"""
    out = {}
    p = os.path.join(ROOT, "build", "port_plain", "EN_Card_Data.csv")
    try:
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            try:
                out[int(r["Card ID"])] = (r.get("Card Name") or "").strip()
            except (KeyError, ValueError):
                pass
    except OSError:
        pass
    return out


def card_delta(a, b, names):
    """a 比 b 多/少哪些卡（多重集差），用于"仅对卡牌做覆盖"。"""
    add = [c for c in a if a[c] > b.get(c, 0)]
    rem = [c for c in b if b[c] > a.get(c, 0)]
    fmt = lambda cs: ", ".join(f"{c}·{names.get(c, '?')}×{a.get(c, 0) - b.get(c, 0)}" if c in a
                               else f"{c}·{names.get(c, '?')}×{b.get(c, 0) - a.get(c, 0)}" for c in sorted(cs))
    return fmt(add), fmt(rem)


def short(name):
    return name.split("__")[-1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--decks", default="", help="逗号分隔的新卡组文件名（默认 2026-11-19 那 7 副）")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)

    NEW = [d.strip() for d in a.decks.split(",") if d.strip()] or DEFAULT_NEW
    pool = pool_decks()
    order = load_order()
    cov = load_pct(REQ, "4hex", "对手覆盖率")
    ceil = load_num(INV, "卡组", "天花板")
    inv = load_num(INV, "卡组", "投资分")
    M = master_n()
    NAMES = card_names()

    # 表内牌 → 池子文件名（按短名匹配，长短横线都试）
    def pool_name_for(h4):
        for n in pool:
            if short(n).lower() == h4.lower():
                return n
        return None

    sim_rows, plan = [], []
    for n in NEW:
        if n not in pool:
            print(f"  ✗ 池子里没有 {n}，跳过")
            continue
        h4 = short(n)
        table_pool = {pool_name_for(x) for x in order if pool_name_for(x)}
        cand = sorted((diff(pool[n], pool[o]), o) for o in table_pool if o != n)
        for d, o in cand[:3]:
            sim_rows.append([h4, n, o, short(o), d])
        best_d, best_o = (cand[0] if cand else (None, ""))
        verdict = "合并为变体" if (best_d is not None and best_d <= MAX_DIFF) else "新增到表"
        plan.append(dict(h4=h4, name=n, best=best_o, best_d=best_d, verdict=verdict,
                         cov=cov.get(h4.lower()), ceil=max([x for x in (ceil.get(h4),) if x] or [0]) or None,
                         inv=inv.get(h4)))

    adds = [p for p in plan if p["verdict"] == "新增到表"]
    new_order = order + [p["h4"] for p in adds if p["h4"] not in order]

    # 补种子清单：新牌 × 全部现有牌，双向
    lst = []
    for p in plan:
        if p["verdict"] != "新增到表":
            plus, minus = card_delta(pool[p["name"]], pool[p["best"]], NAMES) if p.get("best") in pool else ("", "")
            lst.append([p["h4"], p["name"], p["verdict"], f"与 {short(p['best'])} 只差 {p['best_d']} 张",
                        f"{(p['cov'] or 0)*100:.1f}%", p["ceil"] or "", 0,
                        f"多出：{plus or '无'}；缺少：{minus or '无'}",
                        f"不新增表行；复用 {short(p['best'])} 的驾驶员与结构；只把差异卡并进卡牌覆盖；天花板取 max({p['ceil'] or 'NA'}, {ceil.get(short(p['best'])) or 'NA'})"])
            continue
        missing = []
        for other in new_order:
            if other == p["h4"]:
                continue
            for chall, opp in ((p["h4"], other), (other, p["h4"])):
                if M.get((chall, opp), 0) < MIN_N:
                    missing.append(f"{chall}>{opp}")
        lst.append([p["h4"], p["name"], p["verdict"], f"与最近的 {short(p['best'])} 差 {p['best_d']} 张（> {MAX_DIFF}）",
                    f"{(p['cov'] or 0)*100:.1f}%", p["ceil"] or "", len(missing),
                    " ".join(missing[:8]) + (" …" if len(missing) > 8 else ""),
                    f"需补 {len(missing)} 格 × {MIN_N}–400 局；缺的格见左列"])

    print(f"新卡组 {len(plan)} 副：合并 {sum(1 for p in plan if p['verdict']=='合并为变体')} / 新增 {len(adds)}"
          f"（阈值 ≤{MAX_DIFF} 张不同即合并）")
    for p in plan:
        print(f"  {p['h4']:6s}{p['verdict']:8s} 最近 {short(p['best']):8s} 差 {p['best_d']} 张"
              f"  覆盖率 {(p['cov'] or 0)*100:5.1f}%  天花板 {p['ceil'] or 'NA'}")
    if adds:
        print(f"接入后维护表牌序（{len(order)} → {len(new_order)}）：{' '.join(new_order)}")
        need = sum(int(r[6]) for r in lst if r[2] == "新增到表")
        print(f"合计还缺种子格 {need} 个（每格 {MIN_N}–400 局）")

    if a.dry:
        return 0
    if adds:
        with open(ORDER, "w", encoding="utf-8") as fh:
            fh.write("# 维护表牌序（build_from_master.py 读这个文件；一行一副，顺序即矩阵列顺序）\n")
            for h4 in new_order:
                fh.write(h4 + "\n")
        print(f"✓ 写入 {os.path.relpath(ORDER, ROOT)}")
    with open(OUT_LIST, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["卡组", "文件", "判定", "相似度结论", "对手覆盖率", "天花板", "缺种子格数", "缺的格（前8）", "动作"])
        w.writerows(lst)
    with open(OUT_SIM, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["卡组", "文件", "最接近的牌", "短名", "需要换几张"])
        w.writerows(sim_rows)
    print(f"→ {os.path.relpath(OUT_LIST, ROOT)}、{os.path.relpath(OUT_SIM, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
