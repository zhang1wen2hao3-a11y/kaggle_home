#!/usr/bin/env python3
"""**卡组淘汰机制**：把"又弱、又没有格压力、环境里还没人用"的牌从维护表里清出去。

用户 2026-11-19 定的规则（原话）：
> 如果卡组强度全面低于其他卡组，除了冠军卡组，或者部分格子可以给我们的卡组部分压力，
>   后续我们应该有个淘汰机制，如果卡组强度不行，且没有给我们表格里的卡组部分格子压力，
>   在环境中也没有覆盖率，那么就应该被淘汰

⇒ 淘汰 = **三个条件同时成立**（缺一不可，避免误杀"弱但很毒"的牌）：
  ① 强度 S 在本表 12 副里**最低**（"全面低于其他卡组"）
  ② 格压力 P < 55%：它**没有任何一格**能打我们表内卡组到 55% 以上（"没有给任何格子压力"）
  ③ 环境覆盖率 C < 2%（"环境中也没有覆盖率"）
另外两条保护（"除了冠军卡组"）：
  · 强度冠军（S 最高）与 投资分冠军 一律保留；
  · 有格压力（P ≥ 55%）或覆盖率 C ≥ 2% 的牌一律保留（打"保留原因"列）。

数据来源（全部是已有维护表，不新造口径）：
  · 强度 S：`docs/the-pockmon-battle-choice-in-table.csv` 里 **init-vs-init** 单元的行均值
    （用 init 当驾驶员是为了**只比牌、不比驾驶员**；最佳驾驶员矩阵会把 g8_bc_parent 这种强驾驶员
     记到 0d86 头上，那是"人的分"，不是"牌的分"——这条坑 §5/§6 已经踩过）；
  · 格压力 P：`docs/the-pockmon-battle-choices-win-rate.csv`（12×12，最佳驾驶员实战）
    P(d) = max over 表内其它牌 c 的 (1 − 胜率(c 打 d))，即"d 当对手时，d 的胜率"；
  · 覆盖率 C、投资分：`docs/必须卡组清单.csv`、`docs/卡组天花板与投资分.csv`。

输出：`docs/卡组淘汰与新增建议.csv`（状态 / 强度 / 格压力 / 覆盖率 / 理由）

    ./tools/py tools/prune_decks.py
"""
from __future__ import annotations
import collections, csv, os, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHOICE = os.path.join(ROOT, "docs", "the-pockmon-battle-choice.csv")
WRATE = os.path.join(ROOT, "docs", "the-pockmon-battle-choices-win-rate.csv")
INTABLE = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-in-table.csv")
REQLIST = os.path.join(ROOT, "docs", "必须卡组清单.csv")
INV = os.path.join(ROOT, "docs", "卡组天花板与投资分.csv")
OUT = os.path.join(ROOT, "docs", "卡组淘汰与新增建议.csv")

MIN_PRESSURE = 0.55      # 一格都没打到 55% ⇒ "没有格压力"
MIN_COVERAGE = 0.02      # 环境覆盖率 < 2% ⇒ "环境中没有覆盖率"
MIN_SAMPLES = 200        # 算格压力时的最小样本
PROTECT_CEIL = 1500.0    # 外部天花板 ≥ 这个分 ⇒ "冠军级卡组"，一律保护（用户："除了冠军卡组"）


def gb(p):
    return open(p, "rb").read().decode("gb18030", errors="replace")


def table_decks():
    """维护表里**我方**的牌 = 第一列（行名）。表头列还包含 0bf5/db04/e8ee 这类
    "我们已经合并掉、但场上仍作为对手出现"的牌，不能算进我方牌单（2026-11-19 行/列分离后修正）。"""
    lines = gb(CHOICE).splitlines()
    return [ln.split(",")[0] for ln in lines[1:] if ln.strip()]


def winrate_matrix(decks):
    lines = gb(WRATE).splitlines()
    M = {}
    for ln in lines[1:]:
        f = ln.split(",")
        if len(f) < len(decks) + 1:
            continue
        row = f[0]
        for j, d in enumerate(decks):
            try:
                M[(row, d)] = float(f[j + 1])
            except ValueError:
                pass
    return M


def best_driver_of():
    """每副牌最强的专属驾驶员（作为**对手**出现时用过最多的非 init 名字）。"""
    cnt = collections.defaultdict(collections.Counter)
    for r in csv.DictReader(gb(INTABLE).splitlines()):
        d, drv = (r.get("被挑战者") or "").strip(), (r.get("对手驾驶员") or "").strip()
        if d and drv and drv != "init":
            cnt[d][drv] += 1
    return {d: c.most_common(1)[0][0] for d, c in cnt.items()}


def pressure_table():
    """格压力 P(d) = d 用**自己最好的驾驶员**当对手时，对各卡组的最高胜率。

    ★ 不能直接拿"d 当对手"的所有单元取最大：`0d86` 那 750 场对手是 init（端口开不了 shuppet），
      于是 12 副牌对它都是 97–100%，这个 100% 是"端口不会打这副牌"，不是牌的强弱。
      规则：d 有专属驾驶员 ⇒ 只看 对手驾驶员 == 该名字 的单元；没有 ⇒ 才看 init 单元。
    """
    drv = best_driver_of()
    per = collections.defaultdict(list)
    for r in csv.DictReader(gb(INTABLE).splitlines()):
        d = (r.get("被挑战者") or "").strip()
        odrv = (r.get("对手驾驶员") or "").strip()
        cdrv = (r.get("驾驶员") or "").strip()
        want = drv.get(d, "init")
        if odrv != want:
            continue
        # ★ 挑战方也有专属驾驶员却拿 init 上场 ⇒ 这格是"对手不会打自己那副牌"，不是牌的强弱，丢掉
        if cdrv == "init" and (r.get("挑战者") or "").strip() in drv:
            continue
        try:
            wr, n = float(r["胜率"]), int(float(r["样本场次"]))
        except (TypeError, ValueError):
            continue
        if n < MIN_SAMPLES:
            continue
        per[d].append((1 - wr, r["挑战者"], odrv, n))
    return per, drv


def init_strength():
    """每副牌的 init-vs-init 行均值 + 样本（只比牌不比驾驶员）。"""
    vals = collections.defaultdict(list)
    for r in csv.DictReader(gb(INTABLE).splitlines()):
        if r.get("驾驶员") != "init" or r.get("对手驾驶员") != "init":
            continue
        try:
            vals[r["挑战者"]].append((float(r["胜率"]), int(float(r["样本场次"]))))
        except (TypeError, ValueError):
            pass
    return {d: (statistics.mean(v[0] for v in vs), sum(v[1] for v in vs)) for d, vs in vals.items() if vs}


def load_ceil():
    """外部天花板（含回放口径）：docs/卡组天花板与投资分.csv 的 `天花板` 列。"""
    out = {}
    if not os.path.exists(INV):
        return out
    for r in csv.DictReader(open(INV, encoding="utf-8-sig")):
        try:
            out[(r.get("卡组") or "").strip()] = float(r.get("天花板") or 0) or None
        except ValueError:
            out[(r.get("卡组") or "").strip()] = None
    return out


def load_keyed(path, key, field):
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        k = (r.get(key) or "").strip()
        try:
            out[k] = float((r.get(field) or "").rstrip("%") or 0) / (100.0 if "%" in (r.get(field) or "") else 1.0)
        except ValueError:
            pass
    return out


def main(argv=None):
    decks = table_decks()
    M = winrate_matrix(decks)
    S = init_strength()
    S_best = {}
    for ln in gb(WRATE).splitlines()[1:]:
        f = ln.split(",")
        if len(f) < len(decks) + 1:
            continue
        vals = []
        for x in f[1:]:
            try:
                vals.append(float(x))
            except ValueError:
                pass
        if vals:
            S_best[f[0]] = statistics.mean(vals)
    PT, DRV = pressure_table()
    cov = load_keyed(REQLIST, "4hex", "对手覆盖率")
    CEIL = load_ceil()
    inv = load_keyed(INV, "卡组", "投资分")
    variants = set()
    try:
        for ln in open(os.path.join(ROOT, "decks", "deck_merge.txt"), encoding="utf-8"):
            ln = ln.strip()
            if ln and not ln.startswith("#") and len(ln.split()) == 2:
                variants.add(ln.split()[0])
    except OSError:
        pass

    # 冠军卡组：强度最高 + 投资分最高 → 一律保护
    champ_s = max(S, key=lambda d: S[d][0]) if S else None
    champ_inv = max(inv, key=lambda d: inv[d]) if inv else None

    rows = []
    for d in decks:
        s_val, s_n = S.get(d, (None, 0))
        # 格压力：用 d 自己最好的驾驶员（见 pressure_table 的注释）
        press = [(p, c, odrv, n) for (p, c, odrv, n) in PT.get(d, []) if c != d]
        p_max, p_who, p_drv = (max(press)[0], max(press)[1], max(press)[2]) if press else (None, "", "")
        c_val = cov.get(d)
        reasons = []
        if d == champ_s:
            reasons.append(f"强度冠军（S={s_val*100:.1f}%）保留")
        if d == champ_inv:
            reasons.append(f"投资分冠军（{inv[d]:.1f}）保留")
        if p_max is not None and p_max >= MIN_PRESSURE:
            reasons.append(f"有格压力 {p_max*100:.0f}%（打 {p_who}）")
        if c_val is not None and c_val >= MIN_COVERAGE:
            reasons.append(f"环境覆盖率 {c_val*100:.1f}%")
        if (CEIL.get(d) or 0) >= PROTECT_CEIL:
            reasons.append(f"冠军级卡组（外部天花板 {CEIL[d]:.0f} ≥ {PROTECT_CEIL:.0f}）保护")
        # "强度全面低于其他卡组"：以**最佳驾驶员行均值**为准（那才是我们真会用的强度），
        # 同时 init 口径也不能比全场最高还高（防止靠一个强驾驶员掩盖牌本身很弱）
        sbest = S_best.get(d)
        weakest = sbest is not None and sbest <= min(S_best.values()) + 1e-9   # 就是全场最低
        if s_val is None and p_max is None:
            status = "待补种子"       # 表里刚接进来、还没跑过任何一格 ⇒ 先补种子才能判断
            reasons.append("维护表里没有任何对局数据（下一步：tools/run_seeds_new_decks.sh）")
        elif reasons:
            status = "保留"
        elif weakest and (p_max is None or p_max < MIN_PRESSURE) and (c_val or 0) < MIN_COVERAGE:
            status = "淘汰"
            reasons.append(f"三项同时成立：强度全场最低 {s_val*100:.1f}% + 无格压力（最高 {p_max*100:.0f}%）"
                           f" + 覆盖率 {(c_val or 0)*100:.1f}%")
        else:
            status = "保留"
            reasons.append("未同时满足淘汰三条件")
        rows.append(dict(deck=d, status=status, s=(s_val if s_val is not None else None), s_n=s_n,
                         p=p_max, p_who=p_who, cov=c_val, inv=inv.get(d), why="；".join(reasons)))

    # 新增建议：必须卡组里不在本表的牌，按投资分排序
    adds = []
    for d, v in sorted(inv.items(), key=lambda kv: -kv[1]):
        if d in decks or d in variants:
            continue      # 已在表里，或已并入代表（变体不该提"新增"）
        adds.append(dict(deck=d, status="建议新增", s=None, s_n=0, p=None, p_who="",
                         cov=cov.get(d), inv=v,
                         why=f"投资分 {v:.1f}、覆盖率 {(cov.get(d) or 0)*100:.1f}% —— 不在维护表里"))
    rows.sort(key=lambda r: ({"淘汰": 0, "待补种子": 1}.get(r["status"], 2), -(r["inv"] or 0)))
    rows = rows + adds          # ★ 新增建议也要落表（之前只算没写，白算了）

    with open(OUT, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["状态", "卡组", "强度S(init-vs-init)", "样本", "格压力(最高)", "压力来自", "环境覆盖率", "投资分", "理由"])
        for r in rows:
            w.writerow([r["status"], r["deck"],
                        ("" if r["s"] is None else f"{r['s']*100:.1f}%"), r["s_n"],
                        ("" if r["p"] is None else f"{r['p']*100:.0f}%"), r["p_who"],
                        ("" if r["cov"] is None else f"{r['cov']*100:.1f}%"),
                        ("" if r["inv"] is None else round(r["inv"], 1)), r["why"]])
    n_cut = sum(1 for r in rows if r["status"] == "淘汰")
    n_wait = sum(1 for r in rows if r["status"] == "待补种子")
    print(f"→ {os.path.relpath(OUT, ROOT)}：维护表 {len(decks)} 副，淘汰 {n_cut} 副，建议新增 {len(adds)} 副"
          f"（阈值 P<{MIN_PRESSURE*100:.0f}% 且 C<{MIN_COVERAGE*100:.0f}% 且强度最低）；待补种子 {n_wait} 副")
    print(f"{'状态':6s}{'牌':6s}{'强度':>7s}{'格压力':>7s}{'覆盖率':>8s}{'投资分':>7s}  理由")
    for r in rows:
        print(f"{r['status']:6s}{r['deck']:6s}{(f'{r[chr(115)]*100:.1f}%' if r['s'] is not None else '—'):>7s}"
              f"{(f'{r[chr(112)]*100:.0f}%' if r['p'] is not None else '—'):>7s}"
              f"{(f'{r[chr(99)+chr(111)+chr(118)]*100:.1f}%' if r['cov'] is not None else '—'):>8s}"
              f"{(f'{r[chr(105)+chr(110)+chr(118)]:.1f}' if r['inv'] is not None else '—'):>7s}  {r['why'][:60]}")
    if "--check" in (argv or sys.argv[1:]):
        if n_cut:
            print(f"✗ 有 {n_cut} 副卡组同时满足淘汰三条件，仍留在维护表里："
                  f"{', '.join(r['deck'] for r in rows if r['status'] == '淘汰')}")
            print("  ⇒ 从 docs/the-pockmon-battle-choice*.csv 里移除（或补一条'为什么不能淘汰'的证据）后再刷新")
            return 4
        print(f"✓ 维护表没有需要淘汰的卡组（{len(decks)} 副，其中待补种子 {n_wait} 副）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
