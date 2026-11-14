#!/usr/bin/env python3
"""把 the-pockmon-battle-choice.csv 的一行翻译成训练器的 BANKS/BANKDECKS/LEAGUE_PCT。

## 为什么要这个
训练器支持「冻结 bank + 钉死卡组」（`--league-ckpts` / `--bank-decks`），每局按 env tag
选 bank，**bank 的卡组由 bank-decks 钉死**。而 choice 表给的是「对手牌 → 最强驾驶员」。
本脚本把这一维映射成 bank 槽位：**按覆盖率分配槽位**（这样对手卡组分布≈实测场地）。

## 关键约束（写死在训练器里）
    assert league_pct * n_banks < 0.5
⇒ 指定驾驶员最多覆盖 **50% 的对局**，其余 ≥50% 仍是自对弈（learner 自己打覆盖率抽的牌）。
所以总配额默认取 0.49。

## 用法
    ./tools/py tools/choice_to_banks.py --a-deck 26c6
    ./tools/py tools/choice_to_banks.py --a-deck 26c6 --emit-env   # 输出可直接 source 的行

输出示例：
    BANKS=/root/kl/port_w.bin,/root/kl/g6_db04.bin,...
    BANKDECKS=teal_mask_ogerpon_ex_applin__7e39,dunsparce_abra__db04,...
    LEAGUE_PCT=0.0306
"""
import argparse, collections, csv, io, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 驾驶员 → 权重文件（GPU 机上的路径；本地演示可覆盖）
# ★ 名字 → GPU 机上的纯权重 .bin：**唯一来源是 tools/driver_registry.py**
#   （历史上这里手抄过一份，和 make_testset_A 的 DRIVER_AGENT 不同步 ⇒ 表里出现未知名字时
#    静默回退 init，训练其实在打弱端口。以后只认注册表，未知名字直接报错退出。）
sys.path.insert(0, os.path.join(ROOT, "tools"))
import driver_registry as REG

def _registry_pilot_map():
    m = {}
    for _n, _d in REG.DRIVERS.items():
        if _d.get("remote"):
            m[_n] = _d["remote"]
    for _a, _b in REG.ALIASES.items():
        if _b in m:
            m.setdefault(_a, m[_b])
    return m

DEFAULT_PILOT_MAP = {
    "init": "/root/kl/port_w.bin",
    **_registry_pilot_map(),
}
# ★ "每个牌各自最强驾驶员"（模拟天梯上真正会遇到的强驾驶员）。
#   依据 docs/opponent_library.md §D.3③ 的端口能力筛查：
#   同一副 0d86，端口驾驶时 g6_db04 赢 98%，换成 g8_bc_parent 只赢 6.75% —— 难度差一个数量级。
BEST_PILOT = {          # key = 4 位短名（与 choice 表的列名一致）
    "db04": "g6_db04",
    "2617": "ph3_2617_v1",
    "0d86": "shuppet_specialist",   # = g8_bc_parent
    "26c6": "bc_26c6",              # 弱于端口，仅占位；有更好的再换
}
MAX_BANKS = 16
TOTAL_QUOTA = 0.49     # league_pct * n_banks 必须 < 0.5


def read_choice(path, a_deck):
    """返回 {对手卡组hash: 驾驶员}。"""
    txt = open(path, "rb").read().decode("gb18030", errors="replace")
    rows = list(csv.reader(io.StringIO(txt)))
    hdr, body = rows[0], rows[1:]
    for r in body:
        if r[0] == a_deck:
            return {hdr[i]: r[i] for i in range(1, len(hdr))}
    raise SystemExit(f"✗ choice 表里没有挑战者行 {a_deck}")


PILOT_TABLE = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-self.csv")


def drivers_from_table(path=None):
    """读 **docs/the-pockmon-battle-choice-self.csv**（tools/build_pilot_table.py 产出）。

    这是"每副牌自己的最强驾驶员"的唯一真源；训练对手驾驶员一律从这里取。
    返回 {短名: 驾驶员}。
    """
    path = path or PILOT_TABLE
    if not os.path.exists(path):
        return {}
    txt = open(path, "rb").read().decode("gb18030", errors="replace")
    return {r["卡组"]: r["最强驾驶员"] for r in csv.DictReader(io.StringIO(txt)) if r.get("最强驾驶员")}


def drivers_from_choice(choice_path):
    """**从 choice 表反推每副牌自己的最强驾驶员。**

    choice 表是「我方牌 a × 对手牌 b → 我方用谁」；把某一副牌 b 当作"它自己的牌"来看时，
    它在自己那一行里最常出现的非 init 驾驶员，就是"该牌的最强驾驶员"
    （例：db04 行 8 格 g6_db04、26c6 行 2 格 r1A、0d86 行 1 格 shuppet_specialist）。

    返回 {短名: 驾驶员}；没有非 init 驾驶员的牌不出现在结果里（调用方回退 init）。
    """
    txt = open(choice_path, "rb").read().decode("gb18030", errors="replace")
    rows = list(csv.reader(io.StringIO(txt)))
    out = {}
    for row in rows[1:]:
        if not row:
            continue
        a, vals = row[0], row[1:]          # 第一列是行名，后面是 12 格
        cnt = collections.Counter(v for v in vals if v and v != "init")
        if cnt:
            out[a] = cnt.most_common(1)[0][0]
    return out


def coverage():
    """我们 582 局实测场地占比：{hash: share}。"""
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data/our_games.tsv")), delimiter="\t"))
    c = collections.Counter(r["opp_deck"].split("__")[-1]
                            for r in rows if r.get("opp_deck") not in (None, "", "?"))
    tot = sum(c.values())
    return {k: v / tot for k, v in c.items()}


def hash2full():
    m = {}
    for p in os.listdir(os.path.join(ROOT, "decks/pool_field")):
        if p.endswith(".csv") and "__" in p:
            base = p[:-4]
            m.setdefault(base.split("__")[-1][:4], base)
    return m


def allocate(cov, n_slots):
    """按覆盖率做最大余数法分配槽位。"""
    items = sorted(cov.items(), key=lambda t: -t[1])[:n_slots]
    tot = sum(v for _, v in items)
    quota = {k: v / tot * n_slots for k, v in items}
    base = {k: int(q) for k, q in quota.items()}
    rem = n_slots - sum(base.values())
    for k in sorted(quota, key=lambda k: -(quota[k] - base[k]))[:rem]:
        base[k] += 1
    return {k: v for k, v in base.items() if v > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-deck", required=True, help="A 的卡组 hash（如 26c6 / db04）")
    ap.add_argument("--choice", default=os.path.join(ROOT, "docs/the-pockmon-battle-choice.csv"))
    ap.add_argument("--strong", action="store_true",
                    help="对手侧按牌换最强驾驶员（BEST_PILOT），而不是照抄 choice 表")
    ap.add_argument("--pilot-map", default="", help="JSON: 驾驶员→权重路径（默认内置）")
    ap.add_argument("--max-banks", type=int, default=MAX_BANKS)
    ap.add_argument("--pilot-table", default=PILOT_TABLE,
                    help="每副牌最强驾驶员表（tools/build_pilot_table.py 产出）")
    ap.add_argument("--cov-src", choices=["our_games", "weights"], default="our_games",
                    help="槽位分配的覆盖率来源：our_games（我们 582 局实测）或 weights（场地权重文件）")
    ap.add_argument("--quota", type=float, default=TOTAL_QUOTA,
                    help="总配额（league_pct × 槽位数）；引擎上限 1.0 = 全冻结对手")
    ap.add_argument("--emit-env", action="store_true", help="只输出 BANKS=... 三行")
    a = ap.parse_args()

    pmap = dict(DEFAULT_PILOT_MAP)
    if a.pilot_map and os.path.exists(a.pilot_map):
        pmap.update(json.load(open(a.pilot_map)))

    pilots = read_choice(a.choice, a.a_deck)
    if a.strong:                      # 对手侧：按牌换成该牌的最强驾驶员
        merged = dict(BEST_PILOT)
        merged.update(drivers_from_choice(a.choice))        # ① 现算（choice 表反推）
        merged.update(drivers_from_table(a.pilot_table))    # ② 落盘表优先（build_pilot_table 的产物）
        pilots = {h: merged.get(h[-4:], "init") for h in pilots}
    if a.cov_src == "weights":
        cov = {}
        for line in open(os.path.join(ROOT, "decks/weights_field_b05.txt")):
            sp = line.split()
            if len(sp) == 2 and "__" in sp[0]:
                cov[sp[0].split("__")[-1][:4]] = float(sp[1])
        tot = sum(cov.values())
        cov = {k: v / tot for k, v in cov.items()}
    else:
        cov = coverage()
    h2n = hash2full()

    # 只考虑 choice 表里出现、且我们有覆盖率的对手牌
    decks = [d for d in pilots if d in cov]
    cov_sub = {d: cov[d] for d in decks}
    slots = allocate(cov_sub, min(a.max_banks, len(decks)))

    banks, bankdecks, bankw, unknown_pilot, missing_deck = [], [], [], [], []
    for d, k in sorted(slots.items(), key=lambda t: -t[1]):
        pilot = pilots[d]
        if pilot not in pmap:
            unknown_pilot.append((d, pilot))
            sys.exit(f"✗ 对手卡组 {d} 的驾驶员 {pilot!r} 在 tools/driver_registry.py 里没有可用 .bin；\n"
                     f"  先在注册表里登记（或把该格改回 init），不要静默回退 —— 那会让训练打的是弱端口。")
        full = h2n.get(d)
        if not full:
            missing_deck.append(d)
            continue
        for _ in range(k):
            banks.append(pmap[pilot])
            bankdecks.append(full)
            bankw.append(round(cov[d], 4))

    n = len(banks)
    pct = round(a.quota / n, 5)
    if not a.emit_env:
        print(f"A 的卡组 = {a.a_deck}   槽位 {n}/{a.max_banks}   总配额 {a.quota}")
        print(f"{'对手卡组':<12}{'槽位':>4}{'覆盖率':>8}{'驾驶员':<20}{'权重'}")
        for d, k in sorted(slots.items(), key=lambda t: -t[1]):
            print(f"{d:<12}{k:>4}{cov[d]*100:>7.1f}%{pilots[d]:<20}{pmap.get(pilots[d],'?')}")
        print(f"\n对手卡组里非 init 的：{[d for d in slots if pilots[d]!='init'] or '无'}")
        if unknown_pilot:
            print(f"⚠️ 没有权重文件的驾驶员（回退 init）：{unknown_pilot}")
        if missing_deck:
            print(f"⚠️ pool_field 里找不到的卡组：{missing_deck}")
        print(f"\nBANKS={','.join(banks)}")
        print(f"BANKDECKS={','.join(bankdecks)}")
        print(f"LEAGUE_PCT={pct}   （= {a.quota}/{n}，须 <0.5）")
        print(f"BANK_WEIGHTS={','.join(str(x) for x in bankw)}")
        print(f"\n剩余 {100*(1-a.quota):.0f}% 对局仍为自对弈（引擎硬限制）")
    else:
        print(f"BANKS={','.join(banks)}")
        print(f"BANKDECKS={','.join(bankdecks)}")
        print(f"LEAGUE_PCT={pct}")
        print(f"BANK_WEIGHTS={','.join(str(x) for x in bankw)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
