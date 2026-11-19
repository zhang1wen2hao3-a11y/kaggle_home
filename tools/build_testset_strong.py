#!/usr/bin/env python3
"""**建"强对手"真实场测试集**（2026-11-20 用户定：评估必须始终用最强的对手驾驶员）

背景（用户原话）：「我理解的真实场，是用我们手头的每套卡组的最强对位驾驶员，和天梯各卡组的出场概率，
来模拟真实场的情况……我们应该始终使用强度最高的对手来评估我方的卡组强度，流程要改。」

之前的 `decks/testset_meta_real.tsv` 里 **17 格有 15 格的对手是端口权重**（最弱的一档），
等于拿最软的对手量自己的强度 ⇒ 面板系统性偏乐观（9d6e 面板 67.6%，真实 LB 只有 964）。

本脚本为每个格挑"最强对手"，依据（按优先级）：
  1. **实测证据**：母表里 `被挑战者 = 该牌` 且 `对手驾驶员 = 候选` 的记录中，
     我们胜率**最低**的那个（打得越狠 = 越强），要求 n ≥ MIN_N；
  2. **有训练权重**：`build/agents` 里解开同样 60 张卡、且 model 与端口权重不同的 agent；
  3. 都没有 ⇒ 退回端口/init（该牌确实没有训练驾驶员）。

输出：
  · `decks/testset_meta_real.tsv`      ← **新的标准口径（强对手）**
  · `decks/testset_meta_real_port.tsv` ← 旧口径（端口对手），只作历史对照
  · `docs/对手强度审计.csv`            ← 每格候选、实测胜率、最终选谁、为什么

    ./tools/py tools/build_testset_strong.py --dry
    ./tools/py tools/build_testset_strong.py
"""
from __future__ import annotations
import argparse, collections, csv, glob, hashlib, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = os.path.join(ROOT, "decks", "testset_meta_real.tsv")
TS_PORT = os.path.join(ROOT, "decks", "testset_meta_real_port.tsv")
POOL = os.path.join(ROOT, "decks", "pool_field")
AGENTS = os.path.join(ROOT, "build", "agents")
INTABLE = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-in-table.csv")
AUDIT = os.path.join(ROOT, "docs", "对手强度审计.csv")
MIN_N = 200


def ids_of(p):
    try:
        return tuple(sorted(int(x) for x in open(p) if x.strip()))
    except (OSError, ValueError):
        return None


def deck_index():
    """60 张卡号 → 池子牌名；4hex → 池子牌名。"""
    by_ids, by_h4 = {}, {}
    for p in glob.glob(os.path.join(POOL, "*.csv")):
        k = ids_of(p)
        if not k:
            continue
        name = os.path.basename(p)[:-4]
        by_ids[k] = name
        by_h4[name.split("__")[-1][:4].lower()] = name
    return by_ids, by_h4


def driver_agent_map():
    """注册表里的驾驶员名 → agent 目录名（母表里的 对手驾驶员 用的是注册表名，如 r1A / deck_f5f8_）。"""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    try:
        import driver_registry as REG
    except Exception:
        return {}
    out = {}
    for name, info in REG.DRIVERS.items():
        ag = info.get("agent") or name
        out[name] = ag
        out[ag] = ag
    return out


def agent_pool(port_md5):
    """池子牌名 → [(agent 名, 是否训练权重)]"""
    out = collections.defaultdict(list)
    by_ids, _ = deck_index()
    for d in sorted(glob.glob(os.path.join(AGENTS, "*", "deck.csv"))):
        k = ids_of(d)
        if not k or k not in by_ids:
            continue
        name = d.split(os.sep)[-2]
        m = os.path.join(os.path.dirname(d), "model.pt")
        trained = os.path.exists(m) and hashlib.md5(open(m, "rb").read()).hexdigest() != port_md5
        out[by_ids[k]].append((name, trained))
    return out


def hardness():
    """硬度证据：{(我方牌, 对手牌, 对手驾驶员): 胜率}。

    ★ 必须**同一个挑战者**内比较才有意义：母表里"我们打 init 对手"的记录是所有我方驾驶员混在一起的，
      直接平均会把不同驾驶员的能力差异当成对手强度差异。所以下面只用
      `对手驾驶员 != init` 的那 17 条实测，逐条找同 (挑战者, 对手) 的 init 基线做对比。
    """
    tbl = {}
    try:
        txt = open(INTABLE, "rb").read().decode("gb18030", errors="replace")
    except OSError:
        return tbl
    for r in csv.DictReader(txt.splitlines()):
        try:
            wr, n = float(r["胜率"]), int(float(r["样本场次"]))
        except (KeyError, ValueError):
            continue
        if n < MIN_N:
            continue
        tbl[(r["挑战者"], r["被挑战者"], (r.get("对手驾驶员") or "init").strip())] = wr
    return tbl


def harder_than_port(tbl, opp, dname_map):
    """对手牌 opp：有没有**实测比端口更硬**的驾驶员？返回 (driver_name, delta) 或 (None, None)。

    delta = 打训练对手的胜率 − 打端口(init)的胜率；<0 表示更难打。
    """
    best = (None, 0.0)
    for (ch, o, od), wr in tbl.items():
        if o != opp or od == "init":
            continue
        base = tbl.get((ch, o, "init"))
        if base is None:
            continue
        d = wr - base
        if d < best[1]:
            best = (od, d)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)

    port_md5 = hashlib.md5(open(os.path.join(ROOT, "build", "port_plain", "model.pt"), "rb").read()).hexdigest()
    DNAME = driver_agent_map()
    _, by_h4 = deck_index()
    agents = agent_pool(port_md5)
    tbl = hardness()

    rows = []
    for ln in open(TS_PORT if os.path.exists(TS_PORT) else TS, encoding="utf-8"):
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.rstrip("\n").split("\t")
        rows.append(f)

    out, audit = [], []
    for f in rows:
        cell = f[0]
        if cell == "tail_mix":
            out.append(f)
            continue
        h4 = cell.split("_")[-1].lower()
        deck = by_h4.get(h4)
        cands = agents.get(deck, []) if deck else []
        # 去掉纯端口重复项：只留训练权重 + 一个端口兜底
        trained = [c for c, t in cands if t]
        ports = [c for c, t in cands if not t]
        # 只有在"同挑战者"实测里比端口更硬时才换（Δ<−3pp），否则保留端口
        pick, why = None, ""
        od, delta = harder_than_port(tbl, h4, DNAME)
        if od and delta < -0.03:
            pick = DNAME.get(od, od)
            why = f"实测比端口更硬 {delta*100:+.1f}pp（{od}）"
        elif trained and od is None:
            # 没有实测数据：取最近训练出来的那个（mtime 最新）
            def mt(a):
                m = os.path.join(AGENTS, a, "model.pt")
                return os.path.getmtime(m) if os.path.exists(m) else 0
            pick = max(trained, key=mt)
            why = "无实测硬度数据，取最近训练的驾驶员"
        elif ports:
            pick = ports[0]
            why = "只有端口权重（该牌无训练驾驶员）"
        if pick is None:
            out.append(f)
            audit.append([cell, h4, "", "", "保持原样（找不到 agent）"])
            continue
        out.append([cell, f"build/agents/{pick}"] + f[2:])
        audit.append([cell, h4, f[1].split("/")[-1], pick, why])

    print(f"{'格':16s}{'原对手':22s}{'新对手':22s}原因")
    for r in audit:
        print(f"{r[0]:16s}{r[2]:22s}{r[3]:22s}{r[4]}")
    if a.dry:
        return 0
    # 旧口径另存一份（如果还没备份过）
    if not os.path.exists(TS_PORT):
        open(TS_PORT, "w", encoding="utf-8").write(open(TS, encoding="utf-8").read())
        print(f"✓ 旧口径已备份 → {os.path.relpath(TS_PORT, ROOT)}")
    header = ["# 真实场口径（2026-11-20 起为**强对手**版）：每格用该牌实测最硬的对手驾驶员；"
              "出场概率权重见第 5 列；旧端口口径见 testset_meta_real_port.tsv"]
    with open(TS, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n")
        for f in out:
            fh.write("\t".join(f) + "\n")
    with open(AUDIT, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["格", "4hex", "原对手", "新对手", "选它的理由"])
        w.writerows(audit)
    print(f"✓ 写入 {os.path.relpath(TS, ROOT)}、{os.path.relpath(AUDIT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
