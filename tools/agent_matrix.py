#!/usr/bin/env python3
"""**Agent 强弱表**（可实时刷新）：统一身份、算显著性、控制规模、支持反查。

## 解决什么

建训练环境时要问："让对手打这副牌，用哪个策略最强？"
以前要翻 100 多个 logs/**/*.json，而数字来自不同种子/对手集合，只能当先验、不能排序。

## 三条设计原则（都是踩过坑才定的）

① 身份必须显式，不能靠猜。
   agent = <name>@<deck8>，name 取自 build/agents/<dir> 的真实目录名，deck8 从 deck.csv 算。
   **认不出来就计入被拒行并报数**，绝不静默并进某个格子 —— data/arena_matches.tsv 里
   policy_id 混着 '0'/'1'/'db04'/'learner' 这种脏值，自动并过一次就会把结论带歪。

② 规模不爆炸：只把"有稳定作用"的记在案。
   admitted 需同时满足：n >= --min-n（默认 400）、该格 95% Wilson CI 排除 0.5、
   对手不是未识别标签。其余全部标 probe：原始数据留在矩阵里，但不参与排序、不进 profile。

③ 不同可比组不合并。
   logs 里每次运行有自己的 seed_base；跨组同格子只并列不相加，group 列写明来源。
   排序只在同组内才有意义。

## 用法

    ./tools/py tools/agent_matrix.py                        # 刷新 + 汇总
    ./tools/py tools/agent_matrix.py --query 0904e46f       # 谁最会打这副牌
    ./tools/py tools/agent_matrix.py --query bc_abra76k     # 谁最会打这个 agent
    ./tools/py tools/agent_matrix.py --min-n 800            # 更严的入表门槛

产物：data/agent_matrix.tsv（全部格子）、data/agent_strength.tsv（每 agent 一行）
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha12(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except OSError:
        return ""


def deck8_of(deck_csv):
    try:
        ids = [int(x) for x in open(deck_csv) if x.strip()]
    except (OSError, ValueError):
        return ""
    return hashlib.sha256(",".join(str(x) for x in sorted(ids)).encode()).hexdigest()[:8]


def agent_table():
    t = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "build", "agents", "*"))):
        name = os.path.basename(d)
        dc, mp = os.path.join(d, "deck.csv"), os.path.join(d, "model.pt")
        if not os.path.isfile(dc):
            continue
        t[name] = dict(name=name, pol=sha12(mp) if os.path.isfile(mp) else "",
                       deck8=deck8_of(dc))
    d2h = {}
    for pat in ("decks/pool_field/*.csv", "decks/top/*.csv", "decks/*.csv"):
        for f in glob.glob(os.path.join(ROOT, pat)):
            h = deck8_of(f)
            if h:
                d2h[os.path.basename(f)[:-4]] = h
    return t, d2h


def resolve_name(name, agents, d2h):
    if name in agents:
        return agents[name]["name"], agents[name]["deck8"]
    for stem, h in sorted(d2h.items(), key=lambda kv: -len(kv[0])):
        short = stem.rsplit("__", 1)[-1]
        for tail in (stem, short):
            if name.endswith("_" + tail):
                return name[: -(len(tail) + 1)].rstrip("_"), h
    if len(name) == 8 and all(c in "0123456789abcdef" for c in name):
        return "?", name
    return None, ""


def opp_fmt(r):
    return r["on"] + ("@" + r["od"] if r.get("od") else "")


def wilson(k, n, z=1.96):
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def load_logs(agents, d2h, cells, junk):
    n = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "logs", "**", "*.json"), recursive=True)):
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict) or "per_opp" not in d:
            continue
        base = os.path.basename(str(d.get("cand", "")).rstrip("/"))
        # 机械别名：历史运行把 cand 记成 ag_xxx / agent_xxx，去掉已知前缀即可，不做猜测
        for pre in ("ag_", "agent_", "build_agents_"):
            if base not in agents and base.startswith(pre):
                base = base[len(pre):]
        if base not in agents:
            junk["cand 认不出"] += 1
            continue
        a = agents[base]
        seed = str((d.get("protocol") or {}).get("seed_base", "?"))
        for label, v in d["per_opp"].items():
            if not isinstance(v, dict) or "score" not in v or not v.get("n"):
                continue
            name = label.split("=")[0] if "=" in label else label
            oname, odeck = resolve_name(name, agents, d2h)
            if oname is None:
                junk["对手标签认不出"] += 1
                oname = "label:" + name
            cells.append(dict(an=a["name"], ad=a["deck8"], apol=a["pol"],
                              on=oname, od=odeck, n=int(v["n"]),
                              wins=int(round(v["score"] * v["n"])),
                              group="seed:" + seed, src=os.path.relpath(f, ROOT)))
        n += 1
    return n


def load_arena(agents, cells, junk):
    p = os.path.join(ROOT, "data", "arena_matches.tsv")
    if not os.path.exists(p):
        return 0
    agg = collections.defaultdict(lambda: [0, 0])
    used = 0
    for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"):
        try:
            w0 = int(r.get("winner") or "") == 0
        except ValueError:
            junk["arena 胜负字段坏"] += 1
            continue
        pd = (r.get("policy_id") or "").strip()
        od = (r.get("opponent_policy_id") or "").strip()
        if pd not in agents or od not in agents:
            junk["arena 策略名认不出"] += 1
            continue
        a, b = agents[pd], agents[od]
        key = (a["name"], a["deck8"], b["name"], b["deck8"])
        agg[key][1] += 1
        agg[key][0] += 1 if w0 else 0
        used += 1
    for (an, ad, bn, bd), (k, n) in agg.items():
        cells.append(dict(an=an, ad=ad, apol=agents[an]["pol"], on=bn, od=bd,
                          n=n, wins=k, group="arena", src="data/arena_matches.tsv"))
    return used


def merge(cells):
    out = {}
    for c in cells:
        key = (c["an"], c["ad"], c["on"], c["od"], c["group"])
        if key in out:
            out[key]["n"] += c["n"]
            out[key]["wins"] += c["wins"]
        else:
            out[key] = dict(c)
    for c in out.values():
        c["score"], c["lo"], c["hi"] = wilson(c["wins"], c["n"])
    return sorted(out.values(), key=lambda c: -c["n"])


def judge(rows, min_n):
    prof = collections.defaultdict(list)
    for c in rows:
        c["admitted"] = bool(c["n"] >= min_n
                             and not c["on"].startswith("label:")
                             and not c["on"].startswith("?")
                             and (c["lo"] > 0.5 or c["hi"] < 0.5))
        prof[(c["an"], c["ad"])].append(c)
    return prof


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-n", type=int, default=400)
    ap.add_argument("--query", default="")
    ap.add_argument("--out", default="data/agent_matrix.tsv")
    ap.add_argument("--profile", default="data/agent_strength.tsv")
    a = ap.parse_args()

    agents, d2h = agent_table()
    cells, junk = [], collections.Counter()
    nlog = load_logs(agents, d2h, cells, junk)
    nar = load_arena(agents, cells, junk)
    rows = merge(cells)
    prof = judge(rows, a.min_n)

    print("身份表 %d 个 agent | eval 日志 %d 个 | arena 可用 %d 场" % (len(agents), nlog, nar))
    if junk:
        print("被拒的行（不静默合并）：" + "，".join("%s %d" % (k, v) for k, v in junk.most_common()))

    with open(os.path.join(ROOT, a.out), "w") as fh:
        fh.write("agent\tdeck\tpolicy\topp\topp_deck\tn\twins\tscore\t"
                 "ci_low\tci_high\tgroup\tadmitted\tsource\n")
        for r in rows:
            fh.write("\t".join([r["an"] + "@" + (r["ad"] or "-"), r["ad"] or "-",
                                r["apol"] or "-", r["on"], r["od"] or "-", str(r["n"]),
                                str(r["wins"]), "%.4f" % r["score"], "%.4f" % r["lo"],
                                "%.4f" % r["hi"], r["group"], str(int(r["admitted"])),
                                r["src"]]) + "\n")

    with open(os.path.join(ROOT, a.profile), "w") as fh:
        fh.write("agent\tdeck\tadmitted\tn_cells\tn_games\tn_sig\tbest_opp\tbest_score\t"
                 "best_n\tworst_opp\tworst_score\n")
        for (an, ad), rs in sorted(prof.items(), key=lambda kv: -sum(x["n"] for x in kv[1])):
            adm = sorted([x for x in rs if x["admitted"]], key=lambda x: -x["score"])
            b = adm[0] if adm else None
            w = adm[-1] if adm else None
            fh.write("\t".join([
                an + "@" + (ad or "-"), ad or "-", str(int(bool(adm))), str(len(rs)),
                str(sum(x["n"] for x in rs)), str(len(adm)),
                opp_fmt(b) if b else "-", ("%.4f" % b["score"]) if b else "-",
                str(b["n"]) if b else "-",
                opp_fmt(w) if w else "-", ("%.4f" % w["score"]) if w else "-"]) + "\n")

    adm_agents = sum(1 for rs in prof.values() if any(x["admitted"] for x in rs))
    print("\n格子 %d 个 → admitted %d 个" % (len(rows), sum(1 for r in rows if r["admitted"])))
    print("agent %d 个 → **admitted %d 个**，其余 %d 个只标 probe（不进表、不参与排序）"
          % (len(prof), adm_agents, len(prof) - adm_agents))
    print("\n%-28s%4s%7s%7s  最强格" % ("agent", "格", "总场", "显著格"))
    for (an, ad), rs in sorted(prof.items(), key=lambda kv: -sum(x["n"] for x in kv[1]))[:16]:
        adm = sorted([x for x in rs if x["admitted"]], key=lambda x: -x["score"])
        tag = "%-24s %.3f (n=%d)" % (opp_fmt(adm[0]), adm[0]["score"], adm[0]["n"]) if adm else "—"
        print("%-28s%4d%7d%7d  %s" % (an + "@" + (ad or "-"), len(rs),
                                      sum(x["n"] for x in rs), len(adm), tag))

    if a.query:
        q = a.query.lower()
        # 允许直接用卡组名查（"z_ogerpon_toolbox_miwa__e1d7" / "e1d754a7"）
        if a.query in d2h:
            q = d2h[a.query].lower()
        hit = [r for r in rows if q in r["on"].lower() or q in (r["od"] or "").lower()
               or q in r["an"].lower() or q in (r["ad"] or "").lower()]
        print("\n=== 谁最会打 %s（n>=%d 才排序）===" % (a.query, a.min_n))
        ok = [r for r in hit if r["n"] >= a.min_n]
        if not ok:
            print("（命中 %d 格，都不到 n=%d，样本不足，不给排序）" % (len(hit), a.min_n))
        else:
            print("%-28s%7s%16s%6s  组" % ("候选", "胜率", "95% CI", "n"))
            for r in sorted(ok, key=lambda r: -r["score"]):
                print("%-28s%7.3f%16s%6d  %s" % (r["an"] + "@" + (r["ad"] or "-"), r["score"],
                                                 "[%.3f,%.3f]" % (r["lo"], r["hi"]),
                                                 r["n"], r["group"]))
    print("\n矩阵 → %s\n档案 → %s" % (a.out, a.profile))


if __name__ == "__main__":
    main()
