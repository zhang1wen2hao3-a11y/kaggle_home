#!/usr/bin/env python3
"""生成 `对位总表.md` —— 我们的卡组 × 对手原型 的 2D 胜负表 + 胜负手机制表。

设计原则：
  1. **一张 2D 表一个数据来源**，不混口径（真实天梯 / gate real / gate ours 分开列）
  2. 每个数字都带 n，不然没法判断可信度
  3. 下一行/下一列只需在 REGISTRY 或 CSV 里加一条，重跑本工具即可

用法：
    python3 tools/matchup_table.py            # 打印
    python3 tools/matchup_table.py --write    # 写 对位总表.md
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- 数据来源注册表
# (文件, 我方卡组标签, 口径)
GATE_REGISTRY = [
    ("logs/gate_ala_ours.json", "9d6e", "ours"),
    ("logs/gate_rr_dunsparce_abra__9d6e.json", "9d6e", "real"),
    ("logs/gate_grookey_thwackey__2617.json", "grookey", "ours"),
    ("logs/gate_rr_db04.json", "db04", "real"),
    ("logs/gate_rr_db04_ours.json", "db04", "ours"),
]

# 真实天梯：CSV -> 我方卡组标签
REAL_REGISTRY = [
    ("data/episodes_ala.csv", "9d6e"),
    ("data/episodes_grookey.csv", "grookey"),
    ("data/episodes_db04.csv", "db04"),
]

# 局部矩阵（多行 × 少列，n 小）
MATRIX_REGISTRY = [
    ("logs/roundrobin_dunsparce.json", "变体内筛(ours)"),
]

# 胜负手机制：手工维护（这是知识，不是数据）—— key 为对手原型
MECHANISM = {
    "teal_mask_ogerpon_ex_applin": "速攻铺场。无效果免疫 ⇒ `Powerful Hand` 有效，**我们最强的对位**。"
                                   "代价：db04 交出 Fezandipiti ex/Boss's Orders 后变慢 ⇒ −2.7~−6.5pp",
    "abra_kadabra": "真实野区**最大份额**。同为效果伤害牌 ⇒ 比谁先凑够手牌与进化。"
                    "Togekiss 的 `Wonder Kiss` 让奖品竞赛反超 ⇒ +21.7pp",
    "grookey_thwackey": "铺场 + `Festival Grounds`（免疫特殊状态）。我们靠手牌放大伤害，"
                        "但它铺得比我们快 ⇒ 拉锯；`Wonder Kiss` 后 +8.3pp",
    "dunsparce_abra": "**同型对拼**（Alakazam 家族）⇒ 长局、慢。"
                      "`Wonder Kiss` 每击倒多拿 1 奖品 ⇒ +42.0pp（碾压性）",
    "shuppet_dhelmise": "★ **结构性洞**：核心 Shuppet/Banette/Sinistcha/Poltchageist 带 "
                        "`Hide 'n' Sneak`（防所有效果，伤害除外）⇒ `Powerful Hand` **零伤害**。"
                        "9d6e 真实 0/9（只拿 0.4/6 奖品）。Togekiss 的 `Speed Wing` 140 是真实伤害 ⇒ 理论可修，**待真实天梯验证**",
    "mega_kangaskhan_ex_dwebble": "`Mist Energy` 同样挡效果 ⇒ `Powerful Hand` 归零。"
                                  "但**真实野区里几乎不出现**（118 局里 0 次）⇒ 别为它优化",
    "munkidori_marnie_s_impidimp": "样本极少（n=12~16），结论不稳",
    "dreepy_drakloak": "LB#1 Dipam 用的原型。9d6e 尚可（真实 0.700），**grookey 极差**"
                       "（真实 0.333 / gate 0.062）",
    "mega_lucario_ex_solrock": "样本少，db04 略降",
    "不在 185 池中": "对手池过期证据：占真实野区 ~9%，多为 Spiky Energy / Legacy+Neo Upper 构筑",
}


def gate_rows() -> dict[str, dict[str, dict[str, dict]]]:
    out: dict[str, dict[str, dict[str, dict]]] = collections.defaultdict(dict)
    for rel, deck, band in GATE_REGISTRY:
        p = ROOT / rel
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        agg = collections.defaultdict(lambda: {"w": 0, "n": 0})
        for k, v in d.get("per_deck", {}).items():
            a = k.split("__")[0]
            agg[a]["w"] += v["wins"]
            agg[a]["n"] += v["matches"]
        out[band][deck] = {
            "total": {"score": d["score"], "n": d["matches"], "err": d.get("errors", 0),
                      "lo": d.get("ci_low"), "hi": d.get("ci_high")},
            "arch": dict(agg),
        }
    return out


def real_rows() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for rel, deck in REAL_REGISTRY:
        p = ROOT / rel
        if not p.exists():
            continue
        agg = collections.defaultdict(lambda: {"w": 0, "n": 0})
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        for r in rows:
            a = r["op_deck"].split("__")[0] if r["op_deck"] != "?" else "不在 185 池中"
            agg[a]["w"] += int(r["won"])
            agg[a]["n"] += int(r["won"]) + int(r["loss"])
        W = sum(v["w"] for v in agg.values())
        N = sum(v["n"] for v in agg.values())
        out[deck] = {"total": {"score": W / max(N, 1), "n": N}, "arch": dict(agg)}
    return out


def matrix_rows() -> list[tuple[str, dict, dict]]:
    res = []
    for rel, label in MATRIX_REGISTRY:
        p = ROOT / rel
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        res.append((label, {r: {c: v["score"] for c, v in cols.items()
                                if isinstance(v, dict) and "score" in v}
                            for r, cols in d.items() if not r.startswith("_")}, d))
    return res


def pct(w, n):
    return f"{100*w/n:5.1f}% ({w}/{n})" if n else "—"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    G = gate_rows()
    R = real_rows()
    M = matrix_rows()

    arche = ["abra_kadabra", "teal_mask_ogerpon_ex_applin", "grookey_thwackey",
             "dunsparce_abra", "shuppet_dhelmise", "dreepy_drakloak",
             "mega_kangaskhan_ex_dwebble", "munkidori_marnie_s_impidimp",
             "mega_lucario_ex_solrock", "不在 185 池中"]
    short = {x: x.replace("teal_mask_ogerpon_ex_applin", "Ogerpon")
              .replace("abra_kadabra", "abra").replace("grookey_thwackey", "grookey")
              .replace("dunsparce_abra", "dunsparce").replace("shuppet_dhelmise", "shuppet")
              .replace("dreepy_drakloak", "dreepy").replace("mega_kangaskhan_ex_dwebble", "kangaskhan")
              .replace("munkidori_marnie_s_impidimp", "munkidori")
              .replace("mega_lucario_ex_solrock", "lucario_solrock")
              .replace("不在 185 池中", "池外") for x in arche}

    L = ["# 对位总表（我方差 × 对手原型）", "",
         "> 本文件由 `tools/matchup_table.py --write` 自动生成。**加新卡组/新数据只需在脚本的",
         "> `REAL_REGISTRY` / `GATE_REGISTRY` 里加一行，重跑即可。**", "",
         "## 0. 野区权重（决定一切结论的分母）", "",
         "| 来源 | abra | Ogerpon | grookey | dunsparce | dreepy | shuppet | 合计覆盖 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    tot_real = collections.Counter()
    for deck, v in R.items():
        for k, c in v["arch"].items():
            tot_real[k] += c["n"]
    N = sum(tot_real.values())
    L.append("| **真实天梯**（" + str(N) + " 局实测） | " +
             " | ".join(f"{100*tot_real[k]/N:.1f}%" for k in
                        ["abra_kadabra", "teal_mask_ogerpon_ex_applin", "grookey_thwackey",
                         "dunsparce_abra", "dreepy_drakloak", "shuppet_dhelmise"]) +
             f" | {100*sum(tot_real[k] for k in ['abra_kadabra','teal_mask_ogerpon_ex_applin','grookey_thwackey','dunsparce_abra','dreepy_drakloak','shuppet_dhelmise'])/N:.1f}% |")
    w = {}
    for line in (ROOT / "decks" / "weights_ours.txt").read_text().splitlines():
        p = line.split()
        if len(p) == 2 and p[0] not in ("version", "default"):
            w[p[0].split("__")[0]] = w.get(p[0].split("__")[0], 0) + float(p[1])
    tw = sum(w.values())
    L.append("| `weights_ours.txt`（旧语料） | " +
             " | ".join(f"{100*w.get(k,0)/tw:.1f}%" for k in
                        ["abra_kadabra", "teal_mask_ogerpon_ex_applin", "grookey_thwackey",
                         "dunsparce_abra", "dreepy_drakloak", "shuppet_dhelmise"]) +
             f" | {100*sum(w.get(k,0) for k in ['abra_kadabra','teal_mask_ogerpon_ex_applin','grookey_thwackey','dunsparce_abra','dreepy_drakloak','shuppet_dhelmise'])/tw:.1f}% |")
    og_real = 100 * tot_real["teal_mask_ogerpon_ex_applin"] / N
    og_ours = 100 * w.get("teal_mask_ogerpon_ex_applin", 0) / tw
    L += ["", f"⚠️ 两把尺子的 Ogerpon 权重差 **{og_ours - og_real:.1f}pp**"
              f"（{og_ours:.1f}% vs {og_real:.1f}%）。**用错尺子会选错牌**"
              "（实例：`546f`/`5b12` 把 kangaskhan 从 0.075 修到 0.45，但该格真实野区几乎不出现）。",
          "",
          "⚠️ **同一行的数字必须来自同一台机器**。跨机比较会产生假的差值"
          "（实例：db04 本机 ours 0.7275 曾被拿去向 4090 上的 9d6e 0.7425 对比，"
          "误报成 −1.50pp；同机同源实为 0.7275 vs 0.7300 = −0.25pp）。", ""]

    for band, title in (("real", "gate `real` 档（真实野区权重，n=400，seed=7）"),
                        ("ours", "gate `ours` 档（旧语料权重，历史口径）")):
        if band not in G:
            continue
        decks = list(G[band])
        L += [f"## {'1' if band=='real' else '2'}. {title}", "",
              "| 我方卡组 | " + " | ".join(short[k] for k in arche) + " | **总** |",
              "|---|" + "---:|" * len(arche) + "---:|"]
        for dk in decks:
            v = G[band][dk]
            cells = [(pct(v["arch"][k]["w"], v["arch"][k]["n"]).split(" (")[0]
                      if k in v["arch"] else "—") for k in arche]
            t = v["total"]
            L.append(f"| **{dk}** | " + " | ".join(cells) +
                     f" | **{t['score']:.4f}** (n={t['n']}) |")
        L.append("")

    if R:
        L += ["## 3. ★ 真实天梯（最有价值，但 n 小）", "",
              "| 我方卡组 | " + " | ".join(short[k] for k in arche) + " | **总** |",
              "|---|" + "---:|" * len(arche) + "---:|"]
        for dk, v in R.items():
            cells = [pct(v["arch"][k]["w"], v["arch"][k]["n"]) if k in v["arch"] else "—"
                     for k in arche]
            L.append(f"| **{dk}** | " + " | ".join(cells) +
                     f" | **{v['total']['score']:.3f}** (n={v['total']['n']}) |")
        L.append("")

    if M:
        for label, d, _raw in M:
            cols = sorted({c.split("__")[0] for cols in d.values() for c in cols})
            L += [f"## 4. {label}（小样本 n=40/格，仅用于排位）", "",
                  "| 变体 | " + " | ".join(short.get(c, c) for c in cols) + " | 简单均值 |",
                  "|---|" + "---:|" * len(cols) + "---:|"]
            for row, cols_ in d.items():
                vals = []
                for c in cols:
                    hit = [v for k, v in cols_.items() if k.startswith(c)]
                    vals.append(f"{hit[0]:.3f}" if hit else "—")
                nums = [float(x) for x in vals if x != "—"]
                L.append(f"| `{row.split('__')[-1]}` | " + " | ".join(vals) +
                         f" | {sum(nums)/len(nums):.3f} |" if nums else
                         f"| `{row.split('__')[-1]}` | " + " | ".join(vals) + " | — |")
            L.append("")

    L += ["## 5. 胜负手机制（知识，手工维护）", "",
          "| 对手原型 | 机制 / 胜负手 |", "|---|---|"]
    for k, txt in MECHANISM.items():
        L.append(f"| **{short.get(k, k)}** | {txt} |")
    L += ["", "## 6. 更新规程", "",
          "1. 新卡组提交后 → 抓真实对局到 `data/episodes_<标签>.csv`，在 `REAL_REGISTRY` 加一行",
          "2. 新卡组跑 gate → 在 `GATE_REGISTRY` 加一行（写清 band）",
          "3. 发现新机制 → 加进 `MECHANISM`",
          "4. `python3 tools/matchup_table.py --write` 重出本文件",
          "5. **永远先看第 0 节的权重，再看表**——用错尺子会得出相反结论", ""]

    text = "\n".join(L)
    if a.write:
        (ROOT / "对位总表.md").write_text(text, encoding="utf-8")
        print("写入 对位总表.md")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
