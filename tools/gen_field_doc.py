#!/usr/bin/env python3
"""从 `cg/api.py` 自动生成《引擎字段清单》—— 每个字段「用了没 / 在哪用 / 漏了什么」。

为什么要有这个：我们踩过两次同类坑，都是"引擎给了字段但我们以为没有"：

1. `PLAY(7)` / `EVOLVE(9)` 选项**只有 `index`、没有 `area`**，隐含区是手牌
   ⇒ 5 个 `opt_src_*` 特征对每个 PLAY/EVOLVE 恒为 0（其中一个还被注释称作"★核心"）
2. `Pokemon.energies` 只被**数张数**、没查**属性** ⇒ 2 张《超》被判成"付得起 {W}{W}"

这类坑的共同点：**引擎的字段语义藏在 enum 的注释里**，不读注释就会漏。
所以这份清单直接从源码注释抽字段说明，并统计我们代码里的使用次数。

用法：
    ./run.sh tools/gen_field_doc.py                 # 写到 引擎字段清单.md
    ./run.sh tools/gen_field_doc.py --out /tmp/x.md
"""
import argparse
import ast
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = os.path.join(ROOT, "cg", "api.py")

# 我们自己的代码（统计使用次数时排除 cg/ 和生成物）
SEARCH_DIRS = ("framework", "agents", "tools", "tests")
SEARCH_FILES = ("main.py", "main_rollout.py", "local_match.py", "evaluate.py",
                "train_bc.py", "train_rl.py", "train_cf_q.py", "train_value.py",
                "gen_counterfactual.py", "gen_pool_data.py", "make_line_decks.py",
                "race_decks.py", "diagnose_rollout.py")

# 已知的"我们有意识地实现/忽略"的注释
NOTES = {
    "serial": "卡实例唯一编号；未用。可用于判断“这张是不是刚抽到的”",
    "appearThisTurn": "**未用**。刚上场的宝可梦不能进化，这条能直接影响 EVOLVE 判断",
    "inPlayArea": "**features.py 未用**（plan_features 用了 ATTACH 的）。ATTACH/EVOLVE 的目标宝可梦所在区",
    "inPlayIndex": "**features.py 未用**。ATTACH/EVOLVE 的目标宝可梦下标",
    "remainEnergyCost": "**未用**。引擎直接给出“还差几点能量”，ENERGY 子决策可直接用",
    "remainDamageCounter": "**未用**。放伤害指示物时剩余个数",
    "contextCard": "**未用**。本次选择在结算哪张牌 —— 判断条件牌是否满足的关键",
    "effect": "**未用**。正在生效的牌",
    "poisoned": "已用（但和另外 4 种揉成 1 个布尔 st_special_*）。麻痹/睡眠直接导致不能攻击，值得拆开",
    "energies": "已用，但**只数张数不查属性** —— 已于 2026-10-23 修成 _can_pay() 按属性判定",
    "search_begin_input": "**未用**（这是 search_begin 的入参，不是可读状态）",
}


def parse_api(path):
    """返回 (enums, dataclasses)。enums: {name: [(member, value, comment_fields)]}"""
    src = open(path).read()
    lines = src.splitlines()
    enums, dataclasses = {}, {}
    i = 0
    while i < len(lines):
        m = re.match(r"class (\w+)\(IntEnum\):", lines[i])
        if m:
            name = m.group(1)
            members, pending = [], []
            i += 1
            while i < len(lines) and (lines[i].startswith((" ", "\t")) or not lines[i].strip()):
                ln = lines[i]
                c = ln.strip()
                if c.startswith("#"):
                    pending.append(c.lstrip("# ").strip())
                else:
                    mm = re.match(r"\s*([A-Z_0-9]+)\s*=\s*([0-9]+)\s*,?\s*(?:#\s*(.*))?", ln)
                    if mm:
                        members.append((mm.group(1), mm.group(2),
                                        pending + ([mm.group(3)] if mm.group(3) else [])))
                        pending = []
                    elif c and not c.startswith(("global", "def", "return")):
                        pending = []
                i += 1
            enums[name] = members
            continue
        m = re.match(r"class (\w+):", lines[i]) or re.match(r"@dataclass", lines[i])
        if m and "dataclass" in lines[max(i - 1, 0)] + lines[i]:
            j = i + 1 if re.match(r"@dataclass", lines[i]) else i
            if j < len(lines) and re.match(r"class (\w+):", lines[j]):
                name = re.match(r"class (\w+):", lines[j]).group(1)
                fields, k = [], j + 1
                while k < len(lines) and (lines[k].startswith((" ", "\t")) or not lines[k].strip()):
                    mm = re.match(r"\s*(\w+)\s*:\s*(.+?)\s*(?:#\s*(.*))?$", lines[k])
                    if mm and not lines[k].strip().startswith("#"):
                        fields.append((mm.group(1), mm.group(2), mm.group(3) or ""))
                    k += 1
                dataclasses[name] = fields
                i = k
                continue
        i += 1
    return enums, dataclasses


def usage_count(field):
    """统计字段名在我们代码里出现的次数（排除 cg/）。"""
    pat = r"\b" + re.escape(field) + r"\b|[\"']" + re.escape(field) + r"[\"']"
    args = ["grep", "-rIoE", pat]
    total = 0
    for d in SEARCH_DIRS:
        p = os.path.join(ROOT, d)
        if os.path.isdir(p):
            r = subprocess.run(args + [p], capture_output=True, text=True)
            total += len([x for x in r.stdout.splitlines() if x])
    for f in SEARCH_FILES:
        p = os.path.join(ROOT, f)
        if os.path.isfile(p):
            r = subprocess.run(args + [p], capture_output=True, text=True)
            total += len([x for x in r.stdout.splitlines() if x])
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "引擎字段清单.md"))
    args = ap.parse_args()
    enums, dcs = parse_api(API)

    L = []
    L.append("# 引擎字段清单（自动生成）\n")
    L.append("> 由 `tools/gen_field_doc.py` 从 `cg/api.py` 直接抽取，"
             "并统计我们代码里的使用次数。**引擎更新后重跑一次即可刷新。**\n")
    L.append("> 为什么需要它：我们踩过两次同类坑，都是「引擎给了字段但我们以为没有」——\n")
    L.append("> ① `PLAY`/`EVOLVE` 选项只有 `index`、隐含区是手牌；"
             "② `Pokemon.energies` 只数张数不查属性。\n")
    L.append("> 这类语义**只写在 enum 的注释里**，不读注释就会漏。\n")

    # ---- OptionType
    if "OptionType" in enums:
        L.append("\n## 1. 选项类型 → 需要哪些字段\n")
        L.append("| 值 | 名称 | 引擎注释里要求的字段 |")
        L.append("|---:|---|---|")
        for name, val, comments in enums["OptionType"]:
            flds = []
            for c in comments:
                mm = re.match(r"(\w+)\s*\(([^)]+)\)\s*:\s*(.*)", c)
                if mm:
                    flds.append(f"**{mm.group(1)}** ({mm.group(2)})")
                elif re.match(r"^[A-Z_]+ = ", c) is None and c:
                    flds.append("_" + c + "_")
            L.append(f"| {val} | `{name}` | {'；'.join(flds) if flds else '—'} |")
        L.append("\n> ⚠️ **只有 `PLAY(7)` 缺 `area`** —— 实测 5,340 个 PLAY 选项全部是 "
                 "`{index, type}`（隐含区 = 手牌）。不兜这一层会让 `opt_src_*` 特征"
                 "对每个 PLAY 恒为 0（已踩，已修）。\n"
                 "> ✅ `EVOLVE(9)` **有** `area` + `inPlayArea/inPlayIndex`（实测 840/840），"
                 "不是同一个坑。\n"
                 "> ⚠️ `ATTACH`/`EVOLVE` 的 `inPlayArea/inPlayIndex` 是**目标宝可梦**，"
                 "和 `area/index`（要操作的牌）是两回事 —— 而 `features.py` 里**这两个字段都没用**"
                 "（只有 `plan_features.py` 用了 ATTACH 的）。\n")

    # ---- 其他 enum：LogType 单独展开
    if "LogType" in enums:
        L.append("\n## 2. LogType（`obs['logs']` 的事件类型）\n")
        L.append("`memory.py` 目前只用了 4 种：`MOVE_CARD` / `ATTACH` / `PLAY` / `DISCARD`。\n")
        L.append("| 值 | 事件 | 携带字段 |")
        L.append("|---:|---|---|")
        for name, val, comments in enums["LogType"]:
            flds = [c for c in comments if "(" in c]
            desc = [c for c in comments if "(" not in c]
            cells = "；".join(f"`{x}`" for x in flds) if flds else "—"
            L.append(f"| {val} | `{name}`{('：' + desc[0]) if desc else ''} | {cells} |")
        L.append("\n> ★ **`MOVE_CARD` 带 `cardId` + `fromArea`→`toArea`，"
                 "等于「对手打出的每一张牌」** —— 这是唯一不加噪声的对手推断信号。\n"
                 "> `DRAW_REVERSE`（对手抽牌）**没有 `cardId`**，所以用它不泄露信息。\n")

    # ---- 数据结构字段 + 使用次数
    L.append("\n## 3. 数据结构字段与我们的使用情况\n")
    for dc in ("Option", "SelectData", "Observation", "State", "PlayerState",
               "Pokemon", "Card", "Skill", "Attack", "CardData"):
        if dc not in dcs:
            continue
        L.append(f"\n### `{dc}`\n")
        L.append("| 字段 | 类型 | 引擎注释 | 我们用了几次 |")
        L.append("|---|---|---|---:|")
        for name, typ, comment in dcs[dc]:
            n = usage_count(name)
            mark = "**0**" if n == 0 else str(n)
            note = NOTES.get(name, "")
            cm = comment.strip()
            if note:
                cm = (cm + " —— " + note) if cm else note
            L.append(f"| `{name}` | `{typ[:40]}` | {cm} | {mark} |")
    L.append("\n> 「0 次」不等于必须用，但**每一个 0 都要有理由**。\n")

    txt = "\n".join(L) + "\n"
    open(args.out, "w").write(txt)
    print(f"已生成 {os.path.relpath(args.out, ROOT)}（{len(txt.splitlines())} 行）")
    zero = sum(1 for dc in dcs for (n, _, _) in dcs[dc] if usage_count(n) == 0)
    print(f"  字段总数 {sum(len(v) for v in dcs.values())}，其中我们**从未使用** {zero} 个")


if __name__ == "__main__":
    main()
