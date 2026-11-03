#!/usr/bin/env python3
"""卡组变异（<4 张改动），并把 §17.5 的「引擎守恒量」检查机械化。

1st place 的约束（writeup §5.2.1）：
  · 改动 **<4 张**
  · 依据 = 策略分析 + 常见卡牌组合 + 反制对位
  · 目标 = 削弱弱点 **且** 保住其它对位
本工具负责第 1 条与"组合协同"那一半的**自动量化**，不负责想点子。

用法：
    # 1) 先看一副牌的守恒量画像
    python3 tools/mutate_deck.py --deck dunsparce_abra__9d6e --audit

    # 2) 声明改动（会校验 60 张、卡名存在、改动数 <4，并报守恒量变化）
    python3 tools/mutate_deck.py --deck dunsparce_abra__9d6e \
        --swap "1x Sacred Ash -> 1x Fezandipiti ex" \
        --out dunsparce_abra__9d6e_v1

    # 3) 落到池里（默认只打印，--write 才落盘）
    ... --write
"""
from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
POOL = ROOT / "decks" / "pool_field"
CARD_CSV = ROOT / "data" / "EN_Card_Data_R2_full.csv"

# 判定"引擎卡"的启发式：卡面里出现抽牌/检索描述 => 它参与"手牌数"守恒量
ENGINE_RE = re.compile(r"\bdraw\b|\bsearch your deck\b|\bput .{0,30}into your hand\b", re.I)


def load_cards() -> tuple[dict[str, list[int]], dict[int, str], dict[int, str]]:
    byname: dict[str, list[int]] = collections.defaultdict(list)
    name: dict[int, str] = {}
    text: dict[int, str] = {}
    with open(CARD_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                cid = int(r["Card ID"])
            except (TypeError, ValueError):
                continue
            nm = (r["Card Name"] or "").strip()
            name[cid] = nm
            byname[nm].append(cid)
            # ⚠️ 卡表同一 Card ID 会出现**多行**（特性一行、招式一行）。
            #    必须累加而不是覆盖，否则会丢掉特性文本（Dudunsparce 的
            #    "Run Away Draw: draw 3 cards" 就是这样丢的）。
            frag = " ".join(filter(None, [r.get("Effect Explanation", ""),
                                          r.get("Move Name", ""),
                                          r.get("Rule", "")]))
            text[cid] = (text[cid] + " " + frag) if cid in text else frag
    # 注意：**不按卡名跨版本合并** —— 同名不同印刷版（不同 Card ID）是不同卡，
    # 合并会把别的版本的抽牌文本算进来（Alakazam 就会被误判成引擎卡）。
    # 同一 Card ID 的多行已经在上面累加了，这就够了。
    return byname, name, text


def load_deck(spec: str) -> tuple[str, list[int]]:
    p = pathlib.Path(spec)
    if not p.exists():
        cands = sorted(POOL.glob(spec + "*.csv"))
        if len(cands) != 1:
            raise SystemExit(f"卡组 {spec!r} 找不到或有歧义：{[c.stem for c in cands[:5]]}")
        p = cands[0]
    ids = [int(x) for x in p.read_text().split() if x.strip()]
    return p.stem, ids


def roles(ids: list[int], name: dict[int, str], text: dict[int, str]) -> dict[str, int]:
    cnt = collections.Counter(ids)
    eng = sum(c for i, c in cnt.items() if ENGINE_RE.search(text.get(i, "")))
    return {"总张数": len(ids), "引擎卡(抽/检索)": eng, "不同卡种": len(cnt)}


def parse_swap(s: str) -> tuple[int, str, int, str]:
    m = re.match(r"^\s*(\d+)\s*x?\s*(.+?)\s*->\s*(\d+)\s*x?\s*(.+?)\s*$", s)
    if not m:
        raise SystemExit(f"改写语法错误：{s!r}（应为 '1x 旧卡 -> 1x 新卡'）")
    return int(m.group(1)), m.group(2).strip(), int(m.group(3)), m.group(4).strip()


def resolve(nm: str, byname: dict[str, list[int]], deck_ids: list[int], want_in_deck: bool) -> int:
    if nm not in byname:
        near = [k for k in byname if nm.lower() in k.lower()][:5]
        raise SystemExit(f"卡名 {nm!r} 不在卡表里。相近：{near}")
    cands = byname[nm]
    if len(cands) == 1:
        return cands[0]
    # 同名多版本 -> 优先选牌里已经在用的那张
    inside = [c for c in cands if c in deck_ids]
    if inside:
        return inside[0]
    if want_in_deck:
        raise SystemExit(f"{nm!r} 有 {len(cands)} 个版本且都不在牌里：{cands}")
    print(f"  ⚠️ {nm!r} 有 {len(cands)} 个版本 {cands}，默认用 {cands[0]}")
    return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True)
    ap.add_argument("--swap", action="append", default=[], help="可多次：'1x 旧卡 -> 1x 新卡'")
    ap.add_argument("--out", default=None, help="变体名（写入池时用）")
    ap.add_argument("--write", action="store_true", help="真的写进 pool_field，否则只预览")
    ap.add_argument("--audit", action="store_true", help="只输出这副牌的守恒量画像")
    a = ap.parse_args()

    byname, name, text = load_cards()
    stem, ids = load_deck(a.deck)
    before = roles(ids, name, text)

    print(f"=== {stem} ===")
    print(f"  总张数 {before['总张数']}   引擎卡 {before['引擎卡(抽/检索)']}   卡种 {before['不同卡种']}")
    if a.audit:
        cnt = collections.Counter(ids)
        print(f"\n{'张':>3} {'卡名':34s} 角色")
        for cid, c in sorted(cnt.items(), key=lambda kv: (-kv[1], name.get(kv[0], ""))):
            role = "引擎(抽/检索)" if ENGINE_RE.search(text.get(cid, "")) else ""
            print(f"{c:3d}x {name.get(cid,'?')[:34]:34s} {role}")
        return 0

    if not a.swap:
        print("没有 --swap，也没有 --audit：什么也没做")
        return 0
    if len(a.swap) > 3:
        raise SystemExit(f"改动 {len(a.swap)} 处，超过 1st place 的 <4 张上限")

    cur = list(ids)
    for s in a.swap:
        n_out, nm_out, n_in, nm_in = parse_swap(s)
        cid_out = resolve(nm_out, byname, cur, True)
        cid_in = resolve(nm_in, byname, cur, False)
        if cur.count(cid_out) < n_out:
            raise SystemExit(f"{nm_out} 只有 {cur.count(cid_out)} 张，减不了 {n_out} 张")
        for _ in range(n_out):
            cur.remove(cid_out)
        cur += [cid_in] * n_in
        print(f"  改动：{n_out}x {nm_out} -> {n_in}x {nm_in}")
        if n_in - n_out != 0:
            print(f"  ⚠️ 这条改动改变总张数 {n_in - n_out:+d}，最终必须仍是 60 张")

    if len(cur) != 60:
        raise SystemExit(f"✗ 结果 {len(cur)} 张，不是 60")
    after = roles(cur, name, text)
    print(f"\n  改前：引擎卡 {before['引擎卡(抽/检索)']}   卡种 {before['不同卡种']}")
    print(f"  改后：引擎卡 {after['引擎卡(抽/检索)']}   卡种 {after['不同卡种']}")
    delta = after["引擎卡(抽/检索)"] - before["引擎卡(抽/检索)"]
    if delta < 0:
        print(f"\n  ⚠️ 引擎卡减少 {-delta} 张。对 Alakazam 这类「手牌数 = 伤害」的牌，")
        print(f"     等于主引擎输出下降 {-delta * 20} 点（见 下一步.md §17.5）。")
        print(f"     除非能证明弱格涨得更多，否则违反 §17.6 第 1 条「守恒量中性」。")
    elif delta > 0:
        print(f"\n  ✅ 引擎卡 +{delta}，主引擎不被削弱（但仍需过 §17.6 其余三条）")
    else:
        print("\n  ✅ 引擎卡不变（守恒量中性）")

    dest = POOL / f"{a.out or (stem + '_v1')}.csv"
    print(f"\n  目标文件：{dest}")
    body = "\n".join(str(x) for x in cur) + "\n"
    if a.write:
        dest.write_text(body, encoding="utf-8")
        print(f"  ✅ 已写入（记得同步更新 decks/weights_*.txt，否则 gate 抽不到它）")
    else:
        print("  （预览模式，加 --write 才落盘）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
