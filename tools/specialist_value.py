#!/usr/bin/env python3
"""League 角色化 · **Specialist Value** 计算器（2026-11-02 与用户对齐）。

## 定义

对 Champion `c` 和 Specialist `s`，两者在对手 `i` 上的差：

    G(s, i) = W(s, i) − W(c, i)          # 正数 = s 在那个对位上更好

按天梯分布（我们 640 场实地对局）加权的**潜在补位价值**（只看增益，不算损失）：

    V(s) = Σ_i P(i) · max(0, G(s, i))

同时给出**替代成本**（整体替换时的净变化，可正可负）：

    ΔEV_full(s) = Σ_i P(i) · G(s, i)

## ⚠️ 一个必须写在最前面的部署约束

**天梯一次提交只有一副牌**（`deck.csv` 固定），权重也不能合并。
所以：

  * Specialist 与 Champion **不同卡组** ⇒ `V(s)` 是**研究价值 / 联赛价值**，
    **不能**通过"换一副牌"在同一个提交里兑现。
  * Specialist 与 Champion **同卡组** ⇒ 才有可能替换部署（比如两个 abra 策略）。

所以本工具把价值拆成三列，别混读：

    V_league    Σ P(i)·max(0, G)      联赛/训练环境价值（可跨卡组）
    ΔEV_full    Σ P(i)·G              同卡组替换时的净期望变化（跨卡组无意义）
    V_same_deck 同卡组替换的价值上限    （只有同位候选才算）

用法：
    ./tools/py tools/specialist_value.py --champion db04
    ./tools/py tools/specialist_value.py --champion port_abra --all
"""
from __future__ import annotations

import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIELD = {
    "teal_mask_ogerpon_ex_applin": 0.356,
    "abra_kadabra": 0.149,
    "grookey_thwackey": 0.129,
    "dunsparce_abra": 0.113,
    "dreepy_drakloak": 0.096,
    "shuppet_dhelmise": 0.055,
}
ALIAS = {   # 日志里的对手名 → 场地原型
    "ogerpon": "teal_mask_ogerpon_ex_applin",
    "abra": "abra_kadabra",
    "grookey": "grookey_thwackey",
    "dunsparce": "dunsparce_abra",
    "db04": "dunsparce_abra",
    "dreepy": "dreepy_drakloak",
    "shuppet": "shuppet_dhelmise",
    "g8bc": "shuppet_dhelmise",
    "g8": "shuppet_dhelmise",
    "sp": "shuppet_dhelmise",
}


def load_cells():
    """(policy, opponent_arch) -> win_rate，从两批日志合并。"""
    cells = {}
    pats = [("logs/matrix2/*.json", "__vs__"), ("logs/bc_screen/*.json", "__vs_"),
            ("logs/abra_shuppet/*.json", "_vs_"), ("logs/bc_verify/*.json", "_vs_")]
    for pat, sep in pats:
        for f in glob.glob(os.path.join(ROOT, pat)):
            base = os.path.basename(f)[:-5]
            if sep not in base:
                continue
            a, b = base.split(sep, 1)
            b = b.strip("_")
            if b.startswith("self_"):
                b = b[5:]
            arch = ALIAS.get(b)
            if arch is None:
                continue
            try:
                o = json.load(open(f))["overall"]
            except Exception:
                continue
            key = (a, arch)
            prev = cells.get(key)
            if prev is None or o["n"] > prev["n"]:
                cells[key] = o
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default="db04")
    ap.add_argument("--all", action="store_true", help="列出所有候选（默认只列同卡组或显著更好的）")
    a = ap.parse_args()

    cells = load_cells()
    pols = sorted({p for p, _ in cells})
    if a.champion not in pols:
        raise SystemExit(f"✗ 没有 champion={a.champion} 的矩阵数据。当前可用：{', '.join(pols)}")
    field_sum = sum(FIELD.values())

    def w(pol, arch):
        c = cells.get((pol, arch))
        return (c["score"], c["n"]) if c else (None, 0)

    # champion 的卡组（用于判定"同卡组可替换"）
    champ_deck = {"db04": "dunsparce_abra", "port_abra": "abra_kadabra",
                  "bc_abra76k": "abra_kadabra", "bc_abra2": "abra_kadabra",
                  "port_grookey": "grookey_thwackey", "bc_grookey3": "grookey_thwackey",
                  "dreepy": "dreepy_drakloak", "abra": "abra_kadabra",
                  "grookey": "grookey_thwackey"}.get(a.champion, a.champion)

    rows = []
    for s in pols:
        if s == a.champion:
            continue
        g = {}
        for arch, p in FIELD.items():
            ws, ns = w(s, arch)
            wc, nc = w(a.champion, arch)
            if ws is None or wc is None:
                continue
            g[arch] = (ws - wc, ws, wc, p)
        if not g:
            continue
        v_league = sum(p * max(0.0, d) for d, _, _, p in g.values()) / field_sum
        dev_full = sum(p * d for d, _, _, p in g.values()) / field_sum
        same = any(A and s in ("bc_" + A, A) for A in [champ_deck])
        rows.append((s, champ_deck, v_league, dev_full, g, same))

    print(f"Champion = {a.champion}（卡组 {champ_deck}）")
    print(f"场地分布：{', '.join(f'{k.split(chr(95))[0]} {v:.1%}' for k, v in FIELD.items())}\n")
    print(f"{'specialist':14s}{'V_league':>10}{'ΔEV_full':>10}{'同卡组':>7}  主要增益对位")
    for s, deck, vl, dv, g, same in sorted(rows, key=lambda r: -r[2]):
        best = sorted(g.items(), key=lambda kv: -kv[1][0])[:2]
        note = ", ".join(f"{k.split('_')[0]} {100*v[0]:+.0f}pp" for k, v in best if v[0] > 0) or "（无正增益）"
        print(f"{s:14s}{vl:>10.3f}{dv:>+10.3f}{'✅' if same else '—':>7}  {note}")

    print("\n⚠️ 只有「同卡组 ✅」的行才可能在一次提交里替换 Champion；")
    print("   其余行的 V_league 是**联赛/训练环境价值**（例如它能否作为 PPO 的陪练）。")


if __name__ == "__main__":
    main()
