#!/usr/bin/env python3
"""**BC 分类别模仿准确率**（D1 验收用：不看总 val_acc，看"这副牌的操作"学没学会）。

## 为什么需要它

`bc_dragapult_dreepy_pc` 的教训：val_acc 0.567→0.687（+12pp）看起来很好，
但同卡组对打只有 **0.3825** —— 因为它学会了"普通动作"（攻击/结束/上能量），
却没学会真正决定胜负的**工具箱组合决策**（检索、支援者调度、场地切换、能力链）。

总准确率会被占比大的普通动作淹没。本工具把准确率按**动作语义**拆开统计。

## 分类口径（以出手方那一刻的**真实动作**为准）

| 类别 | 判定 |
|---|---|
| `search` 检索/搜索 | 使用 Item，且卡名在检索白名单里（Poffin/Ultra Ball/Tera Orb/Bug Catching Set/Poké Pad/Night Stretcher…） |
| `resource` 资源调度 | 使用 **Supporter**（Lillie's Determination / Ciphermaniac / N's Plan / Boss's Orders…） |
| `stadium` 场地切换 | 使用 **Stadium**（Area Zero Underdepths / Watchtower…） |
| `ability` 能力/组合 | `OT_ABILITY`（Munkidori / Meowth / Fezandipiti / Area Zero 之类的能力链） |
| `evolve` 进化 | `OT_EVOLVE` |
| `attach` 上能量 | `OT_ATTACH` |
| `attack` 攻击 | `OT_ATTACK` |
| `end` 结束回合 | `OT_END` |
| `other` | 其余（retreat / discard / skill / play-其他） |

同时给出**预测分布**：如果模型把 `search` 帧大多预测成 `attack`，那就是"压根没学会检索"，
即使这些帧被算进某个高准确率类别也不代表会打。

## 用法

    ./tools/py tools/bc_category_acc.py \
        --npz data/bc_miwa_e1d7.npz --model /path/bc_miwa_e1d7.pt \
        --split val --out logs/bc_cat_miwa_e1d7.tsv

    # 只统计某一类（调试拆分）
    ./tools/py tools/bc_category_acc.py ... --only-cat search
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.environ.get("PTCG_BUNDLE") or os.path.join(ROOT, "build", "port_plain")
sys.path.insert(0, BUNDLE)
sys.path.insert(0, ROOT)

from ptcg.rl.search_agent import SearchAgent  # noqa: E402

ROW_F = 10944
OPT_OFF = 9760          # OFFSETS["opt_int"] 起点
OPT_INT = 6             # 每个选项 6 个 int：[ot, p1, p2, card_id, attackId, number]
MAX_OPTIONS = 64

OT_PLAY, OT_ATTACH, OT_EVOLVE = 7, 8, 9
OT_ABILITY, OT_DISCARD, OT_RETREAT = 10, 11, 12
OT_ATTACK, OT_END, OT_SKILL = 13, 14, 15
# 引擎 OptionType：0~6 是**子选择提示**（NUMBER/YES/NO/CARD/TOOL_CARD/ENERGY_CARD/ENERGY），
# 7~17 才是主动作。混在一起统计会让"选牌"这种廉价提示淹没真正的操作能力。
OT_SELECT_NAMES = {0: "number", 1: "yes", 2: "no", 3: "card", 4: "tool_card",
                   5: "energy_card", 6: "energy"}
OT_NAME = {**OT_SELECT_NAMES,
           OT_PLAY: "play", OT_ATTACH: "attach", OT_EVOLVE: "evolve", OT_ABILITY: "ability",
           OT_DISCARD: "discard", OT_RETREAT: "retreat", OT_ATTACK: "attack",
           OT_END: "end", OT_SKILL: "skill"}
MAIN_TYPES = (OT_PLAY, OT_ATTACH, OT_EVOLVE, OT_ABILITY, OT_DISCARD,
              OT_RETREAT, OT_ATTACK, OT_END, OT_SKILL)

# 检索类 Item 白名单（卡名子串，小写匹配）
SEARCH_ITEMS = ("buddy-buddy poffin", "ultra ball", "tera orb", "bug catching set",
                "poké pad", "poke pad", "night stretcher", "nest ball", "level ball",
                "great ball", "quick ball", "hisuian heavy ball", "earthen vessel",
                "buddy poffin")

CATS = ["search", "resource", "stadium", "ability", "evolve", "attach", "attack", "end",
        "other", "select"]


def load_card_categories():
    """card_id → (name, kind)，kind 取自 `Stage (Pokémon)/Type (Energy and Trainer)`。

    ⚠️ 不能用 `Category` 那列 —— 它在 R2 卡表里全是 `n/a`（踩过）。
    kind ∈ {Item, Supporter, Stadium, Pokémon, Basic Energy, Special Energy, ...}
    """
    import csv
    path = os.path.join(ROOT, "data", "EN_Card_Data_R2_full.csv")
    info = {}
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                cid = int(row["Card ID"])
            except (KeyError, ValueError):
                continue
            kind = (row.get("Stage (Pokémon)/Type (Energy and Trainer)") or "").strip()
            info[cid] = (row.get("Card Name") or "", kind, row.get("Rule") or "")
    return info


def classify(ot, card_id, cards):
    """→ (类别, 细分标签)。类别口径见文件头表格。"""
    name, cat, _rule = cards.get(int(card_id), ("", "", ""))
    low = name.lower()
    if ot == OT_PLAY:
        if cat == "Supporter":
            return "resource", name
        if cat == "Stadium":
            return "stadium", name
        if cat in ("Item", "Pokémon Tool"):
            if any(k in low for k in SEARCH_ITEMS):
                return "search", name
            return "other", name          # 非检索类 Item（如 Crushing Hammer / Energy Switch）
        return "other", name              # 从手牌放上场的宝可梦等
    if ot in OT_SELECT_NAMES:
        return "select", OT_SELECT_NAMES[ot]
    if ot == OT_ABILITY:
        return "ability", name
    if ot == OT_EVOLVE:
        return "evolve", name
    if ot == OT_ATTACH:
        return "attach", name
    if ot == OT_ATTACK:
        return "attack", name
    if ot == OT_END:
        return "end", name
    return "other", name


def build_policy(model_path, bundle, device):
    ag = SearchAgent(model_path=str(model_path), deck_path=os.path.join(bundle, "deck.csv"),
                     cfg=None, seed=0, device=device, data_dir=bundle)
    return ag.policy


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--x-dat", default=None, help="X 流式 f32（默认与 npz 同名 .f32）")
    ap.add_argument("--model", required=True, help="BC 训练产出的 .pt")
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="只算前 N 帧（调试）")
    ap.add_argument("--only-cat", default="", help="只打印这一类")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import torch
    z = np.load(a.npz)
    y, nopt, split = z["y"], z["nopt"], z["split"]
    xdat = a.x_dat or os.path.splitext(a.npz)[0] + ".f32"
    if not os.path.exists(xdat):
        raise SystemExit(f"✗ 缺 {xdat}")
    X = np.memmap(xdat, dtype=np.float32, mode="r", shape=(len(y), ROW_F))
    idx = np.arange(len(y)) if a.split == "all" else np.where(split.astype(str) == a.split)[0]
    if a.limit:
        idx = idx[: a.limit]
    if len(idx) == 0:
        raise SystemExit(f"✗ split={a.split} 没有帧")
    print(f"帧 {len(idx):,}（split={a.split}）  X={X.shape}  模型={a.model}")

    cards = load_card_categories()
    policy = build_policy(a.model, a.bundle, a.device)
    policy.eval()

    per = collections.defaultdict(lambda: [0, 0])          # 类别 → [正确, 总数]
    pred_of = collections.defaultdict(collections.Counter)  # 类别 → 预测类别分布
    per_ot = collections.defaultdict(lambda: [0, 0])        # 真实 option type → 正确/总数

    with torch.no_grad():
        for i in range(0, len(idx), a.bs):
            sel = idx[i:i + a.bs]
            rows = np.asarray(X[sel], dtype=np.float32)
            xb = torch.from_numpy(rows).to(a.device)
            logits = policy.forward_eval(xb)[0]
            pred = logits.argmax(dim=1).cpu().numpy()
            ys = y[sel].astype(np.int64)
            base = OPT_OFF
            for r, (yy, pp) in enumerate(zip(ys, pred)):
                row = rows[r]
                ot_t = int(row[base + int(yy) * OPT_INT + 0])
                card_t = int(row[base + int(yy) * OPT_INT + 3])
                ot_p = int(row[base + int(pp) * OPT_INT + 0])
                card_p = int(row[base + int(pp) * OPT_INT + 3])
                ct, _ = classify(ot_t, card_t, cards)
                cp, _ = classify(ot_p, card_p, cards)
                per[ct][1] += 1
                per_ot[OT_NAME.get(ot_t, str(ot_t))][1] += 1
                if int(yy) == int(pp):
                    per[ct][0] += 1
                    per_ot[OT_NAME.get(ot_t, str(ot_t))][0] += 1
                pred_of[ct][cp] += 1
            if (i // a.bs) % 20 == 0:
                done = min(i + a.bs, len(idx))
                print(f"  {done:,}/{len(idx):,}", end="\r", flush=True)
    print()

    tot_c = sum(v[0] for v in per.values())
    tot_n = sum(v[1] for v in per.values())
    lines = ["cat\tn\tcorrect\tacc\ttop_pred", ]
    print(f"\n总一致率 {tot_c / tot_n:.4f}（{tot_c}/{tot_n}）\n")
    print(f"{'类别':10s}{'帧数':>8}{'占比':>7}{'准确率':>9}   主要被错判成")
    for c in CATS:
        if c not in per:
            continue
        k, n = per[c]
        wrong = pred_of[c].most_common(3)
        wrong = [(x, v) for x, v in wrong if x != c][:2]
        tag = " ".join(f"{x}:{v / n:.0%}" for x, v in wrong) or "—"
        print(f"{c:10s}{n:>8}{n / tot_n:>7.1%}{k / n:>9.3f}   {tag}")
        lines.append(f"{c}\t{n}\t{k}\t{k / n:.4f}\t{tag}")
    print(f"\n{'动作类型(option type)':22s}{'帧数':>8}{'准确率':>9}")
    for ot, (k, n) in sorted(per_ot.items(), key=lambda kv: -kv[1][1]):
        print(f"{ot:22s}{n:>8}{k / n:>9.3f}")
        lines.append(f"ot:{ot}\t{n}\t{k}\t{k / n:.4f}\t")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write("\n".join(lines) + "\n")
        print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
