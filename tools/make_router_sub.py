#!/usr/bin/env python3
"""生成**多模型路由提交件**：一个包里带泛用模型 + N 个 specialist，由规则模型在线选择。

和普通提交件的区别只有一个：`models/` 目录里放多个 `.pt`，`routes.json` 说明
「哪个原型族用哪个模型」，`main.py` 换成 `build/router_main.py`（自包含，内联分类器）。

**硬约束**：路由只能在同一副牌内部切换。包的 `deck.csv` 是唯一的，所以所有模型必须
按这副牌训练（我们的 `s2`/`s3` 都是 db04；`s1`/`s4` 学的是别的牌，**不能**塞进同一个包）。

    # 看有哪些可用的 specialist，以及它们的对位优势
    ./tools/py tools/make_router_sub.py list

    # 打包：泛用 = g6_db04，专家 = s3 打 ogerpon、s2 打 dunsparce 族
    ./tools/py tools/make_router_sub.py build --name router_db04 \
        --general g6_db04 --deck dunsparce_abra__db04 \
        --route teal_mask_ogerpon_ex_applin=s3_db04_vs7e39 \
        --route dunsparce_abra=s2_db04_vs0bf5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAIN = os.path.join(ROOT, "build", "port_plain")
ROUTER_MAIN = os.path.join(ROOT, "build", "router_main.py")
# 提交件骨架：这些是 main.py 与推理必需的
SKELETON = ["cg", "ptcg", "data", "EN_Card_Data.csv", "r2e_manifest.json"]

# 已知 specialist 的实测对位优势（真实权重面板，k 次均值）—— 只用于 list 提示，
# 选谁路由由 --route 明确指定，避免脚本替人做决定
KNOWN = {
    "s3_db04_vs7e39": dict(family="teal_mask_ogerpon_ex_applin", cell="vs 7e39", gain="0.790→0.870(on panel)"),
    "s2_db04_vs0bf5": dict(family="dunsparce_abra", cell="vs 0bf5", gain="0.805→0.830(on panel)"),
}


def families():
    """family → 卡 id 集合（用于路由识别），与 tools/router_core.load_families 同源。"""
    import glob

    fam = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "decks", "pool_field", "*.csv"))):
        base = os.path.basename(f)[:-4]
        if "__" not in base:
            continue
        name = base.split("__")[0]
        ids = set()
        for line in open(f):
            line = line.strip()
            if line:
                ids.add(int(line))
        fam.setdefault(name, set()).update(ids)
    return fam


def cmd_list(a):
    print("== 可用 specialist（面板实测对位收益）==")
    for k, v in KNOWN.items():
        ok = os.path.exists(os.path.join(ROOT, "models/ppo", k + ".pt"))
        print(f"  {k:20s} {'✓' if ok else '✗ 缺 .pt'}  family={v['family']:34s} {v['cell']:10s} {v['gain']}")
    print("\n== 池内原型族（前 10，供 --route 用）==")
    fam = families()
    for name in sorted(fam, key=lambda x: -len(fam[x]))[:10]:
        print(f"  {name:40s} {len(fam[name])} 种卡")


def cmd_build(a):
    stage = os.path.join(ROOT, "dist", f"router_{a.name}")
    if os.path.exists(stage):
        if not a.force:
            raise SystemExit(f"✗ {stage} 已存在（--force 覆盖）")
        shutil.rmtree(stage)
    os.makedirs(os.path.join(stage, "models"), exist_ok=True)

    # ① 骨架（不含 port_plain 自带的 model.pt，避免多塞 20 MB）
    for item in SKELETON:
        src = os.path.join(PLAIN, item)
        if not os.path.exists(src):
            raise SystemExit(f"✗ 骨架缺 {item}")
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(stage, item), symlinks=True)
        else:
            shutil.copy2(src, os.path.join(stage, item))
    # cg/ptcg 目录里可能有大的 .so，保留；但去掉 port_plain 自己的 model.pt
    for junk in ("model.pt", "deck.csv"):
        p = os.path.join(stage, junk)
        if os.path.exists(p):
            os.remove(p)

    # ② 模型：general 排 0，其余按 --route 顺序排
    route_pairs = [r.split("=", 1) for r in a.route]      # "--route FAMILY=MODEL"
    names = [a.general] + [v for _, v in route_pairs]
    seen = []
    for n in names:
        if n not in seen:
            seen.append(n)
    idx = {}
    for i, n in enumerate(seen):
        src = os.path.join(ROOT, "models/ppo", n + ".pt")
        if not os.path.exists(src):
            raise SystemExit(f"✗ 缺 {src}（先让远端训出来并拉回）")
        dst = os.path.join(stage, "models", f"{i:02d}_{n}.pt")
        shutil.copy2(src, dst)
        idx[n] = i
    # ③ 路由表
    route = {}
    fam = families()
    for family, model in route_pairs:
        if model not in idx:
            raise SystemExit(f"✗ route 里的 {model} 没在模型列表里")
        if family not in fam:
            raise SystemExit(f"✗ 池里没有原型族 {family}（用 list 子命令看）")
        route[family] = idx[model]
    payload = dict(
        models=[f"{i:02d}_{n}.pt" for i, n in enumerate(seen)],
        model_names=seen,
        deck=a.deck,
        route=route,
        # ★ 必须带**全部**族的卡表：分类器的 margin = 第一名与第二名覆盖率之差，
        #   只带路由用到的族时"第二名"不存在，margin 检查形同虚设，任何牌都可能被误判。
        #   全部 40 个族 × 几十张 id，只有几 KB，没有省的必要。
        family_cards={f: sorted(c) for f, c in fam.items()},
        note="family -> 模型序号；未命中一律回退到 0（泛用）",
    )
    json.dump(payload, open(os.path.join(stage, "routes.json"), "w"), indent=1)

    # ④ main.py + deck.csv
    shutil.copy2(ROUTER_MAIN, os.path.join(stage, "main.py"))
    deck_csv = os.path.join(ROOT, "decks", "pool_field", a.deck + ".csv")
    if not os.path.exists(deck_csv):
        raise SystemExit(f"✗ 没有卡组 {deck_csv}")
    shutil.copy2(deck_csv, os.path.join(stage, "deck.csv"))

    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(stage) for f in fs)
    print(f"✅ {stage}  {size/2**20:.1f} MB（上限 197.7 MiB）")
    print(f"   模型: {payload['models']}")
    print(f"   路由: {route}  （family → 模型序号）")
    print(f"   卡组: {a.deck}（{sum(1 for l in open(deck_csv) if l.strip())} 张）")
    if a.pack:
        out = f"ptcg_sub_{a.name}"
        r = subprocess.run(["bash", os.path.join(ROOT, "tools", "make_sub.sh"), a.name],
                           cwd=ROOT, capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
    return 0


def main():
    ap = argparse.ArgumentParser(description="打包多模型路由提交件")
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("list")
    l.set_defaults(fn=cmd_list)
    b = sub.add_parser("build")
    b.add_argument("--name", required=True, help="提交件名（build/agents/<name>）")
    b.add_argument("--general", default="g6_db04", help="泛用/兜底模型")
    b.add_argument("--deck", required=True, help="唯一的卡组（pool_field 名字）")
    b.add_argument("--route", action="append", default=[], metavar="FAMILY=MODEL",
                   help="原型族 → 模型，可重复")
    b.add_argument("--pack", action="store_true", help="顺手打成 dist/ptcg_sub_<name>.tar.gz")
    b.add_argument("--force", action="store_true")
    b.set_defaults(fn=cmd_build)
    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
