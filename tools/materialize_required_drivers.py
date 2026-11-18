#!/usr/bin/env python3
"""**补齐"必须卡组"的驾驶员**：没有模型的卡组，用 BC 权重（有就优先）或端口 init 权重铺一个标准
agent 目录，并登记进 `tools/driver_registry.py`，这样母表/投资分表/测试集都能引用到它。

用户 2026-11-19 要求：
  "刷新脚本也需要支持这些必须卡组能出现在我们维护的 csv 文件里，初始如果没有模型，可以选择 BC 或者 init。"

规则（按顺序取第一个可用的）：
  1. **同名训练驾驶员**：`build/agents/*` 里已经有解开这 60 张卡的 agent（model 与端口不同）⇒ 不动它；
  2. **BC 权重**：某 archetype 有 BC 模型（`build/agents/bc_*` / `models/ppo/*bc*.pt`）⇒ 用它铺 `bc_<4hex>`；
  3. **init/端口权重**：`build/port_plain/model.pt`，铺 `init_<4hex>`。

产出：
  · `build/agents/{bc,init}_<4hex>/`（骨架软链 + model.pt + deck.csv）
  · `tools/driver_registry.py` 里新增登记（幂等：已存在就跳过）
  · 打印一张"谁用什么补的"清单

    ./tools/py tools/materialize_required_drivers.py --dry-run
    ./tools/py tools/materialize_required_drivers.py
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIST = os.path.join(ROOT, "docs", "必须卡组清单.csv")
REG = os.path.join(ROOT, "tools", "driver_registry.py")
POOL = os.path.join(ROOT, "decks", "pool_field")
PORT_MODEL = os.path.join(ROOT, "build", "port_plain", "model.pt")


def ids_of(p):
    try:
        return tuple(sorted(int(x) for x in open(p) if x.strip()))
    except (OSError, ValueError):
        return None


def norm_arch(s: str) -> str:
    return s.lower().replace("-", "_").replace("__", "_")


def bc_index():
    """archetype → BC 权重路径（从 build/agents/bc_* 与 models/ppo/*bc*.pt 两边拼）。"""
    out = {}
    for d in glob.glob(os.path.join(ROOT, "build", "agents", "*")):
        name = os.path.basename(d)
        m = os.path.join(d, "model.pt")
        if "bc" not in name.lower() or not os.path.exists(m):
            continue
        deck = os.path.join(d, "deck.csv")
        ids = ids_of(deck)
        if not ids:
            continue
        # 用这副 60 张卡去池子里反查原型
        for p in glob.glob(os.path.join(POOL, "*.csv")):
            if ids_of(p) == ids:
                out[norm_arch(os.path.basename(p).split("__")[0])] = m
                break
    for p in glob.glob(os.path.join(ROOT, "models", "ppo", "*bc*.pt")):
        tok = os.path.basename(p)[:-3].lower()
        for a in ("e1d7", "0d86", "26c6", "3eaa"):
            if a in tok:
                out.setdefault(a, p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(LIST, encoding="utf-8-sig")))
    BC = bc_index()
    plan = []
    for r in rows:
        if (r.get("已训练驾驶员") or "").strip():
            continue                                  # 已经有训练过的驾驶员，不动
        h4, name, arch = r["4hex"], r["卡组文件"], r["原型"]
        deck_csv = os.path.join(POOL, f"{name}.csv")
        if not os.path.exists(deck_csv):
            print(f"  ✗ 池子里找不到 {name}.csv，跳过 {h4}")
            continue
        bc = BC.get(norm_arch(arch))
        if bc:
            plan.append(dict(h4=h4, name=name, agent=f"bc_{h4}", model=bc, how="BC"))
        else:
            plan.append(dict(h4=h4, name=name, agent=f"init_{h4}", model=PORT_MODEL, how="init"))
    print(f"需要补的卡组 {len(plan)} 副：BC {sum(1 for p in plan if p['how']=='BC')} / init {sum(1 for p in plan if p['how']=='init')}")
    for p in plan:
        print(f"   {p['h4']}  {p['name']:44s} → {p['agent']:10s}（{p['how']}）")
    if a.dry_run:
        return 0

    reg_src = open(REG, encoding="utf-8").read()
    add = []
    for p in plan:
        dst = os.path.join(ROOT, "build", "agents", p["agent"])
        if not os.path.isdir(dst):
            subprocess.run(["bash", os.path.join(ROOT, "tools", "make_agent_dir.sh"), dst,
                            p["model"], os.path.join(POOL, f"{p['name']}.csv")],
                           check=True, capture_output=True, text=True)
            # ★ 这些补位 agent 用的都是同一份权重（端口/BC），用**硬链接**代替 20 MB 的拷贝
            try:
                mp = os.path.join(dst, "model.pt")
                os.remove(mp)
                os.link(p["model"], mp)
            except OSError:
                pass
        if f'"{p["agent"]}": dict(' not in reg_src:
            add.append(
                f'    "{p["agent"]}": dict(agent="{p["agent"]}", weights={p["model"].replace(ROOT + "/", "")!r}, remote=None,\n'
                f'{" " * (len(p["agent"]) + 14)}deck="{p["h4"]}", fidelity=None,\n'
                f'{" " * (len(p["agent"]) + 14)}note="必须卡组自动补位（{p["how"]}）；{p["name"]}"),\n'
            )
    if add:
        anchor = "DRIVERS = {\n"
        reg_src = reg_src.replace(anchor, anchor + "    # ── 必须卡组自动补位（tools/materialize_required_drivers.py 生成，勿手改）──\n" + "".join(add), 1)
        open(REG, "w", encoding="utf-8").write(reg_src)
        print(f"✓ 注册表新增 {len(add)} 条")
    else:
        print("· 注册表无新增（都登记过）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
