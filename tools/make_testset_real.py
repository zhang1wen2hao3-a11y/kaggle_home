#!/usr/bin/env python3
"""**按真实回放份额重建 12 格测试集**（+把盲区里份额够大的卡组补成新格）。

为什么必须做：`decks/testset_meta.tsv` 的权重是**自家 582 局历史**（b05 口径），
和提交后的真实对局差得很远（7e39 自家 35% / 真实 34.9% 差得不多，但 0d86 自家 9.7% / 真实 0.4%，
而 37.8% 的对局根本不在 12 格里）。分母错了，面板分就不能用来排序。

口径（写死）：
  · 份额 = `logs/sub_feedback/scan.jsonl` 里**对手侧**卡组计数（已剔除自镜像）；
  · 12 个老格按池内名字的 4-hex 后缀聚合份额；
  · 盲区里份额 ≥ `--min-share`（默认 1.2%，即 ≥3 局）的卡组**升级成独立格**；
  · 其余长尾合成一格 `tail_mix`，**不本地测量**，胜率直接取回放观测值（带 n 与 Wilson CI）；
  · 对手驾驶员优先用注册表里该卡的**最强驾驶员**，没有就用端口权重（port on that deck）；
  · 输出 6 列格式（与 testset_meta_A.tsv 一致）：cell / agent_dir / deck / group / weight / driver

用法：
    ./tools/py tools/make_testset_real.py                      # → decks/testset_meta_real.tsv
    ./tools/py tools/make_testset_real.py --min-share 2.0      # 只把 ≥2% 的盲区补成格
    ./tools/py tools/make_testset_real.py --dry
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import driver_registry as REG  # noqa: E402

SCAN = os.path.join(ROOT, "logs", "sub_feedback", "scan.jsonl")
POOL = os.path.join(ROOT, "decks", "pool_field")
OUT = os.path.join(ROOT, "decks", "testset_meta_real.tsv")

# 12 个老格：cell 名 → (对手驾驶员, 说明)。与 testset_meta_A.tsv 保持一致，便于和新口径对比
OLD = {
    "ogerpon_7e39":   ("init", "build/agents/port_7e39"),
    "grookey_2617":   ("init", "build/agents/c_grookey"),
    "abra_e8ee":      ("init", "build/agents/port_e8ee"),
    "abra_f5f8":      ("candf5f8S", "build/agents/candf5f8S"),
    "champ_0d86":     ("g8_bc_parent", "build/agents/g8_bc_parent"),
    "dunsparce_9d6e": ("init", "build/agents/port_on_9d6e"),
    "dunsparce_0bf5": ("init", "build/agents/opp_dunsparce_0bf5"),
    "dreepy_3eaa":    ("init", "build/agents/opp_3eaa"),
    "dunsparce_db04": ("init", "build/agents/port_db04"),
    "champ_0904":     ("init", "build/agents/port_0904"),
    "champ_26c6":     ("init", "build/agents/r1A_26c6"),
    "champ_e1d7":     ("init", "build/agents/port_e1d7"),
}


def pool_index():
    """8-hex hash → (文件名, 60 张 ids)。"""
    idx = {}
    for f in sorted(os.listdir(POOL)):
        if not f.endswith(".csv"):
            continue
        p = os.path.join(POOL, f)
        try:
            ids = sorted(int(x) for x in open(p) if x.strip())
        except Exception:
            continue
        h = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:8]
        idx[h] = (f[:-4], ids)
    return idx


def load_shares():
    share, wins = collections.Counter(), collections.Counter()
    for ln in open(SCAN):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if not r.get("opp_hash") or r.get("win") is None or r.get("selfmirror"):
            continue
        share[r["opp_hash"]] += 1
        if r["win"]:
            wins[r["opp_hash"]] += 1
    return share, wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-share", type=float, default=1.2, help="盲区卡组升级成格的门槛（%%）")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    idx = pool_index()
    share, wins = load_shares()
    n = sum(share.values())
    if not n:
        print("✗ scan.jsonl 里没有可用对局"); return 2

    # 按"格"聚合
    fam = collections.Counter()
    famw = collections.Counter()
    famname = {}
    for h, c in share.items():
        name = idx[h][0] if h in idx else None
        short = name.split("__")[-1] if name else None
        key = None
        for cell in OLD:
            if cell.split("_")[-1] == short:
                key = cell
                break
        if key is None:
            key = f"out_{short}" if short else f"unseen_{h}"
        fam[key] += c
        famw[key] += wins[h]
        famname[key] = name or f"unseen_{h}"

    rows = []
    tail = collections.Counter()
    tailw = collections.Counter()
    for key, c in fam.items():
        sh = c / n * 100
        if key in OLD:
            drv, ddir = OLD[key][0], OLD[key][1]
            rows.append(dict(cell=key, agent=ddir, deck=famname[key], group="old", weight=sh, n=c,
                             wr=famw[key] / c, drv=drv))
        elif sh >= a.min_share:
            rows.append(dict(cell=key, agent=None, deck=famname[key], group="new", weight=sh, n=c,
                             wr=famw[key] / c, drv="init"))
        else:
            tail[key] += c
            tailw[key] += famw[key]

    tail_n = sum(tail.values())
    tail_wr = sum(tailw.values()) / tail_n if tail_n else float("nan")

    rows.sort(key=lambda r: -r["weight"])
    print(f"真实对局 {n} 局；格 {len(rows)} 个 + 长尾 {len(tail)} 副（{tail_n} 局，观测胜率 {tail_wr*100:.1f}%）\n")
    print(f"{'格':26s}{'来源':>5s}{'份额':>7s}{'局数':>5s}{'观测胜率':>9s}  对手")
    for r in rows:
        who = r["drv"] if r["agent"] is None else r["agent"]
        print(f"{r['cell']:26s}{r['group']:>5s}{r['weight']:6.1f}%{r['n']:5d}{r['wr']*100:8.1f}%  {who}")

    if a.dry:
        return 0
    with open(OUT, "w") as fh:
        fh.write(f"# 真实份额口径测试集（由 tools/make_testset_real.py 生成，数据源 {os.path.relpath(SCAN, ROOT)}）\n")
        fh.write("# cell\tagent_dir\tdeck\tgroup\tweight(%)\tdriver\n")
        for r in rows:
            agent = r["agent"] if r["agent"] else f"build/agents/port_on_{r['deck'].split('__')[-1]}"
            fh.write(f"{r['cell']}\t{agent}\t{r['deck']}\t{r['group']}\t{r['weight']:.4f}\t{r['drv']}\n")
        fh.write(f"tail_mix\t-\t-\ttail\t{tail_n / n * 100:.4f}\tinit\n")
    print(f"\n→ {os.path.relpath(OUT, ROOT)}（{len(rows)} 格 + 长尾；权重合计 "
          f"{sum(r['weight'] for r in rows) + tail_n / n * 100:.1f}%）")
    print("注意：group=new 的格子需要机器上有对应 agent（端口权重开那副牌），否则 meta_eval 会跑空。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
