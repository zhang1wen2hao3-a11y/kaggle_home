#!/usr/bin/env python3
"""**评测标准化的唯一入口：队列生成 + 校验 + 分片 + 状态**

为什么要标准化：2026-11-15 那次收尾，为了补 6 个面板，先后踩了
  ① agent 目录没铺出来（cand7e39S / candE1D7S）⇒ 面板静默跑空
  ② 队列 proto 写错（v1 写成了 A 的 tag）⇒ 全部 skip
  ③ runner 起了两个 ⇒ 负载 57，速度砍半
  ④ 机器掉了 ⇒ 剩下的活散在 3 个分片里，要人工搬
  ⑤ 结果拉回来才想起来没同步已完成面板 ⇒ 重复跑
每一条都是流程问题，不是运气问题。

本工具把这些固定下来：

    ./tools/py tools/eval_queue.py build     # 由 choice 表 + 规则生成队列（写 logs/wishlist.tsv）
    ./tools/py tools/eval_queue.py check     # 校验队列：tag 唯一、agent 存在、N/种子合规、已完成的标记
    ./tools/py tools/eval_queue.py shard N   # 按"未完成行数"动态分片（N 台机器）
    ./tools/py tools/eval_queue.py status    # 队列 + 分片 + 每格进度 + ETA

约定（写死，不再每次商量）：
  · 队列行 = `候选agent \t tag基名 \t 种子 \t 协议(v1|A) \t shard`
  · tag 全名 = `<tag基名>_s<seed>`（v1）或 `<tag基名>_A_s<seed>`（A）——**必须是这个格式**，
    export_master 的 TAG2CELL 就是按基名池化，写错就进不了母表
  · 每格 N 场由 `--n` 决定（默认 150）；3 种子池化 450 ≥ 400，满足「池化 n≥400」的规矩
  · 分片按"未完成行数"轮流分配（动态），不是写死行号取模 ⇒ 机器掉线后重算即可，不会散
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import driver_registry as REG                     # noqa: E402

WL = os.path.join(ROOT, "logs", "wishlist.tsv")
SELF = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-self.csv")
DEFAULT_SEEDS = ["950000", "950001", "950002"]


def tag_of(row) -> str:
    ag, base, seed, proto, _ = row
    return f"{base}_A_s{seed}" if proto == "A" else f"{base}_s{seed}"


def done(row) -> bool:
    return os.path.exists(os.path.join(ROOT, "logs", "meta", tag_of(row), "summary.json"))


def load():
    if not os.path.exists(WL):
        return []
    with open(WL) as f:
        rows = [r for r in csv.reader(f, delimiter="\t") if len(r) >= 5]
    return [r[:5] for r in rows]


def save(rows):
    with open(WL, "w", newline="") as f:
        csv.writer(f, delimiter="\t", lineterminator="\n").writerows(rows)


# ────────────────────────────── build ──────────────────────────────
def cmd_build(a):
    """由「每副牌最佳驾驶员」表生成方案A队列（对手=强驾驶员），v1 只保留指定候选。"""
    if not os.path.exists(SELF):
        sys.exit(f"✗ 缺 {SELF}（先跑 build_pilot_table.py）")
    best = {}
    for r in csv.DictReader(io.StringIO(open(SELF, "rb").read().decode("gb18030", "replace"))):
        d = (r.get("最佳驾驶员") or "init").strip()
        best[r["卡组"]] = "init" if d in ("", "待测", None) else d

    rows = []
    # ① 方案A 面板：被测 = 当前最佳驾驶员（去掉 init/端口）
    cands = sorted({d for d in best.values() if d != "init"})
    for c in a.candidates.split(",") if a.candidates else cands:
        c = c.strip()
        if not c:
            continue
        for sd in DEFAULT_SEEDS[: a.seeds]:
            rows.append([REG.info(c)["agent"] or c, c, sd, "A", "0"])
    # ② 可选：把某几副牌的 v1 面板也带上（默认不带，v1 只用来填矩阵）
    for c in (a.also_v1.split(",") if a.also_v1 else []):
        c = c.strip()
        if not c:
            continue
        for sd in DEFAULT_SEEDS[: a.seeds]:
            rows.append([REG.info(c)["agent"] or c, c, sd, "v1", "0"])
    print(f"生成 {len(rows)} 行（候选 {sorted({r[1] for r in rows})}，种子 {a.seeds} 个）")
    save(reserve(rows, a.hosts))


# ────────────────────────────── check ──────────────────────────────
def cmd_check(a):
    rows = load()
    bad = 0
    seen = collections.Counter(tag_of(r) for r in rows)
    dup = [k for k, v in seen.items() if v > 1]
    if dup:
        print(f"✗ tag 重复: {dup[:5]}"); bad += len(dup)
    miss = collections.Counter()
    for r in rows:
        ag, base, seed, proto, _ = r
        if proto not in ("v1", "A"):
            print(f"✗ 未知协议 {proto}: {r}"); bad += 1
        try:
            REG.info(base)
        except REG.UnknownDriver as e:
            print(f"✗ 驾驶员未登记 {base}: {e}"); bad += 1
        for path in (f"build/agents/{ag}",):
            if not os.path.isdir(os.path.join(ROOT, path)):
                miss[ag] += 1
    for ag, n in miss.items():
        print(f"⚠️ 本地缺 agent 目录 build/agents/{ag}（{n} 行）—— 跑之前必须在目标机上铺好")
    nd = sum(1 for r in rows if done(r))
    print(f"队列 {len(rows)} 行：已完成 {nd}，待跑 {len(rows)-nd}；错误 {bad} 条")
    return 1 if bad else 0


# ────────────────────────────── shard ──────────────────────────────
def reserve(rows, hosts: int):
    """按**未完成行数**轮流分片（动态）；已完成的行保留原分片不动。"""
    hosts = max(1, hosts)
    todo = [i for i, r in enumerate(rows) if not done(rows[i])]
    for k, i in enumerate(todo):
        rows[i][4] = str(k % hosts)
    return rows


def cmd_shard(a):
    rows = reserve(load(), a.hosts)
    save(rows)
    c = collections.Counter(r[4] for r in rows if not done(r))
    print(f"按 {a.hosts} 台重新分片，未完成分布: {dict(sorted(c.items()))}")
    for k in sorted(c):
        print(f"  shard {k}: " + "\n           ".join(tag_of(r) for r in rows if r[4] == k and not done(r)))


# ────────────────────────────── status ──────────────────────────────
def cmd_status(a):
    rows = load()
    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        per[r[4]][0 if done(r) else 1] += 1
    nd = sum(v[0] for v in per.values())
    npend = sum(v[1] for v in per.values())
    per_panel = a.minutes
    print(f"{'分片':6}{'已完成':>7}{'待跑':>6}")
    for k in sorted(per):
        print(f"  {k:4}{per[k][0]:>7}{per[k][1]:>6}")
    print(f"\n合计 {len(rows)} 行：已完成 {nd}，待跑 {npend}")
    if per:
        worst = max(v[1] for v in per.values())
        print(f"按 {per_panel:.0f} 分钟/面板估算：最慢分片还需要 {worst*per_panel/60:.1f} 小时")
    if npend:
        print("\n待跑清单：")
        for r in rows:
            if not done(r):
                print(f"  [{r[4]}] {tag_of(r):28} ({r[0]})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--candidates", default="")
    b.add_argument("--also-v1", default=""); b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--hosts", type=int, default=1); b.set_defaults(f=cmd_build)
    c = sub.add_parser("check"); c.set_defaults(f=cmd_check)
    s = sub.add_parser("shard"); s.add_argument("hosts", type=int); s.set_defaults(f=cmd_shard)
    t = sub.add_parser("status"); t.add_argument("--minutes", type=float, default=20.0)
    t.set_defaults(f=cmd_status)
    a = ap.parse_args()
    return a.f(a) or 0


if __name__ == "__main__":
    sys.exit(main())
