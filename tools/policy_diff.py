#!/usr/bin/env python3
"""逐决策对比两个策略在**真实回放观测**上选了什么，并按选项类型聚合分歧。

## 为什么需要它（GPU-free 的唯一诊断手段）

判断"训练有没有把策略搞坏"，本来要靠 GPU 上的 A/B 或闸门。但在没有 GPU 的时候，
仍然可以做一件信息量很大的事：把两个 agent 喂**同一批真实对局观测**，逐帧比较选择。
端口推理是纯 CPU 的（`main.py` 用 `device="cpu"`），所以本工具完全不需要 GPU。

它能回答的问题：
  · 分歧率有多高（`check_port_fidelity.py` 已给出总一致率）
  · **分歧集中在哪类决策**——是整体偏移，还是"某类动作被系统性地改掉了"
  · 训练后偏爱/回避了哪些选项类型

## 与 `check_port_fidelity.py` 的分工

后者比的是"agent vs 回放里记录的动作"（判线上等价）。本工具比的是
"agent A vs agent B 在同一帧上的选择"，并做类型级混淆统计。

## 用法

    ./run.sh tools/policy_diff.py --a build/port_plain --b /tmp/staged/f1_og --episodes 6
    ./run.sh tools/policy_diff.py --a build/port_plain --b /tmp/staged/f1_og --dir epsoide_port
"""
import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from check_port_fidelity import load_agent, prepare, OUR_NAME  # noqa: E402
from framework.replay import iter_decisions, our_index           # noqa: E402


def sig(opts, idx):
    """选项 → 语义签名。type 是动作类别；再带上 area/index/attackId 以区分同类的不同目标。"""
    if not idx or not (0 <= idx[0] < len(opts)):
        return None
    o = opts[idx[0]]
    return (o.get("type"), o.get("attackId"), o.get("area"), o.get("inPlayArea"))


def tname(t):
    return f"type{t}" if t is not None else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="build/port_plain", help="基线策略（通常是线上那份）")
    ap.add_argument("--b", required=True, help="候选策略")
    ap.add_argument("--dir", default="epsoide_port")
    ap.add_argument("--episodes", type=int, default=6)
    a = ap.parse_args()

    fa = load_agent(prepare(a.a))
    fb = load_agent(prepare(a.b))
    files = sorted(glob.glob(os.path.join(ROOT, a.dir, "*.json")))[: a.episodes]
    print(f"A = {a.a}\nB = {a.b}\n回放 {len(files)} 局（{a.dir}）\n")

    stat = collections.Counter()
    conf = collections.Counter()
    for f in files:
        d = json.load(open(f))
        me = our_index(d, OUR_NAME)
        if me is None:
            continue
        for t, j, obs, _act in iter_decisions(d.get("steps") or []):
            if j != me:
                continue
            call = dict(obs)
            call.setdefault("logs", [])
            call.setdefault("remainingOverageTime", 600.0)
            call["player"] = me
            call["step"] = t
            try:
                ga = list(fa(dict(call)) or [])
                gb = list(fb(dict(call)) or [])
            except Exception as e:
                stat[f"异常 {type(e).__name__}"] += 1
                continue
            stat["决策"] += 1
            if ga == gb:
                stat["完全相同"] += 1
                continue
            opts = (obs.get("select") or {}).get("option") or []
            sa, sb = sig(opts, ga), sig(opts, gb)
            if sa is not None and sa == sb:
                stat["语义相同(索引不同)"] += 1
                continue
            stat["实质分歧"] += 1
            conf[(tname(sa[0] if sa else None), tname(sb[0] if sb else None))] += 1

    tot = max(stat["决策"], 1)
    print(f"决策帧数 {stat['决策']}")
    for k in ("完全相同", "语义相同(索引不同)", "实质分歧"):
        print(f"  {k:<18} {stat[k]:6d}  ({stat[k] / tot:6.2%})")
    if stat["实质分歧"] == 0:
        print("\n两个策略在真实观测上行为一致。")
        return
    print(f"\n实质分歧按 (A 的选项类型 → B 的选项类型) 聚合，前 12 组：")
    print(f"  {'A type':>10} {'B type':>10} {'帧数':>7} {'占比':>8}")
    for (ta, tb), n in conf.most_common(12):
        print(f"  {ta:>10} {tb:>10} {n:7d} {n / stat['实质分歧']:8.1%}")
    print("\n读法：")
    print("  · 若分歧集中在少数几个 type→type 组，说明是**某类动作被系统性改掉**")
    print("    （例如总是把『攻击』换成『附加能量』，或 STOP 时机整体偏移）。")
    print("  · 若分歧均匀散布在所有组，说明是**整体策略漂移**，不是单点 bug。")


if __name__ == "__main__":
    main()
