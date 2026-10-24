#!/usr/bin/env python3
"""P0：**本地一致性校验** —— 我们本地跑的 agent 是不是和线上跑的是同一个行为？

方法与公开界（busyaprime）相同：把回放里**我们自己的真实观测**按时间顺序逐帧喂给本地 agent，
比对它给出的动作与回放里记录的动作是否一致。

为什么必须先做这一步：如果本地复现不出线上行为（例如权重加载失败、卡片表不一致、
`__file__` 回退路径不同、按局重置没触发），那么后面所有本地 Bo3 结论都不可信。

用法：
    ./run.sh tools/check_port_fidelity.py --agent dist/ptcg_public_port.tar.gz --dir epsoide_port

判据：**一致率 ≥ 99%** 才算本地等价；不一致的帧会打印出来（含选项数与动作），便于定位。
"""
import argparse
import collections
import glob
import json
import os
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from framework.replay import iter_decisions, our_index  # noqa: E402

OUR_NAME = "zhang1wen2hao3"


def prepare(path: str) -> str:
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if os.path.isdir(p):
        return p
    d = tempfile.mkdtemp(prefix="fidelity_agent_")
    with tarfile.open(p) as t:
        t.extractall(d)
    assert os.path.exists(os.path.join(d, "main.py")), "包里没有顶层 main.py"
    return d


def load_agent(folder: str):
    """按 Kaggle 的方式加载：exec、无 __file__、取最后一个 callable。"""
    os.chdir(folder)
    sys.path.insert(0, folder)
    ns = {}
    exec(compile(open("main.py", encoding="utf-8-sig").read(), "main.py", "exec"), ns)
    fn = [v for v in ns.values() if callable(v)][-1]
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="dist/ptcg_public_port.tar.gz")
    ap.add_argument("--dir", default="epsoide_port")
    ap.add_argument("--limit", type=int, default=0, help="只检查前 N 局（调试用）")
    args = ap.parse_args()

    folder = prepare(args.agent)
    fn = load_agent(folder)
    print(f"agent: {args.agent} → {folder}\n入口: {getattr(fn, '__name__', '?')}")

    files = sorted(glob.glob(os.path.join(ROOT, args.dir, "*.json")))
    if args.limit:
        files = files[: args.limit]
    print(f"回放: {len(files)} 局（{args.dir}）")

    stat = collections.Counter()
    mismatches = []
    for f in files:
        d = json.load(open(f))
        steps = d.get("steps") or []
        me = our_index(d, OUR_NAME)
        if me is None:
            stat["找不到我们的座位"] += 1
            continue
        # 按时间顺序重放：**必须整局顺序喂**，否则 agent 内部 tracker 状态不对
        for t, j, obs, act in iter_decisions(steps):
            if j != me:
                continue
            # 补齐 Kaggle 会提供、而回放里没有的字段
            call = dict(obs)
            call.setdefault("logs", [])
            call.setdefault("remainingOverageTime", 600.0)
            call["player"] = me
            call["step"] = t
            try:
                got = fn(call)
            except Exception as e:
                stat["本地抛异常"] += 1
                mismatches.append((os.path.basename(f), t, act, f"EXC {type(e).__name__}"))
                continue
            got = list(got) if got is not None else None
            stat["决策"] += 1
            if got == act:
                stat["完全一致"] += 1
                continue
            # 语义一致（同类型同卡，只是挑的副本不同）也算"行为等价"
            opts = obs["select"].get("option") or []

            def sig(a):
                if not a or not (0 <= a[0] < len(opts)):
                    return None
                o = opts[a[0]]
                return (o.get("type"), o.get("attackId"), o.get("index"), o.get("area"))

            if sig(got) is not None and sig(got) == sig(act):
                stat["语义一致(索引不同)"] += 1
            else:
                stat["不一致"] += 1
                if len(mismatches) < 25:
                    mismatches.append((os.path.basename(f), t, act, got))

    n = max(stat["决策"], 1)
    print(f"\n决策总数 {stat['决策']}")
    for k in ("完全一致", "语义一致(索引不同)", "不一致", "本地抛异常"):
        if stat[k]:
            print(f"  {k}: {stat[k]}  ({stat[k]/n:.2%})")
    agree = (stat["完全一致"] + stat["语义一致(索引不同)"]) / n
    print(f"\n**等价率 = {agree:.4%}**（判据 ≥99%）")
    if mismatches:
        print("\n不一致样例（file, 帧, 回放动作, 本地动作）：")
        for m in mismatches[:15]:
            print("  ", m)
    if agree < 0.99:
        print("\n⚠️ 本地行为与线上不等价 —— 先查：权重/卡表是否一致、__file__ 回退路径、按局重置是否触发。")


if __name__ == "__main__":
    main()
