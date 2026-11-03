#!/usr/bin/env python3
"""量化"提前攻击"假设：在「攻击与非攻击选项同时可用」的决策里，谁更容易选攻击。

## 为什么这个测量比"搜索假设"更硬

PTCG 里**攻击会结束回合**，所以攻击在动作序列中天然应该最后。若对手在还有发展空间时
极少攻击、而我们的策略经常攻击，那分歧的"内容"就是回合内的动作顺序——这可以由
一个推理期动作屏蔽直接修正，不需要训练、也不需要搜索。

## 口径

对每一帧决策：
  has_atk  = 存在 type==13 的选项（type 13 带 attackId ⇒ 攻击）
  has_other= 存在非攻击选项
只在 has_atk AND has_other 的帧上统计，比较"他选攻击的比例"与"我们选攻击的比例"。

用法：
    ./run.sh tools/measure_attack_haste.py --dir epsoide_dipam --agent-name "Dipam Chakraborty" --games 10
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

from framework.replay import iter_decisions, decks_of   # noqa: E402
from agentpool import get_agent                         # noqa: E402

ATK = 13


def classify(opt):
    return "atk" if opt and opt.get("type") == ATK else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="epsoide_dipam")
    ap.add_argument("--agent-name", required=True)
    ap.add_argument("--games", type=int, default=10)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(ROOT, a.dir, "*.json")))
    stat = collections.Counter()
    done = 0
    for fp in files:
        if done >= a.games:
            break
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        steps = d.get("steps") or []
        names = [x.get("Name", "") for x in ((d.get("info") or {}).get("Agents") or [])]
        if a.agent_name not in names:
            continue
        seat = names.index(a.agent_name)
        decks = decks_of(steps)
        if seat not in decks or len(decks[seat]) != 60:
            continue
        fn = get_agent(decks[seat])
        done += 1
        for t, j, obs, act in iter_decisions(steps):
            if j != seat:
                continue
            opts = (obs.get("select") or {}).get("option") or []
            if not act or act[0] >= len(opts):
                continue
            has_atk = any(o.get("type") == ATK for o in opts)
            has_other = any(o.get("type") != ATK for o in opts)
            if not (has_atk and has_other):
                if has_atk:
                    stat["只有攻击可选"] += 1
                continue
            stat["攻守皆可"] += 1
            his = classify(opts[act[0]])
            stat[f"他选_{'攻击' if his == 'atk' else '其他'}"] += 1
            call = dict(obs)
            call.setdefault("logs", [])
            call.setdefault("remainingOverageTime", 600.0)
            call["player"] = seat
            call["step"] = t
            try:
                got = list(fn(dict(call)) or [])
            except Exception:
                stat["我方异常"] += 1
                continue
            if got and got[0] < len(opts):
                ours = classify(opts[got[0]])
                stat[f"我们选_{'攻击' if ours == 'atk' else '其他'}"] += 1
                if his == "atk" and ours == "other":
                    stat["他攻击_我们发展"] += 1
                if his == "other" and ours == "atk":
                    stat["他发展_我们攻击"] += 1

    n = stat["攻守皆可"]
    if not n:
        sys.exit("没有可判定的帧")
    print(f"局数 {done}；「攻击与其他选项同时可用」的决策 {n} 帧\n")
    print(f"  他选攻击   {stat['他选_攻击']:5d}  ({stat['他选_攻击'] / n:6.2%})")
    print(f"  他选其他   {stat['他选_其他']:5d}  ({stat['他选_其他'] / n:6.2%})")
    m = stat["他选_攻击"] + stat["他选_其他"]
    print(f"  我们选攻击 {stat['我们选_攻击']:5d}  "
          f"({stat['我们选_攻击'] / max(m, 1):6.2%})")
    print(f"  我们选其他 {stat['我们选_其他']:5d}  "
          f"({stat['我们选_其他'] / max(m, 1):6.2%})")
    print()
    print(f"  他发展 / 我们攻击 : {stat['他发展_我们攻击']:5d} 帧")
    print(f"  他攻击 / 我们发展 : {stat['他攻击_我们发展']:5d} 帧")
    only = stat["只有攻击可选"]
    print(f"  （另有 {only} 帧只有攻击可选，未计入）")


if __name__ == "__main__":
    main()
