"""从数据反算实际探索率 —— 数据溯源工具。

数据文件里**没有记录** ε，所以用选项的 opt_t_* 独热特征还原 greedy 的选择，
再和数据里实际记录的 action 对比，反解 ε。用于验证"这批数据是在多大随机概率下跑的"。

推导：
    explorer.py 仅在 maxCount==1 且 minCount<=1 时随机化，且在全部 option 中均匀选，
    所以   P(action != greedy) = ε · (1 − 1/n)
    =>     ε = 实测不一致率 / E[1 − 1/n]

用法：
    ./run.sh measure_epsilon.py data/pool_explore_v3.jsonl

⚠️ 重要前提：gen_pool_data.py 里是 play_and_record(agent, agent, ...)，
   **双方共用同一个 agent**。所以 ε 是"每一方各自的偏离率"，
   双方合起来的混沌度远高于 ε 本身（见 README "探索率" 一节）。

explorer.py: 当 hi==1 且 lo<=1 时, 以概率 EPS 在全部 opt 中均匀随机选。
=> P(action != greedy) = EPS * (1 - 1/n)
"""
import json, sys
from collections import Counter

PRI = {"opt_t_PLAY":0,"opt_t_EVOLVE":1,"opt_t_ABILITY":2,"opt_t_ATTACH":3,
       "opt_t_RETREAT":4,"opt_t_DISCARD":5,"opt_t_ATTACK":6,"opt_t_END":9}

def analyze(path, limit=80000):
    n_tot = n_explorable = n_mismatch = 0
    n_multi = 0
    sum_q = 0.0   # sum of (1-1/n)
    dist = Counter()
    with open(path) as fh:
        for line in fh:
            n_tot += 1
            if n_tot > limit: break
            d = json.loads(line)
            opts = d["options"]
            n = len(opts)
            if n < 2: continue
            o0 = opts[0]
            hi, lo = o0.get("opt_max", 1), o0.get("opt_min", 1)
            if hi != 1 or lo > 1:
                n_multi += 1
                continue
            # greedy 会选哪个
            def key(i):
                for k,v in opts[i].items():
                    if k in PRI and v == 1.0: return PRI[k]
                return 99
            g = min(range(n), key=key)
            act = d["action"]
            n_explorable += 1
            sum_q += (1 - 1.0/n)
            if act != [g]:
                n_mismatch += 1
                dist[n] += 1
    q = sum_q / n_explorable if n_explorable else 0
    rate = n_mismatch / n_explorable if n_explorable else 0
    eps = rate / q if q else float('nan')
    return dict(n_tot=n_tot, n_explorable=n_explorable, n_multi=n_multi,
                mismatch=rate, q=q, eps=eps)

import sys as _s
import glob as _g
_DEFAULT = (_s.argv[1:] or sorted(_g.glob("data/*.jsonl")) or ["data/pool_rank_v1.jsonl"])
for f in _DEFAULT:
    r = analyze(f)
    print(f"{f}:")
    print(f"   采样 {r['n_tot']} 条 | 单选取可探索 {r['n_explorable']} | 多选取(全 greedy) {r['n_multi']}")
    print(f"   动作 != greedy 的比例 = {r['mismatch']:.4f}")
    print(f"   E[1-1/n] = {r['q']:.4f}   =>  反算 EPS = {r['eps']:.4f}")
    print()
