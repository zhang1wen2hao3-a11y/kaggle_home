#!/usr/bin/env python3
"""校验 `decks/weights_*.txt` 是否真的表达了实测的天梯分档分布。

为什么需要这个脚本：

1. `version 3` 权重文件是**相对倍率**（`name <weight>`，未列出的名字取 `default`），
   训练器 `native/train_native.py:178-180` 会用 `default` 填满所有未列出的卡组。
   所以"只列出现过的那 20 多副"会让剩下 150+ 副各拿 `default`，把真实集中度稀释掉，
   而这个错误**不会报错、只会静默改分布**。这里按训练器同一口径归一化后再比。
2. 权重文件里出现池子里没有的名字会让**整份请求被拒绝**（`weight_vector()` 返回
   unknown），所以名字必须逐一对上，这里也一并检查。
3. 实测份额本身有采样误差：78 局里 59% 的份额，标准误是 sqrt(p(1-p)/n)≈5.6pp。
   拿 ±5pp 这种绝对阈值卡它是不讲道理的，所以这里同时报标准误与 z。

用法：
    python3 tools/check_pool_weights.py                     # 用内置的实测参照
    python3 tools/check_pool_weights.py --band ours --top 59.0 --n 78
"""
import argparse
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 实测参照（来自 episode dump 分析，见 下一步.md §9.1 / 高分段架构清单.md）
# 键是卡组哈希前 8 位；值是该档内的份额（%）。
MEASURED = {
    # 独立样本：比权重文件的来源更大/更晚的一次定量（78 局 vs 77 次出镜）。
    # 镜像 59% 与 54.5% 差 5.3pp，而这本身有 5.6pp 的标准误 ⇒ 一致。
    "ours": {"7e398437": 59.0},
    # ⚠️ 回归基准：这几个数就是 `epsoide_dipam` 对手侧算出来的（EXP-0127），
    #    所以它**不能自证准确性**，只能当"管道没出错"的回归测试——
    #    正好能抓住 `default` 稀释和改名这类静默错误。真正的样本误差是 SE≈4.4pp。
    "top": {"e8eef8d8": 29.0, "a47a7971": 15.9, "26c6e843": 11.2,
            "f91d57e2": 9.3, "0d86f3a5": 7.5, "7e79ad8f": 5.6},
    "lbfeed": {"0d86f3a5": 23.9, "e8eef8d8": 15.8, "ea39ccb1": 14.7,
               "e1f9bfbd": 14.7, "26c6e843": 13.0},
    "mid": {"b702e251": 29.4},
}
# 各参照的样本量。份额的采样误差约 sqrt(p(1-p)/n)。
MEASURED_N = {"ours": 78, "top": 107, "lbfeed": 184, "mid": 316}
INDEPENDENT = {"ours"}          # 只有这一档是独立于权重来源的样本


def shipped_validate(pool_dir, bands, repo):
    """走**出厂代码路径**校验：`train_native.read_weight_request` +
    `weight_vector` + `pool_deck_names`。

    比本文件自己复刻的解析更可信——`pool_deck_names()` 是"deck id = 排序后位置"
    的唯一定义处（train_native.py:118-135），权重文件里的名字只有经过它才能映射到
    正确的 id。名字对不上时 `weight_vector()` 返回 unknown，而训练器对 unknown 的
    处理是**整份请求作废**。
    """
    import sys
    sys.path.insert(0, os.path.join(repo, "native"))
    import train_native as T                      # noqa: E402

    names, _rows = T.pool_deck_names(pool_dir)
    print(f"\n出厂解析器：pool={pool_dir} ⇒ {len(names)} 副，deck id 0..{len(names) - 1}")
    print(f"  前 3 个 id：{names[:3]}")
    bad = 0
    for band in bands:
        path = os.path.join(repo, "decks", f"weights_{band}.txt")
        if not os.path.exists(path):
            continue
        ver, dflt, per = T.read_weight_request(path)
        w, unknown = T.weight_vector(names, dflt, per)
        if ver != 3 or w is None or unknown:
            print(f"  [{band}] ✗ version={ver} unknown={len(unknown)}"
                  f"{(' 例: ' + unknown[0]) if unknown else ''}"
                  f" ⇒ 训练器会拒绝整份请求")
            bad += 1
            continue
        tot = sum(w)
        top_i = max(range(len(w)), key=lambda i: w[i])
        print(f"  [{band}] ✓ version=3 default={dflt} 列名 {len(per)} 个，"
              f"权重向量 {len(w)} 维，归一化后最大 {w[top_i] / tot:.1%}"
              f"（{names[top_i]}）")
    return bad


def load_weights(path):
    """按 `native/train_native.py:read_weight_request` 的语义读。"""
    ver, dflt, per = 0, 1.0, {}
    for ln in open(path, encoding="utf-8"):
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        key, val = ln.split()
        if key == "version":
            ver = int(val)
        elif key == "default":
            dflt = float(val)
        else:
            per[key] = float(val)
    return ver, dflt, per


def normalized(per, dflt, pool_names):
    """归一化到 sum=1，未列出的池内卡组取 dflt。这正是训练器看到的分布。"""
    raw = {n: float(per.get(n, dflt)) for n in pool_names}
    tot = sum(raw.values())
    if tot <= 0:
        raise SystemExit("权重全为 0：训练器会无法采样")
    return {n: v / tot for n, v in raw.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="decks/pool_field")
    ap.add_argument("--band", default=None, help="只查这一档（ours/mid/top）")
    ap.add_argument("--top", type=float, default=None,
                    help="覆盖内置参照：该档最大卡组的实测份额(%%)")
    ap.add_argument("--n", type=int, default=None, help="覆盖该档样本量")
    ap.add_argument("--repo", default=None,
                    help="训练仓库路径（如 port_repo）；给了就走出厂解析器做端到端校验")
    args = ap.parse_args()

    pool = os.path.join(ROOT, args.pool)
    names = sorted(os.path.basename(p)[:-4] for p in
                   __import__("glob").glob(os.path.join(pool, "*.csv")))
    if not names:
        raise SystemExit(f"{pool} 里没有卡组")
    bands_all = ["ours", "mid", "lbfeed", "top"]
    if args.repo:
        repo = os.path.abspath(os.path.join(ROOT, args.repo))
        # 出厂解析器要读仓库自己的 decks/ 目录，所以先把池与权重对齐过去
        bad = shipped_validate(pool, [args.band] if args.band else bands_all, repo)
        print(f"\n出厂路径判定：{'通过' if bad == 0 else f'{bad} 档不通过'}")
        raise SystemExit(0 if bad == 0 else 1)
    man_p = os.path.join(pool, "_manifest.json")
    h2n = {}
    if os.path.exists(man_p):
        for d in json.load(open(man_p, encoding="utf-8"))["decks"]:
            h2n[d["hash"]] = d["name"]
            h2n.setdefault(d["hash"][:8], d["name"])
    print(f"池 {args.pool}：{len(names)} 副")

    bands = [args.band] if args.band else ["ours", "mid", "lbfeed", "top"]
    bad = 0
    for band in bands:
        path = os.path.join(ROOT, "decks", f"weights_{band}.txt")
        if not os.path.exists(path):
            print(f"\n[{band}] 缺文件，跳过")
            continue
        ver, dflt, per = load_weights(path)
        unknown = sorted(n for n in per if n not in set(names))
        print(f"\n[{band}] {path}")
        print(f"  version={ver} default={dflt} 列名 {len(per)} 个 / 池 {len(names)} 副")
        if ver != 3:
            print("  ✗ version 不是 3")
            bad += 1
        if unknown:
            print(f"  ✗ {len(unknown)} 个名字不在池里（会让整份请求被拒绝），例：{unknown[0]}")
            bad += 1
        dist = normalized(per, dflt, names)
        ess = 1.0 / sum(v * v for v in dist.values())
        top_name = max(dist, key=dist.get)
        print(f"  归一化后：最大 {dist[top_name]:.1%} ({top_name})，ESS {ess:.1f}")

        meas = ({"?": args.top} if args.top is not None and args.band
                else MEASURED.get(band))
        n = args.n if args.n is not None else MEASURED_N.get(band, 600)
        if not meas:
            print("  （无实测参照，只做格式检查）")
            continue
        worst = 0.0
        print(f"  实测参照（n={n} 局）→ 权重：")
        for h, pct in sorted(meas.items(), key=lambda x: -x[1]):
            nm = h2n.get(h) or h2n.get(h[:8])
            got = dist.get(nm, 0.0) * 100 if nm else 0.0
            se = math.sqrt(max(pct, 1e-9) * (100 - pct) / max(n, 1))
            z = (got - pct) / se if se > 0 else 0.0
            worst = max(worst, abs(z))
            print(f"    {h} {pct:5.1f}% → {got:5.1f}%  ({got - pct:+5.1f}pp, "
                  f"SE {se:.1f}pp, z {z:+.2f})  {nm or '不在池中'}")
        kind = "独立样本" if band in INDEPENDENT else "回归基准（同源）"
        verdict = "通过" if worst <= 2.0 else "不通过"
        print(f"  判定（{kind}）：|z|max {worst:.2f} ≤ 2 ⇒ {verdict}")
        if worst > 2.0:
            bad += 1
    raise SystemExit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
