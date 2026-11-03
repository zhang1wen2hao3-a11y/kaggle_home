#!/usr/bin/env python3
"""**评测闸门**：把一个候选 agent 放到"按天梯分档加权的卡组场"上量分。

## 为什么需要它（而不是直接看训练日志）

训练器日志里的回报是**对随机初始化的自己**的胜率，跟比赛分数没有稳定关系。
比赛口径是 `cabt_bo3` 对真实天梯场，所以"变没变好"只能用同一口径量：

1. 对手必须是**按分档权重抽出来的真实卡组**（`decks/weights_*.txt`）；
2. 必须是**真 Bo3**（换边、第 2 局选先后手、每场 deck-request 重置、每局满池）；
3. 必须给 **CI**，并且提前登记"怎样才算赢"——否则每次都能挑出一个好看的数字。

## 与 `tools/bo3_eval.py` 的分工

`bo3_eval.py` 是**成对**评测（一个 A 对一个 B）。本文件负责"按权重抽一批对手、
把场次按权重分配、再跨对手聚合"。底层还是调 `bo3_eval.py`，不另起一套引擎。

## 用法

    # 当前部署端口在"我们这一档"的基线
    python3 tools/eval_gate.py --agent build/port_plain --band ours --matches 200

    # 训练出来的 ckpt 跟基线比
    python3 tools/eval_gate.py --agent /tmp/staged/myckpt --band ours --matches 200

## 判定口径（预先登记，别事后改）

- **绝对门槛**：`score` 的 CI 下界 > 基线点估计 ⇒ 才算赢；
- **两档都要看**：`ours`（现在）与 `top`（上 1200 之后）。只在一档变好不算
  （`9d6ea740` 就是"我们档 0.880 / 顶档 0.267"的反面教材）；
- **`errors` 必须为 0**：出现兜底/异常说明 agent 状态已坏，那个数字没意义；
- 单次比较的噪声：n=200 时 CI 半宽约 ±0.06，所以 **CI 重叠就别宣称胜负**。
"""
import argparse
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

BANDS = ("ours", "mid", "lbfeed", "top", "real")
# 「real」= 由 tools/scrape_episodes.py 从我们自己的真实天梯对局统计出来的
# 对手分布（decks/weights_real.txt）。训练后的验收应该用它，而不是旧语料权重——
# 2026-10-27 实测：ours 档里 Ogerpon 占 57.5%，真实只有 21.2%。

# ⚠️ 汇总口径**不要复刻**：直接取出厂实现。`bo3_eval.py` 模块级只有 stdlib 导入，
#    可以安全 import；两边各写一份 summarize 迟早会漂移成两个不同的"得分"。
from bo3_eval import summarize, wilson            # noqa: E402


def load_band_weights(path, pool_names):
    """按**训练器同口径**读权重：未列出的名字取 default，然后归一化。"""
    ver, dflt, per = 0, 1.0, {}
    for ln in open(path, encoding="utf-8"):
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        k, v = ln.split()
        if k == "version":
            ver = int(v)
        elif k == "default":
            dflt = float(v)
        else:
            per[k] = float(v)
    if ver != 3:
        raise SystemExit(f"{path}: version={ver}，训练器只认 3")
    unknown = [n for n in per if n not in set(pool_names)]
    if unknown:
        # 训练器对 unknown 的处理是整份作废，所以这里也直接失败而不是忽略
        raise SystemExit(f"{path}: {len(unknown)} 个名字不在池里，例 {unknown[0]}")
    raw = {n: float(per.get(n, dflt)) for n in pool_names}
    tot = sum(raw.values())
    return {n: v / tot for n, v in raw.items()}


def sample_opponents(dist, k, rng):
    """按权重**不放回**抽 k 副（权重不可加式抽样，简单可靠）。"""
    items = [(n, w) for n, w in dist.items() if w > 0]
    picked = []
    for _ in range(min(k, len(items))):
        tot = sum(w for _, w in items)
        r = rng.random() * tot
        acc = 0.0
        for i, (n, w) in enumerate(items):
            acc += w
            if r <= acc:
                picked.append(items.pop(i))
                break
        else:
            picked.append(items.pop())
    return picked


def allocate(picks, matches, rng):
    """把场次按权重分配，单位 = 2 场（Bo3 需要偶数）。

    每个被抽中的对手至少 2 场，否则它在聚合里没有发言权却被计入了抽样。
    用最大余数法保证总数精确。
    """
    units = matches // 2
    k = len(picks)
    if units < 1:
        raise SystemExit("--matches 至少要 2")
    tot_w = sum(w for _, w in picks)
    raw = [w / tot_w * units for _n, w in picks]
    # ⚠️ 不要给"每副至少 2 场"的下限：这一档有 161 副是 EPS 近零权重（实测没出现过），
    #    给它们各塞 2 场会凭空把分布拉平，算出来的分数就不再是"这一档的分数"。
    #    分到 0 场的卡组直接丢掉，并在下面报出"覆盖了多少权重质量"。
    base = [int(math.floor(x)) for x in raw]
    diff = units - sum(base)
    order = sorted(range(k), key=lambda i: -(raw[i] - math.floor(raw[i])))
    i = 0
    while diff > 0:
        base[order[i % k]] += 1
        diff -= 1
        i += 1
        if i > 100000:
            raise SystemExit("场次分配不收敛")
    out = [(picks[j][0], picks[j][1], base[j] * 2) for j in range(k) if base[j] > 0]
    dropped = [picks[j][0] for j in range(k) if base[j] == 0]
    return out, dropped


def stage(bundle, outdir, ids):
    """用软链 + 单独 deck.csv 造一个"端口 + 指定卡组"的目录。

    `main.py` 按 `__file__` 目录找 deck.csv，所以软链其余一切、只写一份真的
    deck.csv 就够了；复制整包的话每副要 20 MB。
    """
    os.makedirs(outdir, exist_ok=True)
    for e in os.listdir(bundle):
        if e == "deck.csv":
            continue
        dst = os.path.join(outdir, e)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(os.path.join(bundle, e)), dst)
    with open(os.path.join(outdir, "deck.csv"), "w") as f:
        f.write("\n".join(str(x) for x in ids) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="被测 agent（目录或 .tar.gz）")
    ap.add_argument("--pool", default="decks/pool_field")
    ap.add_argument("--band", default="ours", choices=BANDS)
    ap.add_argument("--bundle", default="build/port_plain",
                    help="对手侧的端口包（软链底座）")
    ap.add_argument("--matches", type=int, default=200, help="总 Bo3 场数")
    ap.add_argument("--opponents", type=int, default=30, help="抽多少个不同卡组")
    ap.add_argument("--workers", type=int, default=3, help="并行面板数（≤10，CPU 敏感）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--json", default=None, help="把汇总写成 JSON")
    args = ap.parse_args()

    pool_dir = os.path.join(ROOT, args.pool)
    deck_files = sorted(glob.glob(os.path.join(pool_dir, "*.csv")))
    names = [os.path.basename(p)[:-4] for p in deck_files]
    if not names:
        raise SystemExit(f"{pool_dir} 里没有卡组")

    wpath = os.path.join(ROOT, "decks", f"weights_{args.band}.txt")
    dist = load_band_weights(wpath, names)
    rng = random.Random(args.seed)

    picks = sample_opponents(dist, args.opponents, rng)
    alloc, dropped = allocate(picks, args.matches, rng)
    kept = sum(w for _, w in dist.items() if w > 0)
    covered = sum(w for _, w, _ in alloc)
    print(f"档位 {args.band}：池 {len(names)} 副，其中 {sum(1 for w in dist.values() if w > 0)} 副有权重"
          f"（权重合计 {kept:.3f}）")
    print(f"抽 {len(picks)} 副，其中 {len(alloc)} 副分到场次，共 {sum(m for _, _, m in alloc)} 场 Bo3"
          f"（场次 ∝ 权重）")
    print(f"这些卡组覆盖该档权重质量 {covered:.1%}"
          + (f"；{len(dropped)} 副近零权重卡组分到 0 场已丢弃" if dropped else ""))
    print(f"{'卡组':<40} {'权重':>8} {'场数':>6}")
    for n, w, m in sorted(alloc, key=lambda x: -x[1])[:12]:
        print(f"  {n:<38} {w:8.3%} {m:6d}")
    if len(alloc) > 12:
        print(f"  …另 {len(alloc) - 12} 副")

    logdir = os.path.abspath(args.logdir or os.path.join(ROOT, "logs", f"gate_{args.band}"))
    os.makedirs(logdir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="gate_")
    bundle = os.path.join(ROOT, args.bundle)
    if not os.path.isdir(bundle):
        raise SystemExit(f"--bundle {bundle} 不存在")

    all_records, per_deck = [], {}
    try:
        for idx, (name, w, m) in enumerate(sorted(alloc, key=lambda x: -x[1])):
            ids = [int(x) for x in open(os.path.join(pool_dir, name + ".csv")).read().split()]
            if len(ids) != 60:
                print(f"⚠️ 跳过 {name}（{len(ids)} 张）")
                continue
            d = os.path.join(work, name)
            stage(bundle, d, ids)
            sub = os.path.join(logdir, name)
            os.makedirs(sub, exist_ok=True)
            cmd = [sys.executable, os.path.join(ROOT, "tools", "bo3_eval.py"),
                   "--a", args.agent, "--b", d, "--matches", str(m),
                   "--workers", str(args.workers), "--logdir", sub]
            print(f"\n[{idx + 1}/{len(alloc)}] {name}  {m} 场 …", flush=True)
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  ✗ bo3_eval 失败（rc={r.returncode}）\n{r.stderr[-800:]}")
                continue
            rec_p = os.path.join(sub, "last_run.jsonl")
            if not os.path.exists(rec_p):
                print("  ✗ 没有 last_run.jsonl")
                continue
            recs = [json.loads(line) for line in open(rec_p) if line.strip()]
            if not recs:
                # 子进程崩了但父进程 rc 仍是 0 —— 不打出 stderr 就会变成"静默 0 场"
                print(f"  ✗ 0 场记录；bo3_eval 输出尾部：\n{(r.stdout or '')[-400:]}"
                      f"\n{(r.stderr or '')[-1200:]}")
                continue
            for rec in recs:
                rec["_deck"] = name
                rec["_w"] = w
            all_records += recs
            per_deck[name] = recs
            s = summarize(recs)
            print(f"  score {s['score']:.3f}  n={s['matches']}  err={s['errors']}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if not all_records:
        raise SystemExit("一场都没跑成")

    s = summarize(all_records)
    print(f"\n{'=' * 64}\n{args.band} 档加权 Bo3 得分率：{s['score']:.4f}"
          f"  95% CI [{s['ci_low']:.4f}, {s['ci_high']:.4f}]")
    print(f"  场数 {s['matches']}  胜 {s['wins']}  平 {s['draws']}"
          f"  兜底/异常 {s['errors']}  不同卡组 {len(per_deck)}")
    if s["errors"]:
        print("  ⚠️ 有兜底/异常：这个数字不可信，先查 agent 状态")

    # 按**卡组等权**的参照：判断"这个分是不是几副高权重卡组撑起来的"。
    # 记录级平均 = 权重加权（因为场次 ∝ 权重）；卡组等权则是另一种口径。
    eq = sum(summarize(r)["score"] for r in per_deck.values()) / max(len(per_deck), 1)
    print(f"  参照（按已抽卡组等权）：{eq:.4f}")

    top = sorted(per_deck.items(), key=lambda kv: -len(kv[1]))[:8]
    print("\n分卡组（按场数）：")
    for n, recs in top:
        t = summarize(recs)
        print(f"  {n:<38} n={t['matches']:>3}  score {t['score']:.3f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"band": args.band, "agent": args.agent, **s,
                       "per_deck": {n: summarize(r) for n, r in per_deck.items()}},
                      f, ensure_ascii=False, indent=1)
        print(f"\n汇总已写 {args.json}")
    print(f"逐场记录在 {logdir}/<deck>/last_run.jsonl")


if __name__ == "__main__":
    main()
