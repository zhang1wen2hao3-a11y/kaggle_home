#!/usr/bin/env python3
"""把**外部测回来的强度阶梯**写进池状态（`state.pfsp`），并重算 `w_i = p_i(1-p_i)`。

这是 `tools/make_pool_kit.py` 打包出去的评测包的**回收端**：
对方机器只跑评测，交回一个几 KB 的 `out/p_i.json`，由这个工具落库。

⚠️ 方向不能搞反（这是这一步唯一的坑）：
    * 评测包量到的 `p_i` = **该成员打赢学习者**的比例
    * `state.pfsp.p[aid]` 存的是 = **学习者打赢该成员**的比例（`run_league.stage_pfsp`
      里 `res[aid] = s["score"]`，而 `s` 是候选=学习者的得分率）
    ⇒ 落库必须用 `p_learner = 1 - p_i`。新旧两种 json（只带 p_i / 带 p_learner）都认。

用法：
    ./tools/py tools/pool_set_pi.py --json /path/to/out/p_i.json --dry-run
    ./tools/py tools/pool_set_pi.py --json /path/to/out/p_i.json
    ./tools/py tools/pool_set_pi.py --from-ladder logs/ladder   # 回收本机 pool_ladder.sh 的产出
    ./tools/py tools/pool_set_pi.py --show          # 只看当前阶梯
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ptcg_league"))

import league as L                                      # noqa: E402


def ladder_dir_to_results(d, reg):
    """`tools/pool_ladder.sh` 的产出：每个成员一个 eval_pair json，`overall.score` = 成员胜率。

    ⚠️ 方向：成员胜率 p_i ⇒ 学习者胜率 = 1 - p_i。
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        name = os.path.basename(f)[:-5]
        aid = ("bc:" + name[3:]) if name.startswith("bc_") else name
        try:
            o = (json.load(open(f, encoding="utf-8")).get("overall") or {})
        except Exception as e:                          # noqa: BLE001
            print(f"  [跳过] {name}: 读不出（{e}）")
            continue
        if "score" not in o:
            print(f"  [跳过] {name}: 没有 overall.score")
            continue
        if aid not in reg.by_id:
            print(f"  [跳过] {name} → {aid}: 不在池里")
            continue
        out[aid] = dict(p_i=float(o["score"]), n=int(o.get("n") or 0),
                        errors=int(o.get("errors") or 0))
    return out


def show(state, rules, title="当前阶梯"):
    p = state.pfsp.get("p") or {}
    w = state.pfsp.get("w") or {}
    roles = (rules.get("banks") or {}).get("roles") or {}
    if not p:
        print("（state.pfsp 是空的：还没跑过 pfsp 阶段）")
        return
    print(f"\n===== {title} =====")
    print(f"{'成员':<30}{'学习者胜率 p':>12}{'PFSP w':>9}  角色")
    for aid, v in sorted(p.items(), key=lambda kv: kv[1]):
        role = "?"
        for name, (lo, hi) in roles.items():
            if lo <= v < hi:
                role = name
                break
        print(f"{aid:<30}{v:>12.4f}{w.get(aid, 0.0):>9.3f}  {role}")
    unmeasured = [a for a, v in p.items() if abs(v - 0.5) < 1e-9
                  and a not in (state.pfsp.get("measured") or [])]
    if unmeasured:
        print(f"\n⚠️ 还是默认 0.5（没实测过）的 {len(unmeasured)} 个：{', '.join(unmeasured)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="评测包回传的 out/p_i.json")
    ap.add_argument("--from-ladder", default=None, help="本机 logs/ladder 目录（pool_ladder.sh 的产出）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--min-n", type=int, default=100, help="场次低于此值不落库（CI 太宽）")
    a = ap.parse_args()

    rules = L.load_rules()
    state = L.LeagueState()

    reg = L.Registry()
    if a.show or not (a.json or a.from_ladder):
        show(state, rules)
        return

    if a.from_ladder:
        data = ladder_dir_to_results(a.from_ladder, reg)
        print(f"从 {a.from_ladder} 读到 {len(data)} 个成员的实测")
    else:
        data = json.load(open(a.json, encoding="utf-8"))
    res, skipped = {}, []
    for aid, rec in data.items():
        if isinstance(rec, (int, float)):
            rec = dict(p_i=float(rec))
        n = int(rec.get("n") or 0)
        if "p_learner" in rec:
            p_l = float(rec["p_learner"])
            why = "p_learner"
        elif "p_i" in rec:
            p_l = 1.0 - float(rec["p_i"])
            why = "1-p_i"
        else:
            skipped.append((aid, "既没有 p_i 也没有 p_learner"))
            continue
        if n and n < a.min_n:
            skipped.append((aid, f"只有 {n} 场（<{a.min_n}），CI 太宽"))
            continue
        if aid not in reg.by_id:
            skipped.append((aid, "不在池里（先跑 pool_adopt.py）"))
            continue
        res[aid] = p_l
        print(f"{aid:<30} p_learner={p_l:.4f}（由 {why} 得到，n={n}）")

    if skipped:
        print("\n跳过：")
        for aid, why in skipped:
            print(f"  {aid}: {why}")
    if not res:
        raise SystemExit("\n✗ 没有任何可落库的条目")

    if a.dry_run:
        print(f"\n[dry-run] 会更新 {len(res)} 个，未写入。")
        return

    state.pfsp["measured"] = sorted(set(state.pfsp.get("measured") or []) | set(res))
    pf = L.update_pfsp(state, res, rules)
    print(f"\n✅ 已写入 {len(res)} 个 → state.pfsp（w = p(1-p)，共 {len(pf['w'])} 项）")
    show(state, rules, "更新后的阶梯")
    print("\n下一步：下一代的 bank 选择会按这些 p_i 分档（A 强 / B 中 / C 弱 / D 特殊）。")


if __name__ == "__main__":
    main()
