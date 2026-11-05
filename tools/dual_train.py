#!/usr/bin/env python3
"""**双模型互训编排器**：两边轮流冻结/训练，用"有没有提升"决定换边与停止。

## 为什么不用头对头当判据

互训最大的陷阱是奖励"专门克制对方"。今天已有实证：G13 按"打梦妖"练，`lgw` 健康，
但顶部加权 **+0.0pp**。所以主判据用 **field-weighted EV**（本地矩阵预测 LB 的误差 ±2pp，
已在 G6@db04 上标定：本地 0.660 ↔ 真实 0.639）。

## 状态机（每轮三步，重活都在 GPU 机）

    ./tools/py tools/dual_train.py init  --a g6_db04 --b e1d7_bc --names db04_vs_e1d7
    ./tools/py tools/dual_train.py pack  --names db04_vs_e1d7        # 打本轮 kit（纯 CPU，秒级）
    #   → 按打印出来的命令传到 GPU 机： bash apply.sh --run
    ./tools/py tools/dual_train.py collect --names db04_vs_e1d7 --bin ~/g15.bin
    ./tools/py tools/dual_train.py status --names db04_vs_e1d7

## 停止判据 = **预算护栏，不是收敛预测**

我们不知道互训要几轮才收敛；下面三条是**在不确定下控制成本**的规则：

1. **"有没有提升"用配对检验，不用点估计相减。**
   每轮评估都跑固定面板 + 固定种子 ⇒ 第 r 轮与第 r−1 轮（同一侧）的对局**逐场可配对**。
   判定用**两层**：① 配对 **McNemar 精确检验** p < `alpha`（默认 0.05，统计显著）；
   ② 同时 `delta ≥ min_delta`（默认 0.02，实际够大）。
   只靠 ① 不够：实测"8 个种子全赢、0 输"就有 p=0.008，但实际只 +0.6pp——
   纯显著性会让循环为噪声级提升一直跑；`min_delta` 是实际意义的地板。
   只靠 ② 也不够：点估计相减会把抽样波动当信号。
2. **自适应换向，不盲目轮换。**
   这一侧本轮**显著提升 ⇒ 下一轮继续练同一侧**（它还在涨，换走浪费）；
   **没显著提升 ⇒ 换到另一侧**。
3. **两侧都停摆才收工。**
   两侧各自连续 `patience`（默认 2）轮都没显著提升 ⇒ 停（再跑性价比不明）。
   另有 `max_rounds`（默认 8）作为硬上限。

> 注意：`patience`/`max_rounds` 是成本参数，不是"预计第几轮收敛"。
> 真收敛只能看 Δ 序列的趋势：**逐轮收窄并趋近 0 = 收敛；来回跳 = 还在噪声里，规则本身不可信。**

## 与既有件的关系（不是新框架）

调用的全是现成的：`make_line_kit.py` 打包、`native_weights.py` bin↔pt、`eval_pair.py` 配对评估。
本文件只做"跨代 driver + 停止判据 + 记账"，正好补上 `ptcg_league/` 缺的那两块。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(ROOT, "ptcg_league", "state")
GUARD = os.path.join(ROOT, "tools", "eval_pair.py")
PY = os.path.join(ROOT, "tools", "py")

# 我们 640 局实测的对手分布 → field-weighted EV 的权重（与 §39 一致）
FIELD = {"ogerpon_7e39": 0.316, "abra_f5f8": 0.125, "grookey_2617": 0.106,
         "shuppet_0d86": 0.050, "dreepy_3eaa": 0.063}
FIELD_OPPS = {
    "ogerpon_7e39": "build/agents/port_7e39",
    "abra_f5f8": "build/agents/c_abra",
    "grookey_2617": "build/agents/c_grookey",
    "shuppet_0d86": "build/agents/opp_shuppet_human",
    "dreepy_3eaa": "build/agents/c_dreepy",
}
DEFAULT_RULES = dict(min_delta=0.02, patience=2, max_rounds=8,
                     matches_per_cell=250, seed_base=950000, h2h_cell=400,
                     alpha=0.05)


def state_path(name):
    return os.path.join(STATE_DIR, f"dual_{name}.json")


def load(name):
    p = state_path(name)
    if not os.path.exists(p):
        raise SystemExit(f"✗ 没有 {p} —— 先 init")
    return json.load(open(p))


def save(name, st):
    os.makedirs(STATE_DIR, exist_ok=True)
    json.dump(st, open(state_path(name), "w"), ensure_ascii=False, indent=1)


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd)[:160])
    return subprocess.run([str(c) for c in cmd], cwd=ROOT, **kw)


def cmd_init(a):
    rules = dict(DEFAULT_RULES)
    for k in rules:
        v = getattr(a, k, None)
        if v is not None:
            rules[k] = v
    st = dict(name=a.names, rules=rules,
              sides=dict(A=dict(agent=a.a, model=a.a, best_ev=None, flat=0),
                         B=dict(agent=a.b, model=a.b, best_ev=None, flat=0)),
              turn="A", round=0, history=[], verdict=None, primary=a.primary)
    save(a.names, st)
    print(f"✅ {state_path(a.names)}")
    print(f"   A = {a.a}   B = {a.b}   先训 A（B 冻结当 bank）")
    print(f"   规则: {rules}")


def cmd_pack(a):
    st = load(a.names)
    if st.get("verdict"):
        raise SystemExit(f"✗ 已经结束：{st['verdict']}")
    side = st["turn"]
    other = "B" if side == "A" else "A"
    learner, bank = st["sides"][side], st["sides"][other]
    rnd = st["round"] + 1
    run_name = f"{a.names}_r{rnd}{side}"
    model = os.path.join(ROOT, learner["model"])
    if not os.path.exists(model):
        raise SystemExit(f"✗ learner 权重不存在：{model}")
    cmd = [PY, "tools/make_line_kit.py",
           "--init", learner["model"],
           "--run", run_name,
           "--anchor", a.anchor,
           "--dwf", a.dwf,
           "--banks", bank["model"],
           "--bank-decks", a.anchor,
           "--league-pct", str(a.league_pct),
           "--track", a.anchor,
           "--purpose", f"{a.names} 第 {rnd} 轮：{side} 学、{other} 冻结当 bank",
           "--extra", f"主判据 field-weighted EV；min_delta={st['rules']['min_delta']}、"
                      f"patience={st['rules']['patience']}",
           "--out", f"/root/{run_name}"]
    if a.full_pool:
        cmd.append("--stage-full-pool")
    r = run(cmd)
    if r.returncode != 0:
        raise SystemExit("✗ 打包失败")
    kit = os.path.join(ROOT, "dist", f"{run_name}_kit.tar.gz")
    print(f"\n✅ 包 → {kit}")
    print(f"   GPU 机上： tar xzf {os.path.basename(kit)} && cd {run_name}_kit")
    print(f"              PTCG_REPO=<训练仓库> bash apply.sh --run")
    print(f"   回传：      /root/{run_name}/{run_name}.bin")
    print(f"   然后本机：  ./tools/py tools/dual_train.py collect --names {a.names} "
          f"--bin <拉回的.bin>")


def mcnemar_exact(n_new_win, n_old_win):
    """精确 McNemar（双侧二项）。返回 (p, n10, n01)。"""
    import math
    n = n_new_win + n_old_win
    if n == 0:
        return 1.0, 0, 0
    k = min(n_new_win, n_old_win)
    p = 2.0 * sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, p), n_new_win, n_old_win


def per_seed_totals(per_opp):
    """把 5 个对手的 per_match 合成 {seed: 本侧赢的场数}，用于跨轮配对。"""
    tot = {}
    for v in per_opp.values():
        for sd, w in (v.get("per_match") or {}).items():
            tot[sd] = tot.get(sd, 0) + (1 if w > 0.5 else 0)
    return tot


def _h2h(cand_dir, opp_label, opp_dir, n, seed, out):
    """候选 vs 指定对手的配对评估 → 胜率。用于衡量"陪练侧有没有变强"。"""
    cmd = [PY, GUARD, "--cand", cand_dir, "--opp", f"{opp_label}={opp_dir}",
           "--matches", str(n), "--workers", "0", "--seed-base", str(seed),
           "--out", out, "--logdir", os.path.join(ROOT, "logs/dual", opp_label)]
    run(cmd)
    d = json.load(open(out))
    v = d["per_opp"][opp_label]
    return v["score"], v["n"], v["ci_low"], v["ci_high"]


def cmd_collect(a):
    st = load(a.names)
    if st.get("verdict"):
        raise SystemExit(f"✗ 已经结束：{st['verdict']}")
    side = st["turn"]
    other = "B" if side == "A" else "A"
    rnd = st["round"] + 1
    run_name = f"{a.names}_r{rnd}{side}"
    src = os.path.abspath(os.path.expanduser(a.bin))
    if not os.path.exists(src) or os.path.getsize(src) != 18205408:
        raise SystemExit(f"✗ {src} 不是 18,205,408 B 的 native blob")
    # bin → pt → agent 目录
    dst_bin = os.path.join(ROOT, "models/ppo", f"{run_name}.bin")
    dst_pt = os.path.join(ROOT, "models/ppo", f"{run_name}.pt")
    os.makedirs(os.path.dirname(dst_bin), exist_ok=True)
    if src != dst_bin:
        subprocess.run(["cp", "-f", src, dst_bin], check=True)
    run([sys.executable, os.path.join(ROOT, "port_repo/native/tools/native_weights.py"),
         "import", dst_bin, dst_pt])
    ag_dir = os.path.join(ROOT, "build/agents", f"ag_{run_name}")
    if not os.path.isdir(ag_dir):
        run(["bash", os.path.join(ROOT, "tools", "make_agent_dir.sh"), ag_dir,
             dst_pt, os.path.join(ROOT, a.deck_csv)])
    # 配对评估：5 个真实场对手（种子固定 ⇒ 可与历史配对）
    out = os.path.join(ROOT, "logs/dual", f"{run_name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [PY, GUARD, "--cand", ag_dir]
    for k, d in FIELD_OPPS.items():
        cmd += ["--opp", f"{k}={d}"]
    cmd += ["--matches", str(st["rules"]["matches_per_cell"] * len(FIELD_OPPS)),
            "--workers", "0", "--seed-base", str(st["rules"]["seed_base"]),
            "--out", out, "--logdir", os.path.join(ROOT, "logs/dual", run_name)]
    r = run(cmd)
    if r.returncode != 0:
        raise SystemExit("✗ 评估失败")
    d = json.load(open(out))
    ev = sum(FIELD[k] * d["per_opp"][k]["score"] for k in FIELD) / sum(FIELD.values())

    # 陪练侧重在"更会打主侧"：额外量一次头对头。主侧仍以 field_ev 为主判据。
    primary = st.get("primary", "A")
    h2h = None
    if side != primary:
        opp = st["sides"][primary]
        h2h_out = os.path.join(ROOT, "logs/dual", f"{run_name}_h2h.json")
        try:
            h2h = _h2h(ag_dir, f"vs_{primary}", opp["agent"], st["rules"]["h2h_cell"],
                       st["rules"]["seed_base"], h2h_out)[0]
        except SystemExit:
            h2h = None

    metric = ev if side == primary else (h2h if h2h is not None else ev)

    # ── 判定"有没有提升"：与**同一侧的上一轮模型**做同种子配对 McNemar ──
    # 两侧各自独立，因为要回答的是"这一侧这一轮是否比它上一轮显著更好"。
    seeds_now = per_seed_totals(d["per_opp"])
    prev_eval = (st["sides"][side].get("evals") or [])
    p_val, n_better, n_worse, delta = 1.0, 0, 0, None
    if prev_eval:
        prev_seeds = prev_eval[-1]["per_seed"]
        common = set(seeds_now) & set(prev_seeds)
        for sd in common:
            a_, b_ = seeds_now[sd], prev_seeds[sd]
            if a_ > b_:
                n_better += 1
            elif a_ < b_:
                n_worse += 1
        p_val = mcnemar_exact(n_better, n_worse)[0]
        delta = metric - prev_eval[-1]["metric"]
    first_round = not prev_eval
    # 双层判定：**统计显著**（配对 McNemar）**且 实际够大**（delta ≥ min_delta）。
    # 只用显著性不够：实测 8 个种子全赢/0 输 ⇒ p=0.008 但实际只 +0.6pp，
    # 纯显著性会让循环为噪声级提升一直跑下去；min_delta 是"实际意义"的地板。
    significant = first_round or (p_val < st["rules"]["alpha"] and n_better > n_worse
                                  and delta is not None
                                  and delta >= st["rules"]["min_delta"])
    st["sides"][side]["flat"] = 0 if significant else st["sides"][side]["flat"] + 1
    side_rec = st["sides"][side]
    side_rec.setdefault("evals", []).append(dict(round=rnd, metric=metric, ev=ev,
                                                 h2h=h2h, per_seed=seeds_now))
    side_rec["last_ev"] = ev
    if h2h is not None:
        side_rec["last_h2h"] = h2h
    key = "best_ev" if side == primary else "best_h2h"
    if first_round or (side_rec.get(key) is None) or metric > side_rec.get(key, -1):
        side_rec.update({key: metric, "agent": ag_dir,
                         "model": f"models/ppo/{run_name}.bin", "best_round": rnd})
    st["round"] = rnd
    st["history"].append(dict(round=rnd, side=side, metric=metric, ev=ev, h2h=h2h,
                              delta=delta, p=p_val, n_better=n_better, n_worse=n_worse,
                              improved=significant, agent=ag_dir,
                              per_opp={k: round(v["score"], 4) for k, v in d["per_opp"].items()}))
    # ── 自适应换向 + 预算护栏 ──────────────────────────────────────────
    flat = {s: st["sides"][s]["flat"] for s in ("A", "B")}
    if all(v >= st["rules"]["patience"] for v in flat.values()):
        st["verdict"] = (f"预算护栏：两侧各连续 {st['rules']['patience']} 轮未显著提升"
                         f"（flat={flat}）⇒ 停止（不是『已收敛』，是『再跑性价比不明』）")
    elif rnd >= st["rules"]["max_rounds"]:
        st["verdict"] = f"预算护栏：达到 max_rounds={st['rules']['max_rounds']} ⇒ 停止"
    else:
        # 还在涨就继续练同一侧；不涨了就换方向
        st["turn"] = side if significant else other
    save(a.names, st)
    tag = "field_ev" if side == primary else "h2h(打主侧)"
    print(f"\n本轮 {side}: field_ev {ev:.4f}"
          + (f"  头对头 {h2h:.4f}" if h2h is not None else "")
          + f"  → 判据 {tag} {metric:.4f}"
          + (f"（Δ {delta:+.4f}）" if delta is not None else "（首轮）"))
    if first_round:
        print("   首轮：作为该侧基准，不做显著性检验")
    else:
        print(f"   与同一侧上一轮配对：本轮赢/输 = {n_better}/{n_worse}，McNemar p={p_val:.4f}"
              f" ⇒ {'✅ 显著提升' if significant else '⚠️ 未显著提升'}")
    print(f"   flat = {flat}")
    if st["verdict"]:
        print(f"\n🛑 {st['verdict']}")
        for s_ in ("A", "B"):
            r = st["sides"][s_]
            print(f"   {s_} 最佳: {r.get('agent')}  ev={r.get('best_ev')}  h2h={r.get('best_h2h')}")
    else:
        why = "本轮还在涨 → 继续练同一侧" if significant else "本轮没涨 → 换方向"
        print(f"   下一轮：训 {st['turn']}（{why}；先 pack）")


def cmd_status(a):
    st = load(a.names)
    print(f"== {a.names} ==  round {st['round']}  下一轮训 {st.get('turn')}  "
          f"规则 {st['rules']}")
    for s in ("A", "B"):
        d = st["sides"][s]
        print(f"  {s}: best_ev={d['best_ev']}  flat={d['flat']}  model={d['model']}")
    if st["history"]:
        print(f"\n{'轮':>3}{'侧':>3}{'field_ev':>10}{'头对头':>9}{'Δ判据':>10}  判定")
        for h in st["history"]:
            dl = "" if h["delta"] is None else f"{h['delta']:+.4f}"
            hh = f"{h['h2h']:.4f}" if h.get("h2h") is not None else "-"
            print(f"{h['round']:>3}{h['side']:>3}{h['ev']:>10.4f}{hh:>9}{dl:>10}  "
                  f"{'✅' if h['improved'] else '⚠️ 无提升'}")
    if st.get("verdict"):
        print(f"\n🛑 {st['verdict']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init"); i.add_argument("--names", required=True)
    i.add_argument("--a", required=True); i.add_argument("--b", required=True)
    i.add_argument("--primary", default="A", choices=["A", "B"],
                   help="我们要提交的那一侧（判据用 field_ev）；另一侧用『打主侧的头对头』")
    for k in DEFAULT_RULES:
        i.add_argument(f"--{k.replace('_','-')}", type=int if k != "min_delta" else float)
    for name, fn in (("init", cmd_init), ("pack", cmd_pack), ("collect", cmd_collect), ("status", cmd_status)):
        p = sub.add_parser(name) if name != "init" else i
        p.set_defaults(fn=fn)
    for name in ("pack",):
        p = sub.choices[name]
        p.add_argument("--names", required=True)
        p.add_argument("--anchor", required=True, help="本轮学习者打的卡组")
        p.add_argument("--dwf", required=True, help="对手卡组权重文件")
        p.add_argument("--league-pct", type=float, default=0.12)
        p.add_argument("--full-pool", action="store_true")
    p = sub.choices["collect"]
    p.add_argument("--names", required=True); p.add_argument("--bin", required=True)
    p.add_argument("--deck-csv", default="")
    p = sub.choices["status"]
    p.add_argument("--names", required=True)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
