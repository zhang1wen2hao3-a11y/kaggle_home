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
GUARD = os.path.join(ROOT, "ptcg_league", "eval_pair.py")   # 不是 tools/：曾指错路径，评估一跑就崩
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


def resolve_model(spec):
    """把 init 参数归一成**存在的 native blob 路径**。

    接受三种写法：
      models/ppo/xxx.bin          （推荐）
      xxx                          → models/ppo/xxx.bin 或 bc_pool/xxx.bin
      build/agents/<dir>           → 找同目录 model.pt 对应的 .bin（不自动转换，报错提示）
    make_line_kit 的 --init 必须是 18,205,408 B 的 native blob，.pt 不行。
    """
    cands = []
    if os.path.isabs(spec) or "/" in spec:
        cands.append(spec if os.path.isabs(spec) else os.path.join(ROOT, spec))
    else:
        cands += [os.path.join(ROOT, "models/ppo", spec + ".bin"),
                  os.path.join(ROOT, "bc_pool", spec + ".bin")]
    for c in cands:
        if os.path.isfile(c) and os.path.getsize(c) == 18205408:
            return os.path.relpath(c, ROOT)
    raise SystemExit(
        f"✗ 认不出 init={spec!r}（需要 18,205,408 B 的 native blob .bin）\n"
        f"   试过: {cands}\n"
        f"   若是 .pt 或 agent 目录，先导出：\n"
        f"   python3 port_repo/native/tools/native_weights.py export <x.pt> <x.bin>")


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
    ma, mb = resolve_model(a.a), resolve_model(a.b)
    st = dict(name=a.names, rules=rules,
              sides=dict(A=dict(agent=a.a, model=ma, deck=a.a_deck, best_ev=None, flat=0),
                         B=dict(agent=a.b, model=mb, deck=a.b_deck, best_ev=None, flat=0)),
              turn="A", round=0, history=[], verdict=None, primary=a.primary)
    save(a.names, st)
    print(f"✅ {state_path(a.names)}")
    print(f"   A = {ma} @ {a.a_deck}")
    print(f"   B = {mb} @ {a.b_deck}   先训 A（B 冻结当 bank）")
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
           # ★ anchor 必须是**这一侧**的卡组；bank 必须钉在**对手那一侧**的卡组上。
           #   写死成同一个 a.anchor 会让交叉卡组对训里"专家打生牌"，陪练强度虚高。
           "--anchor", learner["deck"],
           "--dwf", a.dwf,
           "--banks", bank["model"],
           "--bank-decks", bank["deck"],
           "--league-pct", str(a.league_pct),
           "--track", f"{bank['deck']},{learner['deck']}",
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


def run_pair(cand_dir, opps, n, seed, out, logdir, retries=(0, 4, 2, 1)):
    """跑一次 eval_pair，**回收不全就自动减 worker 重试**，返回 json。

    为什么需要：`--workers 0` 会按内存预算自动开面板；机器被别人占着内存时
    worker 会被 OOM 杀掉，eval_pair 只写一句"回收 0 场，期望 N 场"，被杀的
    worker 的 *.log 还是 0 字节——无人值守的循环会直接中止在一轮评估上。
    实测：同一组对手 workers=4 时 400 场回收 0 场，workers=1 正常。所以这里
    把 worker 数当可退让参数，只在收集数达标（≥80%）时接受结果。
    """
    last = None
    for w in retries:
        if w != retries[0]:
            print(f"[eval] 回收不足，降到 --workers {w} 重试（{out}）")
        cmd = [PY, GUARD, "--cand", cand_dir]
        for lab, dv in opps:
            cmd += ["--opp", f"{lab}={dv}"]
        cmd += ["--matches", str(n), "--workers", str(w), "--seed-base", str(seed),
                "--out", out, "--logdir", logdir]
        r = run(cmd)
        if r.returncode != 0 or not os.path.exists(out):
            last = f"rc={r.returncode}"
            continue
        d = json.load(open(out))
        got = sum(int(v.get("n") or 0) for v in (d.get("per_opp") or {}).values())
        last = f"{got}/{n} 场"
        if got >= 0.8 * n:
            return d
    raise SystemExit(
        f"✗ 评估回收不全（{last}）：{out}\n"
        f"   worker 被 OOM 杀掉的可能性最大 —— 看同目录 *.log（0 字节=worker 没起来）\n"
        f"   手动验：bash loop.sh <names> --dry 之外另开一个纯评估，确认这台机还有多少内存")


def _h2h(cand_dir, opp_label, opp_dir, n, seed, out):
    """候选 vs 指定对手的配对评估 → 胜率。用于衡量"陪练侧有没有变强"。"""
    d = run_pair(cand_dir, [(opp_label, opp_dir)], n, seed, out,
                 os.path.join(ROOT, "logs/dual", opp_label))
    v = d["per_opp"].get(opp_label)
    if not v:
        raise SystemExit(f"✗ h2h 结果里没有 {opp_label}: {out}")
    return v["score"], v["n"], v["ci_low"], v["ci_high"]


def panel_eval(cand_dir, rules, out, logname):
    """在**固定面板 + 固定种子**上评估一个 agent 目录 → (json, field_ev)。"""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = rules["matches_per_cell"] * len(FIELD_OPPS)
    d = run_pair(cand_dir, list(FIELD_OPPS.items()), n, rules["seed_base"], out,
                 os.path.join(ROOT, "logs/dual", logname))
    missing = [k for k in FIELD if k not in (d.get("per_opp") or {})]
    if missing:
        raise SystemExit(
            f"✗ 面板评估不完整：缺 {missing}\n"
            f"   多半是 agent 起不来（看 {out} 同目录的 *.log）\n"
            f"   常见原因：该 python 没装 torch → 用 PYTHON=<装了torch的解释器> 重跑")
    ev = sum(FIELD[k] * d["per_opp"][k]["score"] for k in FIELD) / sum(FIELD.values())
    return d, ev


def agent_dir_from_blob(blob, name, deck_csv):
    """native blob(.bin) 或已有 .pt → 标准 agent 目录。返回目录路径。"""
    blob = blob if os.path.isabs(blob) else os.path.join(ROOT, blob)
    if not os.path.exists(blob):
        raise SystemExit(f"✗ 找不到权重 {blob}")
    ag_dir = os.path.join(ROOT, "build/agents", f"ag_{name}")
    pt = os.path.join(ROOT, "models/ppo", f"{name}.pt")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.join(ROOT, "build", "port_plain"), ROOT, os.environ.get("PYTHONPATH", "")]).strip(os.pathsep))
    if blob.endswith(".bin"):
        os.makedirs(os.path.dirname(pt), exist_ok=True)
        # ★ 必须带 PYTHONPATH=<bundle>：native_weights 要 import ptcg，否则静默失败、后面 cp 也静默失败
        r = subprocess.run([sys.executable, os.path.join(ROOT, "port_repo/native/tools/native_weights.py"),
                            "import", blob, pt], env=env)
        if r.returncode != 0 or not os.path.exists(pt):
            raise SystemExit(f"✗ bin→pt 失败（native_weights import）: {blob}")
    else:
        pt = blob
    if not os.path.isdir(ag_dir) or not os.path.exists(os.path.join(ag_dir, "model.pt")) \
            or not os.path.exists(os.path.join(ag_dir, "deck.csv")):
        r = subprocess.run(["bash", os.path.join(ROOT, "tools", "make_agent_dir.sh"), ag_dir,
                            pt, os.path.join(ROOT, deck_csv)])
        if r.returncode != 0:
            raise SystemExit(f"✗ 建 agent 目录失败: {ag_dir}")
    return ag_dir


def side_dir(rec, tag):
    """把 state 里的某一侧归一成**可评估的 agent 目录**。

    init 时 `agent` 只存了命令行给的名字（如 `g6_db04`），不是目录；这一侧第一次
    被评估、或它更新 `best_*` 之后 `agent` 才是真目录。所以任何拿它去跑 eval_pair
    的地方都必须走这里，否则 eval_pair 会报"无法识别的 agent 路径"。
    """
    cur = rec.get("agent")
    if cur:
        p = cur if os.path.isabs(cur) else os.path.join(ROOT, cur)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "model.pt")):
            return p
    return agent_dir_from_blob(rec["model"], tag, f"decks/pool_field/{rec['deck']}.csv")


def ensure_baseline(st, side, a):
    """**round 0 基线**：该侧还没有任何评估记录时，先把它当前的起始模型
    在同一面板/种子下量一遍。否则第 1 轮只能"当作基准"，训练是否让模型变差看不出来。"""
    rec = st["sides"][side]
    if rec.get("evals"):
        return st
    deck = rec["deck"]
    deck_csv = f"decks/pool_field/{deck}.csv"
    name = f"{a.names}_r0{side}_base"
    ag = agent_dir_from_blob(rec["model"], name, deck_csv)
    out = os.path.join(ROOT, "logs/dual", f"{name}.json")
    d, ev = panel_eval(ag, st["rules"], out, name)
    # ★ 非主侧的判据是"头对头打主侧"，基线也必须量同一个量：否则第 1 轮会拿
    #   h2h 去减基线里的 field_ev，得出 Δ=-0.43 这种跨量纲的假信号。
    primary = st.get("primary", "A")
    h2h, h2h_seeds = None, None
    if side != primary:
        try:
            opp = st["sides"][primary]
            h2h_out = os.path.join(ROOT, "logs/dual", f"{name}_h2h.json")
            h2h = _h2h(ag, f"vs_{primary}", side_dir(opp, f"{a.names}_side{primary}"),
                       st["rules"]["h2h_cell"], st["rules"]["seed_base"], h2h_out)[0]
            h2h_seeds = per_seed_totals(json.load(open(h2h_out))["per_opp"])
        except SystemExit:
            h2h = None
    metric = ev if side == primary else (h2h if h2h is not None else ev)
    seeds0 = per_seed_totals(d["per_opp"]) if side == primary else (h2h_seeds or {})
    rec.setdefault("evals", []).append(dict(round=0, baseline=True, metric=metric, ev=ev,
                                            h2h=h2h, per_seed=seeds0))
    rec["last_ev"] = ev
    key = "best_ev" if side == primary else "best_h2h"
    if rec.get(key) is None:
        rec[key] = metric
    save(a.names, st)
    print(f"[baseline] {side} 起始模型 = {rec['model']}  field_ev = {ev:.4f}"
          + (f"  头对头 {h2h:.4f}" if h2h is not None else "")
          + f"  判据 {metric:.4f}（已记为 round 0）")
    return st


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
    # ★ 先确保 round 0 基线存在（拿"训练前的模型"当对照），否则第 1 轮无法判定
    ensure_baseline(st, side, a)
    # bin → pt → agent 目录
    dst_bin = os.path.join(ROOT, "models/ppo", f"{run_name}.bin")
    os.makedirs(os.path.dirname(dst_bin), exist_ok=True)
    if src != dst_bin:
        subprocess.run(["cp", "-f", src, dst_bin], check=True)
    # deck 默认用**这一侧的卡组**（state 里有），不用人给
    deck_csv = a.deck_csv or f"decks/pool_field/{st['sides'][side]['deck']}.csv"
    ag_dir = agent_dir_from_blob(dst_bin, run_name, deck_csv)
    out = os.path.join(ROOT, "logs/dual", f"{run_name}.json")
    d, ev = panel_eval(ag_dir, st["rules"], out, run_name)

    # 陪练侧重在"更会打主侧"：额外量一次头对头。主侧仍以 field_ev 为主判据。
    primary = st.get("primary", "A")
    h2h, h2h_seeds = None, None
    if side != primary:
        opp = st["sides"][primary]
        h2h_out = os.path.join(ROOT, "logs/dual", f"{run_name}_h2h.json")
        try:
            h2h = _h2h(ag_dir, f"vs_{primary}", side_dir(opp, f"{a.names}_side{primary}"),
                       st["rules"]["h2h_cell"], st["rules"]["seed_base"], h2h_out)[0]
            h2h_seeds = per_seed_totals(json.load(open(h2h_out))["per_opp"])
        except SystemExit:
            h2h = None

    metric = ev if side == primary else (h2h if h2h is not None else ev)
    if side == primary or h2h is None:
        seeds_now = per_seed_totals(d["per_opp"])
    else:
        # 判据是 h2h 时，配对种子必须**也来自 h2h**，否则是拿两组不同对局的场次
        # 去配对，McNemar 的"同种子赢/输"根本不对应。
        seeds_now = h2h_seeds
    # ── 判定"有没有提升"：与**同一侧的上一轮模型**做同种子配对 McNemar ──
    # 两侧各自独立，因为要回答的是"这一侧这一轮是否比它上一轮显著更好"。
    prev_eval = (st["sides"][side].get("evals") or [])
    p_val, n_better, n_worse, delta = 1.0, 0, 0, None
    base = (prev_eval[-1] if prev_eval else None)   # ★ 先抓快照：下面 append 会改动这个 list
    if base:
        prev_seeds = base["per_seed"]
        common = set(seeds_now) & set(prev_seeds)
        for sd in common:
            a_, b_ = seeds_now[sd], prev_seeds[sd]
            if a_ > b_:
                n_better += 1
            elif a_ < b_:
                n_worse += 1
        p_val = mcnemar_exact(n_better, n_worse)[0]
        delta = metric - base["metric"]
    first_round = not prev_eval
    # 双层判定：**统计显著**（配对 McNemar）**且 实际够大**（delta ≥ min_delta）。
    # 只用显著性不够：实测 8 个种子全赢/0 输 ⇒ p=0.008 但实际只 +0.6pp，
    # 纯显著性会让循环为噪声级提升一直跑下去；min_delta 是"实际意义"的地板。
    significant = first_round or (p_val < st["rules"]["alpha"] and n_better > n_worse
                                  and delta is not None
                                  and delta >= st["rules"]["min_delta"])
    st["sides"][side]["flat"] = 0 if significant else st["sides"][side]["flat"] + 1
    side_rec = st["sides"][side]
    side_rec.setdefault("evals", []).append(dict(round=rnd, baseline=False, metric=metric,
                                                 ev=ev, h2h=h2h, per_seed=seeds_now))
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
        print("   （仍无基线，作为基准）")
    else:
        who = "起始模型(round 0)" if base.get("baseline") else f"上一轮 r{base['round']}"
        print(f"   对比 {who}：配对 本轮赢/输 = {n_better}/{n_worse}，McNemar p={p_val:.4f}"
              f"，Δ{tag} {delta:+.4f}"
              f" ⇒ {'✅ 显著提升' if significant else '⚠️ 未显著提升'}")
    if not first_round and p_val < st["rules"]["alpha"] and n_worse > n_better:
        print(f"   ⚠️ **显著变差**：比对照差 {n_worse}:{n_better}（p={p_val:.4f}）"
              f" —— 这一轮不该采用，best 保持原样")
    print(f"   flat = {flat}")
    if st["verdict"]:
        print(f"\n🛑 {st['verdict']}")
        for s_ in ("A", "B"):
            r = st["sides"][s_]
            print(f"   {s_} 最佳: {r.get('agent')}  ev={r.get('best_ev')}  h2h={r.get('best_h2h')}")
    else:
        why = "本轮还在涨 → 继续练同一侧" if significant else "本轮没涨 → 换方向"
        print(f"   下一轮：训 {st['turn']}（{why}；先 pack）")


def cmd_train(a):
    """**本地跑一轮训练**（整包里的自动循环用）：不起 kit、不 tar，
    直接把 state 里的 learner/bank 参数喂给 port_repo/tools/db04_line.sh。"""
    st = load(a.names)
    if st.get("verdict"):
        raise SystemExit(f"✗ 已经结束：{st['verdict']}")
    side = st["turn"]; other = "B" if side == "A" else "A"
    L, B = st["sides"][side], st["sides"][other]
    rnd = st["round"] + 1
    run = f"{a.names}_r{rnd}{side}"
    repo = os.path.abspath(os.path.join(ROOT, a.repo))
    out = os.path.abspath(os.path.join(ROOT, a.out))
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ,
               PYTHON=a.python or "python3",
               INIT=os.path.join(ROOT, L["model"]),
               OUT=out, RUN=run,
               ANCHOR=L["deck"], DWF=a.dwf,
               TRACK=f"{B['deck']},{L['deck']}",
               BANKS=os.path.join(ROOT, B["model"]),
               LEAGUE_PCT=str(a.league_pct),
               BANKDECKS=B["deck"])
    print(f"[train] 第 {rnd} 轮 训 {side}: anchor={L['deck']} bank={B['model']}@{B['deck']}")
    if getattr(a, "stub", False):
        # 自测用：不跑真训练，直接把 init 当成"产物"。用来验循环/判定/换向链路。
        print("[train] ⚠️ STUB 模式：不训练，直接复制 init 作为产物（仅用于链路自测）")
        dst0 = os.path.join(out, run + ".bin")
        os.makedirs(out, exist_ok=True)
        subprocess.run(["cp", "-f", env["INIT"], dst0], check=True)
    else:
        r = subprocess.run(["bash", os.path.join(repo, "tools", "db04_line.sh")],
                           cwd=repo, env=env)
        if r.returncode != 0:
            raise SystemExit(f"✗ 训练失败 rc={r.returncode}")
    binp = os.path.join(out, run + ".bin")
    if not os.path.exists(binp):
        raise SystemExit(f"✗ 没产出 {binp}")
    dst = os.path.join(ROOT, "models/ppo", run + ".bin")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(["cp", "-f", binp, dst], check=True)
    print(f"[train] 产出 → {os.path.relpath(dst, ROOT)}")
    print(f"        下一步： collect --names {a.names} --bin {os.path.relpath(dst, ROOT)}")


def cmd_stop(a):
    """循环用：已判定结束则 exit 0（并打印结论），否则 exit 1。"""
    st = load(a.names)
    if st.get("verdict"):
        print(f"🛑 {st['verdict']}")
        for s_ in ("A", "B"):
            r = st["sides"][s_]
            print(f"   {s_} 最佳: {r.get('agent')}  ev={r.get('best_ev')}  h2h={r.get('best_h2h')}")
        return 0
    print(f"继续：下一轮训 {st.get('turn')}")
    return 1


def cmd_status(a):
    st = load(a.names)
    print(f"== {a.names} ==  round {st['round']}  下一轮训 {st.get('turn')}  "
          f"规则 {st['rules']}")
    for s in ("A", "B"):
        d = st["sides"][s]
        role = "主" if s == st.get("primary", "A") else "练"
        be = "-" if d.get("best_ev") is None else f"{d['best_ev']:.4f}"
        bh = "-" if d.get("best_h2h") is None else f"{d['best_h2h']:.4f}"
        print(f"  {s}({role}): best_ev={be}  best_h2h={bh}  flat={d['flat']}  "
              f"deck={d['deck']}  model={d['model']}")
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
    i.add_argument("--a-deck", required=True, help="A 侧打的卡组（pool_field 里的名字）")
    i.add_argument("--b-deck", required=True, help="B 侧打的卡组")
    i.add_argument("--primary", default="A", choices=["A", "B"],
                   help="我们要提交的那一侧（判据用 field_ev）；另一侧用『打主侧的头对头』")
    for k in DEFAULT_RULES:
        i.add_argument(f"--{k.replace('_','-')}", type=int if k != "min_delta" else float)
    for name, fn in (("init", cmd_init), ("pack", cmd_pack), ("collect", cmd_collect),
                     ("status", cmd_status), ("train", cmd_train), ("stop", cmd_stop)):
        p = sub.add_parser(name) if name != "init" else i
        p.set_defaults(fn=fn)
    for name in ("pack",):
        p = sub.choices[name]
        p.add_argument("--names", required=True)
        p.add_argument("--dwf", required=True, help="对手卡组权重文件")
        p.add_argument("--league-pct", type=float, default=0.12)
        p.add_argument("--full-pool", action="store_true")
    p = sub.choices["collect"]
    p.add_argument("--names", required=True); p.add_argument("--bin", required=True)
    p.add_argument("--deck-csv", default="", help="留空=用该侧在 state 里的卡组")
    p = sub.choices["status"]
    p.add_argument("--names", required=True)
    p = sub.choices["train"]
    p.add_argument("--names", required=True)
    p.add_argument("--repo", default="port_repo", help="训练仓库目录")
    p.add_argument("--out", default="/tmp/loopout", help="训练产物输出目录")
    p.add_argument("--dwf", default="decks/weights_real.txt")
    p.add_argument("--league-pct", type=float, default=0.20)
    p.add_argument("--python", default="python3")
    p.add_argument("--stub", action="store_true", help="自测用：不训练，复制 init 当产物")
    p = sub.choices["stop"]
    p.add_argument("--names", required=True)
    a = ap.parse_args()
    rc = a.fn(a)
    raise SystemExit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
