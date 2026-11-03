#!/usr/bin/env python3
"""**训练联赛调度器**。你只负责 `init` 一次、`next` 反复跑、看 `reports/`。

设计约束（来自用户 spec + 硬件实测）：

* 三个池子互不覆盖：`init`（冻结 BC/通用 init）/ `evo`（专项分支）/ `league`（正式对手）；
* 一次 **generation** = 8 个固定步骤，每步都落盘，任何一步失败都能从断点续；
* **训练可以无限自动跑，晋级只由 `rules.json` 里写死的规则决定**；
* 4090 上 **PPO 并发 = 1**（独占 GPU），评测走 CPU 低优先级。

## 与硬件现实对齐的两处调整（必须知道）

1. **联赛池 8–16 个，但每代只上 2–4 个 bank。** 训练器把对手池 `league-ckpts`
   在 `create_pufferl` 里烘成 cudagraph 指针，**跑起来不能加**；且每 bank ≈ 1.0 GiB
   VRAM（512 agents 时实测 9128→9100 MiB），还有 `league_pct × n_banks < 0.5` 的断言。
   所以不是"池子多大就全上"，而是**每代按 PFSP 权重抽 2–4 个**，靠轮换覆盖整个池。
2. **PFSP 权重动态变化，bank 名额每代重抽** —— 这两件事在"每代重抽 bank"的实现下天然一致，
   不需要改原生训练器（原生不支持权重渐变，改了会有 cudagraph 一致性风险）。

## 命令

    ./tools/py ptcg_league/run_league.py init                 # 一次性：登记初始池、定 bootstrap best
    ./tools/py ptcg_league/run_league.py next --gens 3        # 跑到断点或 3 代
    ./tools/py ptcg_league/run_league.py next --resume        # 从上一代卡住的那步续
    ./tools/py ptcg_league/run_league.py status               # 看池子与 PFSP 权重
    ./tools/py ptcg_league/run_league.py verify               # 自检：规则 sha / 注册表 / 目录
    ./tools/py ptcg_league/run_league.py rollback --to <id>   # 回滚 best 指针

训练那一步默认**只写脚本不执行**（本机没有能跑原生 PPO 的 GPU）：
写 `ptcg_league/pending/gen_NNNN_train.sh`，把 ckpt 放回 `ckpt_out` 后 `next --resume` 即可。
给了 `--train-cmd` 模板（或 `--mock-train`）就自动执行。
"""
import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time

CODE = os.path.dirname(os.path.abspath(__file__))      # 代码在哪（永远是这个目录）
HERE = os.environ.get("PTCG_LEAGUE_HOME") or CODE      # 状态/池子/报告放哪（自测时重定向到 /tmp）
ROOT = os.path.dirname(CODE)
sys.path.insert(0, ROOT)
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import league as L                                    # noqa: E402

PY = os.path.join(ROOT, "tools", "py")
STAGES = ("plan", "train", "stage", "screen", "confirm", "anchor", "decide", "pfsp")


# --------------------------------------------------------------------- 小工具
def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gen_dir(g):
    d = os.path.join(HERE, "gen", f"{g:04d}")
    os.makedirs(d, exist_ok=True)
    return d


def artifact(g, stage):
    return os.path.join(gen_dir(g), f"{stage}.json")


def have(g, stage):
    p = artifact(g, stage)
    return os.path.exists(p) and os.path.getsize(p) > 2


def put(g, stage, obj):
    L.write_json(artifact(g, stage), obj)
    return obj


def get(g, stage):
    return L.read_json(artifact(g, stage))


def run(cmd, log, cwd=None):
    os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
    say("$ " + " ".join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd)))
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(cmd) + f"\n[{L.now()}]\n")
        f.flush()
        p = subprocess.run(cmd, cwd=cwd or ROOT, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(f"命令失败({p.returncode})，日志 {log}")
    return log


# --------------------------------------------------------------------- 卡组
def canonical_deck(path):
    """把任意卡组文件映射到 `decks/pool_field/` 里的规范文件名（按内容 sha 匹配）。

    为什么要做：`build/port_plain/deck.csv` 这种名字放进登记表 → `deck_id="deck"`，
    报告里看不出是哪副牌。按内容找到池里的同名文件，`deck_id` 才有意义。
    """
    if not path or not os.path.exists(path):
        return path
    sha = L.sha256_file(path)
    cache = canonical_deck.__dict__.setdefault("_c", {})
    if not cache:
        pool = os.path.join(ROOT, "decks", "pool_field")
        if os.path.isdir(pool):
            for f in sorted(os.listdir(pool)):
                if f.endswith(".csv"):
                    fp = os.path.join(pool, f)
                    cache[L.sha256_file(fp)] = fp
    return cache.get(sha, os.path.abspath(path))


def deck_archetype(path):
    """卡组 → 原型名。沿用 `tools/scrape_episodes.py` 的口径：去掉 `__abcd` 变体后缀。

    ⚠️ 口径说明：这个粒度是"一副具体卡表"，不是"Ogerpon 大类"。
    两阶段 PFSP（先抽原型再抽个体）只有在池子里出现多副不同卡组时才有意义，
    池子小的时候它会自动退化成按个体抽 —— 不会出错，只是没额外收益。
    """
    stem = os.path.basename(path).split("__")[0]
    return stem or "?"


def pool_deck_by_hash(h):
    """`0d86` → `decks/pool_field/shuppet_dhelmise__0d86.csv`（不在池里返回 None）。"""
    c = glob.glob(os.path.join(ROOT, "decks", "pool_field", f"*__{h}.csv"))
    return c[0] if c else None


def dominant_deck_of_agent(name, npz=None):
    """某个天梯玩家的**主力卡组**（按他所有帧里出现最多的 archetype）。

    为什么要按人配牌：agent 专精 brain 学的是"这个人怎么打"，
    如果给他一副他从没用过的牌，BC 的先验就落空了。
    """
    npz = npz or os.path.join(ROOT, "data", "bc_all.npz")
    if not os.path.exists(npz):
        return None
    import collections
    import numpy as np
    z = np.load(npz, allow_pickle=True)
    arch, agent = z["arch"], z["agent"]
    sel = np.array([name.lower() in str(a).lower() for a in agent])
    if not sel.any():
        return None
    top = collections.Counter(arch[sel]).most_common(1)[0][0]
    return pool_deck_by_hash(str(top))


def bc_brain_specs(pool="bc_pool", bundle="build/port_plain"):
    """读 `bc_pool/registry.tsv` → [(brain_name, ckpt, deck_path, note)]。

    卡组规则：
      * `deck:0d86`  → 池里对应的那副（这个 brain 就是在这副牌上学出来的）；
      * `agent:Raihan` → 该玩家帧里出现最多的 archetype（子串匹配名字）。
    """
    reg = os.path.join(ROOT, pool, "registry.tsv")
    if not os.path.exists(reg):
        raise SystemExit(f"缺 {reg}：先跑 ./tools/py tools/bc_pool_registry.py")
    out = []
    with open(reg, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            if not line.strip():
                continue
            r = dict(zip(head, line.rstrip("\n").split("\t")))
            if r["name"] == "name":
                continue
            spec = r["filter"] or "?"
            if ":" not in spec:            # 半成品/来历不明的 brain（比如只存了 1 个 epoch 的死文件）
                say(f"  ⚠️ 跳过 {r['name']}：filter 不可解析（{spec}）")
                continue
            kind, val = spec.split(":", 1)
            pt = os.path.join(ROOT, pool, r["name"] + ".pt")
            bn = os.path.join(ROOT, pool, r["name"] + ".bin")
            ckpt = bn if os.path.exists(bn) else pt
            if kind == "deck":
                deck = pool_deck_by_hash(val)
            else:
                deck = dominant_deck_of_agent(val)
            if not deck:
                say(f"  ⚠️ 跳过 {r['name']}：找不到配套卡组（{spec}）")
                continue
            va = r.get("val_acc") or "?"
            out.append((r["name"].strip("_"), ckpt, deck,
                        f"{spec} val_acc={va} frames={r.get('frames')}", bn))
    return out


# --------------------------------------------------------------------- 对手池
def restat(old, **kw):
    """从旧登记记录派生一条新记录（rev 由 Registry.add 递增）。

    ⚠️ 不要写 `dict(**old, status=...)` —— old 里本来就有 status，会 TypeError。
    """
    d = {k: v for k, v in old.items() if k != "rev"}
    d.update(kw)
    return d


def league_rows(reg):
    return [r for r in reg.in_pool("league") if r["status"] in ("best", "opponent", "specialist")]


def init_rows(reg):
    return reg.in_pool("init")


def opponent_candidates(reg):
    """可当训练 bank 的对手 = 正式联赛池 + 冻结的初始策略池。

    为什么把 BC 也算进来：用户 spec 的"初始对手池"就是 5 个冻结 BC + init 副本。
    它们不进正式联赛池（因为它们没有被评测晋级），但**必须**能进训练对手池，
    否则第一次训练只会看到端口自己。
    """
    out, seen = [], {}
    for r in league_rows(reg) + reg.in_pool("init"):
        if not r.get("eval_dir"):
            continue
        bp = r.get("bin_path") or ""
        if not bp:
            continue
        sha = L.sha256_file(bp) if os.path.exists(bp) else ""
        # 同一份权重只留一个（正式池优先）：bootstrap 的 `lg_boot:port_plain`
        # 与 `init:port_plain` 就是同一份 .bin，两个都抽等于给端口双倍预算。
        if sha and sha in seen:
            continue
        if sha:
            seen[sha] = r["agent_id"]
        out.append(r)
    return out


def roster_for_anchor(reg, rules):
    """**固定评测集**：冻结的初始池（BC + 端口 + 通用 init）。

    必须用冻结 roster 而不是"当前联赛池"：后者会随晋级漂移，
    那 anchor 分就成了移动靶，"有没有退化"永远量不出来。
    """
    rows = [r for r in init_rows(reg) if r.get("eval_dir")]
    return rows


def current_best(reg, state):
    if state.current_best:
        return reg.get(state.current_best)
    b = reg.best()
    if b:
        state.current_best = b["agent_id"]
        state.save()
        return b
    raise RuntimeError("联赛里还没有 best，先 `init --best <agent_id>`")


# --------------------------------------------------------------------- 1 plan
def stage_plan(g, reg, state, rules, args):
    best = current_best(reg, state)
    rows = opponent_candidates(reg)
    groups = {r["agent_id"]: r.get("archetype") or "?" for r in rows}
    pf = state.pfsp.get("w") or {}
    pv = state.pfsp.get("p") or {}
    if not pf:                                   # 第一次：均分
        pf = {r["agent_id"]: 1.0 / max(1, len(rows)) for r in rows}

    tcfg = rules.get("train", L.DEFAULT_RULES.get("train", {}))
    deck = canonical_deck(args.deck or pick_deck(reg, state, args))
    target = (args.target or (rules.get("specialist_probe") or {}).get("target_archetype") or "").strip()
    tc = next((r for r in (reg.in_pool("evo") + [best]) if r.get("deck_id") == deck), None)
    parent = tc["agent_id"] if tc else best["agent_id"]

    # ⚠️ 把"和 init 同一份权重"的对手剔掉：
    #    历史教训（20M + ladder 1→4 banks）是 `lgw 0.96–0.98` + `ep_len 176→123`，
    #    即训练退化成"跟自己下"。frozen 自快照本身不算错，但它**占掉一个 bank 名额
    #    却不提供新信息**，所以只要候选的 .bin 与 init 同 sha 就不抽它。
    def _sha(r):
        bp = r.get("bin_path") or ""
        return L.sha256_file(bp) if bp and os.path.exists(bp) else ""

    # ---- 角色配额抽 bank（取代"纯按权重随机"，理由见 rules.banks._roles_why）----
    bcfg = rules.get("banks") or {}
    roles = bcfg.get("roles") or {}
    quota = bcfg.get("quota") or {}
    parent_sha = _sha(reg.get(parent))
    # ★ 必须**先**过滤候选池再抽样。原先是"抽完再丢掉 parent 同权重的那个"，
    #   而 parent 本身就在候选池里（bootstrap 时 best == parent），于是有
    #   1-(6/7)^3 ≈ 35% 的概率白抽一个名额、最后只剩 2 个 bank。
    cands = [r for r in rows
             if r["agent_id"] != parent and _sha(r) != parent_sha]
    nbank = min(int(args.banks), len(cands))
    # ★ 强制配额：某些原型必须每代都在场（见 rules.banks.pinned_archetypes）。
    #   理由不是"它强"，而是"不钉住就永远学不到" —— 我们这副牌的
    #   Powerful Hand 是招式效果，被 Hide 'n' Sneak 全额免疫，真实天梯对 shuppet 0/16。
    pinned_arch = [str(x).lower() for x in (rules.get("banks") or {}).get("pinned_archetypes") or []]
    pinned, rest = [], []
    for r in cands:
        a = (r.get("archetype") or r.get("deck_id") or "").lower()
        if any(a.startswith(x) or x in a for x in pinned_arch) and len(pinned) < nbank:
            pinned.append(r["agent_id"])
        else:
            rest.append(r)
    # 每个 pinned 原型只钉一个代表（池里 shuppet 有两个脑子：sue124 与 deck_0d86）
    seen_pin = set()
    uniq_pin = []
    for aid in pinned:
        a = (reg.get(aid).get("archetype") or "").lower()
        if a in seen_pin:
            continue
        seen_pin.add(a)
        uniq_pin.append(aid)
    pinned = uniq_pin[:max(1, nbank - 1)] if pinned_arch else []
    rest = [r for r in cands if r["agent_id"] not in pinned]
    nfree = max(0, nbank - len(pinned))
    if roles and quota:
        # 按 p_i 分档 → 按配额取；档内优先"上一代没用过的 + 权重高的"，跨代轮转
        def role_of(aid):
            v = pv.get(aid, 0.5)          # ← p_i（学习者对它的胜率），不是 w_i
            for name, (lo_, hi_) in roles.items():
                if lo_ <= v < hi_:
                    return name
            return "D"
        used = (getattr(state, "used", None) or {})
        by_role = {}
        for r in rest:
            by_role.setdefault(role_of(r["agent_id"]), []).append(r)
        for k_ in by_role:
            by_role[k_].sort(key=lambda r: (used.get(r["agent_id"], -99), -pf.get(r["agent_id"], 0.5)))
        free = []
        seen_arch = set()
        for role_name in ("A", "B", "C", "D"):
            want = int(quota.get(role_name, 0))
            pool_ = [r for r in by_role.get(role_name, []) if r["agent_id"] not in free]
            # 先取"新卡组"，再取"上一代没用的"
            pool_.sort(key=lambda r: ((r.get("archetype") in seen_arch),
                                      used.get(r["agent_id"], -99),
                                      -pf.get(r["agent_id"], 0.5)))
            for r in pool_[:max(0, nfree - len(free))]:
                if len(free) >= nfree:
                    break
                if want <= 0:
                    break
                free.append(r["agent_id"])
                seen_arch.add(r.get("archetype"))
                want -= 1
        # 配额没填满就用剩下的按权重补
        if len(free) < nfree and rest:
            left = [r for r in rest if r["agent_id"] not in free]
            free += (L.sample_opponents({r["agent_id"]: pf.get(r["agent_id"], 0.5) for r in left},
                                        nfree - len(free), random.Random(args.seed + g),
                                        {r["agent_id"]: groups.get(r["agent_id"], r["agent_id"])
                                         for r in left}) if left else [])
        say(f"  角色配额：{[role_of(a) for a in free]} → {free}")
    else:
        free = (L.sample_opponents({r["agent_id"]: pf.get(r["agent_id"], 0.5) for r in rest},
                                   nfree, random.Random(args.seed + g),
                                   {r["agent_id"]: groups.get(r["agent_id"], r["agent_id"])
                                    for r in rest}) if rest and nfree else [])
    picks = pinned + free
    # best 必须在场（"有没有比现在更好"要有对照）——但 best 就是 parent 时它已无信息量
    if best["agent_id"] != parent and best["agent_id"] not in picks:
        picks = [best["agent_id"]] + picks
    picks = list(dict.fromkeys(picks))[:nbank]
    if not picks:                       # 极端情况：池里全是同权重 → 退回只带 parent
        picks = [best["agent_id"]]

    seed = args.seed + g * 7919
    # 分支名：代 + 卡组 hash + 父模型短名，例 `g0001_7e39_port_plain`
    deck_tag = os.path.basename(deck).split("__")[-1].replace(".csv", "")[:4]
    ptag = "".join(c if c.isalnum() else "_" for c in parent.split(":")[-1])[:10]
    branch = f"g{g:04d}_{deck_tag}_{ptag}"
    out_dir = os.path.join(HERE, "pools", "evo", branch)
    os.makedirs(out_dir, exist_ok=True)
    matches = int(args.matches or tcfg.get("matches_per_gen", 2048))
    steps = matches * int(tcfg.get("steps_per_match", 170))
    plan = dict(
        gen=g, branch=branch, deck=deck, parent_id=parent, seed=seed, steps=steps,
        matches=matches,
        init_ckpt=reg.get(parent)["checkpoint_path"],
        init_bin=reg.get(parent).get("bin_path") or "",
        league_ckpts=[reg.get(p)["checkpoint_path"] for p in picks],
        league_opponents=picks,
        league_pct=round(min(0.4 / max(1, len(picks)), 0.12), 4),
        total_agents=int(tcfg.get("total_agents", 512)),
        minibatch=int(tcfg.get("minibatch", 512)),
        lr=float(args.lr or tcfg.get("lr", 5e-4)),
        checkpoint_interval=int(tcfg.get("checkpoint_interval", 25)),
        deck_pool=tcfg.get("deck_pool", "decks/pool_field"),
        deck_weights_file=(args.deck_weights
                           or (tcfg.get("deck_weights_file", "decks/weights_real.txt")
                               if not target else "")
                           or tcfg.get("deck_weights_file", "decks/weights_real.txt")),
        target_archetype=target,
        run_name=f"lg_{branch}",
        ckpt_out=os.path.join(out_dir, "final.bin"),
        ckpt_dir=out_dir,
        rules_sha=L.rules_sha(rules),
        rules_version=rules.get("version"),
        started=L.now(),
    )
    if target:
        wf = args.deck_weights or os.path.join(ROOT, "decks", f"weights_{target}.txt")
        if not os.path.exists(wf):
            raise SystemExit(f"专精模式需要定向权重文件 {wf}："
                             f"先跑 ./tools/py tools/make_deck_weights.py "
                             f"--boost-prefix {target} --boost 20 --out {wf}")
        plan["deck_weights_file"] = wf
        # 记下目标族在对手座位实际占多少（从权重文件算，别手写死）
        try:
            tot = hit = 0.0
            for line in open(wf, encoding="utf-8"):
                pr = line.split()
                if len(pr) != 2 or pr[0] in ("version", "default"):
                    continue
                tot += float(pr[1])
                if pr[0].startswith(target):
                    hit += float(pr[1])
            plan["_target_share"] = round(hit / tot, 4) if tot else None
        except (OSError, ValueError):
            plan["_target_share"] = None
        # probe 池 = 初始池里打这个原型的对手（专精要量的就是它）
        probe = [r["agent_id"] for r in reg.in_pool("init")
                 if target in (r.get("archetype") or r.get("deck_id") or "").lower()
                 and r.get("eval_dir")]
        plan["probe_opponents"] = probe
        say(f"  专精目标 = {target}：对手座位权重 {os.path.basename(wf)}，"
            f"probe 池 {len(probe)} 个 {probe}")
    else:
        plan["probe_opponents"] = []
    if plan["league_pct"] * max(1, len(picks)) >= 0.5:
        raise RuntimeError(f"league_pct×n ≥ 0.5：{plan['league_pct']}×{len(picks)}")
    st_used = dict(getattr(state, "used", None) or {})
    for a_ in picks:
        st_used[a_] = g
    state.used = st_used
    state.save()
    put(g, "plan", plan)
    say(f"代 {g} 计划：分支 {branch}  卡组 {os.path.basename(deck)}  "
        f"父 {parent}  {steps} 步（{steps//int(tcfg.get('steps_per_match',170))} 局）  "
        f"banks={len(picks)} pct={plan['league_pct']}")
    return plan


def pick_deck(reg, state, args):
    """专精卡组：默认沿用 best 的卡组（"目标卡组继续训练"）。

    `--deck-cycle` 时按 `decks/weights_real.txt` 在真实野区里轮换挑一副，
    保证多个分支覆盖不同卡组（第 20 名方案的做法：多个专项模型）。
    """
    if args.deck:
        p = args.deck
        if not os.path.exists(p):
            for c in (os.path.join(ROOT, "decks", "pool_field", p),
                      os.path.join(ROOT, "decks", "pool_field", p + ".csv")):
                if os.path.exists(c):
                    return c
            raise SystemExit(f"找不到卡组 {p}")
        return p
    rows = [r for r in reg.in_pool("league") if r.get("deck_path")]
    if not rows:
        rows = [r for r in reg.in_pool("init") if r.get("deck_path")]
    seen = sorted({r["deck_path"] for r in rows if r.get("deck_path")})
    if args.deck_cycle and seen:
        i = (state.gen + 1) % len(seen)
        return seen[i]
    best = reg.by_id.get(state.current_best or "")
    if best and best.get("deck_path"):
        return best["deck_path"]
    if seen:
        return seen[0]
    raise SystemExit("没有可用卡组：给 --deck")


# --------------------------------------------------------------------- 2 train
def stage_train(g, reg, state, rules, args):
    plan = get(g, "plan")
    if os.path.exists(plan["ckpt_out"]):
        say(f"代 {g} 训练产出已存在：{plan['ckpt_out']}")
        return plan["ckpt_out"]
    tpl = args.train_cmd or ""
    if args.mock_train or tpl:
        if args.mock_train:
            src = (plan["init_ckpt"] if args.mock_train in (True, "parent")
                   else os.path.abspath(args.mock_train))
            src = src if src and os.path.exists(src) else None
            if src is None:
                raise SystemExit(f"--mock-train 的 ckpt 不存在：{args.mock_train}")
            # 后缀必须跟来源一致：`.pt` 与原生 blob `.bin` 是两种格式，
            # stage_ckpt 按后缀选导入方式（把 .pt 命名成 .bin 会报 "blob has N floats"）。
            out = os.path.join(plan["ckpt_dir"], "final" + os.path.splitext(src)[1])
            os.makedirs(plan["ckpt_dir"], exist_ok=True)
            shutil.copyfile(src, out)
            if out != plan["ckpt_out"]:
                plan["ckpt_out"] = out
                put(g, "plan", plan)
            say(f"代 {g} mock 训练：把 {src} 当作候选 → {out}")
        else:
            cmd = tpl.format(
                init=plan["init_ckpt"], steps=plan["steps"], seed=plan["seed"],
                lr=plan["lr"], deck=plan["deck"], run_name=plan["run_name"],
                league_ckpts=",".join(plan["league_ckpts"]),
                league_pct=plan["league_pct"], out=plan["ckpt_out"],
                ckpt_dir=plan["ckpt_dir"], total_agents=plan["total_agents"],
                minibatch=plan["minibatch"],
                checkpoint_interval=plan["checkpoint_interval"],
                deck_pool=plan["deck_pool"], deck_weights_file=plan["deck_weights_file"],
            )
            run(["bash", "-lc", cmd], os.path.join(gen_dir(g), "train.log"))
            if not os.path.exists(plan["ckpt_out"]):
                cand = sorted(glob.glob(os.path.join(plan["ckpt_dir"], "*.bin")),
                              key=os.path.getmtime)
                if not cand:
                    raise RuntimeError(f"训练结束但没有 ckpt：{plan['ckpt_dir']}")
                shutil.copyfile(cand[-1], plan["ckpt_out"])
        return plan["ckpt_out"]

    # 本机不能训练 → 写待办脚本，交给 GPU 机器
    os.makedirs(os.path.join(HERE, "pending"), exist_ok=True)
    p = os.path.join(HERE, "pending", f"gen_{g:04d}_train.sh")
    with open(p, "w", encoding="utf-8") as f:
        f.write(TRAIN_SH.format(**plan, root=ROOT, here=HERE))
    os.chmod(p, 0o755)
    state.set_awaiting("train", g)
    say(f"代 {g} 需要 GPU 训练 → 已写 {p}")
    say(f"      跑完把 final.bin 放回 {plan['ckpt_out']}，然后 `next --resume`")
    return None


TRAIN_SH = """#!/usr/bin/env bash
# 自动生成：训练联赛 代 {gen} 的 PPO 分支（在 4090 训练机上跑）
#   bash {here}/pending/gen_{gen:04d}_train.sh
set -uo pipefail
REPO="${{PTCG_REPO:-$HOME/work/ptcg_train_kit/port_repo}}"
cd "$REPO" || {{ echo "✗ 找不到训练仓库 $REPO"; exit 1; }}
PY="${{PYTHON:-python3}}"
RUN="{run_name}"
RD="experiments/native-$RUN"
mkdir -p "$RD" logs
cp -f "{init_bin}" "$RD/init.bin"
echo "init.bin = {init_bin}  deck = {deck}"
echo "banks = {league_ckpts}  pct = {league_pct}"

PYTHONPATH=data:.:native PTCG_TABLES=native/ptcg_tables_field.bin OMP_WAIT_POLICY=PASSIVE \\
"$PY" native/train_native.py \\
  --run-name "$RUN" \\
  --timesteps {steps} \\
  --total-agents {total_agents} \\
  --minibatch {minibatch} \\
  --lr {lr} \\
  --mix-self 1.0 \\
  --league-ckpts "{league_ckpts}" \\
  --league-pct {league_pct} \\
  --deck-pool {deck_pool} \\
  --deck-weights-file {deck_weights_file} \\
  --anchor-decks "$(basename {deck} .csv)" \\
  --p-anchor-learner 1.0 \\
  --p-anchor-self-opp 0.0 \\
  --p-anchor-league-opp 0.0 \\
  --checkpoint-interval {checkpoint_interval} \\
  --seed {seed} \\
  2>&1 | tee "logs/$RUN.log"

echo
echo "训练结束。把 final.bin 传回本机："
echo "  scp $RD/final.bin <本机>:{ckpt_out}"
echo "然后在本机跑：./tools/py ptcg_league/run_league.py next --resume"
"""


# --------------------------------------------------------------------- 3 stage
def stage_stage(g, reg, state, rules, args):
    plan = get(g, "plan")
    out = os.path.join(plan["ckpt_dir"], "bundle")
    if os.path.exists(os.path.join(out, "model.pt")) and not args.force_stage:
        return put(g, "stage", dict(bundle=out, cached=True))
    cmd = [PY, os.path.join(ROOT, "tools", "stage_ckpt.py"), plan["ckpt_out"],
           "--out", out, "--bundle", args.bundle, "--deck", plan["deck"], "--force"]
    if args.repo:
        cmd += ["--repo", args.repo]
    run(cmd, os.path.join(gen_dir(g), "stage.log"))
    # 记录候选指纹（权重 sha），重复候选可以直接跳过
    sha = L.sha256_file(os.path.join(out, "model.pt"))
    return put(g, "stage", dict(bundle=out, ckpt=plan["ckpt_out"], model_sha=sha,
                                deck=plan["deck"]))


def is_duplicate(reg, state, st):
    """候选与当前 best 是不是**同一份权重**（sha 相同）。

    这一步必须在评测**之前**做：省一整个筛查的钱，也避免报告里
    "筛查未过"和"sha 相同"两条结论互相打架。
    """
    best = current_best(reg, state)
    best_sha = best.get("model_sha")
    if not best_sha:
        bp = best.get("checkpoint_path")
        if bp and os.path.exists(bp) and os.path.splitext(bp)[1] in (".pt", ".bin"):
            best_sha = L.sha256_file(bp)
    return bool(best_sha) and best_sha == (st or {}).get("model_sha"), best


# --------------------------------------------------------------------- 4-6 eval
def do_eval(g, name, rules, args, opponents, matches, seed_base, logname=None):
    """跑一次 eval_pair，落盘 gen/NNNN/<name>.json。已存在就直接复用（可复现）。"""
    if have(g, name) and not args.force_eval:
        d = get(g, name)
        say(f"  {name}: 复用已有结果 {d['overall']['score']:.4f} (n={d['overall']['n']})")
        return d
    st = get(g, "stage")
    cmd = [PY, os.path.join(CODE, "eval_pair.py"), "--cand", st["bundle"]]
    for aid, d in opponents:
        cmd += ["--opp", f"{aid}={d}"]
    cmd += ["--matches", str(matches), "--workers", str(args.workers),
            "--seed-base", str(seed_base), "--out", artifact(g, name)]
    onejob = os.path.join(ROOT, "tools", "onejob.sh")
    if os.path.exists(onejob):
        cmd = [onejob, "league-eval", "--"] + cmd
    run(cmd, os.path.join(gen_dir(g), logname or f"{name}.log"))
    d = get(g, name)
    o = d["overall"]
    say(f"  {name}: {o['score']:.4f} CI [{o['ci_low']:.4f},{o['ci_high']:.4f}] "
        f"n={o['n']} 异常 {o['errors']}")
    return d


def stage_screen(g, reg, state, rules, args):
    dup, best = is_duplicate(reg, state, get(g, "stage"))
    if dup:
        say("  候选与当前 best 是同一份权重（sha 相同）→ 不做评测，直接淘汰")
        return put(g, "screen", dict(skipped="duplicate_weights",
                                     overall=dict(score=0.0, ci_low=0.0, ci_high=1.0,
                                                  n=0, errors=0)))
    vs = [(best["agent_id"], best["eval_dir"])]
    d = do_eval(g, "screen", rules, args, vs, rules["screen"]["matches"],
                args.seed_base + 1000 + g * 10000)
    return d


def stage_confirm(g, reg, state, rules, args):
    sc = get(g, "screen")["overall"]
    if sc["score"] < rules["screen"]["min_score"] or sc["errors"] > 0:
        say(f"  筛查未过（{sc['score']:.4f} / 异常 {sc['errors']}），跳过确认与晋级评测")
        return put(g, "confirm", dict(skipped="screen_failed", overall=sc))
    best = current_best(reg, state)
    vs = [(best["agent_id"], best["eval_dir"])]
    return do_eval(g, "confirm", rules, args, vs, rules["confirm"]["matches"],
                   args.seed_base + 2000 + g * 10000)


def stage_anchor(g, reg, state, rules, args):
    cf = get(g, "confirm")
    if cf.get("skipped"):
        return put(g, "anchor", dict(skipped="screen_failed"))
    roster = roster_for_anchor(reg, rules)
    if not roster:
        say("  初始池里没有可评测的 eval_dir，跳过 anchor（记为 null）")
        return put(g, "anchor", dict(skipped="empty_roster"))
    per = rules["anchor"]["matches_per_opponent"]
    vs = [(r["agent_id"], r["eval_dir"]) for r in roster]
    d = do_eval(g, "anchor", rules, args, vs, per * len(vs),
                args.seed_base + 3000 + g * 10000)
    return d


# ------------------------------------------------------- 6 decide / 7 pfsp / 8 report
def stage_probe(g, reg, state, rules, args):
    """专精探针：候选**和父模型**在同一批种子上各打目标对位，看配对差值。

    ⚠️ 这**不是**晋级门。晋级仍然要求"总体对 best 显著更强"；
       探针只回答一个问题：那个 0/16 的对位有没有真的变好。
    """
    plan = get(g, "plan")
    cfg = rules.get("specialist_probe") or {}
    opps = plan.get("probe_opponents") or []
    if not cfg or not opps:
        return put(g, "probe", dict(skipped="no_target"))
    n = int(cfg.get("matches", 400))
    seed = args.seed_base + 5000 + g * 10000
    st = get(g, "stage")
    vs = [(aid, reg.get(aid)["eval_dir"]) for aid in opps]
    cand = do_eval(g, "probe", dict(rules, specialist_probe=None) if False else rules,
                   args, vs, n, seed)
    base = L.read_json(os.path.join(HERE, "state", "probe_base.json")) or {}
    key = L.sha256_obj(dict(opps=[a for a, _ in vs], n=n, seed=seed))
    if base.get("key") != key:
        say("  probe 基线（父模型同协议）不存在，跑一次并缓存（以后再跑一代不用重付）")
        best = current_best(reg, state)
        cmd = [PY, os.path.join(CODE, "eval_pair.py"), "--cand", best["eval_dir"]]
        for aid, d in vs:
            cmd += ["--opp", f"{aid}={d}"]
        cmd += ["--matches", str(n), "--workers", str(args.workers),
                "--seed-base", str(seed), "--out",
                os.path.join(HERE, "state", "probe_base.json")]
        onejob = os.path.join(ROOT, "tools", "onejob.sh")
        if os.path.exists(onejob):
            cmd = [onejob, "league-eval", "--"] + cmd
        run(cmd, os.path.join(gen_dir(g), "probe_base.log"))
        base = L.read_json(os.path.join(HERE, "state", "probe_base.json")) or {}
        base["key"] = key
        L.write_json(os.path.join(HERE, "state", "probe_base.json"), base)

    # 配对比较：同一个 mid 的两局对手相同、座位相同 ⇒ 逐场做差
    pc = (cand.get("overall") or {}).get("per_match") or {}
    pb = (base.get("overall") or {}).get("per_match") or {}
    common = sorted(set(pc) & set(pb), key=int)
    if len(common) < max(10, n // 4):     # 小规模试跑（n=24）也要能算，别按死 30
        return put(g, "probe", dict(skipped="no_pairing", cand=cand.get("overall"),
                                    base=base.get("overall")))
    d = [pc[k] - pb[k] for k in common]
    m = sum(d) / len(d)
    import statistics
    sd = statistics.pstdev(d) if len(d) > 1 else 0.0
    se = sd / max(1, len(d)) ** 0.5
    lo, hi = m - 1.96 * se, m + 1.96 * se
    edge = dict(mean=round(m, 4), ci_low=round(lo, 4), ci_high=round(hi, 4),
                n=len(common), wins=int(sum(1 for x in d if x > 0)),
                losses=int(sum(1 for x in d if x < 0)),
                cand=(cand.get("overall") or {}).get("score"),
                base=(base.get("overall") or {}).get("score"))
    say(f"  专精探针（vs {plan.get('target_archetype')}，配对 {len(common)} 场）："
        f"候选 {edge['cand']:.4f} vs 父 {edge['base']:.4f} ⇒ 差 {m:+.4f} "
        f"CI [{lo:+.4f},{hi:+.4f}]（±{edge['wins']}/{edge['losses']}）")
    return put(g, "probe", dict(target=plan.get("target_archetype"), edge=edge,
                                cand=cand, base=base.get("overall"),
                                protocol=dict(matches=n, seed_base=seed)))


def stage_decide(g, reg, state, rules, args):
    plan, st = get(g, "plan"), get(g, "stage")
    sc, cf, an = get(g, "screen"), get(g, "confirm"), get(g, "anchor")
    dup_same, best = is_duplicate(reg, state, st)

    base_anchor = (best.get("eval") or {}).get("anchor")
    if base_anchor is None and an.get("overall"):
        base_anchor = an["overall"]           # best 没量过就用本轮同一套协议量出来的当基准

    # 按对手卡组切分 → 克制价值
    arb = {}
    for aid, s in (an.get("per_opp") or {}).items():
        r = reg.by_id.get(aid) or {}
        k = r.get("archetype") or aid
        c = arb.setdefault(k, dict(n=0, pts=0.0))
        c["n"] += s["n"]
        c["pts"] += s["score"] * s["n"]
    arb = {k: dict(n=v["n"], score=v["pts"] / max(1, v["n"])) for k, v in arb.items()}

    if dup_same:
        verdict = dict(verdict="reject", reason="candidate 与当前 best 是同一份权重（sha 相同）",
                       detail={"model_sha": st["model_sha"][:16]})
    else:
        verdict = L.decide(f"gen{g}", sc.get("overall"), cf.get("overall"),
                           an.get("overall"), base_anchor, rules, arb,
                           probe=get(g, "probe"))

    # 重复度：跟已有正式对手几乎一样 → 只保留，不占 best（但不直接淘汰，避免误杀）
    sim, sim_id = L.uniqueness(dict(fingerprint=list(arb.values()) and
                                    [arb[k]["score"] for k in sorted(arb)]),
                               league_rows(reg))
    verdict["duplicate_of"] = sim_id
    verdict["similarity"] = round(sim, 4)

    rec = dict(gen=g, agent_id=None, verdict=verdict["verdict"], reason=verdict["reason"],
               detail=verdict["detail"], branch=plan["branch"], deck=plan["deck"],
               parent_id=plan["parent_id"], model_sha=st["model_sha"],
               archetype_scores=arb, similarity=round(sim, 4), duplicate_of=sim_id,
               league_opponents=plan["league_opponents"], rules_sha=L.rules_sha(rules),
               at=L.now())

    if verdict["verdict"] in ("promote", "specialist"):
        aid = f"{'lg' if verdict['verdict']=='promote' else 'sp'}:{plan['branch']}"
        pool_dir = os.path.join(HERE, "pools", "league", aid.replace(":", "_"))
        if not os.path.exists(pool_dir):
            shutil.copytree(st["bundle"], pool_dir, symlinks=True)
        row = reg.add(dict(
            agent_id=aid, checkpoint_path=plan["ckpt_out"], model_version=plan["branch"],
            deck_id=os.path.splitext(os.path.basename(plan["deck"]))[0],
            deck_path=plan["deck"], source="ppo_specialist", parent_id=plan["parent_id"],
            seed=plan["seed"], steps=plan["steps"], pool="league",
            bin_path=plan.get("ckpt_out") if str(plan.get("ckpt_out", "")).endswith(".bin") else "",
            status="best" if verdict["verdict"] == "promote" else "specialist",
            eval=dict(screen=sc.get("overall"), confirm=cf.get("overall"),
                      anchor=an.get("overall")),
            traits=dict(archetype_scores=arb, fingerprint=[arb[k]["score"] for k in sorted(arb)]),
            archetype=deck_archetype(plan["deck"]),
            eval_dir=pool_dir, gen_added=g, promoted_at=L.now(),
        ))
        rec["agent_id"] = aid
        rec["eval_dir"] = pool_dir
        if verdict["verdict"] == "promote":
            old = state.current_best
            state.current_best = aid
            if old and old != aid:
                reg.add(restat(reg.by_id[old], status="opponent", demoted_at=L.now()))
            state.push_history(dict(gen=g, action="promote", agent_id=aid, prev_best=old,
                                    rules_sha=L.rules_sha(rules),
        rules_version=rules.get("version"), at=L.now()))
            say(f"  ✅ 晋级：{aid}  （新 best，旧 best {old} 降为历史对手）")
        else:
            state.push_history(dict(gen=g, action="specialist", agent_id=aid, at=L.now()))
            say(f"  ⭐ 保留为特色对手：{aid}")
        state.save()
        rec["eval_dir"] = pool_dir
    else:
        state.push_history(dict(gen=g, action="reject", branch=plan["branch"],
                                reason=verdict["reason"], at=L.now()))
        say(f"  ❌ 淘汰：{verdict['reason']}")

    # 池子上限：超了按（相似度, anchor 分）贪心淘汰
    drop = L.prune_league(league_rows(reg), rules)
    if drop and verdict["verdict"] != "promote":
        for aid in drop:
            reg.add(restat(reg.by_id[aid], status="retired", retired_at=L.now()))
        rec["pruned"] = drop
        say(f"  ✂️ 池子上限淘汰：{drop}")
    return put(g, "decide", rec)


def stage_pfsp(g, reg, state, rules, args):
    """用本轮量到的对局更新每个对手的胜率 → `w_i = p_i(1-p_i)`。"""
    plan = get(g, "plan")
    best_id = current_best(reg, state)["agent_id"]
    res = {}
    for name in ("screen", "confirm"):
        d = get(g, name) or {}
        for aid, s in (d.get("per_opp") or {}).items():
            if s["n"] >= 20 and s["errors"] == 0:
                res[aid] = s["score"]              # 训练策略对该对手的胜率
    # 池里没量过的对手给 0.5（"不确定性最大 ⇒ 该多练"）
    rows = opponent_candidates(reg)
    for r in rows:
        res.setdefault(r["agent_id"], 0.5)
    groups = {r["agent_id"]: r.get("archetype") or r["agent_id"] for r in rows}
    pf = L.update_pfsp(state, res, rules, groups)
    return put(g, "pfsp", dict(p=pf["p"], w=pf["w"], n_opponents=len(rows),
                               best=best_id, updated=pf["updated"]))


def stage_report(g, reg, state, rules, args):
    sys.path.insert(0, CODE)
    import report as R
    plan, st = get(g, "plan"), get(g, "stage")
    dec = get(g, "decide")
    pf = get(g, "pfsp")
    txt = R.render_gen(g, plan, st, {n: get(g, n) for n in ("screen", "confirm", "anchor")},
                       dec, pf, rules, state, reg, probe=get(g, "probe"))
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    p = os.path.join(HERE, "reports", f"gen_{g:04d}.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt)
    say(f"📄 报告 {p}")
    return p


# --------------------------------------------------------------------- 一代
def run_gen(g, reg, state, rules, args):
    t0 = time.perf_counter()
    state.gen = max(state.gen, g)
    state.save()
    say(f"===== 代 {g} 开始 =====")
    plan = get(g, "plan") if have(g, "plan") else stage_plan(g, reg, state, rules, args)
    ck = stage_train(g, reg, state, rules, args)
    if ck is None:
        return "awaiting_train"
    state.clear_awaiting()
    if not have(g, "stage") or args.force_stage:
        stage_stage(g, reg, state, rules, args)
    stage_screen(g, reg, state, rules, args)
    if not have(g, "probe"):
        stage_probe(g, reg, state, rules, args)   # 专精探针（没有目标对位会自行跳过）
    stage_confirm(g, reg, state, rules, args)
    stage_anchor(g, reg, state, rules, args)
    if not have(g, "decide"):
        stage_decide(g, reg, state, rules, args)
    if not have(g, "pfsp"):
        stage_pfsp(g, reg, state, rules, args)
    stage_report(g, reg, state, rules, args)
    say(f"===== 代 {g} 结束，用时 {(time.perf_counter()-t0)/60:.1f} 分钟 =====")
    return "done"


# --------------------------------------------------------------------- 命令
def cmd_init(args):
    rules = L.load_rules(HERE, args.rules)
    reg = L.Registry()
    state = L.LeagueState()
    os.makedirs(os.path.join(HERE, "pools", "init"), exist_ok=True)

    def reg_init(agent_id, ckpt, deck, source, note="", archetype=None, bin_path=None):
        deck = canonical_deck(deck)
        dst_dir = os.path.join(HERE, "pools", "init", agent_id.replace(":", "_"))
        if not os.path.exists(os.path.join(dst_dir, "model.pt")):
            cmd = [PY, os.path.join(ROOT, "tools", "stage_ckpt.py"), ckpt, "--out", dst_dir,
                   "--bundle", args.bundle, "--deck", deck, "--force"]
            if args.repo:
                cmd += ["--repo", args.repo]
            run(cmd, os.path.join(HERE, "logs", f"init_{agent_id.replace(':','_')}.log"))
        frozen = os.path.join(dst_dir, "frozen.json")
        if not os.path.exists(frozen):
            L.write_json(frozen, dict(src=os.path.abspath(ckpt),
                                      sha256=L.sha256_file(ckpt) if os.path.exists(ckpt) else None,
                                      model_pt_sha256=L.sha256_file(os.path.join(dst_dir, "model.pt")),
                                      deck_sha256=L.sha256_file(deck) if os.path.exists(deck) else None,
                                      note=note, frozen_at=L.now()))
        if agent_id in reg.by_id:
            say(f"  已冻结：{agent_id}（初始池不许覆盖，跳过写入）")
            return reg.by_id[agent_id]
        return reg.add(dict(
            agent_id=agent_id, checkpoint_path=os.path.abspath(ckpt), model_version=note or source,
            deck_id=os.path.splitext(os.path.basename(deck))[0], deck_path=os.path.abspath(deck),
            source=source, pool="init", status="opponent", eval_dir=dst_dir,
            bin_path=(bin_path or (os.path.abspath(ckpt)
                                   if str(ckpt).endswith(".bin") else "")),
            eval={}, archetype=archetype or deck_archetype(deck), traits={}))

    say("== 冻结初始策略池 ==")
    if args.bc_pool:
        for aid, ckpt, deck, note, bn in bc_brain_specs(args.bc_pool, args.bundle):
            reg_init(f"bc:{aid}", ckpt, deck, "bc", note=note,
                     bin_path=bn if os.path.exists(bn) else None)
    port_b = os.path.join(ROOT, args.bundle)
    # ⚠️ 实测：build/port_plain/model.pt 导出成 .bin 后与 port_repo/native/pt_model_field.bin
    #    **sha256 完全相同**（3e50b694…）——「出厂端口」和「通用 init」是同一份权重。
    #    所以不再单独登记一个 init:general 对手位（同权重对打只会浪费评测预算），
    #    只把这份 .bin 记在 port_plain 名下，当作 PPO 的 init 来源。
    reg_init("init:port_plain", os.path.join(port_b, "model.pt"),
             os.path.join(port_b, "deck.csv"), "init",
             note="出厂端口 = 通用 init（同一份权重，LB 1048.5 的 9d6e 卡组）",
             archetype="ours",
             bin_path=os.path.join(ROOT, "port_repo", "native", "pt_model_field.bin"))
    for spec in args.bc or []:
        if "=" not in spec:
            raise SystemExit(f"--bc 要 NAME=CKPT: {spec}")
        aid, ckpt = spec.split("=", 1)
        deck = args.deck or os.path.join(port_b, "deck.csv")
        reg_init(f"bc:{aid}", ckpt, deck, "bc", note=os.path.basename(ckpt))
    if args.general:
        deck = args.deck or os.path.join(port_b, "deck.csv")
        reg_init("init:general", args.general, deck, "init", note="通用 init（可持任意卡组）")

    best_init = args.best or "init:port_plain"
    src_row = reg.get(best_init)
    # bootstrap best 必须落在**联赛池**里：初始池的记录一旦写入就永不改状态，
    # 而 best 指针会动（第一次晋级时要把它降级为历史对手）。
    boot_id = "lg_boot:" + best_init.split(":")[-1]
    if boot_id not in reg.by_id:
        reg.add(dict(agent_id=boot_id, checkpoint_path=src_row["checkpoint_path"],
                     model_version=f"bootstrap({best_init})", deck_id=src_row["deck_id"],
                     deck_path=src_row["deck_path"], source=src_row["source"],
                     parent_id=best_init, seed=0, steps=0, pool="league", status="best",
                     eval_dir=src_row["eval_dir"], eval={}, traits={},
                     bin_path=src_row.get("bin_path") or "",
                     archetype=src_row.get("archetype") or deck_archetype(src_row["deck_path"]),
                     promoted_at=L.now()))
    state.current_best = boot_id
    state.gen = state.gen or 0
    state.save()
    L.write_json(L.rules_path(HERE), rules)
    say(f"bootstrap best = {boot_id}  （指向 {best_init}，可在联赛里被晋级顶替）")
    cmd_status(args)


def cmd_next(args):
    rules = L.load_rules(HERE, args.rules)
    reg = L.Registry()
    state = L.LeagueState()
    if not reg.rows and not args.mock_train:
        raise SystemExit("注册表是空的：先跑 `run_league.py init`")
    start = state.gen if args.resume else state.gen + 1
    if args.resume and state.awaiting:
        say(f"从断点续跑：代 {state.awaiting_gen} 卡在 {state.awaiting}")
        start = state.awaiting_gen
    end = start + args.gens - 1
    for g in range(start, end + 1):
        st = run_gen(g, reg, state, rules, args)
        state.gen = max(state.gen, g)
        state.save()
        if st != "done":
            say("停在断点，等 GPU 训练产出后再 --resume")
            return
    cmd_status(args)


def cmd_status(args):
    rules = L.load_rules(HERE, args.rules)
    reg = L.Registry()
    state = L.LeagueState()
    print(f"\n规则 v{rules['version']} sha {L.rules_sha(rules)[:12]}   代 {state.gen}"
          f"   best = {state.current_best}")
    if state.awaiting:
        print(f"⚠️ 断点：代 {state.awaiting_gen} 卡在 {state.awaiting}")
    print("\n## 初始策略池（冻结）")
    for r in init_rows(reg):
        e = (r.get("eval") or {}).get("anchor") or {}
        print(f"  {r['agent_id']:<24} {r['source']:<8} {r.get('deck_id','?'):<34} "
              f"anchor {e.get('score', float('nan')) if e else float('nan')}")
    print("\n## 正式联赛池")
    w = state.pfsp.get("w") or {}
    for r in sorted(league_rows(reg), key=lambda x: -w.get(x["agent_id"], 0)):
        e = (r.get("eval") or {})
        an = (e.get("anchor") or {}).get("score")
        cf = (e.get("confirm") or {}).get("score")
        print(f"  {r['agent_id']:<34} {r['status']:<10} w={w.get(r['agent_id'], 0):.3f} "
              f"h2h={cf if cf is not None else '-':<7} anchor={an if an is not None else '-'}")
    if rows_init := [r for r in opponent_candidates(reg) if r["pool"] == "init"]:
        print(f"\n## 可当训练 bank 的冻结初始对手：{len(rows_init)} 个（PFSP 与正式池一起抽）")
        for r in rows_init:
            print(f"  {r['agent_id']:<24} {r.get('deck_id', '?'):<36} "
                  f"w={w.get(r['agent_id'], '-'):>5}  {os.path.basename(r.get('bin_path') or '-')}")
    ev = reg.in_pool("evo")
    if ev:
        print(f"\n## 进化训练池：{len(ev)} 条分支")
        for r in ev[-8:]:
            print(f"  {r['agent_id']:<34} steps={r.get('steps')} {r['status']}")


def cmd_rollback(args):
    rules = L.load_rules(HERE, args.rules)
    reg = L.Registry()
    state = L.LeagueState()
    target = args.to
    if not target:
        acts = [h for h in state.history if h.get("action") == "promote"]
        if not acts:
            raise SystemExit("历史里没有 promote 记录，没什么可回滚")
        # 优先回到"最近一次晋级之前那个 best"（第一次晋级时 prev_best 就是 bootstrap best），
        # 只有 prev_best 缺失才退到"上一次 promote 的对象"。
        target = acts[-1].get("prev_best") or (acts[-2]["agent_id"] if len(acts) > 1 else None)
        if not target:
            raise SystemExit("最近一次 promote 没记 prev_best，且没有更早的 best")
    reg.get(target)
    old = state.current_best
    if old and old != target:
        reg.add(restat(reg.by_id[old], status="opponent", demoted_at=L.now()))
    reg.add(restat(reg.by_id[target], status="best", promoted_at=L.now()))
    state.current_best = target
    state.push_history(dict(gen=state.gen, action="rollback", agent_id=target,
                            prev_best=old, at=L.now()))
    say(f"⏪ 回滚 best：{old} → {target}")


def cmd_verify(args):
    rules = L.load_rules(HERE, args.rules)
    reg = L.Registry()
    state = L.LeagueState()
    bad = []
    print(f"规则 v{rules['version']} sha {L.rules_sha(rules)[:16]}")
    for r in reg.by_id.values():
        if r["pool"] == "init":
            for k in ("agent_id", "checkpoint_path", "deck_id", "source"):
                if not r.get(k):
                    bad.append(f"init 缺字段 {k}: {r['agent_id']}")
        if r.get("eval_dir") and not os.path.isdir(r["eval_dir"]):
            bad.append(f"eval_dir 不存在: {r['agent_id']} → {r['eval_dir']}")
    bests = [r for r in reg.by_id.values() if r["pool"] == "league" and r["status"] == "best"]
    if len(bests) > 1:
        bad.append(f"联赛池有 {len(bests)} 个 best：{[b['agent_id'] for b in bests]}")
    if state.current_best and state.current_best not in reg.by_id:
        bad.append(f"state.current_best 没登记: {state.current_best}")
    n = len(league_rows(reg))
    if n > rules["pool"]["max_league"]:
        bad.append(f"联赛池 {n} > 上限 {rules['pool']['max_league']}")
    # 规则漂移检测：**同一版本号**下出现两种 sha 才算异常（升版是允许的，且必须留痕）
    by_ver = {}
    for p in sorted(glob.glob(os.path.join(HERE, "gen", "*", "plan.json")))[-10:]:
        d = L.read_json(p) or {}
        if d.get("rules_sha"):
            by_ver.setdefault(d.get("rules_version"), set()).add(d["rules_sha"])
    for ver, shas in by_ver.items():
        if len(shas) > 1:
            bad.append(f"规则版本 v{ver} 下出现 {len(shas)} 种 sha —— 同版本号下改过规则，必须升 version")
    vers = sorted(v for v in by_ver if v is not None)
    if len(vers) > 1:
        print(f"  规则升过版：{vers}（各代记录各自的 sha，可追溯）")
    if vers and vers[-1] != rules.get("version"):
        print(f"  ⚠️ 最近一次计划用的是 v{vers[-1]}，当前 rules.json 是 "
              f"v{rules.get('version')}（下一代会用新的）")
    print(f"池：init {len(init_rows(reg))} / evo {len(reg.in_pool('evo'))} / "
          f"league {n}（上限 {rules['pool']['max_league']}）")
    print("\n".join("  ✗ " + b for b in bad) if bad else "  ✓ 自检通过")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--bundle", default="build/port_plain")
        p.add_argument("--repo", default=None)
        p.add_argument("--deck", default=None)
        p.add_argument("--train-cmd", default=None, help="训练模板（{init} {steps} {out} …）")
        p.add_argument("--mock-train", nargs="?", const="parent", default=None, metavar="CKPT",
                       help="不训练，直接拿某份权重当候选（只验证流程）。省略值=用父模型")
        p.add_argument("--workers", type=int, default=6)
        p.add_argument("--seed", type=int, default=20261016)
        p.add_argument("--seed-base", type=int, default=900000)
        p.add_argument("--banks", type=int, default=3, help="每代上几个对手 bank（2–4）")
        p.add_argument("--lr", type=float, default=None,
                       help="覆盖本代学习率（默认 rules.train.lr=5e-4）。"
                            "专精代建议 2e-4：步数更多但别偏离端口太远")
        p.add_argument("--target", default=None,
                       help="专精目标原型（如 shuppet）：自动用 decks/weights_<target>.txt "
                            "抬该原型的对手采样概率，并对它做配对探针")
        p.add_argument("--deck-weights", default=None, help="直接指定对手座位权重文件")
        p.add_argument("--matches", type=int, default=None,
                       help="单代训练局数（默认 rules.train.matches_per_gen=2048；"
                            "1 局≈170 步。2048≈348k 步≈3 分钟 rollout，8192≈1.4M≈12 分钟）")
        p.add_argument("--deck-cycle", action="store_true", help="按已有卡组轮换挑专精卡组")
        p.add_argument("--force-eval", action="store_true")
        p.add_argument("--force-stage", action="store_true")
        p.add_argument("--gens", type=int, default=1)
        p.add_argument("--rules", default=None,
                       help="规则文件（默认 rules.json；小规模试跑用 rules_pilot.json）")
        p.add_argument("--resume", action="store_true")

    p = sub.add_parser("init")
    common(p)
    p.add_argument("--bc", action="append", help="NAME=CKPT，可重复")
    p.add_argument("--general", default=None, help="通用 init 的 ckpt")
    p.add_argument("--bc-pool", default=None,
                   help="把 bc_pool/registry.tsv 里的脑子全部冻结进初始池（推荐）")
    p.add_argument("--best", default=None)
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("next")
    common(p)
    p.set_defaults(fn=cmd_next)
    p = sub.add_parser("status")
    common(p)
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("verify")
    common(p)
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("rollback")
    common(p)
    p.add_argument("--to", default=None)
    p.set_defaults(fn=cmd_rollback)

    args = ap.parse_args()
    rc = args.fn(args)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
