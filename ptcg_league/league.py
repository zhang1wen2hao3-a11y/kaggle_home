#!/usr/bin/env python3
"""**训练联赛核心库**：注册表、状态、PFSP 采样、晋级判定。纯逻辑，不做 I/O 重的活。

这一层刻意不碰引擎、不碰 GPU、不开子进程，所以可以被单元测试全覆盖。
调度器 `run_league.py` 只负责"按顺序调下面的函数"。

## 三个池子（互不覆盖）

| 池 | 谁在里面 | 允许被改吗 |
|---|---|---|
| `init`   | 冻结的 BC 模型 + 通用 init 副本 | **永不**。原始权重、训练配置、卡组一起存档 |
| `evo`    | 各条 PPO 专项分支的检查点 | 只追加，不覆盖同名 |
| `league` | 通过评测的正式对手 | 只追加；`best` 指针可移动，旧 best 留作历史对手 |

登记表里的模型**不要求总体胜率相同**：有的当强基准，有的专门克制某类卡组。

## 最重要的原则

> 训练可以持续自动进行，**但晋级必须由固定、可复现的评测规则决定**。

所以规则全部写死在 `rules.json` 里，每代把它的 sha256 记进代记录；
判定函数 `decide()` 只读规则、不读"这次看起来还行"。
"""
import hashlib
import json
import math
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# `PTCG_LEAGUE_HOME` 只用于**集成自测**（在 /tmp 里跑一整套 init→晋级→回滚，
# 不碰正式 state）。正式运行永远不设它。
HERE = os.environ.get("PTCG_LEAGUE_HOME") or os.path.join(ROOT, "ptcg_league")

POOLS = ("init", "evo", "league")
STATUSES = ("candidate", "opponent", "best", "specialist", "retired")
SOURCES = ("bc", "init", "ppo_general", "ppo_specialist")

DEFAULT_RULES = {
    "_comment": "晋级规则。改这里 = 改比赛规则，必须在报告里留痕（每代记录 rules_sha256）。",
    "version": 1,
    "unit": "bo3_match",          # 一个计量单位 = 一场 Bo3，胜 1 / 平 0.5 / 负 0
    "screen": {                   # 第一道：快速筛掉明显退化的
        "matches": 400,
        "min_score": 0.52,
        "require_zero_errors": True,
    },
    "confirm": {                  # 第二道：对潜在提升做可靠确认
        "matches": 1600,
        "h2h_ci_low_above": 0.50,   # 候选对当前 best 的配对胜率 CI 下界必须 > 0.50
        "require_zero_errors": True,
    },
    "anchor": {                   # 固定评测集：对冻结 roster 的绝对分，防"只会打自家人"
        "matches_per_opponent": 100,
        "max_regress": 0.02,        # 相对当前 best 的 anchor 分允许的下滑
    },
    "specialist": {               # 总体不如 best，但对某类卡组有独特价值
        "h2h_ci_high_below": 0.50,
        "min_archetype_n": 200,
        "min_archetype_score": 0.58,
    },
    "pfsp": {
        "mode": "p(1-p)",
        "floor_quota": 0.05,        # 纯克制对手（胜率 5%）不许被 PFSP 饿死
        "max_single": 0.35,         # 也不许单个对手吃掉预算
        "two_stage": True,          # 先抽 archetype 再抽该 archetype 下的具体对手
    },
    "train": {                      # 一代 PPO 分支的固定配置
        "matches_per_gen": 2048,    # 单轮 2,048 局完整对局
        "steps_per_match": 170,     # 1 局 ≈ 170 决策（实测 ep_len 176→123 的均值附近）
        "total_agents": 512,
        "minibatch": 512,
        "lr": 0.0005,
        "checkpoint_interval": 25,
        "deck_pool": "decks/pool_field",
        "deck_weights_file": "decks/weights_real.txt",
    },
    "banks": {                      # 每代同时上几个对手 bank（不是池子大小）
        "per_gen": 3,
        "max": 4,
        "_why": "4090 24GB 实测：512 agents 无 bank 15.8GB，+2 bank = 18.2GB（≈1.0GB/bank）；"
                "且训练器断言 league_pct×n_banks<0.5。所以池子 8–16 个、每代只抽 2–4 个轮换。",
    },
    "pool": {"max_league": 16, "min_league": 8, "ckpt_every_gens": 3},
}


# --------------------------------------------------------------------- 工具
def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)


def wilson(wins, n, z=1.959963984540054):
    """Wilson 区间 —— **不自己写第二份**，直接从 `tools/bo3_eval.py` 取，避免漂移。"""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (center - radius, center + radius)


# --------------------------------------------------------------------- 规则
def rules_path(root=None):
    return os.path.join(root or HERE, "rules.json")


def load_rules(root=None, path=None):
    """读规则。`path` 给了就用它（小规模试跑用 `rules_pilot.json`）。

    ⚠️ 换规则文件 = 换比赛规则。代记录里存 `rules_sha`，所以"用 pilot 规则跑出来的
       晋级"在报告里一眼能认出来，不会被当成正式规则下的成绩。
    """
    p = os.path.abspath(path) if path else rules_path(root)
    r = read_json(p)
    if r is None:
        if path:
            raise SystemExit(f"规则文件不存在：{p}")
        write_json(p, DEFAULT_RULES)
        r = dict(DEFAULT_RULES)
    # 缺的键用默认补（老规则文件 + 新加键不至于炸）
    for k, v in DEFAULT_RULES.items():
        if k not in r:
            r[k] = v
        elif isinstance(v, dict):
            for kk, vv in v.items():
                r[k].setdefault(kk, vv)
    return r


def rules_sha(rules):
    return sha256_obj(rules)


# --------------------------------------------------------------------- 注册表
class Registry:
    """追加式登记表（`state/registry.jsonl`）。

    一行一个模型版本，**只追加**。同名 id 不会覆盖：`rev` 递增，
    `pool=init` 的记录一旦写入就再也不会被改（`guard_init()` 会拦）。
    """

    def __init__(self, path=None):
        self.path = path or os.path.join(HERE, "state", "registry.jsonl")
        self.by_id = {}
        self.rows = []
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    self.rows.append(r)
                    self.by_id[r["agent_id"]] = r          # 后写入的 rev 覆盖索引

    # -- 写 -------------------------------------------------------------
    def add(self, rec):
        need = ("agent_id", "checkpoint_path", "deck_id", "source", "pool", "status")
        for k in need:
            if k not in rec:
                raise ValueError(f"登记记录缺字段 {k}: {rec.get('agent_id')}")
        if rec["source"] not in SOURCES:
            raise ValueError(f"source 非法 {rec['source']}")
        if rec["pool"] not in POOLS:
            raise ValueError(f"pool 非法 {rec['pool']}")
        if rec["pool"] == "init" and rec["agent_id"] in self.by_id:
            raise RuntimeError(f"初始策略池不许覆盖：{rec['agent_id']}")
        old = self.by_id.get(rec["agent_id"])
        rec = dict(rec)
        rec.setdefault("parent_id", None)
        rec.setdefault("seed", 0)
        rec.setdefault("steps", 0)
        rec.setdefault("model_version", rec["source"])
        rec.setdefault("eval", {})
        rec.setdefault("traits", {})
        rec.setdefault("created", now())
        rec["rev"] = (old["rev"] + 1) if old else 1
        n = sum(1 for r in self.rows if r["agent_id"] == rec["agent_id"])
        rec["gen_added"] = rec.get("gen_added", n)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        self.rows.append(rec)
        self.by_id[rec["agent_id"]] = rec
        return rec

    # -- 读 -------------------------------------------------------------
    def get(self, agent_id):
        if agent_id not in self.by_id:
            raise KeyError(f"没登记过的 agent：{agent_id}")
        return self.by_id[agent_id]

    def in_pool(self, pool, status=None):
        return [r for r in self.by_id.values()
                if r["pool"] == pool and (status is None or r["status"] == status)]

    def best(self, pool="league"):
        """`status == "best"` 的那个（同池内唯一性由 `set_best()` 保证）。"""
        c = [r for r in self.by_id.values() if r["pool"] == pool and r["status"] == "best"]
        if len(c) > 1:
            c.sort(key=lambda r: r.get("promoted_at", ""))
        return c[-1] if c else None


# --------------------------------------------------------------------- 状态
class LeagueState:
    """`state/league_state.json`：代计数器、当前 best、待办断点、PFSP 统计。

    断点语义：`awaiting` 是"上一代卡在哪一步"。调度器每次启动先读它，
    能续跑就续跑（`--resume`），不会因为一次 GPU OOM 丢掉整代。
    """

    FIELDS = dict(gen=0, current_best=None, awaiting=None, awaiting_gen=None,
                  pfsp={}, history=[], shutdown=None)

    def __init__(self, path=None):
        self.path = path or os.path.join(HERE, "state", "league_state.json")
        d = read_json(self.path, None) or {}
        for k, v in self.FIELDS.items():
            setattr(self, k, d.get(k, v if not isinstance(v, (dict, list)) else type(v)()))
        if d:
            self.gen = d.get("gen", 0)
            self.current_best = d.get("current_best")
            self.awaiting = d.get("awaiting")
            self.awaiting_gen = d.get("awaiting_gen")
            self.pfsp = d.get("pfsp", {}) or {}
            self.history = d.get("history", []) or []
            self.shutdown = d.get("shutdown")

    def dump(self):
        return dict(gen=self.gen, current_best=self.current_best, awaiting=self.awaiting,
                    awaiting_gen=self.awaiting_gen, pfsp=self.pfsp,
                    history=self.history, shutdown=self.shutdown, saved=now())

    def save(self):
        write_json(self.path, self.dump())

    # -- 断点 -----------------------------------------------------------
    def set_awaiting(self, stage, gen=None):
        self.awaiting, self.awaiting_gen = stage, gen if gen is not None else self.gen
        self.save()

    def clear_awaiting(self):
        self.awaiting, self.awaiting_gen = None, None
        self.save()

    def push_history(self, rec):
        self.history.append(rec)
        self.save()


# --------------------------------------------------------------------- PFSP
def pfsp_weights(win_rates, rules=None, groups=None):
    """`w_i = p_i (1 - p_i)`，再归一化，带地板配额与单头上限。

    为什么不能只用 `p(1-p)`：胜率 5% 的纯克制对手权重只有 0.0475，
    在 16 个对手里会被挤到几乎不采样 —— 而它恰好是最该练的那一类。
    所以：先按 `floor_quota` 给每个对手保底，再按 `max_single` 削峰，最后归一化。

    `groups`: {agent_id: archetype}。给了就按两阶段（先抽类别再抽个体）算：
    类别权重 = 该类下个体 `p(1-p)` 的最大值（而不是求和）——
    否则"卡组多的大类"会因为人头多而自动吃满预算。
    """
    rules = rules or load_rules()
    cfg = rules.get("pfsp", DEFAULT_RULES["pfsp"])
    if not win_rates:
        return {}
    raw = {k: max(0.0, min(1.0, float(v))) * (1.0 - max(0.0, min(1.0, float(v))))
           for k, v in win_rates.items()}
    floor = float(cfg.get("floor_quota", 0.0))
    cap = float(cfg.get("max_single", 1.0))

    w = dict(raw)
    if groups:
        by_g = {}
        for k, v in raw.items():
            by_g.setdefault(groups.get(k, "?"), []).append((k, v))
        # 类别权重取组内最大，再平分回组内（组内仍按 p(1-p) 比例）
        for g, items in by_g.items():
            tot = sum(v for _, v in items)
            gmax = max(v for _, v in items)
            for k, v in items:
                w[k] = gmax * (v / tot if tot > 0 else 1.0 / len(items))

    n = len(w)
    s = sum(w.values())
    if s <= 0:
        w = {k: 1.0 / n for k in w}
    else:
        w = {k: v / s for k, v in w.items()}

    # 地板：低于 floor 的抬到 floor（总量不够就统一缩放）
    if floor > 0:
        if floor * n > 1.0:
            floor = 1.0 / n
        w = {k: max(v, floor) for k, v in w.items()}
    # 削峰 + 再把多出来的按比例还给别人，最多迭代 8 轮（收敛很快）
    for _ in range(8):
        s = sum(w.values())
        w = {k: v / s for k, v in w.items()}
        over = {k: v for k, v in w.items() if v > cap + 1e-12}
        if not over:
            break
        excess = sum(v - cap for v in over.values())
        for k in over:
            w[k] = cap
        rest = {k: v for k, v in w.items() if k not in over}
        rs = sum(rest.values())
        if rs <= 0:
            break
        for k in rest:
            w[k] += excess * rest[k] / rs
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def update_pfsp(state, results, rules=None, groups=None):
    """`results`: {agent_id: win_rate}，直接覆盖同名项（用最近一次测量）。"""
    p = dict(state.pfsp.get("p", {}))
    p.update({k: float(v) for k, v in results.items()})
    w = pfsp_weights(p, rules, groups)
    state.pfsp = dict(p=p, w=w, updated=now(),
                      rule=(rules or load_rules()).get("pfsp", {}).get("mode", "p(1-p)"))
    state.save()
    return state.pfsp


def sample_opponents(weights, k, rng=None, groups=None):
    """按权重不放回抽 k 个。给了 `groups` 就走两阶段：先抽类别，再抽该类个体。"""
    import random
    rng = rng or random.Random()
    pool = {i: w for i, w in weights.items() if w > 0}
    picks = []
    if groups and len(pool) > k:
        by_g = {}
        for i in pool:
            by_g.setdefault(groups.get(i, "?"), []).append(i)
        gmax = {}
        for g, items in by_g.items():
            gmax[g] = max(pool[i] for i in items)
        gs = sum(gmax.values())
        gkey = list(gmax)
        gw = [gmax[g] / gs for g in gkey]
        guard = 0
        while len(picks) < k and guard < k * 40:
            guard += 1
            g = rng.choices(gkey, weights=gw, k=1)[0]
            cand = [i for i in by_g[g] if i not in picks]
            if not cand:
                gkey.remove(g)
                if not gkey:
                    break
                tot = sum(gmax[gg] for gg in gkey)
                gw = [gmax[gg] / tot for gg in gkey]
                continue
            cw = [pool[i] for i in cand]
            picks.append(rng.choices(cand, weights=cw, k=1)[0])
            # ★ 抽过的类别先撤下来：否则同一个原型会占掉多个 bank 名额
            #   （本池里 shuppet 有两个脑子，实测随机抽 12.9% 会重复占坑）。
            #   类别抽完了（k > 类别数）才允许重复。
            gkey.remove(g)
            if not gkey:
                break
            tot = sum(gmax[gg] for gg in gkey)
            gw = [gmax[gg] / tot for gg in gkey]
    if len(picks) < k:
        left = [i for i in pool if i not in picks]
        lw = [pool[i] for i in left]
        while left and len(picks) < k:
            j = rng.choices(range(len(left)), weights=lw, k=1)[0]
            picks.append(left.pop(j))
            lw.pop(j)
    return picks


# --------------------------------------------------------------------- 判定
def _stat(d):
    """归一化一份评测结果 dict → (score, ci_low, ci_high, n, errors)。"""
    if not d:
        return None
    n = int(d.get("n") or d.get("matches") or 0)
    score = float(d.get("score", 0.0))
    lo = d.get("ci_low")
    hi = d.get("ci_high")
    if lo is None or hi is None:
        lo, hi = wilson(int(round(score * n)), n)
    return dict(score=score, ci_low=float(lo), ci_high=float(hi), n=n,
                errors=int(d.get("errors", 0)))


def decide(cand_id, screen, confirm, anchor, anchor_base, rules=None,
           archetype_scores=None, probe=None):
    """**唯一的晋级判定入口**。返回 `{verdict, reason, detail}`。

    verdict:
      - `promote`     → 进正式联赛池，并成为新的 best
      - `specialist`  → 进正式联赛池，但不是 best（克制价值）
      - `reject`      → 淘汰

    顺序刻意固定：先看错误（有异常/兜底的数字不算数），再看筛查，
    再看统计显著性，最后看固定评测集有没有退化。
    """
    rules = rules or load_rules()
    sc, cf, an = rules["screen"], rules["confirm"], rules["anchor"]
    sp = rules["specialist"]
    detail = {}

    def fail(reason):
        return dict(verdict="reject", reason=reason, detail=detail)

    s = _stat(screen)
    if s is None:
        return fail("no_screen")
    detail["screen"] = s
    if sc.get("require_zero_errors") and s["errors"] > 0:
        return fail(f"screen_errors={s['errors']}（异常/兜底，数字不算数）")
    if s["n"] < sc["matches"]:
        return fail(f"screen_underpowered n={s['n']}<{sc['matches']}")
    if s["score"] < sc["min_score"]:
        # 总体没赢，但专精对位可能真的修好了 —— 走 specialist 分支（不占 best，进池当对手）
        sp_cfg = rules.get("specialist_probe") or {}
        e = (probe or {}).get("edge") if probe else None
        if e and sp_cfg:
            detail["probe"] = e
            if e["mean"] >= sp_cfg.get("min_edge", 0.05) and e["ci_low"] > 0:
                return dict(verdict="specialist",
                            reason=(f"总体 {s['score']:.4f}<{sc['min_score']}，但"
                                    f"{probe.get('target')} 对位配对提升 {e['mean']:+.4f} "
                                    f"CI 下界 {e['ci_low']:+.4f}>0（n={e['n']}，"
                                    f"±{e['wins']}/{e['losses']}）"),
                            detail=detail)
        return fail(f"screen_score={s['score']:.4f}<{sc['min_score']}"
                    + (f"；专精探针差值 {e['mean']:+.4f} CI[{e['ci_low']:+.4f},"
                       f"{e['ci_high']:+.4f}] 不足以声称改善" if e else ""))

    c = _stat(confirm)
    if c is None:
        return fail("no_confirm")
    detail["confirm"] = c
    if cf.get("require_zero_errors") and c["errors"] > 0:
        return fail(f"confirm_errors={c['errors']}")
    if c["n"] < cf["matches"] * 0.98:
        return fail(f"confirm_underpowered n={c['n']}<{cf['matches']}")

    # 固定评测集（对冻结 roster 的绝对分）：防"只会打自家人"
    a = _stat(anchor)
    ab = _stat(anchor_base) or dict(score=0.0)
    detail["anchor"] = a
    detail["anchor_base"] = ab
    if a is None:
        return fail("no_anchor")
    regress = ab["score"] - a["score"]
    detail["anchor_regress"] = regress
    if regress > an["max_regress"] + 1e-9:
        return fail(f"anchor_regress={regress:+.4f}>{an['max_regress']}")

    # 统计显著吗？
    if c["ci_low"] > cf["h2h_ci_low_above"]:
        return dict(verdict="promote",
                    reason=f"h2h_ci_low={c['ci_low']:.4f}>{cf['h2h_ci_low_above']} "
                           f"且 anchor_regress={regress:+.4f}",
                    detail=detail)

    # 不显著 → 有没有克制价值？
    if c["ci_high"] < sp["h2h_ci_high_below"] and archetype_scores:
        good = {k: v for k, v in archetype_scores.items()
                if (v.get("n") or 0) >= sp["min_archetype_n"]
                and (v.get("score") or 0) >= sp["min_archetype_score"]}
        detail["specialist_edges"] = good
        if good:
            return dict(verdict="specialist",
                        reason=f"总体不显著（ci_high={c['ci_high']:.4f}）但 "
                               f"{len(good)} 类 ≥{sp['min_archetype_score']:.0%}",
                        detail=detail)

    if c["ci_high"] < sp["h2h_ci_high_below"]:
        return fail(f"h2h_ci_high={c['ci_high']:.4f}<0.50（明确更差）")
    return fail(f"h2h_ci=[{c['ci_low']:.4f},{c['ci_high']:.4f}] 跨 0.50，"
                f"不能宣称提升（n={c['n']} 的 CI 半宽约 ±{1.96*0.5/math.sqrt(c['n'])/2*2:.3f}）")


def uniqueness(cand_traits, others, rules=None):
    """对手池重复度检查：跟已有正式对手"几乎一样"的候选不该占预算。

    `traits`: {"fingerprint": [...每类卡组胜率...]}。用 L1 距离，太近就判重复。
    返回 (max_similarity, most_similar_id)。
    """
    f = (cand_traits or {}).get("fingerprint")
    if not f:
        return (0.0, None)
    best_s, best_id = 0.0, None
    for o in others:
        g = (o.get("traits") or {}).get("fingerprint")
        if not g or len(g) != len(f):
            continue
        d = sum(abs(a - b) for a, b in zip(f, g)) / len(f)     # 平均绝对差
        sim = max(0.0, 1.0 - 2.0 * d)                          # 差 0.5 就完全不同
        if sim > best_s:
            best_s, best_id = sim, o["agent_id"]
    return (best_s, best_id)


def prune_league(rows, rules=None, keep=("best",)):
    """池超过上限时按（实力、特征差异、卡组覆盖）淘汰。

    规则：best 永不淘汰；其余按 `traits.fingerprint` 贪心挑"最冗余"的先走，
    再按 anchor 分从低到高。保证留下的多样性最大。
    """
    rules = rules or load_rules()
    mx = rules["pool"]["max_league"]
    if len(rows) <= mx:
        return []
    keep_ids = {r["agent_id"] for r in rows if r["status"] in keep}
    cands = [r for r in rows if r["agent_id"] not in keep_ids]
    # 贪心：每轮删掉"跟别人最像且 anchor 最低"的那个
    out = []
    # 削到**上限**就停（min_league 是"池子太小就没多样性了"的下限，不是削枝目标）
    while len(rows) - len(out) > rules["pool"]["max_league"] and cands:
        worst, worst_key = None, None
        for c in cands:
            sim, _ = uniqueness(c.get("traits"), [x for x in cands if x is not c])
            a = ((c.get("eval") or {}).get("anchor") or {}).get("score", 0.0)
            key = (sim, -a)          # 越像越该删，越弱越该删
            if worst_key is None or key > worst_key:
                worst, worst_key = c, key
        out.append(worst["agent_id"])
        cands.remove(worst)
    return out


# --------------------------------------------------------------------- 自检
if __name__ == "__main__":
    import random
    rules = load_rules()
    print("rules version", rules["version"], "sha", rules_sha(rules)[:12])
    d = dict(a=0.50, b=0.05, c=0.95, d=0.30, e=0.70)
    w = pfsp_weights(d, rules)
    print("pfsp", {k: round(v, 3) for k, v in w.items()}, "sum", round(sum(w.values()), 6))
    w2 = pfsp_weights(d, rules, groups=dict(a="x", b="y", c="y", d="z", e="z"))
    print("pfsp2", {k: round(v, 3) for k, v in w2.items()}, "sum", round(sum(w2.values()), 6))
    rng = random.Random(0)
    print("pick", sample_opponents(w, 3, rng))
    # 判定自检
    base = dict(score=0.55, n=1600, ci_low=0.525, ci_high=0.575, errors=0)
    print("promote?", decide("c", dict(score=0.56, n=400, ci_low=0.51, ci_high=0.61, errors=0),
                             base, dict(score=0.56, n=500), dict(score=0.55))["verdict"])
    print("reject? ", decide("c", dict(score=0.56, n=400, ci_low=0.51, ci_high=0.61, errors=0),
                             dict(score=0.50, n=1600, ci_low=0.475, ci_high=0.525, errors=0),
                             dict(score=0.56, n=500), dict(score=0.55))["reason"][:60])
