#!/usr/bin/env python3
"""生成**反事实** Q 标签数据集 —— 这是训练 Action 网络的唯一途径。

## 为什么必须要有反事实

之前三次训练 Q/Action 都失败了，原因不是网络结构，而是**数据里没有反事实**：

    greedy 是确定性策略  ⇒  a = f(s)
    每个局面只有一条轨迹、只观测到一个动作
    ⇒ P(win | s, a) == P(win | s)      动作不携带任何额外信息
    ⇒ 实测选项特征重要性仅 0.38% / 0.48% / 0.59%，Q 退化成 V

本脚本用引擎**主动构造反事实**：对同一局面，把每个候选动作都打到底。

## 三个正交维度（别混为一谈）

| 维度 | 本脚本的做法 | 为什么 |
|---|---|---|
| **卡组** | `--deck-dir` 两两轮转（**不含镜像**） | 只有跨卡组才能学到"对手不是我"；镜像里"对手用我的卡组"这个假设永远成立，学不到任何对手推断能力 |
| **策略** | `--state-policy greedy\\|rollout` | 状态分布必须贴近**部署策略**，否则 Q 在真实决策点上是 off-policy 外推（V 搜索当年就死在这） |
| **卡组顺序 × 座位** | 四路平衡 | 消除"某副牌永远坐玩家0"的结构偏置 |

## 做法

对每局，用 `--state-policy` 生成对局轨迹；在途中的目标决策点做反事实标注：

    候选 = [状态策略实际的选择, ATTACK(若有), END(若有)]
    对每个 world（determinization 采样，全部候选共享 = CRN）：
        search_begin(world)
        对每个候选 a:  search_step(a) → 用规则策略打到终局 → 胜负
    每个候选的标签 = 该动作的平均胜率

> **注意**：候选里的第 0 个是"状态策略的选择"，不一定是 greedy 的选择。
> 字段名保留 `greedy_idx` 表示"基线动作"，因为它是策略改进的参照点。

## 输出格式（一行 = 一个局面，含多个候选动作）

    {
      "game": 0, "matchup": "sample_official_vs_fire_gouging",
      "step": 3, "turn": 2, "player": 0,
      "state_policy": "greedy",
      "features": {...63 维状态特征...},
      "options":   [{...32 维选项特征...}, ...],   # 仅候选动作
      "labels":    [0.70, 0.30, 0.50],             # 每个候选的 rollout 平均胜率
      "ns":        [10, 10, 10],                   # 每个候选的有效 world 数
      "base_idx":  0,                              # 状态策略实际选的下标
      "action_types": ["PLAY", "ATTACK", "END"],
      "result": 0
    }

同一局面的多个动作放在**同一行**里，训练时既能做回归，也能做同局面内的成对比较，
不会把"局面难度"混进动作信号。

## 成本

成本约 **O(决策数²)**：局面越靠后，每次 rollout 还要打越久。
`--max-per-game` 用来限制每局标注多少个决策点（默认 20）。

参考：greedy 自对弈 + 20 个标注点 ≈ 2~5 s/局（取决于卡组）。

用法：
    # 冒烟
    ./run.sh gen_counterfactual.py --deck-dir decks --games-per-pair 1 \
        --worlds 4 --workers 4 --out data/cf_smoke.jsonl

    # 层 A：广覆盖（greedy 状态策略，便宜）
    ./run.sh gen_counterfactual.py --deck-dir decks --games-per-pair 2 \
        --state-policy greedy --worlds 8 --max-per-game 20 --workers 8 \
        --out data/cf_greedy_v1.jsonl

    # 层 B：对齐部署（rollout 状态策略，贵）
    ./run.sh gen_counterfactual.py --deck-dir decks --games-per-pair 1 \
        --state-policy rollout --state-worlds 6 --worlds 8 --max-per-game 20 \
        --workers 8 --out data/cf_rollout_v1.jsonl
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cg.api import OptionType  # noqa: E402
from cg.game import battle_start, battle_select  # noqa: E402
from framework import features as F  # noqa: E402
from framework.filler import make_filler  # noqa: E402
from framework.memory import GameMemory  # noqa: E402
from framework.rollout import (  # noqa: E402
    choose_action, evaluate_actions, has_type, index_of_type, is_main,
    rule_action, _stats,
)
from local_match import load_deck  # noqa: E402

# ============================================================ worker
_W: dict = {}


def _worker_init(worlds: int, state_policy: str, state_worlds: float,
                 max_per_game: int, trigger: str, schedule: str,
                 quotas) -> None:
    _W["worlds"] = worlds
    _W["state_policy"] = state_policy
    _W["state_worlds"] = state_worlds
    _W["max_per_game"] = max_per_game
    _W["trigger"] = trigger
    _W["schedule"] = schedule
    _W["quotas"] = quotas
    _W["decks"] = {}


def _deck(path: str):
    d = _W["decks"]
    if path not in d:
        d[path] = load_deck(path)
    return d[path]


def _candidates(sel, base):
    """候选动作下标：状态策略的选择 + ATTACK(若有) + END(若有)。"""
    cands = [base]
    for want in (OptionType.ATTACK, OptionType.END):
        i = index_of_type(sel, want)
        if i is not None and i not in cands:
            cands.append(i)
    return cands


def _is_target(sel, trigger: str) -> bool:
    if not is_main(sel):
        return False
    if trigger == "attack":
        return has_type(sel, OptionType.ATTACK)
    return len(sel.get("option") or []) >= 2      # trigger == "main"


def _state_action(obs, sel, my_deck, filler, mem):
    """状态生成策略：返回选项下标列表。"""
    if _W["state_policy"] == "rollout":
        act, _info = choose_action(
            obs, filler=filler, my_deck=my_deck,
            n_worlds=_W["state_worlds"], margin=1.0, memory=mem,
        )
        return act
    return rule_action(sel)


def _select_indices(n: int, max_per: int, schedule: str, quotas) -> list[int]:
    """从 n 个快照里挑出要标注的下标。

    为什么需要这一步：**打局的过程中并不知道这局会跑多长**，
    所以没法中途决定"该采开局还是末期"。先打完整局、存下所有目标局面，
    事后按已知的进度来挑，才不会像早期版本那样被硬上限截断在前半段
    （实测 max_per=20 时一行都没采到 turn 20 以后）。

    实测依据（800 行、40 局，按进度分箱）：
        进度 0~20%   有差异 53.5%  单标签成本高（剩余步数多）
        进度 80~100% 有差异 42.6%  单标签成本低（剩余步数少）
    信息量只差 1.2 倍，但成本差约 9 倍 => 单位算力产出后期高约 7 倍。
    所以默认 strat（分层 + 配额偏后期），同时保留少量开局样本保证覆盖。
    """
    if n <= max_per:
        return list(range(n))
    if schedule == "late":
        return list(range(n - max_per, n))
    if schedule == "uniform":
        if max_per == 1:
            return [n // 2]
        return sorted({round(i * (n - 1) / (max_per - 1)) for i in range(max_per)})

    # ---- strat：按进度切三段，段内均匀取
    w = quotas if (len(quotas) == 3 and sum(quotas) > 0) else (25, 30, 45)
    tot = sum(w)
    bounds = [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]
    picks: set[int] = set()
    for (lo, hi), x in zip(bounds, w):
        cn = hi - lo
        if cn <= 0:
            continue
        k = min(max(1, int(round(max_per * x / tot))), cn)
        if k == 1:
            picks.add(lo + cn // 2)
        else:
            picks.update(lo + round(i * (cn - 1) / (k - 1)) for i in range(k))
    out = sorted(picks)
    if len(out) > max_per:                    # 配额溢出时均匀抽稀
        step = len(out) / max_per
        out = sorted({out[min(int(i * step), len(out) - 1)] for i in range(max_per)})
    return out[:max_per]


def _run_game(job: dict) -> list[dict]:
    """两阶段：① 打完整局并快照所有目标局面 ② 事后挑选并标注。

    快照必须是 (obs, GameMemory) 成对保存 —— 特征里的 16 个 mem_* 来自记忆的
    累积状态，若末期标注时用局末的记忆，早期局面的特征就错了。
    """
    gid = job["gid"]
    my_deck = _deck(job["deck0"])
    op_deck = _deck(job["deck1"])
    worlds, trigger = _W["worlds"], _W["trigger"]
    max_per = _W["max_per_game"]
    sp = _W["state_policy"]

    # filler 的对手先验用【真实】对手卡组：要的是准确的标签，
    # 而 Q(s,a) 推理时只吃可观测特征，让 world 更真实只会让标签更准。
    filler = make_filler("uniform", my_deck=my_deck,
                         opponent_deck_prior=op_deck, seed=gid)

    obs = battle_start(my_deck, op_deck, reverse_player=job["reverse"])[0]
    if obs is None:
        return []

    mem = GameMemory()
    snaps: list[tuple] = []          # [(step, turn, obs_copy, mem_copy)]
    result = -1
    try:
        # ---------------- 阶段 1：打完整局
        for step in range(600):
            cur, sel = obs.get("current"), obs.get("select")
            if cur is None or sel is None or cur.get("result", -1) != -1:
                break
            mem.observe(obs)
            if _is_target(sel, trigger):
                base = rule_action(sel)[0]
                if len(_candidates(sel, base)) >= 2:
                    snaps.append((step, int(cur.get("turn", 0)),
                                  copy.deepcopy(obs), copy.deepcopy(mem)))
            obs = battle_select(_state_action(obs, sel, my_deck, filler, mem))
            if obs is None:
                break
        result = (obs.get("current") or {}).get("result", -1) if obs else -1
    except Exception:
        result = -1

    # ---------------- 阶段 2：知道局长了，才决定标注哪些点
    recs: list[dict] = []
    idx = _select_indices(len(snaps), max_per, _W["schedule"], _W["quotas"])
    for i in idx:
        step, turn, snap, snap_mem = snaps[i]
        sel = snap["select"]
        base = rule_action(sel)[0]
        cands = _candidates(sel, base)
        try:
            out, ev = evaluate_actions(snap, cands, filler=filler,
                                       my_deck=my_deck, n_worlds=worlds,
                                       memory=snap_mem)
        except Exception:
            continue
        if ev["worlds"] < 2:
            continue
        labels, ns = [], []
        for a in cands:
            s = _stats(out[a])
            labels.append(s["mean"] if s else None)
            ns.append(s["n"] if s else 0)
        if not any(x is not None for x in labels):
            continue
        recs.append({
            "game": gid,
            "matchup": job["matchup"],
            "step": step,
            "turn": turn,
            "progress": round(i / max(len(snaps) - 1, 1), 3),
            "player": int((snap.get("current") or {}).get("yourIndex", 0)),
            "state_policy": sp,
            "features": F.state_features(snap, snap_mem),
            "options": [F.option_features(snap, a, snap_mem) for a in cands],
            "labels": labels,
            "ns": ns,
            "base_idx": 0,
            "action_types": [OptionType(sel["option"][a]["type"]).name for a in cands],
            "result": None if result == -1 else result,
        })
    return recs


# ============================================================ 任务编排
def build_jobs(args) -> list[dict]:
    """生成对局任务表。四路平衡：卡组顺序 × 座位。"""
    if args.deck_dir:
        base = args.deck_dir if os.path.isabs(args.deck_dir) else os.path.join(HERE, args.deck_dir)
        paths = sorted(glob.glob(os.path.join(base, "*.csv")))
        if len(paths) < 2:
            raise SystemExit(f"{base} 下至少要 2 套卡组")
        names = {p: os.path.splitext(os.path.basename(p))[0] for p in paths}
        # i<j：不含镜像对。跨卡组才能学到"对手不是我"
        pairs = [(paths[i], paths[j])
                 for i in range(len(paths)) for j in range(i + 1, len(paths))]
        per_pair = args.games_per_pair
    else:
        a = args.deck if os.path.isabs(args.deck) else os.path.join(HERE, args.deck)
        b = (args.deck_opponent or args.deck)
        b = b if os.path.isabs(b) else os.path.join(HERE, b)
        names = {a: os.path.splitext(os.path.basename(a))[0],
                 b: os.path.splitext(os.path.basename(b))[0]}
        pairs = [(a, b)]
        per_pair = args.games

    jobs = []
    gid = 0
    for (a, b) in pairs:
        for g in range(per_pair):
            swap = bool((g // 2) % 2)          # 交换卡组顺序
            d0, d1 = (b, a) if swap else (a, b)
            jobs.append({
                "gid": gid,
                "deck0": d0,
                "deck1": d1,
                "reverse": bool(g % 2),
                "matchup": f"{names[d0]}_vs_{names[d1]}",
            })
            gid += 1
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- 卡组
    ap.add_argument("--deck-dir", default=None,
                    help="两两轮转的卡组目录（推荐）。给了它就忽略 --deck/--deck-opponent")
    ap.add_argument("--deck", default="decks/sample_official.csv")
    ap.add_argument("--deck-opponent", default=None, help="对手卡组（默认同 --deck）")
    ap.add_argument("--games-per-pair", type=int, default=2,
                    help="--deck-dir 模式：每对卡组打几局（四路平衡，建议偶数）")
    ap.add_argument("--games", type=int, default=20,
                    help="单一卡组对模式：总共打几局")
    # ---- 状态生成策略
    ap.add_argument("--state-policy", choices=["greedy", "rollout"], default="greedy",
                    help="生成对局轨迹的策略。rollout=贴近部署但慢约 10 倍")
    ap.add_argument("--state-worlds", type=int, default=6,
                    help="--state-policy rollout 时每步用的 world 数（控制成本）")
    # ---- 标注
    ap.add_argument("--worlds", type=int, default=8,
                    help="每个局面采多少个 determinization（= 每个候选的 rollout 次数）")
    ap.add_argument("--trigger", choices=["attack", "main"], default="main",
                    help="main=所有多选项 MAIN 决策（默认；候选集含 END，覆盖"
                         "『现在结束回合 vs 继续铺场』这个最高杠杆决策）。"
                         "attack=只标注 ATTACK 可选的（约 31%%，更便宜）")
    ap.add_argument("--max-per-game", type=int, default=20,
                    help="每局最多标注多少个决策点（成本 O(决策数²)，必须限流）")
    ap.add_argument("--schedule", choices=["strat", "uniform", "late"], default="strat",
                    help="从整局快照里挑哪些点标注。strat=分层(默认,配额偏后期)；"
                         "uniform=均匀；late=只取最后 N 个（最便宜）")
    ap.add_argument("--quota", default="25,30,45",
                    help="strat 模式下 早/中/晚 三段的配额比例，默认 25,30,45")
    # ---- 运行
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="data/cf_q_v1.jsonl")
    ap.add_argument("--max-games", type=int, default=0,
                    help="总对局上限（0=不限）。冒烟测试时很有用")
    ap.add_argument("--fresh", action="store_true", help="先清空输出文件（默认追加）")
    args = ap.parse_args()

    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if args.fresh and os.path.exists(out):
        os.remove(out)

    # ⚠️ 四路平衡需要 games_per_pair 是 4 的倍数：
    #    swap=(g//2)%2, reverse=g%2 => g=0..3 才覆盖 卡组顺序×座位 全部组合。
    #    games_per_pair=1 时永远 deck0 当玩家0 且 reverse=False，
    #    而"自己选先后手的玩家0"胜率只有 0.417（实测），会被系统性压成劣势方 ——
    #    实测这样跑出来的数据"有差异局面"只有 19%，远低于正常水平。
    if args.deck_dir and args.games_per_pair < 4:
        print(f"  ⚠️  --games-per-pair={args.games_per_pair} < 4：四路平衡不完整，"
              f"数据会有座位/卡组顺序偏置。建议用 4 或 8。", flush=True)

    jobs = build_jobs(args)
    if args.max_games and len(jobs) > args.max_games:
        jobs = jobs[:args.max_games]
    try:
        quotas = tuple(int(x) for x in str(args.quota).split(","))
        if len(quotas) != 3:
            quotas = (25, 30, 45)
    except ValueError:
        quotas = (25, 30, 45)
    initargs = (args.worlds, args.state_policy, args.state_worlds,
                args.max_per_game, args.trigger, args.schedule, quotas)

    print("=" * 70)
    print("反事实 Q 标签生成")
    print("=" * 70)
    print(f"  卡组      : {'两两轮转 ' + args.deck_dir if args.deck_dir else f'{args.deck} vs {args.deck_opponent or args.deck}'}")
    print(f"  对局数    : {len(jobs)}   （四路平衡：卡组顺序 × 座位）")
    print(f"  状态策略  : {args.state_policy}"
          + (f"（每步 {args.state_worlds} worlds）" if args.state_policy == 'rollout' else ""))
    print(f"  标注      : worlds={args.worlds}  trigger={args.trigger}  "
          f"max/局={args.max_per_game}  调度={args.schedule}"
          + (f" 配额{quotas}" if args.schedule == "strat" else ""))
    print(f"  workers   : {args.workers}")
    print(f"  输出      : {out}")
    print("=" * 70, flush=True)

    t0 = time.time()
    stat = {"rows": 0, "labels": 0, "games": 0, "err": 0,
            "spread_rows": 0, "spread_sum": 0.0}

    def _dump(recs):
        with open(out, "a", encoding="utf-8") as f:
            for r in recs:
                if r.get("error"):
                    stat["err"] += 1
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                stat["rows"] += 1
                vals = [x for x in r["labels"] if x is not None]
                stat["labels"] += len(vals)
                if len(vals) >= 2:
                    sp = max(vals) - min(vals)
                    stat["spread_sum"] += sp
                    if sp > 0:
                        stat["spread_rows"] += 1
        stat["games"] += 1

    def _report(k):
        el = time.time() - t0
        var = stat["spread_rows"] / max(stat["rows"], 1)
        print(f"  {k}/{len(jobs)} 局  局面 {stat['rows']:,} 行  "
              f"动作标签 {stat['labels']:,} 个  有差异局面 {var:.0%}  "
              f"已用 {el:.0f}s  预计还需 {el/k*(len(jobs)-k):.0f}s", flush=True)

    if args.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_worker_init,
                      initargs=initargs) as pool:
            for k, recs in enumerate(pool.imap_unordered(_run_game, jobs,
                                                         chunksize=1), 1):
                _dump(recs)
                if k % 5 == 0 or k == len(jobs):
                    _report(k)
    else:
        _worker_init(*initargs)
        for k, job in enumerate(jobs, 1):
            _dump(_run_game(job))
            if k % 5 == 0 or k == len(jobs):
                _report(k)

    meta = {
        "out": os.path.relpath(out, HERE),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kind": "counterfactual_q",
        "deck_source": args.deck_dir or f"{args.deck}|{args.deck_opponent or args.deck}",
        "games": stat["games"],
        "state_policy": args.state_policy,
        "state_worlds": args.state_worlds,
        "worlds": args.worlds,
        "trigger": args.trigger,
        "max_per_game": args.max_per_game,
        "rows": stat["rows"],
        "action_labels": stat["labels"],
        "errors": stat["err"],
        "matchups": len({j["matchup"] for j in jobs}),
        "schedule": args.schedule,
        "quotas": list(quotas),
        "spread_rows": stat["spread_rows"],
        "spread_frac": round(stat["spread_rows"] / max(stat["rows"], 1), 4),
        "mean_spread": round(stat["spread_sum"] / max(stat["rows"], 1), 4),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out + ".meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print(f"写盘 {stat['rows']:,} 局面 / {stat['labels']:,} 动作标签 -> {out}")
    print(f"对局 {stat['games']} 局（{meta['matchups']} 种 matchup），"
          f"异常 {stat['err']} 行，用时 {time.time()-t0:.0f}s")
    print(f"跨动作有差异的局面: {meta['spread_frac']:.1%}  "
          f"平均跨动作差异 {meta['mean_spread']:.3f}   ← 这个数越大，反事实信号越强")
    print(f"元数据 -> {out}.meta.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
