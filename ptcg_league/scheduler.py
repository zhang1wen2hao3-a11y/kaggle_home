#!/usr/bin/env python3
"""**流水线调度器**：把 `run_league.py` 的 8 个阶段按 job 状态分派，实现训练/评测解耦。

## 和 `run_league.py next` 的区别

| | `next`（旧） | `scheduler.py`（新） |
|---|---|---|
| 结构 | 单进程顺序跑完一代 | 每个 job 一个状态，按状态分派一步 |
| GPU | 训完才评，评测 30 分钟 GPU 空转 | 训练 job 交出去（打 kit）后，**立刻能计划下一个分支** |
| CPU | 只在串行某一刻用 | 独立的评估 job 可以在另一台/另一核并行 |
| 失败 | 整代重跑 | 只重跑当前状态那一步 |

⚠️ 判定逻辑**完全不改**：仍然调 `run_league.stage_*` + `league.decide()`（`rules.json` 说了算）。

## 分支（对应你设计的三类训练任务）

| 分支 | 对手 bank | 对手卡组分布 | 谁适合当 bank |
|---|---|---|---|
| `generalist` | 角色配额（A/B 档） | 真实野区 `weights_real` | **通才**：端口、历史 PPO ckpt（`sp:g0002` / `opp:g0001`） |
| `specialist` | 该原型驾驶者 + 定选 | 压向目标原型（`--target` 生成权重文件） | 该**卡组专家** |
| `counter` | —— 暂不支持做训练分支 | —— | 见下 |

⚠️ `counter` 分支需要"墙 bot"这种**规则对手**当训练陪练，但训练器的 bank 只能是**权重 blob**
（`--league-ckpts` 吃 `.bin`），规则 bot 没有权重 ⇒ 现阶段 `counter` 只能做**战术评估**
（`bots/shuppet_wall` 已在做这件事），不能做训练分支。这一点写死在代码里，不假装支持。

## 用法

    # 计划一个分支（只写 plan.json，不训练）
    ./tools/py ptcg_league/scheduler.py plan --branch generalist
    ./tools/py ptcg_league/scheduler.py plan --branch specialist --target shuppet

    # 推进队列：每个 job 前进一个状态；训练 job 会写 kit 并等 GPU
    ./tools/py ptcg_league/scheduler.py dispatch --max-steps 6

    # 看队列
    ./tools/py ptcg_league/scheduler.py status
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import jobs as J                                       # noqa: E402
import league as L                                     # noqa: E402
import run_league as R                                 # noqa: E402

BRANCH_HINT = {
    "generalist": ("通才 bank：端口 / 历史 PPO ckpt（同卡组或同分布），对手吃真实野区", {}),
    "specialist": ("卡组专家 bank + 对手卡组压向 --target 原型", {"need_target": True}),
    "counter": ("只做战术评估（规则 bot 不能当训练 bank），不生成训练 job", {"no_train": True}),
}


def _args(**kw):
    """构造 run_league 各 stage 需要的那套 args（默认值取自 main 的 argparse）。"""
    d = dict(bundle="build/port_plain", repo=None, deck=None, train_cmd=None,
             mock_train=None, workers=6, seed=20261016, seed_base=900000,
             banks=4, deck_cycle=False, force_eval=False, force_stage=False,
             gens=1, resume=False, rules=None, lr=None, target=None,
             deck_weights=None, matches=None)
    d.update(kw)
    return argparse.Namespace(**d)


def cmd_plan(a):
    rules = L.load_rules(HERE, a.rules)
    reg = L.Registry()
    state = L.LeagueState()
    q = J.JobQueue()
    if a.branch not in BRANCH_HINT:
        raise SystemExit(f"未知分支 {a.branch}（可选 {list(BRANCH_HINT)}）")
    desc, opt = BRANCH_HINT[a.branch]
    if opt.get("no_train"):
        raise SystemExit(f"分支 {a.branch}：{desc}\n"
                         f"（规则 bot 不能当训练 bank，用 tools/pool_ladder.sh 或 wall bot 做战术评估）")
    if opt.get("need_target") and not a.target:
        raise SystemExit(f"分支 {a.branch} 需要 --target <原型>")

    g = state.gen + 1
    job_id = f"{a.branch}-g{g:04d}"
    args = _args(deck=a.deck, target=(a.target if a.branch == "specialist" else None),
                 banks=a.banks, matches=a.matches, lr=a.lr)
    plan = R.stage_plan(g, reg, state, rules, args)
    q.add(job_id, dict(branch=a.branch, gen=g, grid=plan["branch"],
                       deck=os.path.basename(plan["deck"]),
                       banks=plan["league_opponents"],
                       target=plan.get("target_archetype") or "",
                       desc=desc))
    q.advance(job_id, "PLANNED", gen=g)
    state.gen = g
    state.save()
    print(f"✅ {job_id}  PLANNED  deck={os.path.basename(plan['deck'])}")
    print(f"   bank={plan['league_opponents']}  steps={plan['steps']}  lr={plan['lr']}")
    print(f"   下一步：scheduler.py dispatch")


def cmd_dispatch(a):
    rules = L.load_rules(HERE, a.rules)
    reg = L.Registry()
    state = L.LeagueState()
    q = J.JobQueue()
    done = 0
    while done < a.max_steps:
        job = q.actionable()
        if job is None:
            print("队列里没有可推进的 job（要么全终态，要么都在等 GPU）")
            break
        jid, st, g = job["job_id"], job["state"], job.get("gen")
        args = _args(workers=a.workers, rules=a.rules,
                     mock_train=("parent" if a.mock_train else None))
        print(f"[{jid}] {st} → ", end="", flush=True)
        try:
            if st == "PLANNED":
                q.advance(jid, "TRAINING")
                print("TRAINING")
            elif st == "TRAINING":
                ck = R.stage_train(g, reg, state, rules, args)
                if ck is None:
                    q.advance(jid, "TRAINING", awaiting_train=True)
                    print("TRAINING（等 GPU 产出；kit 见 ptcg_league/pending/）")
                    done += 1
                    continue
                q.advance(jid, "STAGED", ckpt=ck)
                print(f"STAGED  {os.path.basename(ck)}")
            elif st == "STAGED":
                R.stage_stage(g, reg, state, rules, args)
                q.advance(jid, "SCREENING")
                print("SCREENING")
            elif st == "SCREENING":
                R.stage_screen(g, reg, state, rules, args)
                q.advance(jid, "TACTICAL_EVAL")
                print("TACTICAL_EVAL")
            elif st == "TACTICAL_EVAL":
                R.stage_probe(g, reg, state, rules, args)
                q.advance(jid, "CONFIRMING")
                print("CONFIRMING")
            elif st == "CONFIRMING":
                R.stage_confirm(g, reg, state, rules, args)
                q.advance(jid, "ANCHOR")
                print("ANCHOR")
            elif st == "ANCHOR":
                R.stage_anchor(g, reg, state, rules, args)
                q.advance(jid, "DECIDED")
                print("DECIDED")
            elif st == "DECIDED":
                dec = R.stage_decide(g, reg, state, rules, args)
                R.stage_pfsp(g, reg, state, rules, args)
                R.stage_report(g, reg, state, rules, args)
                v = (dec or {}).get("verdict", "reject")
                term = {"promote": "PROMOTED", "specialist": "SPECIALIST"}.get(v, "ARCHIVED")
                q.advance(jid, term, verdict=v, agent_id=(dec or {}).get("agent_id"))
                print(term)
            done += 1
        except RuntimeError as e:            # 单步失败：留在原状态，下次重试
            q.advance(jid, st, last_error=str(e)[:300])
            print(f"✗ 停在 {st}：{e}")
            done += 1
    return q, state


def cmd_status(a):
    q = J.JobQueue()
    state = L.LeagueState()
    rows = q.all()
    if not rows:
        print("队列是空的：scheduler.py plan --branch generalist")
        return
    print(f"{'job_id':<26}{'state':<16}{'gen':>5}  {'分支':<12}说明")
    for r in sorted(rows, key=lambda x: (x.get("gen") or 0, x["job_id"])):
        sp = r.get("spec") or {}
        print(f"{r['job_id']:<26}{r['state']:<16}{(r.get('gen') or 0):>5}  "
              f"{sp.get('branch', '?'):<12}"
              f"{(sp.get('target') or sp.get('deck') or '')[:34]}")
    print(f"\nstate.gen={state.gen}  best={state.current_best}")
    n = sum(1 for r in rows if r["state"] not in J.TERMINAL)
    print(f"未完成 {n} / 共 {len(rows)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--branch", required=True, choices=list(BRANCH_HINT))
    p.add_argument("--target", default=None)
    p.add_argument("--deck", default=None)
    p.add_argument("--banks", type=int, default=4)
    p.add_argument("--matches", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--rules", default=None)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("dispatch")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--rules", default=None)
    p.add_argument("--mock-train", action="store_true")
    p.set_defaults(fn=cmd_dispatch)

    p = sub.add_parser("status")
    p.set_defaults(fn=cmd_status)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
