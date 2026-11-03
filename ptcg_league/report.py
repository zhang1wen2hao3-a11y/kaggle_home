#!/usr/bin/env python3
"""联赛报告渲染。只读 json，不产生副作用 —— 所以随时可以重渲。

    ./tools/py ptcg_league/report.py            # 重渲最新一代 + 总览
    ./tools/py ptcg_league/report.py --gen 3    # 只渲第 3 代
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import league as L                                    # noqa: E402

VERDICT_MARK = {"promote": "✅ 晋级", "specialist": "⭐ 特色对手", "reject": "❌ 淘汰"}


def fmt(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def eval_row(name, d, extra=""):
    o = (d or {}).get("overall") or {}
    if (d or {}).get("skipped"):
        return f"| {name} | — | — | — | — | 跳过：{d['skipped']} |"
    n = o.get("n", 0)
    return (f"| {name} | {fmt(o.get('score'))} | [{fmt(o.get('ci_low'))}, {fmt(o.get('ci_high'))}] "
            f"| {n} | {o.get('errors', 0)} | {extra} |")


def render_gen(g, plan, stage, evals, dec, pf, rules, state, reg, probe=None):
    sc, cf, an = evals.get("screen"), evals.get("confirm"), evals.get("anchor")
    best = reg.by_id.get(state.current_best) or {}
    out = []
    A = out.append
    A(f"# 训练联赛 · 第 {g} 代")
    A("")
    A(f"- 规则：`rules.json` v{rules['version']}  sha `{L.rules_sha(rules)[:16]}`"
      f"  （每代记录固定 sha，改规则必须显式改版本号）")
    A(f"- 计量单位：{rules['unit']}（一场 Bo3 = 1 分，胜 1 / 平 0.5 / 负 0）")
    A(f"- 结果：**{VERDICT_MARK.get(dec['verdict'], dec['verdict'])}** — {dec['reason']}")
    A(f"- 当前 best：`{state.current_best}`（卡组 `{best.get('deck_id', '?')}`）")
    A("")
    A("## 1. 本轮训练分支")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A(f"| 分支 | `{plan['branch']}` |")
    A(f"| 卡组 | `{os.path.basename(plan['deck'])}` |")
    A(f"| 父模型 | `{plan['parent_id']}` |")
    A(f"| 种子 | {plan['seed']} |")
    A(f"| 步数 | {plan['steps']:,}（约 {plan['steps'] // 170} 局，"
      f"按 170 步/局折算） |")
    A(f"| 对手 bank | {len(plan['league_opponents'])} 个："
      f"{', '.join('`' + x + '`' for x in plan['league_opponents'])} |")
    A(f"| league-pct | {plan['league_pct']} / bank |")
    A(f"| 超参 | lr={plan['lr']} agents={plan['total_agents']} "
      f"minibatch={plan['minibatch']} mix-self=1.0 |")
    dwf = os.path.basename(plan.get("deck_weights_file") or "?")
    tgt = plan.get("target_archetype")
    A(f"| 对手卡组分布 | `{dwf}`"
      + (f"（**专精目标 {tgt}**，占比 {plan.get('_target_share', '?')}）" if tgt
         else "（真实野区）") + " |")
    msha = (stage or {}).get("model_sha")
    A(f"| 权重 sha256 | `{(msha or '?')[:16]}` |")
    A("")
    A("## 2. 评测（固定种子 / 固定卡组 / 座位交替）")
    A("")
    A("| 阶段 | 得分率 | 95% CI | 场数 | 异常 | 预注册判据 |")
    A("|---|---|---|---|---|---|")
    A(eval_row("筛查 screen", sc,
               f"≥{rules['screen']['min_score']} 且异常=0"))
    A(eval_row("确认 confirm", cf,
               f"CI 下界>{rules['confirm']['h2h_ci_low_above']}"))
    e = (probe or {}).get("edge") or {}
    if e:
        sp = (rules.get("specialist_probe") or {})
        A(f"| 专精探针 probe | {e.get('cand'):.4f} vs 父 {e.get('base'):.4f} "
          f"⇒ {e.get('mean'):+.4f} | [{e.get('ci_low'):+.4f}, {e.get('ci_high'):+.4f}] "
          f"| {e.get('n')} 配对 | — | 差值 ≥{sp.get('min_edge', 0.05)} 且 CI 下界 >0"
          f"（±{e.get('wins')}/{e.get('losses')}） |")
    base_anchor = ((best.get("eval") or {}).get("anchor") or {})
    A(eval_row("固定评测集 anchor", an,
               f"相对 best({fmt(base_anchor.get('score'))}) 回落 ≤"
               f"{rules['anchor']['max_regress']}"))
    A("")
    for name, d, key in (("筛查", sc, "screen"), ("确认", cf, "confirm"),
                         ("固定评测集", an, "anchor")):
        po = (d or {}).get("per_opp") or {}
        if not po:
            continue
        A(f"**{name} · 按对手拆**")
        A("")
        A("| 对手 | 得分率 | 95% CI | 场数 |")
        A("|---|---|---|---|")
        for aid, s in po.items():
            r = reg.by_id.get(aid) or {}
            tag = "" if aid == state.current_best else f" `{r.get('status', 'init')}`"
            A(f"| `{aid}`{tag} | {fmt(s['score'])} | "
              f"[{fmt(s['ci_low'])}, {fmt(s['ci_high'])}] | {s['n']} |")
        A("")
    arb = dec.get("archetype_scores") or {}
    if arb:
        A("**按对手卡组 archetype 聚合**（克制价值就在这里看）")
        A("")
        A("| archetype | 得分率 | 场数 |")
        A("|---|---|---|")
        for k, v in sorted(arb.items(), key=lambda kv: -kv[1]["score"]):
            A(f"| `{k}` | {v['score']:.4f} | {v['n']} |")
        A("")
    A("## 3. 判定明细")
    A("")
    A("```")
    A(f"verdict        = {dec['verdict']}")
    A(f"reason         = {dec['reason']}")
    for k, v in (dec.get("detail") or {}).items():
        if isinstance(v, dict):
            A(f"{k:<14} = {json.dumps(v, ensure_ascii=False, sort_keys=True)}")
        else:
            A(f"{k:<14} = {v}")
    if dec.get("similarity") is not None:
        A(f"{'similarity':<14} = {dec['similarity']:.4f}"
          + (f"（最像 {dec['duplicate_of']}）" if dec.get("duplicate_of") else ""))
    A("```")
    A("")
    if dec["verdict"] in ("promote", "specialist"):
        A(f"已写入正式联赛池：`{dec.get('agent_id')}` → `{dec.get('eval_dir')}`")
        A("")
    A("## 4. 对手采样权重（PFSP `w_i = p_i(1-p_i)`）")
    A("")
    A("| 对手 | 对它的胜率 p_i | 采样权重 w_i | 状态 |")
    A("|---|---|---|---|")
    pw = (pf or {}).get("p") or {}
    ww = (pf or {}).get("w") or {}
    for aid in sorted(ww, key=lambda k: -ww[k]):
        r = reg.by_id.get(aid) or {}
        A(f"| `{aid}` | {fmt(pw.get(aid))} | {ww[aid]:.3f} | {r.get('status', '?')} |")
    A("")
    A(f"地板配额 {rules['pfsp']['floor_quota']}（胜率 5% 的纯克制对手也保底采样）"
      f"，单头上限 {rules['pfsp']['max_single']}。")
    A("")
    A("## 5. 下一步")
    A("")
    nxt = state.gen + 1
    A(f"- 继续跑：`./tools/py ptcg_league/run_league.py next`（将开始第 {nxt} 代）")
    if dec["verdict"] == "reject":
        A(f"- 本代淘汰原因：{dec['reason']}")
        if "ci" in dec["reason"]:
            A("- CI 跨 0.50 属于「不能宣称提升」，不是「变差了」："
              "要么加长确认场次，要么换初始化/对手分布再试。")
    A("- 断点续跑：`run_league.py next --resume`；回滚：`run_league.py rollback`")
    A("")
    return "\n".join(out) + "\n"


def render_overview(reg, state, rules):
    rows = [r for r in reg.by_id.values()
            if r["pool"] == "league" and r["status"] in ("best", "opponent", "specialist")]
    out = ["# 训练联赛 · 总览", ""]
    out.append(f"- 规则 v{rules['version']} sha `{L.rules_sha(rules)[:16]}`   代 {state.gen}")
    out.append(f"- 当前 best：`{state.current_best}`")
    out.append(f"- 池：init {len(reg.in_pool('init'))} / evo {len(reg.in_pool('evo'))} / "
               f"league {len(rows)}（上限 {rules['pool']['max_league']}）")
    if state.awaiting:
        out.append(f"- ⚠️ 断点：第 {state.awaiting_gen} 代卡在 `{state.awaiting}`")
    out += ["", "## 正式联赛池", "",
            "| agent | 状态 | 卡组 | h2h 确认 | 固定评测集 | w_i |",
            "|---|---|---|---|---|---|"]
    w = state.pfsp.get("w") or {}
    for r in sorted(rows, key=lambda x: -w.get(x["agent_id"], 0)):
        e = r.get("eval") or {}
        out.append(f"| `{r['agent_id']}` | {r['status']} | `{r.get('deck_id', '?')}` | "
                   f"{fmt((e.get('confirm') or {}).get('score'))} | "
                   f"{fmt((e.get('anchor') or {}).get('score'))} | "
                   f"{w.get(r['agent_id'], 0):.3f} |")
    out += ["", "## 最近 12 条事件", "", "| 代 | 动作 | 对象 | 说明 |", "|---|---|---|---|"]
    for h in state.history[-12:]:
        out.append(f"| {h.get('gen')} | {h.get('action')} | `{h.get('agent_id') or h.get('branch')}` | "
                   f"{(h.get('reason') or '')[:70]} |")
    # 每代的判定一览
    out += ["", "## 每代判定", "", "| 代 | 分支 | 候选 h2h | CI | 判定 | 原因 |",
            "|---|---|---|---|---|---|"]
    for p in sorted(glob.glob(os.path.join(HERE, "gen", "*", "decide.json"))):
        g = int(os.path.basename(os.path.dirname(p)))
        d = L.read_json(p) or {}
        cf = (d.get("detail") or {}).get("confirm") or {}
        ci = ("—" if not cf else f"[{cf.get('ci_low'):.3f}, {cf.get('ci_high'):.3f}]")
        out.append(f"| {g} | `{d.get('branch', '?')}` | {fmt(cf.get('score'))} | {ci} | "
                   f"{VERDICT_MARK.get(d.get('verdict'), d.get('verdict'))} | "
                   f"{(d.get('reason') or '')[:60]} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=None)
    a = ap.parse_args()
    rules = L.load_rules(HERE)
    reg = L.Registry()
    state = L.LeagueState()
    gens = ([a.gen] if a.gen is not None else
            [int(os.path.basename(os.path.dirname(p)))
             for p in sorted(glob.glob(os.path.join(HERE, "gen", "*", "decide.json")))])
    for g in gens[-3:]:
        gd = os.path.join(HERE, "gen", f"{g:04d}")
        get = lambda n: L.read_json(os.path.join(gd, f"{n}.json"))
        plan, st, dec, pf = get("plan"), get("stage"), get("decide"), get("pfsp")
        if not (plan and dec):
            print(f"第 {g} 代还不完整，跳过")
            continue
        evals = {n: get(n) for n in ("screen", "confirm", "anchor")}
        txt = render_gen(g, plan, st or {}, evals, dec, pf or {}, rules, state, reg,
                         probe=get("probe"))
        with open(os.path.join(HERE, "reports", f"gen_{g:04d}.md"), "w", encoding="utf-8") as f:
            f.write(txt)
        print("reports/gen_%04d.md" % g)
    with open(os.path.join(HERE, "reports", "overview.md"), "w", encoding="utf-8") as f:
        f.write(render_overview(reg, state, rules))
    print("reports/overview.md")


if __name__ == "__main__":
    main()
