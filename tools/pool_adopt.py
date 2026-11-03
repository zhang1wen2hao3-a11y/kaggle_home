#!/usr/bin/env python3
"""把**历史 PPO 检查点**收编进对手池（不改 `rules.json`，不跑新评测）。

依据（`下一步.md` §22.9）：真实天梯里 **Ogerpon 占 29.5%**，而池里这一格只有端口自己
⇒ 训练时对手座位永远是"镜像父模型"，没有一个"同卡组但不同强度"的对手。
被拒绝的历史 ckpt 恰好是这个位置最合适的东西：
- 与父模型同卡组（7e39），所以难度天然接近 0.5；
- 且它们是**不同的策略**（PPO 漂移过的版本），行为和第 1 代都不同。

⚠️ 它们**不当 best**（`status="opponent"`），只是训练对手 —— 淘汰是评测给的结论，
   但"淘汰"不等于"不能当陪练"。这一点在报告里也要写清楚。

用法：
    ./tools/py tools/pool_adopt.py --gen 1          # 收编第 1 代
    ./tools/py tools/pool_adopt.py --gen 1 --gen 2  # 多个
    ./tools/py tools/pool_adopt.py --list           # 只看哪些代可收编

    # 通用入口（**不是按代数编名的 ckpt 走这里**，比如 G4 的 A/B/C、
    # 或者任何手工 stage 出来的 agent 目录）：
    ./tools/py tools/pool_adopt.py \
        --from /tmp/ag_g4B --agent-id opp:g4b --note "G4-B 配方（2 冻结 bank + 真实野区）"
    ./tools/py tools/pool_adopt.py --from /tmp/ag_g4B,/tmp/ag_g4C \
        --agent-id opp:g4b,opp:g4c --deck teal_mask_ogerpon_ex_applin__7e39
"""
import argparse
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ptcg_league"))

import league as L                                      # noqa: E402
from run_league import canonical_deck, deck_archetype, restat   # noqa: E402

LEAGUE = os.path.join(HERE, "ptcg_league")


def load_gen(g):
    d = os.path.join(LEAGUE, "gen", f"{g:04d}")
    plan = L.read_json(os.path.join(d, "plan.json"))
    stage = L.read_json(os.path.join(d, "stage.json"))
    screen = L.read_json(os.path.join(d, "screen.json")) or {}
    probe = L.read_json(os.path.join(d, "probe.json")) or {}
    dec = L.read_json(os.path.join(d, "decide.json")) or {}
    return plan, stage, screen, probe, dec


def adopt_from_dir(reg, state, src, aid, deck, status="opponent", note="",
                   score=None, source="ppo_general", gen=0):
    """把一个**已经 stage 好的 agent 目录**收编进池（通用入口）。

    `src` 必须含 `model.pt` 与 `deck.csv`（`tools/stage_ckpt.py` 的产物就是这个形状）。
    目录会被原样复制到 `pools/league/<aid>`（symlink 保留，所以只多占一份权重）。
    """
    src = os.path.abspath(src)
    if not os.path.isdir(src):
        raise SystemExit(f"✗ --from 不是目录：{src}")
    for need in ("model.pt", "deck.csv", "main.py"):
        if not os.path.exists(os.path.join(src, need)):
            raise SystemExit(f"✗ {src} 缺 {need}（先用 tools/stage_ckpt.py 生成）")
    if aid in reg.by_id:
        print(f"[skip] {aid} 已在池里")
        return None
    deck_csv = deck or os.path.join(src, "deck.csv")
    if not os.path.exists(deck_csv):
        raise SystemExit(f"✗ 卡组文件不存在：{deck_csv}")
    dst = os.path.join(LEAGUE, "pools", "league", aid.replace(":", "_"))
    if not os.path.exists(dst):
        shutil.copytree(src, dst, symlinks=True)
    ckpt = os.path.realpath(os.path.join(src, "model.pt"))
    row = reg.add(dict(
        agent_id=aid,
        checkpoint_path=ckpt,
        bin_path="",
        model_version=note or os.path.basename(src),
        deck_id=os.path.splitext(os.path.basename(deck_csv))[0],
        deck_path=os.path.abspath(deck_csv),
        source=source, pool="league", status=status,
        parent_id=None, seed=0, steps=0,
        archetype=deck_archetype(deck_csv),
        eval=({"ladder_score": score} if score is not None else {}),
        eval_dir=dst, gen_added=gen, traits={},
        _why_opponent=(note or "手工 stage 的检查点：不当 best，只当**同卡组陪练**。"),
    ))
    print(f"✅ {aid}  ← {src}  deck={os.path.basename(deck_csv)}  status={status}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, action="append", default=None)
    ap.add_argument("--from", dest="frm", default=None,
                    help="逗号分隔的 agent 目录（通用入口，与 --agent-id 一一对应）")
    ap.add_argument("--agent-id", default=None, help="逗号分隔的池内 id，如 opp:g4b,opp:g4c")
    ap.add_argument("--deck", default=None, help="卡组 csv 或 decks/ 下的名字；默认用目录里的 deck.csv")
    ap.add_argument("--status", default="opponent", choices=["opponent", "specialist"])
    ap.add_argument("--source", default="ppo_general", help="ppo_general / ppo_specialist / bc / init")
    ap.add_argument("--note", default="", help="写进 model_version / _why_opponent 的说明")
    ap.add_argument("--score", type=float, default=None, help="已知的阶梯 p_i（可后补）")
    ap.add_argument("--parent", default=None, help="父 agent_id（可选）")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    reg = L.Registry()
    state = L.LeagueState()
    gens = sorted(int(os.path.basename(os.path.dirname(p)))
                  for p in glob.glob(os.path.join(LEAGUE, "gen", "*", "decide.json")))

    # ── 通用入口：不依赖 `gen/NNNN/` 的目录结构 ──
    if a.frm:
        dirs = [x.strip() for x in a.frm.split(",") if x.strip()]
        ids = [x.strip() for x in (a.agent_id or "").split(",") if x.strip()]
        if len(ids) != len(dirs):
            raise SystemExit(f"✗ --from 给了 {len(dirs)} 个目录，--agent-id 给了 {len(ids)} 个（要一一对应）")
        adopt_from_dir_n = 0
        for src, aid in zip(dirs, ids):
            if ":" not in aid:
                aid = f"opp:{aid}"
            if adopt_from_dir(reg, state, src, aid, a.deck, status=a.status,
                              note=a.note, score=a.score, source=a.source, gen=0):
                adopt_from_dir_n += 1
        if adopt_from_dir_n:
            state.save()
            print(f"\n收编 {adopt_from_dir_n} 个。下一步："
                  f"./tools/py ptcg_league/run_league.py status")
        return

    if a.list or not a.gen:
        print(f"{'代':>4}  {'分支':<26}{'判定':<12}{'收编?':<8}说明")
        for g in gens:
            plan, stage, screen, probe, dec = load_gen(g)
            aid = f"opp:g{g:04d}"
            already = aid in reg.by_id
            sc = (screen.get("overall") or {}).get("score")
            print(f"{g:>4}  {plan['branch'][:24]:<26}{dec.get('verdict', '?'):<12}"
                  f"{'已收' if already else '可收':<8}"
                  f"screen={(sc if sc is None else round(sc, 4))}  deck={os.path.basename(plan['deck'])}")
        return

    added = []
    for g in a.gen:
        plan, stage, screen, probe, dec = load_gen(g)
        aid = f"opp:g{g:04d}"
        if aid in reg.by_id:
            print(f"[skip] 代 {g} → {aid} 已在池里")
            continue
        if not stage or not os.path.isdir(stage.get("bundle", "")):
            print(f"✗ 代 {g} 没有可用的 stage bundle，跳过")
            continue
        dst = os.path.join(LEAGUE, "pools", "league", aid.replace(":", "_"))
        if not os.path.exists(dst):
            shutil.copytree(stage["bundle"], dst, symlinks=True)
        deck = canonical_deck(plan["deck"])
        ev = dict(screen=(screen.get("overall") or None),
                  probe=((probe or {}).get("edge") or None))
        row = reg.add(dict(
            agent_id=aid,
            checkpoint_path=plan["ckpt_out"],
            bin_path=(plan["ckpt_out"] if str(plan.get("ckpt_out", "")).endswith(".bin") else ""),
            model_version=f"gen{g:04d}({dec.get('verdict', '?')})",
            deck_id=os.path.splitext(os.path.basename(deck))[0],
            deck_path=deck, source="ppo_general",
            parent_id=plan["parent_id"], seed=plan["seed"], steps=plan["steps"],
            pool="league", status="opponent",
            archetype=deck_archetype(deck), eval=ev, eval_dir=dst,
            gen_added=g, traits={},
            _why_opponent=("历史 PPO 检查点：不当 best，但当**同卡组陪练**。"
                           "Ogerpon 在真实天梯占 29.5%，池里需要一个『同卡组、不同强度』的对手，"
                           "而被淘汰的 ckpt 正好落在 0.45–0.75 这个训练最需要的难度带。"),
        ))
        added.append(aid)
        sc = (screen.get("overall") or {}).get("score")
        print(f"✅ 代 {g} → {aid}  判定={dec.get('verdict')}  "
              f"学习者对它胜率≈{1 - sc:.4f}" if sc is not None else f"✅ 代 {g} → {aid}")
    if added:
        state.save()
        print(f"\n收编 {len(added)} 个：{added}")
        print("下一步：./tools/py ptcg_league/run_league.py status")


if __name__ == "__main__":
    main()
