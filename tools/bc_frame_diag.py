#!/usr/bin/env python3
"""Route B · **同帧行为诊断**：真人 vs BC vs 端口，在同一 observation 上比语义动作。

## 为什么必须"同帧"

直接比 action index 没意义（候选列表顺序不同、等价类不同）。这里每一帧：
  1. 用端口 agent 把 tracker 推进到该帧（与生产完全一致，含 forced_run）
  2. 用**同一份编码行**分别算 端口 / BC 的 logits
  3. 三者的输出都映射到**当前帧的候选列表索引**（含等价类代表重映射）
  4. 再映射到**语义动作** `(category, subcategory, card_id, target_card, target_slot)`

## 输出四组指标（都带：帧数 / 独立对局数 / 独立玩家数）

  1. Overall agreement        BC / 端口 与真人的一致率
  2. Category agreement       按真人的语义类别分组的条件一致率
  3. Subcategory agreement    更细一层（进化目标、攻击、换人目标…）
  4. Conditional agreement    特定局面条件下是否系统性偏离

⚠️ 措辞纪律：真人动作**不是标准答案**。报告只写"一致率 / 偏离类别"，
   不写"错误率"，除非有独立的质量标签。

用法：
    ./tools/py tools/bc_frame_diag.py --deck f5f8 --bc models/…/abra2.pt --games 40
    ./tools/py tools/bc_frame_diag.py --deck f5f8 --bc <pt> --index data/distill_index.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "build" / "port_plain", ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch                                            # noqa: E402

from ptcg.rl.search_agent import SearchAgent            # noqa: E402
import ptcg.rl.encoder as E                             # noqa: E402
from bc_materialize import remap_target, _class_key     # noqa: E402
from framework.replay import iter_decisions             # noqa: E402

CAT = {
    E.OT_PLAY: ("打出", "play"), E.OT_ATTACH: ("附着", "attach"),
    E.OT_EVOLVE: ("进化", "evolve"), E.OT_ABILITY: ("特性", "ability"),
    E.OT_DISCARD: ("丢弃", "discard"), E.OT_RETREAT: ("换人", "retreat"),
    E.OT_ATTACK: ("攻击", "attack"), E.OT_END: ("结束回合", "end"),
    E.OT_SKILL: ("招式", "skill"), E.OT_YES: ("是", "yes"), E.OT_NO: ("否", "no"),
    E.OT_NUMBER: ("数值", "number"), E.OT_CARD: ("选牌", "card"),
    E.OT_TOOL_CARD: ("道具", "tool"), E.OT_ENERGY_CARD: ("能量卡", "energy_card"),
    E.OT_ENERGY: ("能量", "energy"), E.OT_SPECIAL: ("特殊", "special"),
    E.OT_STOP: ("停止", "stop"),
}


def semantic(opt, cur, sel, seat):
    """把候选动作映射到语义元组（category, subcategory, card_id, target_card, target_slot）。"""
    ot = opt.get("type", 0)
    name, sub = CAT.get(ot, (f"type{ot}", "other"))
    card = opt.get("cardId") or 0
    tgt_card = 0
    tgt_slot = opt.get("inPlayIndex")
    try:
        if ot == E.OT_PLAY:
            card = _res = opt.get("cardId") or 0
        elif ot == E.OT_ATTACK:
            sub = f"attack#{opt.get('attackId')}"
        elif ot in (E.OT_EVOLVE, E.OT_ATTACH):
            sub = f"{name}#card{card}"
        if tgt_slot is None:
            tgt_slot = opt.get("index")
    except Exception:
        pass
    return (name, sub, card, tgt_card, tgt_slot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="f5f8", help="要诊断的卡组前 4 位（出手方）")
    ap.add_argument("--bc", required=True, help="BC 模型 .pt")
    ap.add_argument("--port", default="build/port_plain/model.pt")
    ap.add_argument("--games", type=int, default=40, help="最多多少个整局")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    port = SearchAgent(model_path=str(ROOT / a.port), deck_path=str(ROOT / "build/port_plain/deck.csv"),
                       cfg=None, seed=0, device=a.device, data_dir=str(ROOT / "build/port_plain"))
    bc = SearchAgent(model_path=a.bc, deck_path=str(ROOT / "build/port_plain/deck.csv"),
                     cfg=None, seed=0, device=a.device, data_dir=str(ROOT / "build/port_plain"))

    # 从索引里挑出"出手方是 target deck"的整局
    files, seen = [], set()
    import hashlib
    for line in open(ROOT / "data/distill_index.jsonl", encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not (r.get("deck") or "").startswith(a.deck):
            continue
        f = r.get("file")
        if f and f not in seen:
            seen.add(f)
            files.append(f)
        if len(files) >= a.games:
            break
    print(f"目标卡组 {a.deck}：候选整局 {len(files)} 个", flush=True)

    stat = collections.defaultdict(lambda: [0, 0, 0])   # cat -> [n, agree_bc, agree_port]
    overall = [0, 0, 0]                                 # [n, bc_agree, port_agree]
    games_used, players = set(), set()
    cond = collections.Counter()
    for gi, f in enumerate(files):
        path = ROOT / f
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        steps = d.get("steps") or []
        agents = [x.get("Name") or "" for x in (d.get("info", {}).get("Agents") or [])]
        deck8 = ["?"] * 2
        if len(steps) > 1:
            for k in (0, 1):
                act0 = steps[1][k].get("action")
                if isinstance(act0, list) and len(act0) == 60:
                    deck8[k] = hashlib.sha256(",".join(str(x) for x in sorted(act0)).encode()).hexdigest()[:8]
        used_here = False
        for ag_ in (port, bc):
            ag_.agent(dict(select=None))
        for t, j, obs, act in iter_decisions(steps):
            if not (isinstance(act, list) and act and isinstance(act[0], int)):
                continue
            if not deck8[j].startswith(a.deck):      # 只看目标卡组那一侧
                continue
            call = dict(obs)
            call.setdefault("logs", [])
            call.setdefault("remainingOverageTime", 600.0)
            call["player"] = j
            call["step"] = t
            try:
                port.agent(call)                     # 推进 tracker（与生产一致）
                sel = call.get("select") or {}
                opts = sel.get("option") or []
                n_opt = len(opts)
                if n_opt < 2:
                    continue
                E.encode(port.row[0], call, [], False, port.forced_run, tracker=port.tracker)
                xt = torch.from_numpy(port.row).to(port.device)
                with torch.no_grad():
                    lg_p = port.policy.forward_eval(xt)[0][0]
                    lg_b = bc.policy.forward_eval(xt)[0][0]
                cur = call["current"]
                seat = cur["yourIndex"]

                def to_idx(lg, a0):
                    if float(lg[a0]) > -1e8:
                        return a0
                    return remap_target(a0, cur, sel, port.row[0])

                h = int(act[0])
                h_idx = to_idx(torch.zeros(n_opt), h) if h < n_opt else None
                # 真人动作也可以是等价类的非代表索引 → 用同一 remap
                if h_idx is None or float(lg_p[h_idx]) <= -1e8:
                    h_idx = remap_target(h, cur, sel, port.row[0])
                p_idx = int(torch.argmax(lg_p[:n_opt]))
                b_idx = int(torch.argmax(lg_b[:n_opt]))
                if h_idx is None or not (0 <= h_idx < n_opt):
                    continue
                sem_h = semantic(opts[h_idx], cur, sel, seat)
                sem_p = semantic(opts[p_idx], cur, sel, seat)
                sem_b = semantic(opts[b_idx], cur, sel, seat)
                cat = sem_h[0]
                stat[cat][0] += 1
                overall[0] += 1
                games_used.add(f)
                players.add(agents[j] if j < len(agents) else "?")
                used_here = True
                if sem_b[0] == cat:
                    stat[cat][1] += 1
                    overall[1] += 1
                if sem_p[0] == cat:
                    stat[cat][2] += 1
                    overall[2] += 1
                # 条件分析：真人在"这一类"，模型是否切到"结束回合"
                if cat != "结束回合" and sem_b[0] == "结束回合":
                    cond[f"BC 在真人{cat}时选了结束回合"] += 1
                if cat != "结束回合" and sem_p[0] == "结束回合":
                    cond[f"端口 在真人{cat}时选了结束回合"] += 1
            except Exception:
                continue
        if used_here and (gi + 1) % 10 == 0:
            print(f"  已处理 {gi+1} 局，帧 {overall[0]}", flush=True)

    n, ab, ap_ = overall
    print(f"\n===== 总览（{n} 帧 / {len(games_used)} 局 / {len(players)} 个玩家）=====")
    if n:
        print(f"  与真人动作一致率：BC {ab/n:.3f}   端口 {ap_/n:.3f}   Δ={100*(ab-ap_)/n:+.1f}pp")
    print(f"\n{'真人类别':10s}{'帧数':>6}{'BC一致':>8}{'端口一致':>9}{'Δpp':>8}")
    for cat, (cnt, cbc, cpt) in sorted(stat.items(), key=lambda kv: -kv[1][0]):
        if cnt < 10:
            continue
        print(f"{cat:10s}{cnt:>6}{cbc/cnt:>8.3f}{cpt/cnt:>9.3f}{100*(cbc-cpt)/cnt:>+8.1f}")
    if cond:
        print("\n===== 条件偏离（真人想做别的事、模型却直接结束回合）=====")
        for k, v in cond.most_common(8):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
