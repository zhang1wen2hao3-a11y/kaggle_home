#!/usr/bin/env python3
"""A0：BC 物化 + 校准。

两件事：
  1) **校准**（必须先过）：用现有端口权重，在回放帧上复现"合法动作内的 argmax"，
     按 动作数(单/多) × 原型 × 来源分数段 分组报告一致率。
     口径固定并打印，避免再出现"52.5% 是哪个口径"这种不可复现的数字。
  2) **物化**：把单动作帧编码成扁平行落盘，供 A1 训练。按**整局**切分 train/val。

用法：
    # 校准（小样本先行）
    python3 tools/bc_materialize.py --calibrate --games 6
    # 物化目标子集
    python3 tools/bc_materialize.py --materialize --archetypes slowking,shuppet,abra,dreepy \
        --out data/bc_rows_s4.npz

口径定义（打印进报告，不可含糊）：
  * 一致 = forward_eval(logits) 在前 n_opt 个上的 argmax == 专家 act[0]
  * 只统计 len(act)==1 的帧（多动作帧单独计数，不计入一致率）
  * 每帧的 obs 取自 iter_decisions；tracker 由 SearchAgent 按帧序推进（与生产一致）
  * 切分：sha1(file) % 100 < 20 判为 val（整局为单位，同局帧不跨集）
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "port_repo", ROOT / "tools"):
    sys.path.insert(0, str(p))

import numpy as np            # noqa: E402
import torch                  # noqa: E402

from framework.replay import iter_decisions   # noqa: E402
from ptcg.rl.search_agent import SearchAgent  # noqa: E402


# ── 等价类重映射 ─────────────────────────────────────────────────────────────
# encoder.py:377 只放行每个等价类的**第一个**索引（classes[key]=j; mask[j]=1），
# 专家可能点了同类里的重复索引 ⇒ 该帧被判"非法"。实测占单动作帧 2.04%（minCount=0 占 71%）。
# 修法：把标签重映射到 mask==1 的代表索引；找不到代表才算真非法。
from ptcg.rl import encoder as _E          # noqa: E402  常量都在它的命名空间里
from ptcg.rl.encoder import _slice as _eslice, encode as _encode  # noqa: E402


def _resolve_card(cur, sel, pidx, area, idx):
    """复刻 encoder.encode 内部那个嵌套 resolve_card。"""
    try:
        if area == _E.HAND:
            cards = cur["players"][pidx]["hand"]
        elif area == _E.DISCARD:
            cards = cur["players"][pidx]["discard"]
        elif area == _E.PRIZE:
            cards = cur["players"][pidx]["prize"]
        elif area == _E.DECK:
            cards = sel.get("deck")
        elif area == _E.LOOKING_A:
            cards = cur.get("looking")
        elif area == _E.ACTIVE:
            c = cur["players"][pidx]["active"][idx]
            return 0 if c is None else c["id"]
        elif area == _E.BENCH:
            return cur["players"][pidx]["bench"][idx]["id"]
        elif area == _E.STADIUM_A:
            return cur["stadium"][idx]["id"]
        else:
            return 0
        c = cards[idx] if cards else None
        return 0 if c is None else c["id"]
    except (IndexError, TypeError, KeyError):
        return 0


def _class_key(o, seat, cur, sel):
    """与 encoder.py:365-377 完全一致的等价类 key。"""
    ot = o.get("type", 0)
    area = o.get("area")
    idx = o.get("index")
    pidx = o.get("playerIndex")
    card_id = o.get("cardId") or 0
    if ot == _E.OT_PLAY:
        card_id = _resolve_card(cur, sel, seat, _E.HAND, idx)
    elif ot in (_E.OT_CARD, _E.OT_DISCARD, _E.OT_ABILITY):
        if not card_id:
            card_id = _resolve_card(cur, sel, pidx if pidx is not None else seat, area, idx)
    elif ot in (_E.OT_ATTACH, _E.OT_EVOLVE):
        card_id = _resolve_card(cur, sel, seat, area, idx)
    board_target = (area in (_E.ACTIVE, _E.BENCH)) or o.get("inPlayArea") is not None
    if ot == _E.OT_SKILL:
        return (ot, card_id, o.get("serial"))
    if board_target or ot in (_E.OT_TOOL_CARD, _E.OT_ENERGY_CARD, _E.OT_ENERGY):
        return (ot, area, idx, pidx, o.get("inPlayArea"), o.get("inPlayIndex"),
                card_id, o.get("count"),
                _resolve_card(cur, sel, pidx if pidx is not None else seat, area, idx)
                if ot in (_E.OT_TOOL_CARD, _E.OT_ENERGY_CARD, _E.OT_ENERGY) else None)
    if area in _E._CARD_ZONES or ot == _E.OT_PLAY:
        return (ot, area, pidx, card_id)
    return (ot, o.get("number"), o.get("attackId"), o.get("specialConditionType"))


def remap_target(act0, cur, sel, row):
    """返回等价类代表索引；None = 真非法。"""
    mask = _eslice(row, "opt_mask")
    if mask[act0] > 0:
        return act0
    opts = sel.get("option") or []
    if not (0 <= act0 < len(opts)):
        return None
    seat = cur["yourIndex"]
    key = _class_key(opts[act0], seat, cur, sel)
    for j, o in enumerate(opts[:_E.MAX_OPTIONS]):
        if mask[j] > 0 and _class_key(o, seat, cur, sel) == key:
            return j
    return None


def deck_archetype(deck8: str) -> str:
    return deck8[:4]


def split_of(path: str, val_pct: int) -> str:
    h = int(hashlib.sha1(path.encode()).hexdigest()[:8], 16) % 100
    return "val" if h < val_pct else "train"


def load_agent(device: str) -> SearchAgent:
    AG = ROOT / "build" / "port_plain"
    return SearchAgent(model_path=str(AG / "model.pt"), deck_path=str(AG / "deck.csv"),
                       cfg=None, seed=0, device=device, data_dir=str(AG))


def iter_game_frames(path: pathlib.Path, agents_pool: dict, want_seats=None, keep_fn=None):
    """按帧序推进 tracker，产出 (seat, obs, act, logits_1d, n_opt)。

    用 agent.agent(obs) 推进内部状态（强制选择/forced_run 与生产完全一致），
    然后用同一 tracker 做一次 encode+forward 拿 logits。
    """
    import hashlib as _hl
    d = json.loads(path.read_text())
    steps = d.get("steps") or []
    agents = [a["Name"] for a in (d.get("info", {}).get("Agents") or [])]
    agents = [a or "" for a in agents]
    # 每个座位的牌组 8 位哈希（与 tools/extract_distill.py 同法：sha256(逗号连接的排序 id)[:8]）
    deck8 = ["?"] * 2
    if len(steps) > 1:
        for k in (0, 1):
            act0 = steps[1][k].get("action")
            if isinstance(act0, list) and len(act0) == 60:
                deck8[k] = _hl.sha256(",".join(str(x) for x in sorted(act0)).encode()).hexdigest()[:8]
    rew = d.get("rewards") or [0, 0]
    for ag_ in agents_pool.values():        # 每局只 reset，不重建（重建是主要开销）
        ag_.agent(dict(select=None))
    out = []
    for t, j, obs, act in iter_decisions(steps):
        if not isinstance(act, list) or not act or not isinstance(act[0], int):
            continue
        a = agents_pool[j]
        call = dict(obs)
        call.setdefault("logs", [])
        call.setdefault("remainingOverageTime", 600.0)
        call["player"] = j
        call["step"] = t
        try:
            a.agent(call)                   # 推进 tracker / forced_run
            sel = call.get("select") or {}
            n_opt = len(sel.get("option") or [])
            if n_opt <= 1:
                continue
            encode_row = a.row[0]
            from ptcg.rl.encoder import encode
            encode(encode_row, call, [], False, a.forced_run, tracker=a.tracker)
            xt = torch.from_numpy(a.row).to(a.device)
            with torch.no_grad():
                lg = a.policy.forward_eval(xt)[0][0]
            if keep_fn is not None and not keep_fn(j, t):
                continue
            a0 = int(act[0])
            remap_flag = 0
            if float(lg[a0]) < -1e8:            # 非代表索引 -> 重映射到等价类代表
                rep = remap_target(a0, call["current"], sel, a.row[0])
                if rep is None:
                    out.append((j, t, n_opt, None, len(act), lg[:n_opt].detach().cpu(),
                                a.row[0].copy(), 0, deck8[j], agents[j] if j < len(agents) else "",
                                int(rew[j] > 0) if j < len(rew) else 0))
                    continue
                a0 = rep
                remap_flag = 1
            out.append((j, t, n_opt, a0, len(act), lg[:n_opt].detach().cpu(),
                        a.row[0].copy(), remap_flag, deck8[j], agents[j] if j < len(agents) else "",
                        int(rew[j] > 0) if j < len(rew) else 0))
        except Exception as e:              # noqa: BLE001
            if len(out) < 3:
                print(f"    帧 t={t} seat={j} 失败: {type(e).__name__} {str(e)[:90]}")
            continue
    return out, agents


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/distill_index.jsonl")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--materialize", action="store_true")
    ap.add_argument("--games", type=int, default=6, help="校准用多少个整局文件")
    ap.add_argument("--archetypes", default="", help="限定原型（4 位哈希前缀，逗号分隔）")
    ap.add_argument("--val-pct", type=int, default=20)
    ap.add_argument("--out", default="data/bc_rows.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-games", type=int, default=0, help="最多处理多少局（0=全部）")
    ap.add_argument("--deck-prefix", default="", help="只保留**出手方**卡组匹配的帧（如 0d86）")
    ap.add_argument("--agent-re", default="", help="只保留**出手方**名字匹配的帧（正则）")
    ap.add_argument("--shuffle-games", action="store_true",
                    help="随机打散局序再取前 --max-games 局（避免总是取时间最早的 N 局）")
    a = ap.parse_args()

    # 从索引里收集文件 → 原型（保持整局为单位），
    # 同时建 (file, step, seat) -> (deck, agent) 以支持**按帧**过滤
    files = {}
    per_frame = {}
    for line in open(ROOT / a.index):
        r = json.loads(line)
        files.setdefault(r["file"], r["deck"])
        per_frame[(r["file"], r["step"], r["seat"])] = (r["deck"], r["agent"])
    want = [x.strip() for x in a.archetypes.split(",") if x.strip()]
    if want:
        files = {f: d for f, d in files.items() if deck_archetype(d) in want}
    import re as _re
    pat = _re.compile(a.agent_re) if a.agent_re else None
    paths = sorted(files)
    # 有帧级过滤时：只保留"含匹配帧"的局（大幅减少读盘），并可随机打散
    if a.deck_prefix or pat:
        hit = set()
        for (f, _st, _se), (d, ag) in per_frame.items():
            if (a.deck_prefix and d.startswith(a.deck_prefix)) or (pat and pat.search(ag or "")):
                hit.add(f)
        before = len(paths)
        paths = [x for x in paths if x in hit]
        print(f"含目标帧的局：{len(paths)}/{before}")
    if a.shuffle_games:
        import random as _rnd
        _rnd.Random(a.max_games or 0).shuffle(paths)
        print("已随机打散局序")

    def make_keep(rel):
        dp, pt = a.deck_prefix, pat
        if not dp and not pt:
            return None
        def keep(seat, step):
            v = per_frame.get((rel, step, seat))
            if v is None:
                return False
            d, ag = v
            if dp and not d.startswith(dp):
                return False
            if pt and not pt.search(ag or ""):
                return False
            return True
        return keep
    print(f"索引覆盖 {len(paths)} 局" + (f"（限定原型 {want}）" if want else ""))

    if a.calibrate:
        sample = paths[:a.games]
        print(f"校准样本：{len(sample)} 局（整局）\n")
        agg = collections.defaultdict(lambda: [0, 0])      # key -> [一致, 单动作帧]
        multi = 0
        frames_total = 0
        pool = {0: load_agent(a.device), 1: load_agent(a.device)}
        for n, rel in enumerate(sample, 1):
            p = ROOT / rel
            if not p.exists():
                print(f"  [{n}/{len(sample)}] 缺文件 {rel}")
                continue
            recs, agents = iter_game_frames(p, pool, keep_fn=make_keep(rel))
            d = files[rel]
            for seat, t, n_opt, act0, actlen, lg, _row, _rf, _dk, _ag, _wn in recs:
                if act0 is None:
                    continue
                frames_total += 1
                if actlen > 1:
                    multi += 1
                    continue
                hit = int(int(lg.argmax()) == act0)
                agg[("ALL", "all")][0] += hit
                agg[("ALL", "all")][1] += 1
                agg[(deck_archetype(d), "all")][0] += hit
                agg[(deck_archetype(d), "all")][1] += 1
            print(f"  [{n}/{len(sample)}] {rel.split('/')[-1]}  原型 {deck_archetype(d)}  "
                  f"帧 {len(recs)}")
        print(f"\n总帧 {frames_total}   多动作帧 {multi} ({100*multi/max(frames_total,1):.1f}%，未计入)")
        print(f"\n{'原型':10s}{'单动作帧':>9s}{'一致率':>9s}")
        rows = [(k[0], v) for k, v in agg.items() if k[0] != "ALL" and v[1]]
        for k, (hit, tot) in sorted(rows, key=lambda x: -x[1][1]):
            print(f"{k:10s}{tot:9d}{hit/tot:8.1%}")
        if agg[("ALL", "all")][1]:
            hit, tot = agg[("ALL", "all")]
            print(f"{'合计':10s}{tot:9d}{hit/tot:8.1%}")
            print(f"\n【校准口径】单动作帧 · 合法动作内 argmax · forward_eval 原始输出 · "
                  f"{len(sample)} 整局")
            print(f"【参考值】此前单局单座位各测 40/60 帧得 52.5% / 58.3%，"
                  f"本次为多局多座位合并，若落在 50–70% 区间即视为管线一致")

    if a.materialize:
        import numpy as _np
        todo = paths[: a.max_games] if a.max_games else paths
        print(f"\n物化：{len(todo)} 局，device={a.device}")
        ys, ns, archs, splits, decks, agents_c, wons = [], [], [], [], [], [], []
        xdat = (ROOT / a.out).with_suffix(".f32")
        xf = open(xdat, "wb")
        nframes = 0
        pool = {0: load_agent(a.device), 1: load_agent(a.device)}
        skipped_mask = 0
        remapped = 0
        multi = 0
        for n, rel in enumerate(todo, 1):
            pth = ROOT / rel
            if not pth.exists():
                continue
            arch = deck_archetype(files[rel])
            sp = split_of(rel, a.val_pct)
            try:
                recs, _ = iter_game_frames(pth, pool, keep_fn=make_keep(rel))
            except Exception as e:                     # noqa: BLE001
                print(f"  [{n}/{len(todo)}] {rel} 失败 {type(e).__name__}")
                continue
            keep_n = 0
            for seat, t, n_opt, act0, actlen, lg, row, remap_flag, dk, ag_, wn in recs:
                if actlen != 1:
                    multi += 1
                    continue
                if act0 is None:                        # 真非法（找不到等价代表）
                    skipped_mask += 1
                    continue
                if remap_flag:
                    remapped += 1
                xf.write(row.astype(_np.float32, copy=False).tobytes()); nframes += 1
                ys.append(act0); ns.append(n_opt)
                decks.append(dk); agents_c.append(ag_); wons.append(wn)
                archs.append(arch); splits.append(sp)
                keep_n += 1
            if n % 10 == 0 or n == len(todo):
                print(f"  [{n}/{len(todo)}] 累计 {nframes:,} 帧  ({rel.split('/')[-1][:34]})")
        xf.close()
        row_w = 10944
        y = _np.array(ys, dtype=_np.int16)
        nopt = _np.array(ns, dtype=_np.int16)
        arch = _np.array(archs); split = _np.array(splits)
        out = ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        _np.savez_compressed(out, y=y, nopt=nopt, arch=arch, split=split,
                             deck=_np.array(decks), agent=_np.array(agents_c),
                             won=_np.array(wons, dtype=_np.int8))
        print(f"\n写入 {out}（元数据）+ {xdat.name}（X 流式，{nframes:,}x{row_w} f32，"                        f"{nframes*row_w*4/2**30:.2f} GB）")
        print(f"  重映射 {remapped:,} 帧（等价类代表）  真非法丢弃 {skipped_mask:,}  多动作 {multi:,}")
        tot_single = nframes + skipped_mask
        print(f"  单动作帧合法率 {100*nframes/max(tot_single,1):.2f}%  "
              f"{'✅ >98%' if nframes/max(tot_single,1)>0.98 else '✗ <98%，停下排查'}")
        for s_ in ("train", "val"):
            m = split == s_
            print(f"  {s_}: {int(m.sum()):,} 帧  原型 {sorted(set(arch[m].tolist()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
