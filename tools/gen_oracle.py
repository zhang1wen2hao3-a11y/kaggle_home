#!/usr/bin/env python3
"""导出 **oracle 标签**：把回放里"双方各自看到的观测"合并成真实隐藏状态。

## 为什么需要

我们想做的三件事全都需要"真值标签"：
1. **诊断**："我们现有的特征能预测多少对手手牌？"（判据式诊断，像"学自己的规则"那样用已知答案当尺子）
2. **辅助任务**：训练时用 oracle 训 `aux_hand`/`aux_prize`（他们的做法，预测只在训练期）
3. **以后自己训**：整条 native 训练栈的 `oracle` 块（136 维）就是它

而 `README.md` 里那条"对手推断模型 ❌ 未开始 —— 缺 oracle 数据"的阻碍**不成立**：
回放里**每个座位都能看到自己的手牌**，把同一步两个座位的观测拼起来，
就得到"某一方视角下的全部隐藏信息"——这正是 oracle。

## 关键实现点

* 同一步索引 `t` 上，`steps[t][i].observation` 是**座位 i 自己的视角**；
  所以对手手牌 = `steps[t][opp].observation.current.players[opp].hand`。
* 并**交叉校验**两个视角看到的公开区（场上/弃牌）是否一致 —— 不一致说明对齐错了，直接丢弃并计数。
* 牌库顺序对双方都是隐藏的（`deck` 为 None），所以 oracle 里只有 `deckCount`；奖赏卡若能看则收集 id。

## 用法

    ./run.sh tools/gen_oracle.py --dirs epsoide_port,epsoide_dipam --out data/oracle.jsonl
    ./run.sh tools/gen_oracle.py --dirs epsoide --limit 20      # 快速试
"""
import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _ids(seq):
    return [c.get("id") for c in (seq or []) if c and c.get("id") is not None]


def _public_view(cur, j):
    """座位 j 视角下，**双方共享的公开区**（用来校验两视角是否同一世界）。"""
    pls = (cur or {}).get("players") or []
    out = []
    for k in (0, 1):
        p = pls[k] if len(pls) > k else {}
        if not p:
            out.append(None)
            continue
        out.append((tuple(_ids(p.get("active"))), tuple(_ids(p.get("bench"))),
                    tuple(sorted(_ids(p.get("discard")))), len(p.get("prize") or [])))
    return tuple(out)


def _view_of(cur, j):
    return _public_view(cur, j)


def _find_matching(cur_me, me, steps, t, window):
    """在 [t-window, t+window] 内找对手视角里与 cur_me 同一时刻的那一帧，返回 (cur, 距离)。

    优先取距离最小的（相同则取较晚的），因为"较晚"更可能是最新一帧。
    """
    opp = 1 - me
    target = _view_of(cur_me, me)
    best, best_d = None, None
    for d in range(0, window + 1):
        for tt in ((t + d,) if d == 0 else (t + d, t - d)):
            if not (0 <= tt < len(steps)):
                continue
            st = steps[tt] or []
            if opp >= len(st):
                continue
            c = ((st[opp] or {}).get("observation") or {}).get("current")
            if not c:
                continue
            if _view_of(c, opp) == target:
                return c, d
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="epsoide_port,epsoide_dipam")
    ap.add_argument("--out", default="data/oracle.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--window", type=int, default=24, help="配对搜索窗口（步）")
    args = ap.parse_args()

    files = []
    for d in args.dirs.split(","):
        files += sorted(glob.glob(os.path.join(ROOT, d.strip(), "*.json")))
    if args.limit:
        files = files[: args.limit]
    print(f"扫描 {len(files)} 局回放")

    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stat = collections.Counter()
    pair_hist = collections.Counter()
    WINDOW = args.window
    hand_sizes = []
    n_rec = 0
    with open(out_path, "w", encoding="utf-8") as fo:
        for f in files:
            try:
                d = json.load(open(f))
            except Exception:
                stat["解析失败"] += 1
                continue
            steps = d.get("steps") or []
            names = [a.get("Name") or "" for a in ((d.get("info") or {}).get("Agents") or [])]
            if len(names) < 2:
                continue
            for t in range(len(steps)):
                st = steps[t] or []
                if len(st) < 2:
                    continue
                obs = [(st[0] or {}).get("observation") or {}, (st[1] or {}).get("observation") or {}]
                if not obs[0].get("current") or not obs[1].get("current"):
                    continue
                # 不再要求"同一步索引上两个视角一致"（实测只有约 9% 的步满足，因为观测会被重复投递）
                stat["候选步数"] += 1
                for me in (0, 1):
                    opp = 1 - me
                    cur_me = obs[me]["current"]
                    # ⚠️ 同一步索引上两个座位的观测**并不总是同一时刻**（观测会被重复投递）：
                    #    所以要在窗口内找一个"公开区一致"的对手视角，而不是硬取同一步。
                    cur_op, dist = _find_matching(obs[me]["current"], me, steps, t, WINDOW)
                    if cur_op is None:
                        stat["窗口内找不到同一时刻的对手视角"] += 1
                        continue
                    pair_hist[dist] += 1
                    p_me = (cur_me.get("players") or [])[me] or {}
                    p_op = (cur_op.get("players") or [])[opp] or {}
                    opp_hand = _ids(p_op.get("hand"))
                    my_hand = _ids(p_me.get("hand"))
                    my_prize = _ids(p_me.get("prize"))
                    sel = obs[me].get("select")
                    hand_sizes.append(len(opp_hand))
                    rec = {
                        "file": os.path.basename(f), "step": t, "seat": me,
                        "agent": names[me] if me < len(names) else "",
                        "turn": cur_me.get("turn"),
                        "is_decision": sel is not None,
                        "select_type": (sel or {}).get("type"),
                        "n_options": len((sel or {}).get("option") or []),
                        "my_hand": my_hand,
                        "my_prize_ids": my_prize,
                        "opp_hand": opp_hand,          # ★ oracle 标签（我方视角下的对手手牌）
                        "opp_deck_count": p_op.get("deckCount"),
                        "opp_hand_count": len(opp_hand),
                        "my_deck_count": p_me.get("deckCount"),
                        "my_prize_count": len(p_me.get("prize") or []),
                    }
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_rec += 1
    print(f"写出 {os.path.relpath(out_path, ROOT)}：{n_rec} 条记录")
    print(f"统计：{dict(stat)}")
    if pair_hist:
        print(f"配对距离分布（0=同一步，越大表示对手视角滞后/超前）：{dict(sorted(pair_hist.items()))}")
    if hand_sizes:
        hand_sizes.sort()
        print(f"对手手牌张数：中位 {hand_sizes[len(hand_sizes)//2]}，均值 {sum(hand_sizes)/len(hand_sizes):.2f}，"
              f"最大 {hand_sizes[-1]}")


if __name__ == "__main__":
    main()
