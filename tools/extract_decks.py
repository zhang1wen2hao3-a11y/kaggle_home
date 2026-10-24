#!/usr/bin/env python3
"""从回放里抽出双方**精确的 60 张卡表**，做成卡组语料（deck corpus）。

## 为什么现在才能做

回放里 `steps[0]` 是"交卡组"那一步（`select=None`、`action` 为空），真正的卡表是
**下一步同座位**那个长度 60 的 action（见 `framework/replay.py` 里实测的格式说明）。
以前用"可见牌"重建只有 20~27 张，被迫补 33~40 张基本能量，把"少能量多训练家"的真实构筑
扭曲成"多能量卡组"（`别再试.md` F3），所以那份重建结果不能用作评价池。

现在每份回放都能给出两条精确卡表 ⇒ 可以做三件事：
  1. **份额**：各 archetype 在池子里的占比
  2. **卡组强度**：把 deck 效应和 pilot（队伍）效应分离
  3. **候选卡组池**：给"同一个策略网络换卡组"的本地 gauntlet 提供原料

## 用法

    ./run.sh tools/extract_decks.py --dirs epsoide_teams2,epsoide_field --out data/deck_corpus.csv
"""
import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from framework.replay import decks_of  # noqa: E402


def deck_hash(ids) -> str:
    """排序后的卡 id 列表的短哈希 —— 同一副牌（引擎只看 multiset）得到同一个哈希。"""
    return hashlib.sha256(",".join(str(x) for x in sorted(ids)).encode()).hexdigest()[:8]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="epsoide_teams2,epsoide_field,epsoide")
    ap.add_argument("--out", default="data/deck_corpus.csv")
    args = ap.parse_args()

    files = []
    for d in args.dirs.split(","):
        files += sorted(glob.glob(os.path.join(ROOT, d.strip(), "*.json")))
    print(f"扫描 {len(files)} 局回放")

    rows = []
    missing = 0
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        steps = d.get("steps") or []
        decks = decks_of(steps)
        if len(decks) < 2:
            missing += 1
            continue
        agents = (d.get("info") or {}).get("Agents") or []
        names = [a.get("Name") or "" for a in agents]
        teams = (d.get("info") or {}).get("TeamNames") or []
        rew = d.get("rewards") or []
        for seat, ids in decks.items():
            won = None
            if len(rew) > seat and rew[seat] is not None:
                others = [r for j, r in enumerate(rew) if j != seat and r is not None]
                if others:
                    won = rew[seat] > max(others)
            rows.append({
                "file": os.path.basename(f),
                "seat": seat,
                "agent": names[seat] if seat < len(names) else "",
                "team": (teams[seat] if seat < len(teams) else (names[seat] if seat < len(names) else "")),
                "deck_hash": deck_hash(ids),
                "won": "" if won is None else int(won),
                "deck": " ".join(str(x) for x in ids),
            })
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)) or ".", exist_ok=True)
    out = os.path.join(ROOT, args.out)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "seat", "agent", "team", "deck_hash", "won", "deck"])
        w.writeheader()
        w.writerows(rows)
    print(f"写出 {out}：{len(rows)} 条卡表记录"
          f"（{missing} 局取不到双方卡表）")

    share = collections.Counter(r["deck_hash"] for r in rows)
    print(f"\n不同卡表 {len(share)} 副；最常见的 10 副：")
    print(f"  {'deck':<10}{'出现':>6}{'份额':>8}")
    for h, n in share.most_common(10):
        print(f"  {h:<10}{n:>6}{n / max(len(rows), 1):>7.1%}")

    # 队伍 × 卡表：同一队伍用了几副牌（用来判断"卡组效应能否与 pilot 效应分离"）
    team_decks = collections.defaultdict(set)
    for r in rows:
        team_decks[r["team"]].add(r["deck_hash"])
    multi = sum(1 for t, s in team_decks.items() if len(s) > 1)
    print(f"\n队伍数 {len(team_decks)}；其中用过 ≥2 副不同卡表的队伍 {multi} 个"
          f"（这些队伍是分离 deck/pilot 效应的关键，参见 公开方案综述.md T1）")


if __name__ == "__main__":
    main()
