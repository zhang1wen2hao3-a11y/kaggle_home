#!/usr/bin/env python3
"""**策略差异量化**：训练前后，我们和 1400+ 真人的五项动作差了多少、补上了百分之几。

用户 2026-11-23 要求：「选一个我们和 1400 分以上真人在策略上有比较大差异的模型进行比对…
面对这类策略差异的场景，训练后是否有优化，要量化一下，每个动作的优化比例」。

## 口径

对每个动作指标（贴能/攻击/撤退/进化/首攻回合），给三组数：

| 记号 | 含义 | 数据来源 |
|---|---|---|
| **人类** `h` | 1400+ 真人打同一副牌、同一对手的均值 | 真人回放目录（如 `epsoide_miwa` 且 `--human-player ミワハルキ`） |
| **训练前** `b` | 起点驾驶员 | 训练前录的回放目录（可用 `tools/dump_teacher_games.py` 生成） |
| **训练后** `n` | 新驾驶员 | 同上，换权重 |

输出三类比例（这才是"优化比例"）：

* **补上的缺口** = `(n − b) / (h − b)` —— 训练把我们与人类的差距补掉了百分之多少（最有用的一个）；
* **相对人类** = `n / h` —— 还差多少（判定阈值：贴能/进化 ≥85%，攻击 ≥90%，撤退 ≤130%）；
* **绝对变化** = `n − b`。

## 用法

    # 三步：先各录一批回放（教师/新候选 vs 同一对手），再量化
    ./tools/py tools/dump_teacher_games.py --teacher build/agents/candE1D7R4 \
        --opp build/agents/cand7e39S --matches 40 --out data/style/e1d7_base --tag base
    ./tools/py tools/dump_teacher_games.py --teacher build/agents/candE1D7S0 \
        --opp build/agents/cand7e39S --matches 40 --out data/style/e1d7_s0 --tag s0

    ./tools/py tools/style_gap.py --deck z_ogerpon_toolbox_miwa__e1d7 \
        --opp teal_mask_ogerpon_ex_applin__7e39 \
        --human epsoide_miwa --human-player ミワハルキ \
        --base data/style/e1d7_base --new data/style/e1d7_s0 --csv docs/策略差异_e1d7_S0.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import style_metrics as SM          # noqa: E402  复用同一套扫描口径

# 判定阈值：训练后应达到"人类水平的百分比"（撤退是上限）
GATE = dict(attach=0.85, attack=0.90, evolve=0.80, retreat=1.30)
# ★ 2026-11-23：**一律用"每回合"速率**，不用"每局次数"。
#   原因：我们自己录的回放（本地引擎）与 Kaggle 回放**局长不同**（实测本地约 130 个动作/局、
#   Kaggle 约 467 个），按"每局"比会系统性偏低。按"每我方回合"归一后两边可比。
def _rate(g, key):
    ot = g.get("own_turns") or 0
    v = g.get(key)
    return (v / ot) if (ot and v is not None) else None


METRICS = [("attach", "贴能/回合", "high"), ("attack", "攻击/回合", "high"),
           ("retreat", "撤退/回合", "low"), ("evolve", "进化/回合", "high"),
           ("play", "出牌/回合", "info")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True)
    ap.add_argument("--opp", required=True)
    ap.add_argument("--human", required=True, help="真人回放目录")
    ap.add_argument("--human-player", default="", help="真人名字（子串匹配）")
    ap.add_argument("--base", required=True, help="训练前的回放目录")
    ap.add_argument("--new", required=True, help="训练后的回放目录")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    pool = SM.pool()

    def resolve(spec):
        for h, nm in pool.items():
            if spec in (nm, h) or nm.endswith("__" + spec):
                return h, nm
        raise SystemExit(f"✗ 找不到牌 {spec}")

    my_h, my_name = resolve(a.deck)
    op_h, op_name = resolve(a.opp)
    H = SM.collect(a.human, my_h, op_h, a.human_player or "")
    B = SM.collect(a.base, my_h, op_h, "")
    N = SM.collect(a.new, my_h, op_h, "")
    if not (H and B and N):
        print(f"✗ 缺数据：人类={bool(H)} 训练前={bool(B)} 训练后={bool(N)}", file=sys.stderr)
        return 2

    print(f"我方牌 {my_name}   对手 {op_name}")
    print(f"人类 {a.human}{(' / ' + a.human_player) if a.human_player else ''}：{H['n']} 局 胜 {H['wins']}")
    print(f"训练前 {a.base}：{B['n']} 局 胜 {B['wins']}")
    print(f"训练后 {a.new}：{N['n']} 局 胜 {N['wins']}\n")
    print(f"{'指标':12s}{'人类':>9s}{'训练前':>9s}{'训练后':>9s}{'绝对变化':>10s}{'补上缺口':>10s}{'相对人类':>10s}{'方向':>7s}")
    rows = []
    ok = True
    print(f"{'（每我方回合速率；首攻回合单列）':40s}")
    for key, label, kind in METRICS:
        h, b, n = _rate(H, key), _rate(B, key), _rate(N, key)
        if h is None or b is None or n is None or (kind != "info" and h == 0):
            continue
        gap = h - b
        closed = (n - b) / gap if abs(gap) > 1e-9 else float("nan")
        ratio = n / h if h else float("nan")
        trend = "→"
        if kind == "high":
            trend = "✅" if n > b else ("➖" if abs(n - b) < 1e-9 else "❌")
        elif kind == "low":
            trend = "✅" if n < b else ("➖" if abs(n - b) < 1e-9 else "❌")
        if key in GATE:                     # ★ 按指标名查阈值（原来错用 kind 查，永远查不到 ⇒ 闸门形同虚设）
            good = (ratio <= GATE[key]) if kind == "low" else (ratio >= GATE[key])
            ok = ok and good
            trend += "·" + ("过" if good else "未过")
        print(f"{label:12s}{h:9.2f}{b:9.2f}{n:9.2f}{n-b:+10.2f}{closed*100:9.0f}%{ratio*100:9.0f}%{trend:>7s}")
        rows.append([label, f"{h:.2f}", f"{b:.2f}", f"{n:.2f}", f"{n-b:+.2f}",
                     f"{closed*100:.0f}%", f"{ratio*100:.0f}%", trend])

    # 首攻回合（越低越好）单独一行
    h, b, n = H.get("first_attack"), B.get("first_attack"), N.get("first_attack")
    if None not in (h, b, n):
        gap = h - b
        closed = (n - b) / gap if abs(gap) > 1e-9 else float("nan")
        ratio = n / h if h else float("nan")
        trend = "✅" if n < b else ("➖" if abs(n - b) < 1e-9 else "❌")
        print(f"{'首攻回合':12s}{h:9.2f}{b:9.2f}{n:9.2f}{n-b:+10.2f}{closed*100:9.0f}%{ratio*100:9.0f}%{trend:>7s}")
        rows.append(["首攻回合", f"{h:.2f}", f"{b:.2f}", f"{n:.2f}", f"{n-b:+.2f}",
                     f"{closed*100:.0f}%", f"{ratio*100:.0f}%", trend])
    win_b = B["wins"] / B["n"] if B["n"] else 0.0
    win_n = N["wins"] / N["n"] if N["n"] else 0.0
    print(f"{'胜率':12s}{H['wins']/H['n']*100:8.0f}%{win_b*100:8.0f}%{win_n*100:8.0f}%"
          f"{(win_n-win_b)*100:+9.1f}pp{'':>10s}{'':>10s}{'':>7s}")
    print(f"\n判定（贴能 ≥85% / 攻击 ≥90% / 进化 ≥80% / 撤退 ≤130% 人类水平）：{'✅ 通过' if ok else '❌ 未通过'}")

    if a.csv:
        out = os.path.join(ROOT, a.csv)
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["指标", "人类", "训练前", "训练后", "绝对变化", "补上缺口", "相对人类", "方向"])
            w.writerows(rows)
            w.writerow(["胜率", f"{H['wins']/H['n']*100:.1f}%", f"{win_b*100:.1f}%",
                        f"{win_n*100:.1f}%", f"{(win_n-win_b)*100:+.1f}pp", "", "", "✅" if ok else "❌"])
        print(f"→ {out}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
