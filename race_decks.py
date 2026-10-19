#!/usr/bin/env python3
"""卡组挑战赛：每套卡组各打 n 局，挑战当前擂主（rollout + sample_official）。

为什么要做这个：LB 上同策略换卡组值 **+174**（`sample_official` 562.9 vs
`grass_heracross` 388.9），是唯一还没榨干、且被线上证实的维度。但**本地用规则策略
做的 25 套 round-robin 排名对线上完全没有预测力**（规则下 `grass_heracross` 第一，
线上它垫底）。差别在于：本地那次是**固定规则策略**下测的。

这一次两侧都用 rollout 当评估器 —— rollout 已被 LB 证实比规则强 +205
（同一卡组），所以它当"裁判"相对更有判别力。

用法：
    ./run.sh race_decks.py --games 160 --workers 8
    ./run.sh race_decks.py --games 60  --workers 8      # 先粗筛一轮再精测

判定纪律：n=160 时 SE≈0.040，**只把 ≥0.60（+2.5σ）的当候选**，
0.5 附近一律视为没差别。
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
CHAMPION_DECK = os.path.join(ROOT, "decks", "sample_official.csv")
WIN_RE = re.compile(r"胜率\s*:\s*([0-9.]+)\s*\(W(\d+)\s*L(\d+)")
GAMES_RE = re.compile(r"对局\s*:\s*(\d+)\s*有效\s*/\s*(\d+)\s*总")


def run_one(deck_path, args, out):
    env = dict(os.environ, PTCG_RO_DECK=deck_path, PYTHONUNBUFFERED="1")
    cmd = [
        os.path.join(ROOT, "run.sh"), "evaluate.py",
        "--agent", args.challenger,
        "--opponent", args.champion,
        "--games", str(args.games),
        "--deck", deck_path,
        "--deck-opponent", CHAMPION_DECK,
        "--workers", str(args.workers),
        "--quiet",
    ]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    txt = p.stdout + p.stderr
    m = WIN_RE.search(txt)
    g = GAMES_RE.search(txt)
    name = os.path.basename(deck_path)[:-4]
    if not m:
        out.write(f"  {name:<22} 解析失败\n{txt[-800:]}\n")
        out.flush()
        return {"deck": name, "win": None, "n": 0, "sec": time.time() - t0}
    win, w, l = float(m.group(1)), int(m.group(2)), int(m.group(3))
    n_eff = int(g.group(1)) if g else (w + l)
    out.write(f"  {name:<22} {win:.4f}  (W{w} L{l}, 有效 {n_eff})  {time.time()-t0:.0f}s\n")
    out.flush()
    return {"deck": name, "win": win, "n": n_eff, "sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=160)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--deck-dir", default=os.path.join(ROOT, "decks"))
    ap.add_argument("--champion", default="agents/ro_attack_w10.py")
    ap.add_argument("--challenger", default="agents/ro_deck.py")
    ap.add_argument("--skip", default="sample_official",
                    help="逗号分隔，跳过这些卡组（默认跳过擂主自己）")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "deck_race.log"))
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    decks = sorted(d for d in glob.glob(os.path.join(args.deck_dir, "*.csv"))
                   if os.path.basename(d)[:-4] not in skip)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"挑战赛：{len(decks)} 套卡组 × {args.games} 局，workers={args.workers}")
    print(f"擂主 = rollout + {os.path.basename(CHAMPION_DECK)}，日志 -> {args.out}")

    results = []
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(f"# 卡组挑战赛  {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"# 擂主 rollout+sample_official | 每套 {args.games} 局 | SI≈"
                  f"{0.5/args.games**0.5:.3f}\n")
        for i, d in enumerate(decks, 1):
            print(f"[{i}/{len(decks)}] {os.path.basename(d)}", flush=True)
            results.append(run_one(d, args, out))

        out.write("\n# ===== 排名（胜率降序，只保留有效对局）=====\n")
        ok = sorted((r for r in results if r["win"] is not None),
                    key=lambda r: -r["win"])
        for r in ok:
            se = 0.5 / max(r["n"], 1) ** 0.5
            z = (r["win"] - 0.5) / se if se else 0.0
            flag = "★候选" if z >= 2.5 else ("" if z < 1.0 else "~")
            out.write(f"  {r['deck']:<22} {r['win']:.4f}  n={r['n']:<4} "
                      f"z={z:+.2f}  {flag}\n")
        bad = [r for r in results if r["win"] is None]
        if bad:
            out.write("\n  解析失败：" + ", ".join(r["deck"] for r in bad) + "\n")

    print(f"\n完成，见 {args.out}")
    ok = sorted((r for r in results if r["win"] is not None), key=lambda r: -r["win"])
    for r in ok[:8]:
        print(f"  {r['deck']:<22} {r['win']:.4f}")


if __name__ == "__main__":
    sys.exit(main())
