#!/usr/bin/env python3
"""生成比赛要求的单文件 `submission.py`。

## 为什么不是压缩包

比赛的标准提交格式是**单个 .py 文件**：

    kaggle competitions submit -c <slug> -f submission.py -m "..."

`sample_submission/` 那个目录（含 cg/ 与 deck.csv）是给本地开发/自测用的，
平台运行时会自己提供 `cg/` 与 `deck.csv`（放在 /kaggle_simulations/agent/）。

## 本脚本做两件事

1. 把 `deck.csv` 的 60 张卡**内联**进 submission.py —— 首次调用返回卡组是允许
   任意 60 张的，内联后卡组完全由我们掌控，不依赖平台放的是哪份 deck.csv
2. 复用 `main.py` 的全部逻辑（优先级表 + 多招式择优），逐字节取自 main.py，
   避免"打包版和本地测的版本不一致"这类问题

用法：
    ./run.sh make_submission_py.py --agent main.py --deck deck.csv --out dist/submission.py
    ./run.sh make_submission_py.py --agent main.py --deck decks/grass_heracross.csv \
        --out dist/submission_heracross.py
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="main.py")
    ap.add_argument("--deck", default="deck.csv")
    ap.add_argument("--out", default="dist/submission.py")
    args = ap.parse_args()

    agent_path = args.agent if os.path.isabs(args.agent) else os.path.join(HERE, args.agent)
    deck_path = args.deck if os.path.isabs(args.deck) else os.path.join(HERE, args.deck)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)

    with open(deck_path) as f:
        deck = [int(x) for x in f.read().split() if x.strip()]
    if len(deck) != 60:
        raise SystemExit(f"❌ 卡组必须是 60 张，当前 {len(deck)}")

    with open(agent_path, encoding="utf-8") as f:
        src = f.read()

    # ---- 把 read_deck_csv 换成内联常量：卡组不再依赖平台放哪份 deck.csv
    old_fn = '''def read_deck_csv() -> list[int]:
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path) as f:
        return [int(x) for x in f.read().split() if x.strip()][:60]'''
    if old_fn not in src:
        raise SystemExit("❌ 在 agent 文件里找不到预期的 read_deck_csv 实现，请先确认 main.py 未被改动")

    lines = []
    for i in range(0, 60, 10):
        lines.append("    " + ", ".join(str(x) for x in deck[i:i + 10]) + ",")
    new_fn = ("# 卡组直接内联，不依赖平台的 deck.csv（首次调用返回任意 60 张是允许的）\n"
              f"# 来源: {os.path.basename(args.deck)}\n"
              "DECK = [\n" + "\n".join(lines) + "\n]\n\n\n"
              "def read_deck_csv() -> list[int]:\n"
              "    return list(DECK)")
    src = src.replace(old_fn, new_fn)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(src)

    # ---- 自检：必须能编译、能通过引擎跑一局
    import ast
    ast.parse(src)
    print(f"  ✅ 语法 OK")
    print(f"  ✅ 已生成 {os.path.relpath(out_path, HERE)}  ({os.path.getsize(out_path)/1024:.1f} KB)")
    print(f"     卡组 {len(deck)} 张，来源 {args.deck}")
    print(f"     逻辑来源 {args.agent}（逐字节复用，未改写）")

    # 真有 cg/ 才做运行时自检
    try:
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('sub', %r)\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "from cg.game import battle_start, battle_select\n"
            "d = m.read_deck_csv()\n"
            "assert len(d) == 60, len(d)\n"
            "obs = battle_start(d, d)[0]\n"
            "steps = 0\n"
            "while steps < 500:\n"
            "    cur = obs.get('current')\n"
            "    if cur is None or cur.get('result', -1) != -1: break\n"
            "    obs = battle_select(m.agent(obs)); steps += 1\n"
            "print('   跑完', steps, '步, result =', obs.get('current', {}).get('result'))\n"
        ) % (HERE, out_path)
        import subprocess
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=300, cwd=HERE)
        if r.returncode == 0:
            print("  ✅ 运行时自检通过：", r.stdout.strip())
        else:
            print("  ⚠ 运行时自检失败（线上仍可能可跑，请检查）：")
            print("   ", (r.stderr or "")[-600:])
    except Exception as e:
        print(f"  ⚠ 跳过运行时自检: {e}")


if __name__ == "__main__":
    main()
