#!/usr/bin/env python3
"""⚠️ 本脚本【不是】提交格式 —— 只用于本地隔离自测。

## 比赛的提交格式是单个 .py 文件

    kaggle competitions submit -c <slug> -f submission.py -m "..."

真正用来生成提交文件的是 `make_submission_py.py`。

## 那这个脚本干什么

把 入口 + deck.csv + cg/ 打成一个目录/压缩包，用于：
  1. **本地隔离自测** —— 解压到临时目录，只让该目录在 sys.path 上，
     验证入口文件不依赖本仓库的任何东西（这是提交前最有效的检查）
  2. 静态检查入口的 import 是否只依赖 cg/ 与标准库

## 当初为什么会做 tar.gz（错误的历史）

`main.py` 里曾写"提交时上传的就是 本文件 + cg/ + deck.csv"，
那是把【运行时 /kaggle_simulations/agent/ 目录里有什么】
误读成了【上传时要给什么】。据此做的压缩包方案是错的，已修正。

## 本脚本做三件事

1. **打包**：按上述结构生成 .tar.gz
2. **静态自检**：扫 main.py 的 import，确认只依赖 `cg/` 和标准库。
   一旦有人误加了 `from framework import ...` 这类仓库内部依赖，这里会立刻报出来
   —— 否则要到 Kaggle 上才炸，而且会算作失败
3. **动态自检**：用打包后的 `main.py` 真跑一局，确认它能在"只有包内文件"的环境下工作

用法：
    ./run.sh make_submission.py --agent main.py --out dist/ptcg_greedy.tar.gz
    ./run.sh make_submission.py --agent main.py --out dist/ptcg_greedy.tar.gz --verify-games 20
"""
from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 允许的 import 前缀（标准库由 sys.stdlib_module_names 判定，这里只管本地模块）
# `cg` 恒有；其余由 --extra 实际打包进来的顶层模块自动放行。
BASE_ALLOWED = {"cg"}


def _local_imports(path: str) -> set[str]:
    """列出 main.py 里 import 的、非标准库的顶层模块名。"""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
    std = set(getattr(sys, "stdlib_module_names", ()))
    return {m for m in mods if m not in std}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="main.py", help="入口文件（会改名为 main.py 放进包里）")
    ap.add_argument("--deck", default="deck.csv")
    ap.add_argument("--cg-dir", default="cg")
    ap.add_argument("--out", default="dist/submission.tar.gz")
    ap.add_argument("--extra", action="append", default=[],
                    help="额外要打包的文件或目录（可重复）。"
                         "例如 --extra framework/rollout.py --extra framework/filler.py")
    ap.add_argument("--verify-games", type=int, default=0,
                    help=">0 时用打包后的 main.py 真跑若干局对手池自检")
    ap.add_argument("--verify-opponent", default="agents/greedy.py",
                    help="动态自检时的对手")
    args = ap.parse_args()

    agent_path = os.path.join(HERE, args.agent)
    deck_path = os.path.join(HERE, args.deck)
    cg_dir = os.path.join(HERE, args.cg_dir)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)

    print("=" * 66)
    print("打包 Kaggle 提交")
    print("=" * 66)
    for p, label in ((agent_path, "入口"), (deck_path, "卡组"), (cg_dir, "引擎目录")):
        if not os.path.exists(p):
            raise SystemExit(f"❌ 缺少{label}: {p}")
        print(f"  ✅ {label:6s} {os.path.relpath(p, HERE)}")

    # ---------------- 静态自检
    extras: list[tuple[str, str]] = []      # (源路径, 包内相对路径)
    for e in args.extra:
        src = e if os.path.isabs(e) else os.path.join(HERE, e)
        if not os.path.exists(src):
            raise SystemExit(f"❌ --extra 指定的路径不存在: {e}")
        rel = os.path.relpath(src, HERE)
        extras.append((src, rel if not rel.startswith("..") else os.path.basename(src)))

    allowed = set(BASE_ALLOWED)
    for _src, rel in extras:
        allowed.add(rel.replace("\\", "/").split("/")[0].removesuffix(".py"))

    local = _local_imports(agent_path)
    bad = local - allowed
    print(f"\n  入口文件的本地依赖: {sorted(local) or '（无）'}")
    if bad:
        print(f"  ❌ 检测到仓库内部依赖: {sorted(bad)}")
        print("     提交包里没有这些模块，到 Kaggle 会直接崩。"
              "请把这些功能内联进入口文件，或一起打进包里。")
        raise SystemExit(1)
    print(f"  ✅ 依赖 {sorted(local) or '标准库'}，均在放行名单 {sorted(allowed)} 内")

    # ---------------- 组装
    stage = tempfile.mkdtemp(prefix="ptcg_sub_")
    try:
        shutil.copy2(agent_path, os.path.join(stage, "main.py"))
        shutil.copy2(deck_path, os.path.join(stage, "deck.csv"))
        shutil.copytree(cg_dir, os.path.join(stage, "cg"))
        for src, rel in extras:
            dst = os.path.join(stage, rel)
            os.makedirs(os.path.dirname(dst) or stage, exist_ok=True)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        # 排除 __pycache__ / *.pyc：它们是本机解释器版本的字节码（3.10 和 3.14 都有），
        # 打进包里纯属浪费，且版本不匹配时还可能引起混淆。
        files = []
        for root, dirs, fs in os.walk(stage):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in fs:
                if f.endswith((".pyc", ".pyo")):
                    continue
                files.append(os.path.relpath(os.path.join(root, f), stage))
        print(f"\n  包内文件 {len(files)} 个:")
        for f in sorted(files)[:12]:
            print(f"    {f}")
        if len(files) > 12:
            print(f"    ... 还有 {len(files)-12} 个")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with tarfile.open(out_path, "w:gz") as tf:
            for f in sorted(files):
                tf.add(os.path.join(stage, f), arcname=f)
        size_mb = os.path.getsize(out_path) / 1048576
        print(f"\n  ✅ 已生成 {os.path.relpath(out_path, HERE)}  ({size_mb:.1f} MB)")

        # ---------------- ★ exec 模拟自检（必须）
        # Kaggle 用 `exec(code, env)` 执行提交代码（kaggle_environments/agent.py:57），
        # 那时 __file__ 未定义。若提交文件里用了 __file__，会直接：
        #     NameError: name '__file__' is not defined  ->  status=ERROR
        # 这个坑真实踩过（rollout 版第一次提交就是这样失败的），所以这里强制模拟一次。
        print("\n  用 exec() 模拟 Kaggle 加载方式（无 __file__）...")
        exec_code = (
            "import sys, json\n"
            "sys.path.insert(0, %r)\n"
            "src = open(%r, encoding='utf-8').read()\n"
            "env = {'__name__': 'kaggle_agent'}\n"
            "try:\n"
            "    exec(compile(src, 'main.py', 'exec'), env)\n"
            "except NameError as e:\n"
            "    print('NAMEERROR', e); raise SystemExit(2)\n"
            "assert 'agent' in env, 'no agent() defined'\n"
            "from cg.game import battle_start, battle_select\n"
            "deck = env['agent']({'select': None, 'current': None, 'logs': []})\n"
            "assert isinstance(deck, list) and len(deck) == 60, ('bad deck', deck)\n"
            "obs = battle_start(deck, deck)[0]\n"
            "steps = 0\n"
            "while steps < 500:\n"
            "    cur = obs.get('current')\n"
            "    if cur is None or cur.get('result', -1) != -1: break\n"
            "    obs = battle_select(env['agent'](obs)); steps += 1\n"
            "print('EXEC_OK steps=', steps)\n"
        ) % (os.path.join(stage, "cg"), os.path.join(stage, "main.py"))
        er = subprocess.run([sys.executable, "-c", exec_code], capture_output=True,
                            text=True, timeout=600, cwd=stage)
        if "EXEC_OK" in er.stdout:
            print("  ✅ exec 模拟通过：", er.stdout.strip().splitlines()[-1])
        else:
            print("  ❌ exec 模拟失败 —— 这个提交在 Kaggle 上必然 ERROR：")
            print((er.stderr or er.stdout)[-1200:])
            raise SystemExit(1)

        # ---------------- 动态自检（可选）
        if args.verify_games > 0:
            print(f"\n  用打包后的 main.py 真跑 {args.verify_games} 局做自检...")
            cmd = [sys.executable, os.path.join(HERE, "evaluate.py"),
                   "--agent", os.path.join(stage, "main.py"),
                   "--opponent", (args.verify_opponent
                                  if os.path.isabs(args.verify_opponent)
                                  else os.path.join(HERE, args.verify_opponent)),
                   "--games", str(args.verify_games), "--workers", "1", "--quiet"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            tail = [l for l in r.stdout.splitlines() if "胜率" in l or "对局" in l or "耗时" in l]
            for l in tail:
                print("   ", l.strip())
            if r.returncode != 0:
                print("  ❌ 自检运行失败：")
                print(r.stderr[-1500:])
                raise SystemExit(1)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print()
    print("=" * 66)
    print("打包完成。上传方式：Kaggle → Submit Predictions → 选择该压缩包")
    print("=" * 66)


if __name__ == "__main__":
    main()
