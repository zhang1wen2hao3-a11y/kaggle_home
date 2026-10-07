#!/usr/bin/env python3
"""实验记录表：为 PTCG AI Battle Challenge 追加/查看实验结果。

和分类比赛不同，这里的"好"是**对局胜率**，不是 F1。
所以记录表的主指标是 win_rate + n_games，便于判断改动是否真的有效。

文件（都在脚本同目录）：
  experiments.csv   实验记录表（可用 Excel 打开，utf-8-sig）
  replays/          对局回放（gitignore）
  submissions/      每次提交的 agent 快照

命令行：
  python exp_log.py list                          # 按胜率降序列出全部实验
  python exp_log.py md                            # 生成 实验记录.md
  python exp_log.py add --name v1 --win 0.42 --games 200
  python exp_log.py lb --exp EXP-0001 --score 350.0

评估脚本里自动记录：
  from exp_log import log_experiment
  log_experiment(name="v1_rulebase", agent="rulebase", win_rate=0.42, n_games=200,
                 eval_scheme="vs-random-bo3", changes="最小可跑 agent")
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "experiments.csv"
REPLAY_DIR = HERE / "replays"
SUB_DIR = HERE / "submissions"
MD_PATH = HERE / "实验记录.md"

COLUMNS = [
    "exp_id", "date", "name", "agent", "eval_scheme",
    "win_rate", "n_games", "lb_score",
    "submission", "changes", "notes",
]
# 跨实验的公共基准：不同实验必须用同一套对手与局数才可比
# 注意：evaluate.py 会按 --opponent 文件名自动生成 eval_scheme（如 vs-random_baseline-n200），
# 这里必须与之一致，否则 exp_log list 的"可与基准比较"提示会失效。
EVAL_REF = "vs-random_baseline-n200"

# ---- 当前基线（改基线时同步更新这里）----
BASELINE = {
    "submission": "main.py",              # 真正提交的文件（自包含）
    "agent": "agents/greedy.py",          # 等价的实验版
    "config": "rulebase.make_agent(order='develop_first', go_first=True, bench_full=True)",
    "opponent": "agents/random_baseline.py",
    # 等价性以 EXP-0004 的实际记录为准。手工跑过一次 0.490、入库那次 0.5067，
    # 二者都在 0.5 附近 —— 结论是"同一策略"，不是某个精确数字。
    "equivalence": "main.py vs agents/greedy.py ≈ 0.5（EXP-0004，n=300）→ 二者是同一策略",
}


def _ensure() -> None:
    REPLAY_DIR.mkdir(exist_ok=True)
    SUB_DIR.mkdir(exist_ok=True)
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()


def read_all() -> list[dict]:
    _ensure()
    with CSV_PATH.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def next_id() -> str:
    rows = read_all()
    if not rows:
        return "EXP-0001"
    nums = [int(r["exp_id"].split("-")[1]) for r in rows if r.get("exp_id", "").startswith("EXP-")]
    return f"EXP-{max(nums) + 1:04d}"


def log_experiment(*, name: str, agent: str = "", eval_scheme: str = EVAL_REF,
                   win_rate: float | None = None, n_games: int | None = None,
                   submission: str = "", changes: str = "", notes: str = "") -> str:
    """追加一条实验记录，返回 exp_id。"""
    _ensure()
    exp_id = next_id()
    row = {
        "exp_id": exp_id,
        "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "name": name, "agent": agent, "eval_scheme": eval_scheme,
        "win_rate": "" if win_rate is None else f"{win_rate:.4f}",
        "n_games": "" if n_games is None else str(n_games),
        "lb_score": "", "submission": submission,
        "changes": changes, "notes": notes,
    }
    with CSV_PATH.open("a", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)
    return exp_id


def set_lb(exp_id: str, score: float) -> None:
    """为某条实验记录写入 Kaggle 排行榜分数。"""
    rows = read_all()
    hit = False
    for r in rows:
        if r.get("exp_id") == exp_id:
            r["lb_score"] = f"{score:.5f}"
            hit = True
    if not hit:
        raise SystemExit(f"未找到 {exp_id}")
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except ValueError:
        return 0.0


def cmd_list(_args) -> None:
    rows = read_all()
    if not rows:
        print("还没有实验记录。")
        return
    rows.sort(key=lambda r: -_f(r, "win_rate"))
    best = _f(rows[0], "win_rate")
    print(f"{'ID':9s} {'name':22s} {'agent':14s} {'eval':22s} {'胜率':>7s} {'局数':>6s} "
          f"{'LB':>9s} {'对比':>8s}")
    print("-" * 104)
    for r in rows:
        delta = _f(r, "win_rate") - best
        print(f"{r['exp_id']:9s} {r['name'][:22]:22s} {r['agent'][:14]:14s} "
              f"{r['eval_scheme'][:22]:22s} {_f(r,'win_rate'):7.4f} "
              f"{(r.get('n_games') or '-'):>6s} {(r.get('lb_score') or '-'):>9s} {delta:+8.4f}")
    print(f"\n共 {len(rows)} 条记录；基准(评估) = {EVAL_REF}")
    print(f"表文件: {CSV_PATH}")


def cmd_md(_args) -> None:
    rows = read_all()
    rows.sort(key=lambda r: -_f(r, "win_rate"))
    lines = [
        "# 实验记录（PTCG AI Battle Challenge）",
        "",
        f"> 基准评估：`{EVAL_REF}`（所有实验必须同一套对手与局数才可比）  ",
        f"> 生成时间：{dt.datetime.now():%Y-%m-%d %H:%M}  ·  共 {len(rows)} 条",
        "",
        "## 当前基线",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 提交入口（真正上传的） | `{BASELINE['submission']}` |",
        f"| 等价实验版 | `{BASELINE['agent']}` |",
        f"| 策略配置 | `{BASELINE['config']}` |",
        f"| 基准对手 | `{BASELINE['opponent']}` |",
        f"| 等价性验证 | {BASELINE['equivalence']} |",
        "",
        "> `main.py` 必须自包含（提交时只上传它 + `cg/` + `deck.csv`，不能 import `agents/`），",
        "> 所以它与 `agents/greedy.py` 是同一策略的两份实现，改动时**两边都要改**。",
        "",
        "> ⚠️ EXP-0002（greedy 0.850）与 EXP-0003（main 0.805）是**同一策略**的两次测量，",
        "> 差异属抽样噪声（n=200 时标准误≈0.025），不要解读为版本差异。",
        "",
        "| ID | 实验 | Agent | 评估方案 | 胜率 | 局数 | **LB** | 改动点 | 备注 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| {exp_id} | {name} | {agent} | {eval_scheme} | **{win_rate}** | "
                     "{n_games} | {lb_score} | {changes} | {notes} |".format(**{
                         k: str(r.get(k, "")).replace("|", "/") for k in COLUMNS}))
    lines += [
        "",
        "## 说明",
        "",
        "- **本地胜率**是唯一可信的对比依据；总共只有 2 次计分提交，不要用 LB 反复试探。",
        "- **框架自检优先**：任何策略对比前，先确认 `random vs random ≈ 0.5`。",
        "  本项目曾因 `agents/random.py` 遮蔽标准库 `random`（对手开局即崩、崩溃被静默记为输）",
        "  导致全部配置被误记为 100% 胜率。`evaluate.py` 现在会红字警告并排除崩溃局。",
        "- 每次评估的对手和局数必须固定（见 `eval_scheme`），否则胜率不可比。",
        "- 新实验只有胜率 ≥ 当前最佳才保留，否则回滚。",
        "- 提交前先在本地跑够局数（建议 ≥200 局），Bo3 的方差比单局大。",
    ]
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成 {MD_PATH}")


def cmd_add(args) -> None:
    exp_id = log_experiment(
        name=args.name, agent=args.agent, eval_scheme=args.eval,
        win_rate=args.win, n_games=args.games, submission=args.sub or "",
        changes=args.changes or "", notes=args.notes or "")
    print(f"已记录 {exp_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="列出实验（按胜率降序）").set_defaults(func=cmd_list)
    sub.add_parser("md", help="生成 实验记录.md").set_defaults(func=cmd_md)

    lb = sub.add_parser("lb", help="为某条实验写入 LB 分数")
    lb.add_argument("--exp", required=True)
    lb.add_argument("--score", type=float, required=True)
    lb.set_defaults(func=lambda a: (set_lb(a.exp, a.score), print(f"{a.exp} LB={a.score:.5f}"))[-1])

    a = sub.add_parser("add", help="手工追加一条实验")
    a.add_argument("--name", required=True)
    a.add_argument("--agent", default="")
    a.add_argument("--eval", default=EVAL_REF)
    a.add_argument("--win", type=float, required=True)
    a.add_argument("--games", type=int, default=None)
    a.add_argument("--sub", default=None)
    a.add_argument("--changes", default=None)
    a.add_argument("--notes", default=None)
    a.set_defaults(func=cmd_add)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
