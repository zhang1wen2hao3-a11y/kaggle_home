#!/usr/bin/env python3
"""**两机流水线**：训练机出模型 → 评估机跑评估 → 本地判定。

    训练机（GPU，2080Ti）   每 13 分钟一个 400k 包，本地不下载也能排队
    评估机（CPU 32 核）      1250 局 ≈ 12 分钟，16 个面板
    本地                    只管判定与记账（dual_train ingest）

为什么要三地分工：训练 13 分钟 / 评估 12 分钟，但评估机只有一个 —— 训练机会先跑完
空着，所以我们**在训练机上排队**（一次性排 6 个包，训完直接从面板关机），
评估机慢慢消化。

用法：

    ./tools/py tools/two_node.py status --names db04_vs_grookey
    ./tools/py tools/two_node.py eval --name r1A_db04_g6            # 评估一个模型
    ./tools/py tools/two_node.py eval --name r1A2_db04_g6 --push    # 先从训练机拉回再推过去
    ./tools/py tools/two_node.py ingest --names db04_vs_grookey --name r1A_db04_g6
    ./tools/py tools/two_node.py loop --names db04_vs_grookey \\
            --models r1A_db04_g6,r1A2_db04_g6,r1A3_db04_g6          # 串起来跑

连接信息：`ptcg_league/remote.json`（训练机）+ `ptcg_league/evalnode.json`（评估机）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PL = os.path.join(ROOT, "ptcg_league")
PY = os.path.join(ROOT, "tools", "py")
BLOB = 18205408
PANEL = {  # 评估机上的面板对手（与 dual_train.FIELD 的键一致）
    "ogerpon_7e39": "build/agents/ogerpon_7e39",
    "abra_f5f8": "build/agents/abra_f5f8",
    "grookey_2617": "build/agents/grookey_2617",
    "shuppet_0d86": "build/agents/shuppet_0d86",
    "dreepy_3eaa": "build/agents/dreepy_3eaa",
}


def cfg(fn, defaults):
    p = os.path.join(PL, fn)
    d = dict(defaults)
    if os.path.exists(p):
        d.update(json.load(open(p)))
    if not d.get("host"):
        raise SystemExit(f"✗ 缺 {p}（需要 host/port）")
    return d


def trainer():
    return cfg("remote.json", dict(host="", port=22, python="python3",
                                   repo="/root/ptcg_train_kit/port_repo",
                                   out="/root/autodl-tmp/models",
                                   logs="/root/autodl-tmp/runlogs"))


def evaluator():
    return cfg("evalnode.json", dict(host="", port=22, python="python3",
                                     root="/root/ptcg_eval"))


def ssh(c, cmd, timeout=3600):
    full = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-p", str(c["port"]), c["host"], cmd]
    return subprocess.run(full, text=True, capture_output=True, timeout=timeout)


def sh(cmd, **kw):
    print("  $", " ".join(str(x) for x in cmd)[:160], flush=True)
    return subprocess.run([str(x) for x in cmd], check=True, **kw)


def state(names):
    p = os.path.join(PL, "state", f"dual_{names}.json")
    if not os.path.exists(p):
        raise SystemExit(f"✗ 没有 state：{p}（先 dual_train.py init）")
    return json.load(open(p))


def cmd_status(a):
    t, e = trainer(), evaluator()
    st = state(a.names)
    print(f"== state {a.names} == round {st['round']} 下一轮训 {st['turn']} "
          f"primary={st.get('primary')}")
    for s in ("A", "B"):
        d = st["sides"][s]
        print(f"  {s}: model={d['model']} deck={d['deck']} flat={d['flat']}")
    rt = ssh(t, f"ls -l --time-style=+%H:%M {t['out']}/*.bin 2>/dev/null; "
                f"tail -2 {t['logs']}/chain.log {t['logs']}/chain2.log 2>/dev/null")
    print("\n-- 训练机 --")
    print(rt.stdout.rstrip() or "（无产物）")
    re_ = ssh(e, f"ls -l --time-style=+%H:%M /root/ev_*.json 2>/dev/null | tail -6; "
                 f"cat /root/run_eval.log 2>/dev/null | tail -4")
    print("\n-- 评估机 --")
    print(re_.stdout.rstrip() or "（无结果）")
    return 0


def pull_from_trainer(name, t):
    """把训练机上的 .bin 拉到本地 models/ppo/（已存在就跳过）。"""
    dst = os.path.join(ROOT, "models/ppo", name + ".bin")
    if os.path.exists(dst) and os.path.getsize(dst) == BLOB:
        print(f"  本地已有 {name}.bin")
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    r = ssh(t, f"test -f {t['out']}/{name}.bin && stat -c%s {t['out']}/{name}.bin")
    if r.stdout.strip() != str(BLOB):
        raise SystemExit(f"✗ 训练机上没有 {name}.bin（或大小不对：{r.stdout.strip()!r}）")
    sh(["scp", "-P", str(t["port"]), "-o", "BatchMode=yes",
        f"{t['host']}:{t['out']}/{name}.bin", dst])
    if os.path.getsize(dst) != BLOB:
        raise SystemExit("✗ 拉回的大小不对")
    return dst


def eval_on_node(name, deck, e, matches, force=False):
    """把模型推到评估机、建 agent 目录、跑评估、把 json 拉回本地。"""
    rpath = f"{e['root']}/models/{name}.pt"
    lpt = os.path.join(ROOT, "models/ppo", name + ".pt")
    ljson = os.path.join(PL, "logs", f"{name}_remote.json")
    if os.path.exists(ljson) and not force:
        d = json.load(open(ljson))
        if d.get("per_opp"):
            print(f"  评估机已有 {name} 的结果，跳过（--force 重跑）")
            return ljson
    # ① 上传 .pt（评估机只需要 pt，不需要 .bin）
    if not os.path.exists(lpt):
        sh([PY, "-c", (
            "import sys;sys.path.insert(0,'.');"
            "from tools.dual_train import agent_dir_from_blob;"
            f"agent_dir_from_blob('models/ppo/{name}.bin','{name}',"
            f"'decks/pool_field/{deck}.csv')")])
    rr = ssh(e, f"test -f {rpath} && stat -c%s {rpath} || echo missing")
    if rr.stdout.strip() == "missing" or force:
        sh(["mkdir", "-p", os.path.join(ROOT, "dist")])
        sh(["scp", "-o", "BatchMode=yes", "-P", str(e["port"]), lpt,
            f"{e['host']}:{rpath}"])
    # ② 建 agent 目录
    r = ssh(e, f"cd {e['root']} && PYTHON={e['python']} bash tools/make_agent_dir.sh "
               f"build/agents/ag_{name} {rpath} decks/pool_field/{deck}.csv")
    if r.returncode != 0:
        raise SystemExit(f"✗ 评估机建 agent 目录失败：{r.stderr[-400:]}")
    # ③ 评估
    opps = " ".join(f"--opp {k}={v}" for k, v in PANEL.items())
    cmd = (f"cd {e['root']} && {e['python']} ptcg_league/eval_pair.py --cand build/agents/ag_{name} "
           f"{opps} --matches {matches} --workers 0 --seed-base 950000 "
           f"--out /root/ev_ag_{name}.json --logdir /root/ev_ag_{name}")
    # ★ 不能用一个 ssh 会话干等：实测评估机的 sshd 会在 ~10 分钟处把连接掐掉
    #   （client_loop: send disconnect: Broken pipe），前台跑的 eval_pair 收 HUP 就死，
    #   结果白跑。所以改成 nohup 起 + 轮询结果文件，ssh 只做短交互。
    import time as _t
    print(f"  [评估机] {name}: {matches} 局，预计 {matches/104:.0f} 分钟")
    launch = (f"cd {e['root']} && rm -f /root/ev_ag_{name}.json && "
              f"nohup {cmd} > /root/ev_ag_{name}.log 2>&1 & echo started")
    r = ssh(e, launch, timeout=120)
    if "started" not in r.stdout:
        raise SystemExit(f"✗ 启动失败：{r.stdout} {r.stderr[-300:]}")
    t0 = _t.time()
    deadline = t0 + (matches / 104.0) * 60 * 2.5 + 300
    while _t.time() < deadline:
        _t.sleep(20)
        rr = ssh(e, f"test -f /root/ev_ag_{name}.json && echo DONE || "
                    f"(pgrep -f 'eval_pair[.]py --cand build/agents/ag_{name}' >/dev/null && echo RUN || echo GONE)",
                timeout=60)
        state_ = rr.stdout.strip()
        if state_ == "DONE":
            break
        if state_ == "GONE":
            tail = ssh(e, f"tail -6 /root/ev_ag_{name}.log").stdout
            raise SystemExit(f"✗ 评估进程消失且没有结果：\n{tail}")
        if int(_t.time() - t0) % 120 < 21:
            print(f"    …{name} 已跑 {int(_t.time()-t0)}s")
    else:
        raise SystemExit(f"✗ 评估超时：{name}")
    print(f"  [评估机] {name} 完成，用时 {int(_t.time()-t0)}s")
    os.makedirs(os.path.dirname(ljson), exist_ok=True)
    sh(["scp", "-o", "BatchMode=yes", "-P", str(e["port"]),
        f"{e['host']}:/root/ev_ag_{name}.json", ljson])
    d = json.load(open(ljson))
    n = sum(v["n"] for v in d["per_opp"].values())
    print(f"  ✅ {name}: {n} 场，per_opp={ {k: round(v['score'],3) for k,v in d['per_opp'].items()} }")
    return ljson


def cmd_eval(a):
    t, e = trainer(), evaluator()
    st = state(a.names)
    side = st["turn"]
    deck = st["sides"][side]["deck"]
    if a.push or not os.path.exists(os.path.join(ROOT, "models/ppo", a.name + ".bin")):
        pull_from_trainer(a.name, t)
    eval_on_node(a.name, a.deck or deck, e, a.matches, force=a.force)
    return 0


def cmd_ingest(a):
    st = state(a.names)
    side = st["turn"]
    ljson = a.json or os.path.join(PL, "logs", f"{a.name}_remote.json")
    base = a.base_json or os.path.join(PL, "logs", f"{a.names}_r0{side}_base_remote.json")
    if not os.path.exists(base):
        raise SystemExit(f"✗ 缺基线 json：{base}（训练前那一侧模型的评估结果）")
    run_bin = os.path.join(ROOT, "models/ppo", a.name + ".bin")
    sh([PY, "tools/dual_train.py", "ingest", "--names", a.names, "--bin", run_bin,
        "--cand-json", ljson, "--base-json", base])
    return 0


def cmd_loop(a):
    for name in [x.strip() for x in a.models.split(",") if x.strip()]:
        print(f"\n########## {name} ##########")
        ns = argparse.Namespace(names=a.names, name=name, deck=a.deck,
                                matches=a.matches, push=True, force=False)
        cmd_eval(ns)
        ni = argparse.Namespace(names=a.names, name=name, json="", base_json=a.base_json)
        cmd_ingest(ni)
        print(f"  → {name} 已入库")
    return 0


def cmd_table(a):
    """把已经评估过的所有模型与基线做**同种子配对**对比，输出训练长度曲线。

    注意：这条曲线里的模型是**同侧的不同检查点**，不是 dual 循环的「轮次」，
    所以只做统计比较、不入 state（state 的轮次语义是「这一侧这一轮是否更强」）。
    """
    import math

    def load(fp):
        d = json.load(open(fp))
        return d["per_opp"]

    def mcnemar(b, c):
        n = b + c
        if n == 0:
            return 1.0
        return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

    sys.path.insert(0, ROOT)
    from tools.dual_train import field_ev

    base = load(a.base_json)
    rows = []
    for name in [x.strip() for x in a.models.split(",") if x.strip()]:
        fp = os.path.join(PL, "logs", f"{name}_remote.json")
        if not os.path.exists(fp):
            rows.append((name, None, None, None, None, "未评估"))
            continue
        d = load(fp)
        ev = field_ev(d)
        b = c = 0
        for k in base:
            pb, pc = base[k]["per_match"], d.get(k, {}).get("per_match") or {}
            for sd in set(pb) & set(pc):
                wb, wc = pb[sd] > 0.5, pc[sd] > 0.5
                if wc and not wb:
                    b += 1
                if wb and not wc:
                    c += 1
        rows.append((name, ev, ev - field_ev(base), b, c, mcnemar(b, c)))
    bev = field_ev(base)
    print(f"基线 field_ev = {bev:.4f}   （seeds 950000，1250 局，5 原型面板）")
    print(f"{'模型':22s}{'field_ev':>9}{'Δpp':>8}{'赢/输':>10}{'p':>8}  判定")
    for name, ev, d_, b, c, p in rows:
        if ev is None:
            print(f"{name:22s}{'-':>9}{'-':>8}{'-':>10}{'-':>8}  {p}")
            continue
        verdict = "✅ 显著更好" if (p < 0.05 and d_ >= 0.02) else ("⚠️ 同水平" if p >= 0.05 else "⚠️ 显著更差")
        print(f"{name:22s}{ev:9.4f}{d_*100:+8.2f}{f'{b}/{c}':>10}{p:8.4f}  {verdict}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="训练机 → 评估机 → 本地判定")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status")
    s.add_argument("--names", required=True)
    s.set_defaults(fn=cmd_status)

    e = sub.add_parser("eval", help="在评估机上评估一个模型")
    e.add_argument("--names", required=True)
    e.add_argument("--name", required=True, help="模型名（models/ppo/<name>.bin）")
    e.add_argument("--deck", default="", help="留空=用 state 里该侧的卡组")
    e.add_argument("--matches", type=int, default=1250)
    e.add_argument("--push", action="store_true", help="先从训练机把 .bin 拉回来")
    e.add_argument("--force", action="store_true", help="重跑（忽略已有结果）")
    e.set_defaults(fn=cmd_eval)

    i = sub.add_parser("ingest", help="把评估结果灌进 state（判断本轮是否提升）")
    i.add_argument("--names", required=True)
    i.add_argument("--name", required=True)
    i.add_argument("--json", default="", help="候选结果 json（默认 ptcg_league/logs/<name>_remote.json）")
    i.add_argument("--base-json", default="", help="基线结果 json")
    i.set_defaults(fn=cmd_ingest)

    l = sub.add_parser("loop", help="批量：评估 + 入库，一轮一个模型")
    l.add_argument("--names", required=True)
    l.add_argument("--models", required=True, help="逗号分隔，按顺序")
    l.add_argument("--deck", default="")
    l.add_argument("--matches", type=int, default=1250)
    l.add_argument("--base-json", default="")
    l.set_defaults(fn=cmd_loop)

    t = sub.add_parser("table", help="已评估模型 vs 基线的配对对比表")
    t.add_argument("--base-json", required=True)
    t.add_argument("--models", required=True, help="逗号分隔")
    t.set_defaults(fn=cmd_table)

    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
