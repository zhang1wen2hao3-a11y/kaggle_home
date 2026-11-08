#!/usr/bin/env python3
"""**远端只训、本地只评** —— 训练机（GPU）的驱动脚本。

分工（2026-11-06 定）：

    训练机 A：只跑 db04_line.sh（GPU，约 8 分钟/模型），不跑任何评估
    评估机 B：只跑 eval_pair.py（CPU 32 核，1250 局约 8 分钟），不碰 GPU
    本地    ：dual_train 的 state / 判定 / 记账，不跑重活

为什么值得拆：旧循环一轮要占训练机 26 分钟（8 分钟训练 + 18 分钟纯 CPU 评估），
GPU 全程计费却 70% 时间在等 CPU；拆开之后训练机每模型只占 8 分钟。

本脚本只做"训练机"这一半：

    # 看这台机器上现成有哪些权重（避免每轮上传 18 MB）
    ./tools/py tools/remote_train.py find --name g6_db04

    # 按 state 跑一轮（自动算出 INIT/BANK/anchor/run 名）
    ./tools/py tools/remote_train.py run --names db04_vs_grookey --wait

    # 只看进度 / 看日志
    ./tools/py tools/remote_train.py log --names db04_vs_grookey -n 20

连接参数放 `ptcg_league/remote.json`（不进 git）：

    {"host": "root@connect.nmb1.seetacloud.com", "port": 23268,
     "python": "/root/miniconda3/bin/python3",
     "repo": "/root/ptcg_train_kit/port_repo",
     "out": "/root/autodl-tmp/models", "logs": "/root/autodl-tmp/runlogs"}
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(ROOT, "ptcg_league", "remote.json")
STATE_DIR = os.path.join(ROOT, "ptcg_league", "state")
BLOB = 18205408
DEFAULT = dict(host="", port=22, python="python3",
               repo="/root/ptcg_train_kit/port_repo",
               out="/root/autodl-tmp/models", logs="/root/autodl-tmp/runlogs")


def cfg():
    d = dict(DEFAULT)
    if os.path.exists(CFG_PATH):
        d.update(json.load(open(CFG_PATH)))
    if not d["host"]:
        raise SystemExit(
            f"✗ 缺连接信息：{CFG_PATH}\n"
            f"   写一份：{{\"host\": \"root@<主机>\", \"port\": <端口>}}\n"
            f"   （该文件不进 git，见 .gitignore）")
    return d


def ssh(c, cmd, timeout=120, capture=True):
    """在训练机上跑一条命令（BatchMode：失败就是失败，不会卡在密码提示上）。"""
    full = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-p", str(c["port"]), c["host"], cmd]
    r = subprocess.run(full, capture_output=capture, text=True, timeout=timeout)
    return r


def _run_name(st, names):
    side = st["turn"]
    rnd = st["round"] + 1
    return f"{names}_r{rnd}{side}", side


def remote_find(c, name):
    r = ssh(c, f"find /root -maxdepth 5 -name {name}.bin -size {BLOB}c 2>/dev/null | head -3")
    return [x for x in r.stdout.strip().splitlines() if x]


def resolve_remote_blob(c, name, deck_hint=""):
    """把 state 里的本地模型名解析成**训练机上的**绝对路径。

    机器上已有 85 个现成 blob（g6_db04 / grookey / league banks ...），所以默认
    不去上传，先按名字找；找不到才报错，并给上传命令。
    """
    hits = remote_find(c, name)
    if hits:
        return hits[0]
    raise SystemExit(
        f"✗ 训练机上没有 {name}.bin（{BLOB} B）\n"
        f"   本地上传： scp -P {c['port']} {os.path.join(ROOT, 'models/ppo', name + '.bin')} "
        f"{c['host']}:{c['out']}/init_{name}.bin\n"
        f"   然后用 --init-remote {c['out']}/init_{name}.bin 指过去")


def cmd_find(a):
    c = cfg()
    names = [a.name] if a.name else []
    if not names:
        r = ssh(c, f"find /root -maxdepth 5 -name '*.bin' -size {BLOB}c 2>/dev/null | sort")
        print(r.stdout.strip() or "（没有 18,205,408 B 的 blob）")
        return 0
    for n in names:
        hits = remote_find(c, n)
        print(f"{n}: {hits[0] if hits else '✗ 没有'}")
    return 0


def cmd_run(a):
    if a.t < 32768:
        raise SystemExit(f"✗ --t {a.t} < 32768（训练器一个 epoch）⇒ 产出全 NaN 权重。改成 ≥ 32768")
    c = cfg()
    st = json.load(open(os.path.join(STATE_DIR, f"dual_{a.names}.json")))
    if st.get("verdict"):
        raise SystemExit(f"✗ 已判定结束：{st['verdict']}")
    run, side = _run_name(st, a.names)
    other = "B" if side == "A" else "A"
    L, B = st["sides"][side], st["sides"][other]
    init_name = os.path.basename(L["model"])[:-4]
    bank_name = os.path.basename(B["model"])[:-4]
    init_r = a.init_remote or resolve_remote_blob(c, init_name)
    bank_r = a.bank_remote or resolve_remote_blob(c, bank_name)
    T = a.t
    env = dict(
        PYTHON=c["python"], INIT=init_r, OUT=c["out"], RUN=run, T=str(T), THREADS=str(a.threads),
        ANCHOR=L["deck"], TRACK=f"{B['deck']},{L['deck']}", DWF="decks/weights_real.txt",
        BANKS=bank_r, LEAGUE_PCT=str(a.league_pct), BANKDECKS=B["deck"],
    )
    sets = " ".join(f"{k}={v}" for k, v in env.items())
    log = f"{c['logs']}/{run}.log"
    binp = f"{c['out']}/{run}.bin"
    print(f"[remote] {run}: 训 {side}  anchor={L['deck']}  init={init_r}")
    print(f"[remote] bank={bank_r}@{B['deck']}  league={a.league_pct}  T={T}  日志={log}")
    if a.dry:
        print("[dry] 不执行：\n  cd", c["repo"], "&&", sets, "bash tools/db04_line.sh")
        return 0
    cmd = (f"mkdir -p {c['out']} {c['logs']} && cd {c['repo']} && "
           f"rm -f {binp} && nohup env {sets} bash tools/db04_line.sh > {log} 2>&1 & echo started")
    r = ssh(c, cmd)
    print(" ", r.stdout.strip() or r.stderr.strip())
    if not a.wait:
        print(f"[remote] 已挂后台。查进度： ./tools/py {os.path.relpath(__file__, ROOT)} log --names {a.names}")
        return 0
    t0 = time.time()
    last = ""
    while time.time() - t0 < a.timeout:
        time.sleep(a.poll)
        rr = ssh(c, f"if [ -f {binp} ]; then stat -c%s {binp}; else tail -1 {log} 2>/dev/null | cut -c1-100; fi")
        line = rr.stdout.strip()
        if line != last:
            print(f"  [{time.time()-t0:6.0f}s] {line}")
            last = line
        if line.isdigit():
            break
    else:
        raise SystemExit(f"✗ 超时 {a.timeout}s 还没产物，看： ssh ... 'tail -40 {log}'")
    size = int(line)
    if size != BLOB:
        raise SystemExit(f"✗ 产物大小 {size} ≠ {BLOB}（坏包）")
    print(f"✅ 训练完成 {time.time()-t0:.0f}s  产物 {binp} ({size} B)")
    if a.pull:
        dst = os.path.join(ROOT, "models/ppo", run + ".bin")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.run(["scp", "-P", str(c["port"]), "-o", "BatchMode=yes",
                        f"{c['host']}:{binp}", dst], check=True)
        got = os.path.getsize(dst)
        print(f"✅ 已拉回 {os.path.relpath(dst, ROOT)}  {got} B  "
              f"md5={subprocess.run(['md5sum', dst], capture_output=True, text=True).stdout[:12]}")
        if got != BLOB:
            raise SystemExit("✗ 拉回的文件大小不对")
        print(f"   下一步（评估机上）：  eval_pair.py --cand build/agents/ag_{run} ...")
    return 0


def cmd_log(a):
    c = cfg()
    if a.run:
        run = a.run
    else:
        sp = os.path.join(STATE_DIR, f"dual_{a.names}.json")
        if not os.path.exists(sp):
            raise SystemExit(f"✗ 没有 state {sp}（用 --run <名字> 直接看日志）")
        run, _ = _run_name(json.load(open(sp)), a.names)
    r = ssh(c, f"tail -{a.n} {c['logs']}/{run}.log 2>/dev/null")
    print(r.stdout.rstrip() or f"（没有日志 {c['logs']}/{run}.log）")
    return 0


def main():
    ap = argparse.ArgumentParser(description="远端只训：驱动 GPU 机器跑一轮 db04_line.sh")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("find", help="看训练机上现成有哪些权重")
    f.add_argument("--name", default="")
    f.set_defaults(fn=cmd_find)

    r = sub.add_parser("run", help="按 state 跑一轮训练")
    r.add_argument("--names", required=True, help="dual_<names>.json 的名字")
    r.add_argument("--t", type=int, default=400000,
                   help="训练步数（默认 400k）。★ 必须 ≥ 32768：训练器按 epoch(32768) 采集，"
                        "T 小于一个 epoch 时第一批是半满缓冲 → 产出全 NaN 权重")
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--league-pct", type=float, default=0.2)
    r.add_argument("--init-remote", default="", help="覆盖：训练机上的 INIT 路径")
    r.add_argument("--bank-remote", default="", help="覆盖：训练机上的 BANKS 路径")
    r.add_argument("--wait", action="store_true", help="等训练结束并校验产物")
    r.add_argument("--pull", action="store_true", help="结束后 scp 拉回 models/ppo/")
    r.add_argument("--timeout", type=int, default=2400)
    r.add_argument("--poll", type=int, default=30)
    r.add_argument("--dry", action="store_true", help="只打印将执行的命令")
    r.set_defaults(fn=cmd_run)

    g = sub.add_parser("log", help="看训练日志尾部")
    g.add_argument("--names", required=True)
    g.add_argument("--run", default="")
    g.add_argument("-n", type=int, default=20)
    g.set_defaults(fn=cmd_log)

    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
