#!/usr/bin/env python3
"""**中央调度器**：任务排队，哪台机器空下来就立刻派下一个（自愈、不空转）。

背景（2026-11-16 用户要求"把队伍排起来"）：早先给每台机器写死一份任务列表，
跑完脚本就退出 ⇒ 机器空转要人盯。这里改成中央队列：
  · 任务 = (候选, 格, 种子)；完成判据 = 远端存在 logs/meta/<tag>/vs_<cell>.json
  · **每轮每台机器只发 1 条 ssh**（同时取"是否在跑"和"已完成清单"），避免 42 次往返
  · 空闲机器 → 取 pending 任务 `setsid nohup` 派过去；派出去但进程没了又没结果 ⇒ 重新入队
  · 全部完成打印 DISPATCH_DONE

★ 两个必须记住的坑：
  1) `pgrep -f "ptcg_league/eval_pair"` 会匹配到**自己这条 ssh 命令行** ⇒ 每台机永远"忙"，
     调度器一个活都不派（2026-11-16 卡了半小时）。必须写成 `ptcg_league/[e]val_pair`。
  2) 判活别用 `pgrep -c` 的退出码，用输出数字；空输出要当 0 处理。

    ./tools/py tools/dispatch_panel.py --cands candf5f8G candf5f8S --cells out_07be ...
    ./tools/py tools/dispatch_panel.py ... --once     # 只跑一轮（调试）
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = "/root/eval_cpu_kit"
HOSTS = ["root@113.44.216.143", "root@1.92.97.36", "root@113.44.139.218"]
STATE = os.path.join(ROOT, "logs", "dispatch_state.json")


def ssh(host, cmd, timeout=30):
    try:
        # ★ 2026-11-19：少了 StrictHostKeyChecking/UserKnownHostsFile，
        #   新机器（如 123.249.80.18）第一次被调度时会因"未知 host key"直接失败 ⇒ 判活返回 -1（连不上），
        #   调度器认为"0 台空闲机"，任务一个都派不出去（本地 40 分钟白等）。两台机器都要带上这两个选项。
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                            host, cmd],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except Exception:
        return ""


def remote_state(host):
    """一次 ssh：返回 (busy, {'tag/cell'}) —— 在跑进程数 + 已完成面板清单。"""
    cmd = (f'cd {KIT} 2>/dev/null; echo "BUSY=$(pgrep -fc \'ptcg_league/[e]val_pair\' || true)"; '
           f'ls logs/meta/*/vs_*.json 2>/dev/null')
    out = ssh(host, cmd)
    if not out:
        return -1, set()          # ★ -1 = 机器连不上（≠ 空闲 0）
    busy = 0
    inv = set()
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("BUSY="):
            v = line[5:].strip()
            busy = int(v) if v.isdigit() else 0
        elif line.endswith(".json"):
            p = line.split("/")
            if len(p) >= 4:
                inv.add(f"{p[2]}/{p[3][3:-5]}")
    return busy, inv


CELL_AGENTS: dict = {}


def dispatch(host, cand, cell, seed, matches, workers):
    tag = f"{cand}_real_s{seed}"
    # 先在本地把 agent 目录解析出来（--cell-agents 覆盖优先，其次查真实口径表），
    # 再把**确定的路径**发到远端——早先远端自己 awk 查表，导致 --cell-agents 指定的格
    # （champ_0d86 / dunsparce_db04）在远端永远 MISSING_AGENT。
    dirp = CELL_AGENTS.get(cell)
    if not dirp:
        try:
            for ln in open(os.path.join(ROOT, "decks", "testset_meta_real.tsv")):
                if ln.startswith("#") or not ln.strip():
                    continue
                f = ln.rstrip("\n").split("\t")
                if f[0] == cell:
                    dirp = f[1]
                    break
        except OSError:
            pass
    if not dirp:
        return f"NO_DIR {cell}"
    cmd = (f"mkdir -p {KIT}/logs/meta/{tag}; cd {KIT} && dir={dirp}; "
           f"[ -d \"$dir\" ] || {{ echo 'MISSING_AGENT ' $dir; exit 0; }}; "
           f"setsid nohup ./tools/py ptcg_league/eval_pair.py --cand build/agents/{cand} "
           f"--opp {cell}=$dir --matches {matches} --workers {workers} --seed-base {seed} "
           f"--out {KIT}/logs/meta/{tag}/vs_{cell}.json "
           f"> /root/dispatch_{cand}_{cell}_{seed}.log 2>&1 < /dev/null & echo GO {tag} {cell}")
    return ssh(host, cmd, timeout=25).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[950000])
    ap.add_argument("--hosts", nargs="+", default=HOSTS)
    ap.add_argument("--matches", type=int, default=150)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--once", action="store_true")
    # ★ 2026-11-19：默认 200 轮 × 15 s ≈ 50 分钟就到顶，102 格的种子队列必被腰斩
    #   （123 的反向跑到 58/102 就 DISPATCH_TIMEOUT 了）。按"每格 ~60 s"算，1000 轮 ≈ 4 小时够用。
    ap.add_argument("--max-rounds", type=int, default=1000)
    ap.add_argument("--cell-agents", default="",
                    help="补充 格子=agent目录 映射（逗号分隔），用于真实口径表里没有、但属于 12 格网格的格，"
                         "如 champ_0d86=build/agents/g8_bc_parent,dunsparce_db04=build/agents/port_db04")
    ap.add_argument("--local-after", type=int, default=3,
                    help="全部机器连不上连续 N 轮后改在本地跑（0=禁用）")
    a = ap.parse_args()

    for pair in (a.cell_agents or "").split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            CELL_AGENTS[k.strip()] = v.strip()
    tasks = [dict(cand=c, cell=x, seed=s, host=None, status="pending")
             for c in a.cands for x in a.cells for s in a.seeds]
    dead_rounds = 0
    print(f"队列 {len(tasks)} 个任务，机器 {len(a.hosts)} 台", flush=True)

    for rnd in range(1, a.max_rounds + 1):
        st = {h: remote_state(h) for h in a.hosts}          # 每机 1 次 ssh
        for t in tasks:                                     # ① 结算
            if t.get("status") == "failed" and t.get("retries", 0) >= 3:
                continue
            key = f"{t['cand']}_real_s{t['seed']}/{t['cell']}"
            if t["status"] != "done" and any(key in st[h][1] for h in a.hosts):
                t.update(status="done")
            elif t["status"] == "running":
                h = t["host"]
                if key in st[h][1]:
                    t["status"] = "done"
                elif st[h][0] == 0:
                    t["retries"] = t.get("retries", 0) + 1
                    if t["retries"] >= 3:
                        # ★ 2026-11-19：agent 目录缺失时 eval 会瞬间失败 → 无限"重排"、机器空转。
                        #   连续 3 次空手而归就判 failed，别再占着队列。
                        t.update(status="failed", host=None)
                        print(f"  ✗ 放弃 {t['cand']} {t['cell']}（连续 3 次派出去没结果，八成远端缺 agent）", flush=True)
                    else:
                        t.update(status="pending", host=None)
                        print(f"  ↺ 重排 {t['cand']} {t['cell']}（{h.split('@')[-1]} 空了但没结果，第 {t['retries']} 次）", flush=True)
        for h in a.hosts:                                   # ② 派活
            if st[h][0] != 0:                               # >0 忙；-1 连不上
                continue
            pend = [t for t in tasks if t["status"] == "pending"]
            if not pend:
                break
            t = pend[0]
            r = dispatch(h, t["cand"], t["cell"], t["seed"], a.matches, a.workers)
            if r.startswith("GO"):
                t.update(status="running", host=h)
                print(f"  → {t['cand']:10s} {t['cell']:14s} → {h.split('@')[-1]}", flush=True)
            elif r:
                print(f"  ✗ {h.split('@')[-1]}: {r[:80]}", flush=True)
        # ③ 机器全掉线 ⇒ 本地兜底（同一 eval_pair、同一口径，只是 worker 少）
        if a.local_after and all(st[h][0] == -1 for h in a.hosts):
            dead_rounds += 1
        else:
            dead_rounds = 0
        if a.local_after and dead_rounds >= a.local_after:
            pend = [t for t in tasks if t["status"] != "done"]
            print(f"⚠️ 连续 {dead_rounds} 轮所有机器都连不上 ⇒ 剩余 {len(pend)} 个任务改在**本地**跑", flush=True)
            for t in pend:
                tag = f"{t['cand']}_real_s{t['seed']}"
                out = os.path.join(ROOT, "logs", "meta", tag, f"vs_{t['cell']}.json")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                if os.path.exists(out):
                    t["status"] = "done"; continue
                dirp = CELL_AGENTS.get(t["cell"], "")
                for ln in ([] if dirp else open(os.path.join(ROOT, "decks", "testset_meta_real.tsv"))):
                    if ln.startswith("#") or not ln.strip():
                        continue
                    f = ln.rstrip("\n").split("\t")
                    if f[0] == t["cell"]:
                        dirp = f[1]; break
                print(f"  → 本地跑 {t['cand']} {t['cell']} s{t['seed']}", flush=True)
                subprocess.run([sys.executable or "python3", os.path.join(ROOT, "ptcg_league", "eval_pair.py"),
                                "--cand", f"build/agents/{t['cand']}", "--opp", f"{t['cell']}={dirp}",
                                "--matches", str(a.matches), "--workers", "8",
                                "--seed-base", str(t["seed"]), "--out", out],
                               cwd=ROOT, capture_output=True, text=True)
                t["status"] = "done" if os.path.exists(out) else "pending"
            dead_rounds = 0
        done = sum(1 for t in tasks if t["status"] == "done")
        running = sum(1 for t in tasks if t["status"] == "running")
        json.dump(tasks, open(STATE, "w"), ensure_ascii=False, indent=1)
        print(f"[{time.strftime('%H:%M:%S')}] 完成 {done}/{len(tasks)} 在跑 {running} "
              f"空闲机 {[h.split('@')[-1] for h in a.hosts if st[h][0] == 0]}", flush=True)
        if done == len(tasks):
            print("DISPATCH_DONE", flush=True)
            return 0
        if a.once:
            return 0
        time.sleep(a.interval)
    print("DISPATCH_TIMEOUT", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
