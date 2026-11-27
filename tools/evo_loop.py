#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**进化式选育循环（evo loop）** —— 用户 2026-11-27 方案

一轮 = 4 个"族"（lineage），每族跑 4M 步，产出 **4 个 breakpoint + 1 个族内 SWA** = 5 个候选，
4 族共 **20 个候选**。选育规则（用户原话）：

    我原来是 A1 模型分化的 r1A1，之前 A1 对 bank B 的平均胜率是 50，
    那你至少得有 51.5 才能晋级下一轮训练；
    而 swa 模型则需要对模型中胜率最高的标准高 1.5，比如 A1 50, A2 51，
    那么 r1Aswa 至少需要 52.5 对 bank 的胜率才行。

⇒ 晋级闸门（全部用「对冻结 bank 的 450 局平均胜率」）：
    · breakpoint  i 的父模型是族首 P_i：  w(bp) ≥ w(P_i) + GATE_PP
    · 族内 SWA                          ：  w(swa) ≥ max_i w(bp_i) + GATE_PP
    · 没过闸门的候选不参与晋级；晋级候选按 w 排序取前 4 个作为下一轮的起始族首。
    · 一个都没过 ⇒ 停止，拉回产物。
    · 最多 ROUNDS 轮。

起始池（第 1 轮）不限于"最强 agent"：把**同牌表**的全部 agent 都拿进来（含特化产物 / SWA /
strong 矩阵上的驾驶员），各跑 450 局对 bank，取前 4 个作为族首 —— 所以"特化后没涨但也没掉"的
模型天然有机会进来。

训练配方（每族）：INIT=族首，KL_ANCHOR=族首（KL_COEF=0.005，把训练方向钉在它附近），
cosine 退火 + warm-up，checkpoint 每 STEPS/4 存一个（= 4 个 breakpoint）。

用法
----
    # 干跑：只打印命令与选育逻辑，不占任何机器
    ./tools/py tools/evo_loop.py --dry-run

    # 真跑（GPU 提供训练、CPU 机提供面板评估）
    GPU=root@connect.nmb2.seetacloud.com GPU_PORT=38237 \\
    EVAL_HOST=root@113.44.139.218 \\
    ./tools/py tools/evo_loop.py

    # 便宜的自检：2 族 × 少步数
    ROUNDS=1 LINEAGES=2 STEPS=65536 ./tools/py tools/evo_loop.py --dry-run

产物：models/ppo/evo_*、logs/evo/round<r>/、logs/evo_state.json（断点续跑用）、
      logs/evo_loop_<ts>.log。**关机器由调用方负责**（见结尾打印的兜底命令）。
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

PY = os.path.join(ROOT, "build", "port_plain", "python")
if not os.path.exists(PY):
    PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": f"{ROOT}/build/port_plain:{ROOT}"}

W_PANEL = os.path.join("decks", "testset_meta_real_v2.tsv")
BIN_BYTES = 18205408


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sh(cmd, check=True, capture=False, quiet=False, timeout=None):
    if not quiet:
        log("  $ " + cmd)
    r = subprocess.run(cmd, shell=True, env=ENV, capture_output=capture, text=True, timeout=timeout)
    if capture:
        return r
    if check and r.returncode != 0:
        raise SystemExit(f"✗ 命令失败（rc={r.returncode}）：{cmd}")
    return r


# ── 环境参数 ─────────────────────────────────────────────────────────────
def cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deck", default=os.environ.get("DECK", "dunsparce_abra__9d6e"))
    ap.add_argument("--rounds", type=int, default=int(os.environ.get("ROUNDS", 5)))
    ap.add_argument("--lineages", type=int, default=int(os.environ.get("LINEAGES", 4)))
    ap.add_argument("--steps", type=int, default=int(float(os.environ.get("STEPS", 4_000_000))))
    ap.add_argument("--breakpoints", type=int, default=int(os.environ.get("BREAKPOINTS", 4)))
    ap.add_argument("--gate-pp", type=float, default=float(os.environ.get("GATE_PP", 1.5)))
    ap.add_argument("--bank-n", type=int, default=int(os.environ.get("BANK_N", 450)))
    ap.add_argument("--panel-seeds", default=os.environ.get("PANEL_SEEDS", "950000,950001,950002"))
    ap.add_argument("--panel-matches", type=int, default=int(os.environ.get("PANEL_MATCHES", 150)))
    ap.add_argument("--lr", default=os.environ.get("LR", "1e-4"))
    ap.add_argument("--kl-coef", default=os.environ.get("KL_COEF", "0.005"))
    ap.add_argument("--league-pct", default=os.environ.get("LEAGUE_PCT", "0.08"))
    ap.add_argument("--ag", default=os.environ.get("AG", "512"))
    ap.add_argument("--seed-base", type=int, default=int(os.environ.get("SEED_BASE", 970000)))
    ap.add_argument("--eval-seed", type=int, default=int(os.environ.get("EVAL_SEED", 960000)))
    ap.add_argument("--local-workers", type=int, default=int(os.environ.get("LOCAL_W", 8)))
    ap.add_argument("--panel-workers", type=int, default=int(os.environ.get("PANEL_W", 12)))
    ap.add_argument("--bank-lo", type=float, default=float(os.environ.get("BANK_LO", 0.1)),
                    help="A vs bank 胜率低于此 ⇒ bank 太强，踢出训练对手")
    ap.add_argument("--bank-hi", type=float, default=float(os.environ.get("BANK_HI", 0.8)),
                    help="A vs bank 胜率高于此 ⇒ bank 太弱，踢出训练对手")
    ap.add_argument("--max-per-lineage", type=int, default=int(os.environ.get("MAX_PER_LINEAGE", 2)),
                    help="同一族最多几个候选能晋级（防 4 个 SWA 全来自一族）")
    ap.add_argument("--panel-gate", type=int, default=int(os.environ.get("PANEL_GATE", 0)),
                    help="1 = 面板配对差 < +GATE_PP 也判失败并停止（默认只报告）")
    ap.add_argument("--max-total-steps", type=int, default=int(float(os.environ.get("MAX_TOTAL_STEPS", 80_000_000))))
    return ap.parse_args()


# ── 远程/本地边界 ────────────────────────────────────────────────────────
def ssh_opts(gpu=False):
    port = os.environ.get("GPU_PORT", "22")
    o = ("-o BatchMode=yes -o LogLevel=ERROR -o StrictHostKeyChecking=no "
         "-o UserKnownHostsFile=/dev/null -o ConnectTimeout=25")
    return o + (f" -p {port}" if gpu else "")


DP = None  # dry-run 时不 ssh


def gpu_sh(remote_cmd, check=True):
    """在 GPU 机上执行（非交互 shell ⇒ 必须显式 export PATH/PYTHON）。"""
    if DP:
        log("  [dry] ssh GPU :: " + remote_cmd[:160].replace("\n", " "))
        return subprocess.CompletedProcess(remote_cmd, 0, "", "")
    return sh(f"ssh {ssh_opts(True)} {GPU} '{remote_cmd}'", check=check)


def gpu_pull(remote_path, local_path):
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    if DP:
        log(f"  [dry] scp GPU:{remote_path} → {local_path}")
        return
    subprocess.run(f"scp -q {ssh_opts(True)} {GPU}:{remote_path} {local_path}",
                   shell=True, env=ENV, check=False)


def eval_host_up():
    return subprocess.run(f"ssh {ssh_opts()} {EVAL_HOST} true", shell=True,
                          capture_output=True).returncode == 0


# ── 模型/agent 工具 ──────────────────────────────────────────────────────
def deck_md5(deck_path):
    return hashlib.md5(open(deck_path, "rb").read()).hexdigest()


def same_deck_agents(deck_path):
    h = deck_md5(deck_path)
    out = []
    for d in sorted(glob.glob("build/agents/*")):
        f = os.path.join(d, "deck.csv")
        if os.path.isfile(f) and hashlib.md5(open(f, "rb").read()).hexdigest() == h \
                and os.path.isfile(os.path.join(d, "model.pt")):
            out.append(os.path.basename(d))
    return out


def bin_to_pt(bin_path, name):
    pt = f"models/ppo/{name}.pt"
    if DP:
        return pt
    if os.path.exists(pt):
        os.remove(pt)
    sh(f"{PY} tools/native_weights.py import {bin_path} {pt}", quiet=True)
    assert os.path.exists(pt), f"✗ import 失败：{bin_path}"
    return pt


def stage(name, pt, deck_csv):
    if DP:
        log(f"  [dry] stage build/agents/{name} ← {os.path.basename(pt)}")
        return f"build/agents/{name}"
    sh(f"bash tools/make_agent_dir.sh build/agents/{name} {pt} {deck_csv}", quiet=True)
    return f"build/agents/{name}"


def stage_deck_agent(agent_name):
    """同牌 agent 本身就能用：它已有 main.py/model.pt/deck.csv。"""
    return f"build/agents/{agent_name}"


def bank_agents(deck_csv):
    """冻结 bank → 可评测 agent（镜像牌表）。缺 .bin 就先从 GPU 拉。"""
    if DP:
        return [f"build/agents/{n}" for n in _bank_names()]
    out = []
    for b in BANKS.split(","):
        nm = "bank_" + os.path.basename(b).split(".")[0].split("_")[-1]
        local = f"models/ppo/{nm}.bin"
        if not os.path.exists(local) or os.path.getsize(local) != BIN_BYTES:
            gpu_pull(b, local)
            assert os.path.getsize(local) == BIN_BYTES, f"✗ bank 拉回不完整：{local}"
        out.append(stage(nm, bin_to_pt(local, nm), deck_csv))
    return out


def _bank_names():
    return [("bank_" + os.path.basename(b).split(".")[0].split("_")[-1]) for b in BANKS.split(",")]


def eval_bank(cand_dir, opp_dirs, name, n=None, seed=None):
    """候选 vs 冻结 bank 集合，返回平均胜率（0~100，百分比）。"""
    n = n or BANK_N
    seed = seed or EVAL_SEED
    out = f"logs/evo/eval_{name}_bank.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    opp_args = " ".join(f'--opp "b{i}={d}"' for i, d in enumerate(opp_dirs))
    if DP:
        log(f"  [dry] eval {name} vs {len(opp_dirs)} banks × {n}")
        return None
    sh(f'{PY} ptcg_league/eval_pair.py --cand {cand_dir} {opp_args} --matches {n} '
       f'--workers {LOCAL_W} --seed-base {seed} --out {out}', quiet=True)
    d = json.load(open(out))
    ov = d.get("overall") or {}
    v = ov.get("score", ov.get("mean"))
    assert v is not None, f"✗ {name} 评测无分数：{out}"
    return float(v) * 100.0


def audit_banks(rnd, lineages):
    """A vs 每个 bank 分别 450 局；胜率落在 (BANK_LO, BANK_HI) 之外的 bank 不能当训练对手。"""
    banks = bank_agents(DECK_CSV)
    names = _bank_names()
    allowed, rows = {}, []
    log(f"bank 强度体检：{len(lineages)} 族 × {len(banks)} bank（n={BANK_N}，合格带 "
        f"({BANK_LO*100:.0f}%, {BANK_HI*100:.0f}%)）")
    for i, (name, _w, _b, _x) in enumerate(lineages):
        oks = []
        for bd, bk in zip(names, banks):
            wr = eval_bank(stage_deck_agent(name), [bk], f"audit_r{rnd}_L{i}_{bd}")
            if DP:
                wr = 55.0
            frac = wr / 100.0
            ok = BANK_LO < frac < BANK_HI
            tag = "✅" if ok else ("❌太弱" if frac >= BANK_HI else "❌太强")
            log(f"    {name:22s} × {bd:8s} = {wr:6.2f}%  {tag}")
            rows.append(dict(round=rnd, lineage=name, bank=bd, wr=wr, ok=ok, tag=tag))
            if ok:
                oks.append(bd)
        allowed[i] = oks
        if not oks:
            log(f"    ⚠️ 族 {name} 没有任何合格 bank ⇒ 本轮该族不用 league bank（退化为自博弈+加权抽样）")
        else:
            log(f"    族 {name} 可用 bank：{', '.join(oks)}")
    os.makedirs("logs/evo", exist_ok=True)
    json.dump(rows, open(f"logs/evo/round{rnd}_bank_audit.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return allowed, rows


def parse_lgw(log_text):
    """从训练日志取最后的 lgw 与逐 bank 在线胜率（[i:w/n ...]，w 就是 vs bank_i 的胜率）。"""
    m = re.findall(r"lgw\s+([\d.]+)\s+\[([^\]]*)\]", log_text or "")
    if not m:
        return None, "", []
    lgw, block = float(m[-1][0]), m[-1][1]
    per = [(int(i), float(w), float(n)) for i, w, n in
           re.findall(r"(\d+):([\d.]+)/([\d.]+)", block)]
    return lgw, block, per


def pull_train_log(run):
    if DP:
        return ""
    loc = f"logs/evo/{run}.log"
    os.makedirs("logs/evo", exist_ok=True)
    subprocess.run(f"scp -q {ssh_opts(True)} {GPU}:{PTCG_REMOTE}/logs/{run}.log {loc}",
                   shell=True, env=ENV, check=False)
    return open(loc, encoding="utf-8", errors="ignore").read() if os.path.exists(loc) else ""


def swa(bins, name):
    out = f"models/ppo/{name}.bin"
    if DP:
        log(f"  [dry] SWA {name} ← {len(bins)} 个 breakpoint")
        return out
    sh(f"{PY} tools/swa_avg.py --bins {' '.join(bins)} --out-bin {out}", quiet=True)
    assert os.path.getsize(out) == BIN_BYTES, f"✗ SWA 产物大小不对：{out}"
    return out


# ── 训练 ─────────────────────────────────────────────────────────────────
def ckpt_interval_epochs():
    """每多少 epoch 存一个快照，使每族正好 BREAKPOINTS 个。1 epoch = AG×64 步。"""
    per = int(AG) * 64
    want = STEPS / max(1, BREAKPOINTS) / per
    return max(1, int(round(want)))


def train_lineage(rnd, i, init_bin, init_name, banks_str=None):
    run = f"evo_r{rnd}_L{i}"
    out_dir = f"/root/evo_r{rnd}_L{i}"
    remote_init = f"/root/evo_r{rnd}_L{i}_init.bin"
    local_bin = f"models/ppo/{run}.bin"
    if os.path.exists(local_bin):
        log(f"  族 {i}: {local_bin} 已在本地，跳过训练")
        return run, local_bin
    if DP:
        log(f"  [dry] 训练 族{i}: INIT={init_name} T={STEPS} ckpt_every={ckpt_interval_epochs()}ep "
            f"KL_ANCHOR=自身({KL_COEF}) → {out_dir}")
        return run, local_bin, [f"models/ppo/{run}_bp{k}.bin" for k in range(BREAKPOINTS)]
    subprocess.run(f"scp -q {ssh_opts(True)} {init_bin} {GPU}:{remote_init}",
                   shell=True, env=ENV, check=True)
    env_prefix = (
        "export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH; "
        "export PYTHON=/root/miniconda3/bin/python3; "
        f"cd {PTCG_REMOTE} && "
    )
    cmd = (
        f"ANCHOR={DECK} TRACK={DECK} INIT={remote_init} "
        f"OUT={out_dir} RUN={run} T={STEPS} LR={LR} AG={AG} SEED={SEED_BASE + rnd * 10 + i} "
        f"CHECKPOINT_INTERVAL={ckpt_interval_epochs()} ANNEAL_LR=1 MIN_LR_RATIO=0.1 WARMUP_EPOCHS=1 "
        f"BANKS={banks_str if banks_str is not None else BANKS} LEAGUE_PCT={LEAGUE_PCT} "
        f"KL_ANCHOR={remote_init} KL_COEF={KL_COEF} "
        f"bash tools/db04_line.sh"
    )
    gpu_sh(env_prefix + cmd)
    # 拉回 final + breakpoints
    ls = gpu_sh(f"ls -1 {PTCG_REMOTE}/experiments/native-{run}/ 2>/dev/null", capture=True)
    files = [f for f in (ls.stdout or "").split() if f.endswith(".bin")]
    snaps = sorted(f for f in files if re.fullmatch(r"\d{16}\.bin", f))
    gpu_pull(f"{PTCG_REMOTE}/experiments/native-{run}/final.bin", local_bin)
    picked = snaps[-BREAKPOINTS:] if len(snaps) >= BREAKPOINTS else snaps
    bps = []
    for s in picked:
        loc = f"models/ppo/{run}_bp_{s[:-4]}.bin"
        gpu_pull(f"{PTCG_REMOTE}/experiments/native-{run}/{s}", loc)
        if not DP:
            assert os.path.getsize(loc) == BIN_BYTES, f"✗ 快照不完整 {loc}"
        bps.append(loc)
    assert len(bps) >= 2, f"✗ 族 {i} 没拿到足够 breakpoint（{len(bps)}）"
    log(f"  族 {i} 训练完成：{len(bps)} 个 breakpoint + final")
    return run, local_bin, bps


# ── 面板（v2 口径）──────────────────────────────────────────────────────
def panel_run(cand, tag="v2"):
    """在 EVAL_HOST 上跑整张 v2 面板（3 种子），拉回 logs/meta/<cand>_v2_s<seed>/。"""
    if DP:
        log(f"  [dry] panel {cand}（{cand} × {PANEL_SEEDS}）")
        return
    sh(f"bash tools/panel_on_host.sh {cand} {EVAL_HOST} {W_PANEL} {tag} {PANEL_W}", quiet=True)
    n = len(glob.glob(f"logs/meta/{cand}_{tag}_s{PANEL_SEEDS.split(',')[0]}/vs_*.json"))
    log(f"  panel {cand}: {n} 格已落盘")


def panel_gain(cand, base):
    """返回 (加权配对差 pp, CI 显示行)。同种子逐局配对，剔除 7e39。"""
    if DP:
        return None
    r = subprocess.run(f"{PY} tools/panel_paired.py --a {cand} --b {base} --tag v2",
                       shell=True, env=ENV, capture_output=True, text=True)
    m = re.search(r"加权配对差 = ([+-][\d.]+)pp\s+SE ([\d.]+)pp\s+95% CI \[([+-][\d.]+), ([+-][\d.]+)\]",
                  r.stdout or "")
    if not m:
        return None
    return float(m.group(1)), r.stdout.strip().splitlines()[-2:]


# ── 选育 ─────────────────────────────────────────────────────────────────
def pick_initial(lineages):
    """第 1 轮：同牌全部 agent → 450 局对 bank → 取前 lineages 个。"""
    agents = same_deck_agents(DECK_CSV)
    log(f"起始池：{len(agents)} 个同牌 agent → 各 {BANK_N} 局对 bank")
    banks = bank_agents(DECK_CSV)
    scored = []
    for a in agents:
        v = eval_bank(stage_deck_agent(a), banks, f"init_{a}")
        if DP:
            v = 50.0  # 干跑占位
        scored.append((v, a))
    scored.sort(reverse=True)
    top = scored[:lineages]
    for v, a in scored:
        log(f"    {a:24s} {v:6.2f}")
    log(f"族首（第 1 轮）：" + "，".join(f"{a}({v:.2f})" for v, a in top))
    return [(a, v) for v, a in top]


def round_select(rnd, lineages, state):
    """训练 → 候选 → 闸门 → 选 4 个；返回 (promoted, stop)。"""
    allowed, _rows = audit_banks(rnd, lineages)
    cands = []  # (kind, lineage_idx, name, bin)
    for i, (name, base_w, init_bin, bps) in enumerate(lineages):
        banks_str = ",".join(b for b in BANKS.split(",")
                             if "bank_" + os.path.basename(b).split(".")[0].split("_")[-1] in allowed[i])
        run, final_bin, bp_bins = train_lineage(rnd, i, init_bin, name, banks_str)
        lgw, dist, per = parse_lgw(pull_train_log(run))
        if lgw is not None:
            log(f"    族 {i} 训练日志 lgw={lgw:.3f}  逐bank={dist}")
            for b, w, n in per:
                if BANK_LO < w < BANK_HI:
                    continue
                log(f"    ⚠️ 族 {i} × bank_{b} 在线胜率 {w:.2f}（份额≈{n:.2f}）不在 "
                    f"({BANK_LO*100:.0f}%, {BANK_HI*100:.0f}%) 带内 ⇒ 下轮体检应剔除该 bank")
            STATE.setdefault("lgw", []).append(
                dict(round=rnd, lineage=name, lgw=lgw, dist=dist,
                     per_bank=[dict(bank=b, wr=w, n=n) for b, w, n in per]))
            save_state()
        for k, b in enumerate(bp_bins):
            cands.append(dict(kind="bp", i=i, name=f"{run}_bp{k}", bin=b, parent_w=base_w))
        swa_bin = swa(bp_bins, f"{run}_swa")
        cands.append(dict(kind="swa", i=i, name=f"{run}_swa", bin=swa_bin, parent_w=base_w))
    # 评测全部候选 vs 冻结 bank
    banks = bank_agents(DECK_CSV)
    for c in cands:
        c["agent"] = stage(c["name"], bin_to_pt(c["bin"], c["name"]), DECK_CSV)
        c["w"] = eval_bank(c["agent"], banks, c["name"])
        if DP:
            c["w"] = 50.0 + (2.0 if c["kind"] == "swa" else 1.0) * (0.5 + c["i"] / 10)
    # 闸门
    promoted = []
    by_lineage = {}
    for c in cands:
        by_lineage.setdefault(c["i"], []).append(c)
    for c in cands:
        if c["kind"] == "bp":
            need = c["parent_w"] + GATE_PP
            ok = c["w"] >= need
            base_txt = f"父 {c['parent_w']:.2f}+{GATE_PP}"
        else:
            best_bp = max(x["w"] for x in by_lineage[c["i"]] if x["kind"] == "bp")
            need = best_bp + GATE_PP
            ok = c["w"] >= need
            base_txt = f"本族最高bp {best_bp:.2f}+{GATE_PP}"
        c["need"], c["ok"] = need, ok
        log(f"    候选 {c['name']:22s} {c['kind']:3s} w={c['w']:6.2f}  晋级线 {base_txt}={need:6.2f}  "
            f"{'✅' if ok else '❌'}")
        if ok:
            promoted.append(c)
    promoted.sort(key=lambda x: -x["w"])
    # 族级限流：默认每族最多 MAX_PER_LINEAGE 个，避免 4 个 SWA 全来自同一族
    if MAX_PER_LINEAGE > 0:
        seen, keep = {}, []
        for c in promoted:
            if seen.get(c["i"], 0) >= MAX_PER_LINEAGE:
                log(f"    （族限流：{c['name']} 让位，本族已入选 {seen[c['i']]} 个）")
                continue
            seen[c["i"]] = seen.get(c["i"], 0) + 1
            keep.append(c)
        promoted = keep
    state["history"].append(dict(
        round=rnd,
        cands=[{k: c[k] for k in ("name", "kind", "i", "w", "need", "ok")} for c in cands],
        promoted=[c["name"] for c in promoted]))
    save_state()
    return promoted


def save_state():
    os.makedirs("logs/evo", exist_ok=True)
    json.dump(STATE, open("logs/evo_state.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


# ── main ─────────────────────────────────────────────────────────────────
def main():
    global DECK, DECK_CSV, ROUNDS, LINEAGES, STEPS, BREAKPOINTS, GATE_PP, BANK_N, PANEL_SEEDS
    global PANEL_MATCHES, LR, KL_COEF, LEAGUE_PCT, AG, EVAL_SEED, LOCAL_W, PANEL_W, MAX_TOTAL_STEPS
    global GPU, EVAL_HOST, PTCG_REMOTE, BANKS, LOG, DP, STATE, SEED_BASE, MAX_PER_LINEAGE, PANEL_GATE
    global BANK_LO, BANK_HI

    a = cfg()
    DP = a.dry_run
    AG = str(a.ag)
    DECK = a.deck
    DECK_CSV = f"decks/pool_field/{DECK}.csv"
    ROUNDS, LINEAGES, STEPS = a.rounds, a.lineages, a.steps
    BREAKPOINTS, GATE_PP, BANK_N = a.breakpoints, a.gate_pp, a.bank_n
    PANEL_SEEDS, PANEL_MATCHES = a.panel_seeds, a.panel_matches
    LR, KL_COEF, LEAGUE_PCT, AG = a.lr, a.kl_coef, a.league_pct, a.ag
    EVAL_SEED, LOCAL_W, PANEL_W = a.eval_seed, a.local_workers, a.panel_workers
    MAX_TOTAL_STEPS, SEED_BASE = a.max_total_steps, a.seed_base
    MAX_PER_LINEAGE, PANEL_GATE = a.max_per_lineage, a.panel_gate
    BANK_LO, BANK_HI = a.bank_lo, a.bank_hi
    GPU = os.environ.get("GPU", "")
    EVAL_HOST = os.environ.get("EVAL_HOST", "")
    PTCG_REMOTE = os.environ.get("PTCG_REMOTE", "/root/ptcg_train_kit/port_repo")
    BANKS = os.environ.get("BANKS", "/root/league_g0003/bank_0.bin,/root/league_g0003/bank_1.bin")
    LOG = f"logs/evo_loop_{time.strftime('%Y%m%d_%H%M')}.log"
    STATE = {"round": 1, "lineages": [], "history": []}

    if os.path.exists(DECK_CSV):
        log(f"卡组 = {DECK}  （{DECK_CSV}）")
    else:
        raise SystemExit(f"✗ 找不到卡组 {DECK_CSV}")
    log(f"计划：{ROUNDS} 轮 × {LINEAGES} 族 × {STEPS/1e6:.2f}M 步 = {ROUNDS*LINEAGES*STEPS/1e6:.0f}M 步"
        f"（上限 {MAX_TOTAL_STEPS/1e6:.0f}M）；每族 {BREAKPOINTS} bp + 1 SWA；闸门 +{GATE_PP}pp；"
        f"bank {BANK_N} 局/候选；面板 {PANEL_SEEDS}")
    if not DP:
        assert GPU, "✗ 需要 GPU=root@host（或 --dry-run）"
        assert EVAL_HOST, "✗ 需要 EVAL_HOST=root@cpu-host（或 --dry-run）"
        if not eval_host_up():
            raise SystemExit(f"✗ 评估机 {EVAL_HOST} 连不上")
        gpu_sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader")

    # 第 1 轮族首：现有同牌 agent（最强 / 特化产物 / strong 驾驶员都在里面）
    if STATE["lineages"]:
        lineages = STATE["lineages"]
    else:
        top = pick_initial(LINEAGES)
        lineages = []
        for name, w in top:
            b = f"models/ppo/{name}.bin"
            if not os.path.exists(b):
                sh(f"{PY} tools/native_weights.py export build/agents/{name}/model.pt {b}", quiet=True)
            assert os.path.getsize(b) == BIN_BYTES
            lineages.append([name, w, b, []])
        STATE["lineages"] = lineages
        save_state()

    r = STATE["round"]
    while r <= ROUNDS:
        log(f"===== 第 {r} 轮：{len(lineages)} 族 =====")
        lin = [(n, w, b, []) for n, w, b, _ in lineages]
        promoted = round_select(r, lin, STATE)
        if not promoted:
            log("✗ 本轮 20 个候选全部没过闸门（+%.1fpp）⇒ 停止训练" % GATE_PP)
            break
        # 面板复核：晋级第一 vs 本轮族首最强
        best = promoted[0]
        base_name = max(lineages, key=lambda x: x[1])[0]
        panel_run(best["name"])
        panel_run(base_name) if not glob.glob(f"logs/meta/{base_name}_v2_s{PANEL_SEEDS.split(',')[0]}/*.json") else None
        g = panel_gain(best["name"], base_name)
        if g:
            log(f"面板复核 {best['name']} vs {base_name}：{g[0]:+.1f}pp（配对，剔除 7e39）")
            if PANEL_GATE and g[0] < GATE_PP:
                log(f"✗ 面板闸门未过（{g[0]:+.1f} < +{GATE_PP}pp）⇒ 停止")
                break
        # 下一轮族首 = 晋级前 4
        lines = []
        for c in promoted[:LINEAGES]:
            lines.append([c["name"], c["w"], c["bin"], []])
        STATE["lineages"] = lines
        STATE["round"] = r + 1
        save_state()
        total = sum(x["w"] for x in promoted)
        log(f"晋级 {len(promoted)} 个：{', '.join(c['name'] for c in promoted[:LINEAGES])}")
        r = STATE["round"]
        if r * LINEAGES * STEPS > MAX_TOTAL_STEPS:
            log(f"⚠️ 已达总步数上限 {MAX_TOTAL_STEPS/1e6:.0f}M，停止")
            break

    log("===== 收尾 =====")
    log(f"状态：{json.dumps({'round': STATE['round'], 'lineages': [l[0] for l in STATE['lineages']]}, ensure_ascii=False)}")
    log("产物：models/ppo/evo_*、logs/evo/、logs/meta/*_v2_*；本地已全量。")
    log("关机器（由你/调用方执行）：")
    log("  HOST=%s PORT=%s AT=+2h bash tools/force_finish_deadline.sh   # GPU 兜底" % (GPU, os.environ.get("GPU_PORT", "22")))
    log("  ssh %s 'shutdown -h now'                                     # CPU 评估机" % EVAL_HOST)


if __name__ == "__main__":
    main()
