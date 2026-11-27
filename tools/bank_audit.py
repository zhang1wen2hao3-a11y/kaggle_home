#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**bank 强度体检**（2026-11-27，用户口径）

> 如果 A 对战 bank B 的胜率高于 0.8 或者低于 0.1，都需要关注对手强度是否在线，
> 不符合的需要筛选掉，不能作为训练对手。

含义：bank 太弱（我们 >80%）＝给的梯度信号是"白捡"，白练；bank 太强（我们 <10%）＝
learner 在这个池子里拿不到可用梯度，只学会"认输"。两种都该踢出训练对手集合。

用法
----
    # 体检几个候选 vs 两个冻结 bank（每个 450 局）
    ./tools/py tools/bank_audit.py \\
        --agents cand9D6ER4 cand9D6E_SWA4 ab9d6e_r2A \\
        --banks /root/league_g0003/bank_0.bin /root/league_g0003/bank_1.bin \\
        --bank-decks dunsparce_abra__9d6e dunsparce_abra__9d6e \\
        --n 450 --out logs/bank_audit.json

    # 只输出"可用的 bank"（喂给 BANKS=）
    ./tools/py tools/bank_audit.py --agents X --banks a.bin b.bin --emit-ok

判定：0.1 < 胜率 < 0.8 为合格（边界值判不合格，与"高于/低于"的措辞对齐）。
一条 bank 对**任一**被测候选不合格，就判该 bank 对这台 learner 不可用；
多候选时按"每个候选各自可用集合"输出，不强行取交集。
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

PY = os.path.join(ROOT, "build", "port_plain", "python")
if not os.path.exists(PY):
    PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": f"{ROOT}/build/port_plain:{ROOT}"}
BIN_BYTES = 18205408


def sh(cmd, quiet=True):
    r = subprocess.run(cmd, shell=True, env=ENV, text=True,
                       capture_output=quiet, timeout=7200)
    if r.returncode != 0:
        raise SystemExit(f"✗ {cmd}\n{(r.stdout or '')[-800:]}{(r.stderr or '')[-800:]}")
    return r


def ensure_local_bin(path, gpu=""):
    """远端 .bin 先拉回本地；已经在本地的原样返回。"""
    if os.path.exists(path) and os.path.getsize(path) == BIN_BYTES:
        return path
    if not gpu:
        raise SystemExit(f"✗ 本地没有且没给 --gpu，无法取 {path}")
    loc = os.path.join("models", "ppo", "audit_" + os.path.basename(path))
    os.makedirs(os.path.dirname(loc), exist_ok=True)
    sh(f"scp -q -P {os.environ.get('GPU_PORT', '22')} -o StrictHostKeyChecking=no "
       f"-o UserKnownHostsFile=/dev/null {gpu}:{path} {loc}", quiet=False)
    if os.path.getsize(loc) != BIN_BYTES:
        raise SystemExit(f"✗ 拉回的 {path} 大小不对")
    return loc


def stage(name, pt, deck_csv):
    sh(f"bash tools/make_agent_dir.sh build/agents/{name} {pt} {deck_csv}")
    return f"build/agents/{name}"


def to_pt(bin_path, name):
    pt = f"models/ppo/{name}.pt"
    sh(f"{PY} tools/native_weights.py import {bin_path} {pt}")
    return pt


def eval_pair(cand_dir, opp_dir, n, seed, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    sh(f'{PY} ptcg_league/eval_pair.py --cand {cand_dir} --opp "b={opp_dir}" '
       f'--matches {n} --workers {WORKERS} --seed-base {seed} --out {out}')
    d = json.load(open(out))
    ov = d.get("overall") or {}
    v = ov.get("score", ov.get("mean"))
    assert v is not None, f"✗ 无分数：{out}"
    return float(v) * 100.0, ov


def main():
    global WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", required=True, help="候选 agent 名（build/agents 下）")
    ap.add_argument("--banks", nargs="+", required=True, help="bank .bin（本地或 /root/… 远端）")
    ap.add_argument("--bank-decks", nargs="+", default=[], help="与 banks 一一对应的牌表名；缺省=镜像候选牌表")
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--seed", type=int, default=960000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lo", type=float, default=0.1, help="低于此判太强（不合格）")
    ap.add_argument("--hi", type=float, default=0.8, help="高于此判太弱（不合格）")
    ap.add_argument("--gpu", default="", help="给了就把 /root/… 的 bank 从这台拉回本地")
    ap.add_argument("--out", default="logs/bank_audit.json")
    ap.add_argument("--emit-ok", action="store_true", help="只打印每个候选可用的 bank 路径")
    a = ap.parse_args()
    WORKERS = a.workers

    # 准备 bank agent：(名字, 本地 bin, 牌表 csv 或 None, 原始路径)
    bank_agents = []
    for k, b in enumerate(a.banks):
        loc = ensure_local_bin(b, a.gpu)
        nm = f"auditbank_{k}"
        deck_csv = None
        if k < len(a.bank_decks):
            deck_csv = f"decks/pool_field/{a.bank_decks[k]}.csv"
            assert os.path.exists(deck_csv), f"✗ 牌表不存在：{deck_csv}"
        bank_agents.append((nm, loc, deck_csv, b))

    report = {"lo": a.lo, "hi": a.hi, "n": a.n, "rows": [], "ok_by_agent": {}}
    for ag in a.agents:
        cand = f"build/agents/{ag}"
        assert os.path.exists(os.path.join(cand, "main.py")), f"✗ 缺 {cand}/main.py"
        cand_deck = os.path.join(cand, "deck.csv")
        oks = []
        print(f"\n== {ag}  vs  {len(bank_agents)} 个 bank（n={a.n}）")
        for nm, loc, deck_csv, orig in bank_agents:
            bank_dir = stage(f"{ag}__{nm}", to_pt(loc, f"{ag}__{nm}"), deck_csv or cand_deck)
            out = f"logs/bank_audit/{ag}__{nm}.json"
            wr, ov = eval_pair(cand, bank_dir, a.n, a.seed, out)
            frac = wr / 100.0
            ok = a.lo < frac < a.hi
            verdict = "✅ 可用" if ok else ("❌ 太弱(我们>80%)" if frac >= a.hi else "❌ 太强(我们<10%)")
            print(f"   {os.path.basename(loc):32s} {wr:6.2f}%  "
                  f"CI[{ov.get('ci_low', 0)*100:.1f},{ov.get('ci_high', 0)*100:.1f}]  {verdict}")
            report["rows"].append(dict(agent=ag, bank=os.path.basename(loc), wr=wr,
                                       ci_low=ov.get("ci_low"), ci_high=ov.get("ci_high"),
                                       verdict=verdict, ok=bool(ok)))
            if ok:
                oks.append(orig)
        report["ok_by_agent"][ag] = oks

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n体检表 → {a.out}")
    if a.emit_ok:
        for ag, paths in report["ok_by_agent"].items():
            print(f"BANKS_{ag}={','.join(paths)}")
    bad = [r for r in report["rows"] if not r["ok"]]
    if bad:
        print(f"\n⚠️ {len(bad)} 条不合格（这些 bank 不能当训练对手）：")
        for r in bad:
            print(f"   {r['agent']} × {r['bank']} = {r['wr']:.2f}%")
        return 1
    print("\n✅ 全部 bank 强度在线")
    return 0


if __name__ == "__main__":
    sys.exit(main())
