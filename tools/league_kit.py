#!/usr/bin/env python3
"""**打一个联赛训练包**：把某一代需要的 init + 对手 bank + 启动脚本装成一个 tar.gz。

本机（RTX 3050 / 4 GB）跑不了原生 PPO，训练必须在 4090 机器上做。缺的东西只有两类：
**权重文件**和**一条准确的命令行**。这个脚本把两件事都装好，并在 manifest 里记 sha256 ——
传坏了会当场发现，而不是训练 40 分钟后才发现 init 不对。

    ./tools/py tools/league_kit.py --gen 1              # 需要先 run_league.py next 生成 plan
    ./tools/py tools/league_kit.py --gen 1 --banks 4    # 改每代 bank 数（2–4）
    ./tools/py tools/league_kit.py --list               # 只打印这一代要带什么

输出：`dist/league_g0001.tar.gz`，解压后 `bash league_g0001/run.sh` 即可。

⚠️ 两个硬件事实写在包里：
1. **每代只上 2–4 个 bank**：训练器把对手池烘成 cudagraph 指针，跑起来不能加；
   每 bank ≈ 1.0 GiB（512 agents 实测 15.8 GB → +2 bank 18.2 GB），
   且断言 `league_pct × n_banks < 0.5`。所以池子 8–16 个、每代抽 2–4 个轮换。
2. `--timesteps` 是**绝对**步数：从 `state_latest.bin` 续跑时如果 ≤ 当前步数会"nothing to do"。
   本包的 run 脚本默认 `--no-resume` 是**故意**的：联赛每一代都是新分支。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "ptcg_league")
sys.path.insert(0, HERE)

import league as L                                     # noqa: E402

RUN_SH = r"""#!/usr/bin/env bash
# 自动生成（tools/league_kit.py）：训练联赛 第 {gen} 代
#   bash run.sh
# 产出：final.bin（约 18 MB）—— 传回本机 ptcg_league/pools/evo/{branch}/final.bin
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${{PTCG_REPO:-$HOME/work/ptcg_train_kit/port_repo}}"
[ -f "$REPO/native/build_native.sh" ] || {{ echo "✗ 找不到训练仓库 $REPO（用 PTCG_REPO= 指定）"; exit 1; }}
cd "$REPO"
PY="${{PYTHON:-python3}}"
RUN="{run_name}"
RD="experiments/native-$RUN"
mkdir -p "$RD" logs

# ── 校验带过来的权重没有传坏（传坏一小时训练就白跑了）──
bad=0
for f in "$HERE"/init.bin "$HERE"/bank_*.bin; do
  [ -f "$f" ] || continue
  bn=$(basename "$f")
  exp=$(awk -v n="$bn" '$2==n {{print $1}}' "$HERE/MANIFEST.txt")
  got=$(sha256sum "$f" | cut -d' ' -f1)
  if [ -z "$exp" ]; then echo "✗ $bn 不在 MANIFEST 里"; bad=1
  elif [ "$exp" != "$got" ]; then echo "✗ $bn sha 不符：期望 $exp 实得 $got"; bad=1; fi
done
[ "$bad" = 0 ] || {{ echo "包不完整，重新传"; exit 1; }}
echo "✓ 权重校验通过（{nbanks} 个 bank + init）"

# ── 卡表尺寸常量：5 份副本里漏改一份，训练器会在开跑几秒后 Aborted（看不出是哪份）──
if [ -f "$HERE/check_consts.py" ]; then
  "$PY" "$HERE/check_consts.py" native/ptcg_tables_field.bin >/dev/null \
    || {{ echo "✗ 卡表常量不一致，先修再跑"; exit 1; }}
  echo "✓ 卡表常量一致"
fi

# ── 显存检查：残留进程会直接 CUDA OOM（之前踩过三次）──
if command -v nvidia-smi >/dev/null; then
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "显存已用 ${{used}} MiB（512 agents + {nbanks} bank 约需 18–19 GB）"
  if [ "${{used:-0}}" -gt 4000 ]; then
    echo "⚠️ 显存已被占用，可能是残留训练进程："
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv
    echo "   先 kill -9 掉再跑，否则一定 OOM"
  fi
fi

cp -f "$HERE/init.bin" "$RD/init.bin"

# 定向野区权重（专精训练）：拷进仓库，训练器只吃**仓库内的相对路径**
DWF="{deck_weights_file}"
if [ -f "$HERE/deck_weights.txt" ]; then
  cp -f "$HERE/deck_weights.txt" "decks/league_gen{gen:04d}.txt"
  DWF="decks/league_gen{gen:04d}.txt"
  echo "deck weights = $DWF（专精目标 {target_archetype}）"
else
  echo "deck weights = $DWF（通用分布）"
fi
BANKS=$(ls "$HERE"/bank_*.bin | sort | paste -sd, -)
echo "init  = $HERE/init.bin"
echo "banks = $BANKS  (league_pct={league_pct})"

PYTHONPATH=data:.:native PTCG_TABLES=native/ptcg_tables_field.bin OMP_WAIT_POLICY=PASSIVE \
"$PY" native/train_native.py \
  --run-name "$RUN" \
  --timesteps {steps} \
  --total-agents {total_agents} \
  --minibatch {minibatch} \
  --lr {lr} \
  --no-anneal-lr \
  --mix-self 1.0 \
  --league-ckpts "$BANKS" \
  --league-pct {league_pct} \
  --deck-pool {deck_pool} \
  --deck-weights-file "$DWF" \
  --anchor-decks "{deck_stem}" \
  --p-anchor-learner 1.0 \
  --p-anchor-self-opp 0.0 \
  --p-anchor-league-opp 0.0 \
  --checkpoint-interval {checkpoint_interval} \
  --seed {seed} \
  --no-resume \
  2>&1 | tee "logs/$RUN.log"

if [ ! -f "$RD/final.bin" ]; then
  echo "✗ 没有 final.bin；看上面日志。可能原因：CUDA OOM（先 kill 掉残留进程）、"\
"或 --timesteps 太小（绝对步数语义）"
  exit 1
fi
echo
echo "── 训练健康指标（预注册看这两个）──"
grep -oE "lgw[ =:]+[0-9.]+|ep_len[ =:]+[0-9.]+" "logs/$RUN.log" | tail -6
echo "  lgw 0.4–0.7 健康；>0.9 且 ep_len 下降 = 退化成自己跟自己下（历史问题）"

cp -f "$RD/final.bin" "$HERE/final.bin"
echo
echo "✅ 完成：$HERE/final.bin"
echo "传回本机："
echo "  scp $HERE/final.bin <本机>:{ckpt_out}"
echo "然后本机跑：./tools/py ptcg_league/run_league.py next --resume"
"""

README = """联赛训练包 · 第 {gen} 代（{branch}）

## 在 4090 机器上
    cd ~/work
    tar xzf league_g{gen:04d}.tar.gz
    cd league_g{gen:04d}
    tmux new -s lg{gen:04d}          # 训练要 40 分钟以上，别让它挂在 SSH 里
    bash run.sh
    # 另开一个窗口看进度：
    tail -f ~/work/ptcg_train_kit/port_repo/logs/{run_name}.log

## 跑完
    scp ~/work/league_g{gen:04d}/final.bin <本机>:{ckpt_out}

## 本机
    ./tools/py ptcg_league/run_league.py next --resume

## 这一代的预注册判据（不要事后改）
- 筛查 {screen_n} 场对当前 best：得分 ≥ {screen_min} 且异常 = 0
- 确认 {confirm_n} 场：对 best 的配对胜率 CI 下界 > {ci_low}
- 固定评测集（{anchor_opps} 个冻结对手 × {anchor_n} 场）：相对 best 回落 ≤ {anchor_reg}
- 三条全过才晋级；CI 跨 0.50 是"不能宣称提升"，不是"变差了"

## 本代是**专精代**（目标对位：{target_archetype}）
- 对手座位按 `decks/league_gen{gen:04d}.txt` 抽牌：{target_archetype} 族占 {target_share}
- 额外有一把**专精探针**（不进晋级门）：候选与父模型在同一种子上各打目标对位 N 场，做配对比较
- 判定：总体 <0.52 但探针配对提升 ≥+5pp 且 CI 下界 >0 ⇒ 记为 `specialist`（进池当对手，不当 best）
- ⚠️ 探针是**弱代理**：我们的 BC shuppet 脑子太弱（端口打它们已 0.92），
  真实 0/16 的裁决只能靠天梯提交 + 抓回放看 shuppet 那格

## 这一代带的东西
init （PPO 起点，来自父模型 {parent}）：
{banks_text}
卡组（学习者恒定）：{deck}
超参：lr={lr} agents={total_agents} minibatch={minibatch} steps={steps}（≈{matches} 局）
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=None)
    ap.add_argument("--banks", type=int, default=None, help="覆盖每代 bank 数（2–4）")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "dist"))
    a = ap.parse_args()

    rules = L.load_rules(HERE)
    reg = L.Registry()
    state = L.LeagueState()
    g = a.gen if a.gen is not None else state.gen + 1
    plan_path = os.path.join(HERE, "gen", f"{g:04d}", "plan.json")
    if not os.path.exists(plan_path):
        raise SystemExit(f"没有 {plan_path}：先跑 ./tools/py ptcg_league/run_league.py next --gens 1")
    plan = L.read_json(plan_path)

    rows = []
    for aid in plan["league_opponents"]:
        r = reg.get(aid)
        b = r.get("bin_path") or r.get("checkpoint_path") or ""
        if not b:
            raise SystemExit(f"{aid} 没有 .bin（训练器只吃原生 blob）：{r}")
        rows.append((aid, b))
    parent = reg.get(plan["parent_id"])
    init_bin = plan.get("init_bin") or parent.get("bin_path") or ""
    if not init_bin:
        raise SystemExit(f"父模型 {plan['parent_id']} 没有 .bin，无法当 PPO init")

    if a.list:
        print(f"第 {g} 代 {plan['branch']}")
        print(f"  init      {init_bin}")
        for aid, b in rows:
            print(f"  bank      {aid}  {b}")
        print(f"  卡组       {plan['deck']}")
        print(f"  league_pct {plan['league_pct']}  steps {plan['steps']}")
        return

    tag = f"league_g{g:04d}"
    stage = os.path.join(a.outdir, tag)
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage, exist_ok=True)

    deck_wf = plan.get("deck_weights_file") or ""
    files = [("init.bin", init_bin)]
    for i, (aid, b) in enumerate(rows):
        files.append((f"bank_{i}.bin", b))
    if deck_wf and os.path.exists(deck_wf) and not deck_wf.startswith("decks/weights_real"):
        files.append(("deck_weights.txt", deck_wf))
    man = []
    for name, src in files:
        dst = os.path.join(stage, name)
        shutil.copyfile(src, dst)
        man.append(f"{L.sha256_file(dst)}  {name}  <- {src}")
    with open(os.path.join(stage, "MANIFEST.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(man) + f"\n# pack {L.now()}  gen {g}  rules {L.rules_sha(rules)[:16]}\n")
    L.write_json(os.path.join(stage, "plan.json"), plan)

    deck_stem = os.path.basename(plan["deck"]).replace(".csv", "")
    runsh = RUN_SH.format(gen=g, branch=plan["branch"], run_name=plan["run_name"],
                          league_pct=plan["league_pct"], steps=plan["steps"],
                          total_agents=plan["total_agents"], minibatch=plan["minibatch"],
                          lr=plan["lr"], deck_pool=plan["deck_pool"],
                          # ⚠️ 必须带 `__<hash>` 全名：训练器里
                          # `resolve_anchor_indices()` 是 `assert n in pool_names`（**精确匹配**，
                          # 没有前缀兜底）。写成不带 hash 的名字会直接 assert 挂掉。
                          deck_stem=os.path.basename(plan["deck"]).replace(".csv", ""),
                          checkpoint_interval=plan["checkpoint_interval"],
                          nbanks=len(rows),
                          deck_weights_file=(f"league_gen{g:04d}.txt"
                                             if plan.get("deck_weights_file")
                                             and not plan["deck_weights_file"].startswith("decks/weights_real")
                                             else plan["deck_weights_file"]),
                          target_archetype=plan.get("target_archetype") or "无",
                          seed=plan["seed"], ckpt_out=plan["ckpt_out"])
    cc = os.path.join(ROOT, "dist", "autodl_kit", "scripts", "check_consts.py")
    if os.path.exists(cc):
        shutil.copyfile(cc, os.path.join(stage, "check_consts.py"))
    with open(os.path.join(stage, "run.sh"), "w", encoding="utf-8") as f:
        f.write(runsh)
    os.chmod(os.path.join(stage, "run.sh"), 0o755)
    banks_text = "\n".join(f"  bank{i} ({aid}): {os.path.basename(b)}"
                           for i, (aid, b) in enumerate(rows))
    with open(os.path.join(stage, "README.txt"), "w", encoding="utf-8") as f:
        f.write(README.format(
            gen=g, branch=plan["branch"], run_name=plan["run_name"], ckpt_out=plan["ckpt_out"],
            screen_n=rules["screen"]["matches"], screen_min=rules["screen"]["min_score"],
            confirm_n=rules["confirm"]["matches"],
            ci_low=rules["confirm"]["h2h_ci_low_above"],
            anchor_opps=len([r for r in reg.in_pool("init") if r.get("eval_dir")]),
            anchor_n=rules["anchor"]["matches_per_opponent"],
            anchor_reg=rules["anchor"]["max_regress"], parent=plan["parent_id"],
            target_share=(f"{plan['_target_share']:.0%}" if plan.get("_target_share") else "—"),
            target_archetype=plan.get("target_archetype") or "无",
            banks_text=banks_text, deck=os.path.basename(plan["deck"]),
            lr=plan["lr"], total_agents=plan["total_agents"],
            minibatch=plan["minibatch"], steps=plan["steps"],
            matches=plan["steps"] // int(rules["train"]["steps_per_match"])))

    tar = os.path.join(a.outdir, tag + ".tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(stage, arcname=tag)
    sz = os.path.getsize(tar) / 1048576
    print(f"✅ {tar}  {sz:.1f} MB  sha {L.sha256_file(tar)[:16]}")
    print(f"   init {init_bin}")
    for aid, b in rows:
        print(f"   bank {aid} → {os.path.basename(b)}")
    print(f"   第 {g} 代 卡组 {os.path.basename(plan['deck'])}  steps {plan['steps']}")


if __name__ == "__main__":
    main()
