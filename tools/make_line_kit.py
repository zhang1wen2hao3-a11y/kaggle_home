#!/usr/bin/env python3
"""打包**卡组线 · 任意 anchor 的 PPO 一代**训练包（GPU 机器上跑）。

同一套机制只换参数，所以以后不用再打补丁：
    anchor 卡组、对手卡组分布、父权重 INIT、观察用 TRACK —— 全是命令行参数。

本仓库已在用的两代：
    # A：学习者打 db04，对手座位里 shuppet 抬到 31%（针对真实天梯 0 胜 23 负的对位）
    ./tools/py tools/make_line_kit.py --init port_repo/native/pt_model_field.bin \
        --anchor dunsparce_abra__db04 --dwf decks/weights_shuppet31.txt \
        --run g7a_db04_shuppet --out /root/g7a --parent "端口原始权重"

    # B：学习者打 shuppet，对手按真实野区（造一个强 shuppet 对手进池）
    ./tools/py tools/make_line_kit.py --init port_repo/native/pt_model_field.bin \
        --anchor shuppet_dhelmise__0d86 --dwf deck_repo/decks/weights_real.txt \
        --run g7b_shuppet --out /root/g7b --parent "端口原始权重"
"""
import argparse
import hashlib
import os
import shutil
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "port_repo")

APPLY_SH = r"""#!/usr/bin/env bash
# 自动生成（tools/make_line_kit.py）：铺文件 → 预检（不花 GPU）→ 可选开训。
#   PTCG_REPO=/root/ptcg_train_kit/port_repo bash apply.sh
#   PTCG_REPO=/root/ptcg_train_kit/port_repo bash apply.sh --run
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
PY="${PYTHON:-python3}"
REPO="${PTCG_REPO:-$HOME/work/ptcg_train_kit/port_repo}"
[ -f "$REPO/native/train_native.py" ] || { echo "✗ 找不到训练仓库：$REPO"; echo "  用 PTCG_REPO=/root/ptcg_train_kit/port_repo bash apply.sh"; exit 1; }
echo "repo = $REPO"

bad=0
while read -r sha name; do
  [ -f "$name" ] || { echo "✗ 缺 $name"; bad=1; continue; }
  [ "$sha" = "$(sha256sum "$name" | cut -d' ' -f1)" ] || { echo "✗ $name sha 不符（传坏了）"; bad=1; }
done < <(grep -v '^#' MANIFEST.txt | awk '{print $1, $2}')
[ "$bad" = 0 ] || { echo "包不完整，重传"; exit 1; }
echo "✓ 包完整性通过"

grep -q "n_track" "$REPO/native/ptcg/binding.c" 2>/dev/null \
  || { echo "✗ $REPO 没有 deck-seat 补丁（binding.c 里找不到 n_track）⇒ --track-decks 会静默失效"; exit 1; }
echo "✓ 训练仓库已有 v2 座位计数补丁"

mkdir -p "$REPO/decks/pool_field" "$REPO/tools" "$REPO/init" "$HERE/banks_used" "$REPO/native"
for f in decks/pool_field/*.csv; do cp -f "$f" "$REPO/decks/pool_field/"; done
cp -f decks/@DWFNAME@ "$REPO/decks/@DWFNAME@"
cp -f tools/db04_line.sh "$REPO/tools/db04_line.sh"; chmod +x "$REPO/tools/db04_line.sh"
cp -f tools/patch_bank_decks.py "$REPO/tools/patch_bank_decks.py" 2>/dev/null || true
cp -f banks/*.bin "$HERE/banks_used/"
cp -f init/@INITNAME@ "$REPO/init/@INITNAME@"
# 关键：把打包机上的 tables blob 原样装上。卡组池加了新卡组后（anchor 不在旧池里），
# GPU 机器自己那份 blob 还是旧 185 行，不覆盖就会在 resolve_anchor_indices 处报
# “pool 行 ≠ blob 行”。blob 随包走，deck id 才永远和打包机一致。
cp -f native/ptcg_tables_field.bin "$REPO/native/ptcg_tables_field.bin"
echo "✓ 已铺：卡组(anchor) + decks/@DWFNAME@ + tables blob + tools/db04_line.sh + init/@INITNAME@ + bank(s)"

cd "$REPO"
RUN_ENV="PYTHON=$PY INIT=$REPO/init/@INITNAME@ OUT=@OUT@ RUN=@RUN@ \
ANCHOR=@ANCHOR@ DWF=decks/@DWFNAME@ TRACK=@TRACK@ \
BANKS=@BANKLIST@ \
LEAGUE_PCT=@LEAGUEPCT@ \
BANKDECKS=@BANKDECKS@"
env $RUN_ENV bash tools/db04_line.sh --check
rc=$?
[ $rc -eq 0 ] || { echo; echo "✗ 预检没过 —— 别开训，把上面整段发回来"; exit $rc; }

if [ "${1:-}" = "--run" ]; then
  echo; echo "===== 预检通过，开训：anchor=@ANCHOR@ 父权重=@INITNAME@ ====="
  env $RUN_ENV bash tools/db04_line.sh
  echo
  echo "拉回本机：@OUT@/@RUN@.bin  与  $REPO/logs/@RUN@.log"
else
  echo
  echo "预检通过。开训命令："
  echo "  cd $REPO && env $RUN_ENV bash tools/db04_line.sh"
fi
"""

RUNBOOK = """{run}：卡组线一代（anchor = {anchor}）
==================================================
父权重 `{initname}`，对手卡组分布 `{dwfname}`，{nbank} 个冻结 bank（{banksnote}）@ league-pct {lpct}（合计 {banktotal}，须 <0.5），
T=400k、AG=512、MB=512、lr=2e-4、SEED=424242、THREADS=8（与历史各代逐项相同，便于对照）。

## 这一代要回答什么

{purpose}

## 跑法

```bash
tar xzf {tag}.tar.gz && cd {tag}
PTCG_REPO=/root/ptcg_train_kit/port_repo bash apply.sh          # 只预检，不花 GPU
PTCG_REPO=/root/ptcg_train_kit/port_repo bash apply.sh --run    # 预检通过直接开训
```

预检会打印：`pool_field == tables blob`、`anchor → deck id`、
`weights version=3 / 生效副数 / ESS`、**anchor 权重（必须 >0，否则训练器 REJECTED）**、
镜像对局率、父权重 18,205,408 B、{nbank} 个 bank 且逐位 PINNED、仓库有没有 v2 座位补丁。

## 预期与硬门槛

- 耗时 ≈ **8 分钟**（400k 步，~930 steps/s）；显存 ≈17 GB / 22.5 GB —— **一次只跑一支**
- 日志必须有三行，缺一行该支作废：
  1. `anchor decks resolved: {{'{anchor}': <id>}}`
  2. `deck weights -> v3, ... active, ESS ...`
  3. `tl0:≈1.00/to0:≈...` ← 学习者确实在打 anchor 牌的**证据**
- 退化信号：`lgw → 1.00`、`d0 → 0.95+`、`ep_len` 断崖

## 交回来

```
{out}/{run}.bin
<repo>/logs/{run}.log
```

评估：cand@{anchor} vs port@{anchor}，1600 场配对、种子基准 900000（与历史各代同批种子 ⇒ 可逐场配对）；
判据 **CI 下界 > 0.50**。{extra}
"""


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="父权重（native blob，18,205,408 B）")
    ap.add_argument("--run", required=True, help="运行名，如 g7a_db04_shuppet")
    ap.add_argument("--anchor", default="dunsparce_abra__db04", help="学习者 100% 打的卡组")
    ap.add_argument("--dwf", default="port_repo/decks/weights_real.txt", help="对手卡组分布文件")
    ap.add_argument("--track", default="", help="只观测的卡组（默认 = anchor,dunsparce_abra__db04）")
    ap.add_argument("--parent", default="", help="RUNBOOK 里显示的父代说明")
    ap.add_argument("--banks", default=None,
                    help="两个 bank 的源文件（逗号分隔，native blob 18,205,408 B）；"
                         "默认从 dist/league_g0003.tar.gz 取")
    ap.add_argument("--banks-note", default="", help="RUNBOOK 里说明这对 bank 是什么")
    ap.add_argument("--bank-decks", default="", help="与 --banks 逐位对应的卡组名，'-' 不绑定（需 --bank-decks 补丁）")
    ap.add_argument("--stage-full-pool", action="store_true",
                    help="把 decks/pool_field 全部 187 副都打进包（约 34 KB）。"
                         "默认只带 anchor+track，靠目标机已有池子补齐；"
                         "但 blob 是 187 行的，目标池若有漂移会 fail-fast。"
                         "要保证 pool==blob，用这个开关。")
    ap.add_argument("--league-pct", type=float, default=0.08, help="每个 bank 占的行比例（1 个 bank 时可到 0.49）")
    ap.add_argument("--purpose", default="", help="RUNBOOK 里这一代要回答的问题")
    ap.add_argument("--extra", default="", help="RUNBOOK 结尾追加的判据说明")
    ap.add_argument("--out", default=None, help="远端输出目录（默认 /root/<run 前缀>）")
    ap.add_argument("--tag", default=None, help="包名（默认 {run}_kit）")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "dist"))
    a = ap.parse_args()

    init = os.path.abspath(a.init)
    if not os.path.exists(init):
        raise SystemExit(f"✗ 父权重不存在：{init}")
    if os.path.getsize(init) != 18205408:
        raise SystemExit(f"✗ 父权重大小 {os.path.getsize(init)} ≠ 18,205,408（不是 native blob）")
    initname = os.path.basename(init)
    deck_csv = os.path.join(REPO, "decks", "pool_field", a.anchor + ".csv")
    if not os.path.exists(deck_csv):
        raise SystemExit(f"✗ anchor 卡组不在 pool_field：{deck_csv}")
    dwf = a.dwf if os.path.isabs(a.dwf) else os.path.join(ROOT, a.dwf)
    if not os.path.exists(dwf):
        raise SystemExit(f"✗ 卡组权重文件不存在：{dwf}")
    dwfname = os.path.basename(dwf)
    _tk = a.track or f"{a.anchor},dunsparce_abra__db04"
    _seen, _out = set(), []
    for _n in [x.strip() for x in _tk.split(",") if x.strip()]:
        if _n not in _seen:
            _seen.add(_n); _out.append(_n)
    track = ",".join(_out)
    tag = a.tag or f"{a.run}_kit"
    out = a.out or f"/root/{a.run.split('_')[0]}"

    stage = os.path.join(a.outdir, tag)
    shutil.rmtree(stage, ignore_errors=True)
    for d in ("tools", "decks/pool_field", "banks", "init", "native"):
        os.makedirs(os.path.join(stage, d), exist_ok=True)

    shutil.copyfile(os.path.join(ROOT, "tools", "db04_line.sh"),
                    os.path.join(stage, "tools", "db04_line.sh"))
    # 补丁脚本也要随包走：apply.sh 缺补丁时打印的修复命令就是跑它，
    # 包里没有这个文件，操作员在 GPU 机上无路可走（2026-11-05 真实踩到）。
    shutil.copyfile(os.path.join(ROOT, "tools", "patch_bank_decks.py"),
                    os.path.join(stage, "tools", "patch_bank_decks.py"))
    os.chmod(os.path.join(stage, "tools", "db04_line.sh"), 0o755)
    if a.stage_full_pool:
        for f in sorted(os.listdir(os.path.join(REPO, "decks", "pool_field"))):
            if f.endswith(".csv"):
                shutil.copyfile(os.path.join(REPO, "decks", "pool_field", f),
                                os.path.join(stage, "decks", "pool_field", f))
    shutil.copyfile(deck_csv, os.path.join(stage, "decks", "pool_field", a.anchor + ".csv"))
    # TRACK 里用到的卡组也带过去（apply.sh 会把池里缺的补进去，顺序不变就不会错位）
    for name in [x.strip() for x in track.split(",") if x.strip()]:
        src = os.path.join(REPO, "decks", "pool_field", name + ".csv")
        if os.path.exists(src) and name != a.anchor:
            shutil.copyfile(src, os.path.join(stage, "decks", "pool_field", name + ".csv"))
    shutil.copyfile(dwf, os.path.join(stage, "decks", dwfname))
    shutil.copyfile(init, os.path.join(stage, "init", initname))
    shutil.copyfile(os.path.join(REPO, "native", "ptcg_tables_field.bin"),
                    os.path.join(stage, "native", "ptcg_tables_field.bin"))

    if a.banks:
        srcs = [x.strip() for x in a.banks.split(",") if x.strip()]
        if not (1 <= len(srcs) <= 16):
            raise SystemExit(f"✗ --banks 需要 1..16 个，给了 {len(srcs)}")
        for i, src in enumerate(srcs):
            if not os.path.exists(src):
                raise SystemExit(f"✗ bank 不存在：{src}")
            if os.path.getsize(src) != 18205408:
                raise SystemExit(f"✗ {src} 不是 18,205,408 B（不是 native blob）")
            shutil.copyfile(src, os.path.join(stage, "banks", f"bank_{i}.bin"))
        banks_note = a.banks_note or (" + ".join(os.path.basename(x) for x in srcs))
        _nbank = len(srcs)
    else:
        tgz = os.path.join(ROOT, "dist", "league_g0003.tar.gz")
        if not os.path.exists(tgz):
            raise SystemExit("✗ 找不到 dist/league_g0003.tar.gz（或用 --banks 指定两个 .bin）")
        with tarfile.open(tgz) as t:
            for i, member in enumerate(("league_g0003/bank_0.bin", "league_g0003/bank_1.bin")):
                dst = os.path.join(stage, "banks", f"bank_{i}.bin")
                with t.extractfile(member) as fsrc, open(dst, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
        banks_note = a.banks_note or "league_g0003 的两个冻结快照"
        _nbank = 2

    apply_txt = (APPLY_SH.replace("@DWFNAME@", dwfname).replace("@INITNAME@", initname)
                 .replace("@RUN@", a.run).replace("@OUT@", out)
                 .replace("@ANCHOR@", a.anchor).replace("@TRACK@", track)
                 .replace("@BANKDECKS@", a.bank_decks)
                 .replace("@LEAGUEPCT@", str(a.league_pct))
                 .replace("@BANKLIST@", ",".join(f"$HERE/banks_used/bank_{i}.bin"
                                                 for i in range(_nbank))))
    p = os.path.join(stage, "apply.sh")
    open(p, "w", encoding="utf-8").write(apply_txt)
    os.chmod(p, 0o755)
    open(os.path.join(stage, "RUNBOOK.md"), "w", encoding="utf-8").write(
        RUNBOOK.format(run=a.run, anchor=a.anchor, initname=initname, dwfname=dwfname,
                       out=out, tag=tag, purpose=a.purpose or "（未填写）", extra=a.extra,
                       banksnote=banks_note, nbank=_nbank,
                       lpct=f"{a.league_pct:g}", banktotal=f"{a.league_pct*_nbank:.2f}"))

    man = []
    _pool_rels = ([f"decks/pool_field/{f}" for f in sorted(os.listdir(os.path.join(stage, "decks", "pool_field")))
                   if f.endswith(".csv")] if a.stage_full_pool else
                  [f"decks/pool_field/{n}.csv" for n in [a.anchor] + [x.strip() for x in track.split(",")
                                                                     if x.strip() and x.strip() != a.anchor]])
    for rel in (["tools/db04_line.sh", "tools/patch_bank_decks.py"] + _pool_rels +
                [f"decks/{dwfname}", f"init/{initname}"] + [f"banks/bank_{i}.bin" for i in range(_nbank)] + [
                 "native/ptcg_tables_field.bin"]):
        f = os.path.join(stage, rel)
        man.append(f"{sha256(f)}  {rel}  {os.path.getsize(f)}")
    open(os.path.join(stage, "MANIFEST.txt"), "w", encoding="utf-8").write(
        "# sha256  path  bytes\n" + "\n".join(man) + "\n")

    tar = os.path.join(a.outdir, f"{tag}.tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(stage, arcname=tag)
    print(f"✅ {tar}  {os.path.getsize(tar) / 1048576:.0f} MB  sha {sha256(tar)[:16]}")
    print(f"   run={a.run}  anchor={a.anchor}  权重={dwfname}  父={initname} → {out}/{a.run}.bin")
    for line in man:
        print("   " + "  ".join(line.split()[:2]))


if __name__ == "__main__":
    main()
