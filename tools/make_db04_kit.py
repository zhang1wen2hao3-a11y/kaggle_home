#!/usr/bin/env python3
"""打包 **db04 卡组线 · 下一代** 训练包（GPU 机器上跑）。

这张表只有一个变量：**父权重 INIT**。其余（anchor=db04、weights_real、2 个冻结 bank、
league-pct、lr、AG、seed、T）逐项固定，所以 G5 → G6 的差异只可能来自"从哪起步"。

包里放：运行器 + 父权重 + 卡组 + 权重文件 + 两个冻结 bank + 表格 blob（供预检核对）
+ apply.sh（校验→铺文件→预检→可选开训）+ RUNBOOK。

用法：
    ./tools/py tools/make_db04_kit.py \
        --init models/ppo/g5_db04.bin --run g6_db04 --parent "G5（confirm 0.5550 [0.5305,0.5792]）"
"""
import argparse
import hashlib
import os
import shutil
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "port_repo")
DECK = "dunsparce_abra__db04"

APPLY_SH = r"""#!/usr/bin/env bash
# 自动生成（tools/make_db04_kit.py）：把本期需要的文件铺进训练仓库，并跑一次不花 GPU 的预检。
#   PTCG_REPO=/root/ptcg_train_kit/port_repo bash apply.sh          # 只预检
#   PTCG_REPO=/root/ptcg_train_kit/port_repo bash apply.sh --run    # 预检通过直接开训
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
PY="${PYTHON:-python3}"
REPO="${PTCG_REPO:-$HOME/work/ptcg_train_kit/port_repo}"
[ -f "$REPO/native/train_native.py" ] || { echo "✗ 找不到训练仓库：$REPO"; echo "  用 PTCG_REPO=/root/ptcg_train_kit/port_repo bash apply.sh"; exit 1; }
echo "repo = $REPO"

# ① 完整性（传坏了别浪费一小时 GPU）
bad=0
while read -r sha name; do
  [ -f "$name" ] || { echo "✗ 缺 $name"; bad=1; continue; }
  [ "$sha" = "$(sha256sum "$name" | cut -d' ' -f1)" ] || { echo "✗ $name sha 不符（传坏了）"; bad=1; }
done < <(grep -v '^#' MANIFEST.txt | awk '{print $1, $2}')
[ "$bad" = 0 ] || { echo "包不完整，重传"; exit 1; }
echo "✓ 包完整性通过"

# ② 训练仓库必须已经打了 v2 座位计数补丁（没有它 --track-decks 会静默失效）
grep -q "n_track" "$REPO/native/ptcg/binding.c" 2>/dev/null \
  || { echo "✗ $REPO 没有 deck-seat 补丁（binding.c 里找不到 n_track）"; echo "  先用 dist/ptcg_train_kit.tar.gz 的补丁：tools/patch_deck_seat_counters.py"; exit 1; }
echo "✓ 训练仓库已有 v2 座位计数补丁"

# ③ 铺文件
mkdir -p "$REPO/decks/pool_field" "$REPO/tools" "$REPO/init" "$HERE/banks_used"
cp -f "decks/pool_field/@DECK@.csv" "$REPO/decks/pool_field/@DECK@.csv"
cp -f tools/db04_line.sh "$REPO/tools/db04_line.sh"; chmod +x "$REPO/tools/db04_line.sh"
cp -f banks/bank_0.bin banks/bank_1.bin "$HERE/banks_used/"
cp -f init/@INITNAME@ "$REPO/init/@INITNAME@"
echo "✓ 已铺：decks/pool_field/@DECK@.csv, tools/db04_line.sh, init/@INITNAME@, banks_used/（两个 bank）"

# ④ 预检（不花 GPU）
cd "$REPO"
RUN_ENV="PYTHON=$PY INIT=$REPO/init/@INITNAME@ OUT=@OUT@ RUN=@RUN@ \
BANKS=$HERE/banks_used/bank_0.bin,$HERE/banks_used/bank_1.bin"
env $RUN_ENV bash tools/db04_line.sh --check
rc=$?
[ $rc -eq 0 ] || { echo; echo "✗ 预检没过 —— 别开训，把上面整段发回来"; exit $rc; }

if [ "${1:-}" = "--run" ]; then
  echo; echo "===== 预检通过，开训（父权重 = @INITNAME@） ====="
  env $RUN_ENV bash tools/db04_line.sh
  echo
  echo "拉回本机：@OUT@/@RUN@.bin  与  $REPO/logs/@RUN@.log"
else
  echo
  echo "预检通过。开训命令："
  echo "  cd $REPO && env $RUN_ENV bash tools/db04_line.sh"
fi
"""

RUNBOOK = """db04 卡组线 · {run}（父权重 = {initname}）
================================================
这张表**只有一个变量**：父权重 INIT。其余与 G5 逐项固定 ——
anchor=`{deck}`、对手卡组分布 `weights_real`（185 副）、2 个冻结 bank @ league-pct 0.08、
T=400k、AG=512、MB=512、lr=2e-4、**SEED=424242**、THREADS=8。

| 代 | 父权重 | 本地 confirm（vs 端口，同卡组 1600 场） |
|---|---|---|
| G5 | 端口原始权重 | **0.5550 [0.5305, 0.5792]** ✅ 通过 |
| **{run}** | **{parent}** | 待测（判据同 G5：CI 下界 > 0.50） |

## 跑法

```bash
tar xzf {tag}.tar.gz && cd {tag}
PTCG_REPO=/root/ptcg_train_kit/port_repo bash apply.sh          # 只预检，不花 GPU
PTCG_REPO=/root/ptcg_train_kit/port_repo bash apply.sh --run    # 预检通过直接开训
```

预检会打印/校验：`pool_field 185 副 == tables blob 185 行`、`anchor → deck id 66`、
`weights_real version=3 覆盖 185/185 ESS 28.9`、`anchor 权重 2.11%（镜像对局率 2.1%）`、
**父权重 `{initname}` 必须是 18,205,408 B**、两个 bank、仓库里有没有 v2 座位补丁。

## 预期与硬门槛

- 耗时：2080 Ti / AG=512 ≈ **8 分钟**（400k 步，~930 steps/s）；显存 ≈17 GB / 22.5 GB
- 日志里必须出现三行，缺一行这支作废：
  1. `anchor decks resolved: {{'dunsparce_abra__db04': 66, ...}}`
  2. `deck weights -> v3, 185/185 active, ESS 28.9`
  3. `tl0:≈1.00/to0:≈0.02` ← **证据：学习者确实 100% 在打 db04，对手抽到 db04 ≈2.1%**
- 退化信号（这几条出现就别指望涨）：`lgw → 1.00`、`d0 → 0.95+`、`ep_len` 大幅下降

## 交回来

```
{out}/{run}.bin
<repo>/logs/{run}.log
```

我用**与 G5 完全相同的冻结评估**打分（cand@{deck} vs port@{deck}，1600 场配对，种子基准 900000），
判据不变：**CI 下界 > 0.50**。若 G6 与 G5 都通过，用同一批种子做**配对差**选更稳的那个去提交
（注意前期教训：同源候选的配对差 CI 很难压窄，别用配对差硬分胜负，看不重叠再选）。
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
    ap.add_argument("--run", required=True, help="运行名，如 g6_db04")
    ap.add_argument("--parent", default="", help="RUNBOOK 里显示的父代说明")
    ap.add_argument("--out", default=None, help="远端输出目录（默认 /root/<run>）")
    ap.add_argument("--tag", default=None, help="包名（默认 {run}_kit）")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "dist"))
    a = ap.parse_args()

    init = os.path.abspath(a.init)
    if not os.path.exists(init):
        raise SystemExit(f"✗ 父权重不存在：{init}")
    if os.path.getsize(init) != 18205408:
        raise SystemExit(f"✗ 父权重大小 {os.path.getsize(init)} ≠ 18,205,408（不是 native blob）")
    initname = os.path.basename(init)
    tag = a.tag or f"{a.run}_kit"
    out = a.out or f"/root/{a.run.split('_')[0]}"
    stage = os.path.join(a.outdir, tag)
    shutil.rmtree(stage, ignore_errors=True)
    for d in ("tools", "decks/pool_field", "banks", "init", "native"):
        os.makedirs(os.path.join(stage, d), exist_ok=True)

    shutil.copyfile(os.path.join(ROOT, "tools", "db04_line.sh"),
                    os.path.join(stage, "tools", "db04_line.sh"))
    os.chmod(os.path.join(stage, "tools", "db04_line.sh"), 0o755)
    shutil.copyfile(os.path.join(REPO, "decks", "pool_field", DECK + ".csv"),
                    os.path.join(stage, "decks", "pool_field", DECK + ".csv"))
    shutil.copyfile(os.path.join(REPO, "decks", "weights_real.txt"),
                    os.path.join(stage, "decks", "weights_real.txt"))
    shutil.copyfile(init, os.path.join(stage, "init", initname))
    for rel in (("native", "ptcg_tables_field.bin"),):
        shutil.copyfile(os.path.join(REPO, *rel), os.path.join(stage, *rel))

    tgz = os.path.join(ROOT, "dist", "league_g0003.tar.gz")
    if not os.path.exists(tgz):
        raise SystemExit("✗ 找不到 dist/league_g0003.tar.gz（两个冻结 bank 在里面）")
    with tarfile.open(tgz) as t:
        for i, member in enumerate(("league_g0003/bank_0.bin", "league_g0003/bank_1.bin")):
            dst = os.path.join(stage, "banks", f"bank_{i}.bin")
            with t.extractfile(member) as fsrc, open(dst, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)

    apply_txt = (APPLY_SH.replace("@DECK@", DECK).replace("@INITNAME@", initname)
                 .replace("@RUN@", a.run).replace("@OUT@", out))
    p = os.path.join(stage, "apply.sh")
    with open(p, "w", encoding="utf-8") as f:
        f.write(apply_txt)
    os.chmod(p, 0o755)
    with open(os.path.join(stage, "RUNBOOK.md"), "w", encoding="utf-8") as f:
        f.write(RUNBOOK.format(run=a.run, initname=initname, deck=DECK, out=out,
                               parent=a.parent or initname, tag=tag))

    man = []
    for rel in ("tools/db04_line.sh", f"decks/pool_field/{DECK}.csv", "decks/weights_real.txt",
                f"init/{initname}", "banks/bank_0.bin", "banks/bank_1.bin",
                "native/ptcg_tables_field.bin"):
        f = os.path.join(stage, rel)
        man.append(f"{sha256(f)}  {rel}  {os.path.getsize(f)}")
    with open(os.path.join(stage, "MANIFEST.txt"), "w", encoding="utf-8") as f:
        f.write("# sha256  path  bytes\n" + "\n".join(man) + "\n")

    tar = os.path.join(a.outdir, f"{tag}.tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(stage, arcname=tag)
    print(f"✅ {tar}  {os.path.getsize(tar) / 1048576:.0f} MB  sha {sha256(tar)[:16]}")
    print(f"   父权重 {initname}  →  run={a.run}  远端输出 {out}/{a.run}.bin")
    for line in man:
        print("   " + "  ".join(line.split()[:2]))


if __name__ == "__main__":
    main()
