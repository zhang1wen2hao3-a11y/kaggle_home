#!/usr/bin/env python3
"""打包 **G5 · db04 卡组线** 训练包（GPU 机器上跑）。

为什么这个包很小（~56 MB）：GPU 机器上已经有打 v2 补丁的训练仓库、185 副卡组池和表格 blob。
包里只放**这次实验真正会变的东西** + 一份能在开训前把坑全踩出来的预检：
    1. `tools/g5_db04.sh`      —— 运行器（预检 → 训练 → 硬门槛检查）
    2. `decks/...db04.csv`     —— 学习者要用的那副牌（**必须**在池里第 66 位，池子顺序 = deck id）
    3. `decks/weights_real.txt`—— 对手卡组分布（与 G4-B 完全相同的那份）
    4. `banks/bank_{0,1}.bin`  —— G4-B 用过的那两个冻结 bank（受控：配方不变）
    5. `init/pt_model_field.bin` —— 父权重（= 端口 field 模型，G4 用的同一个 init）
    6. `native/ptcg_tables_field.bin` —— 用来在对方机器上核对"池 == blob"，1.9 MB
    7. `apply.sh` / `RUNBOOK_g5.md` / `MANIFEST.txt`

用法：
    ./tools/py tools/make_g5_kit.py
"""
import argparse
import hashlib
import os
import shutil
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "port_repo")

DECK = "dunsparce_abra__db04"
DECK_CSV = os.path.join(REPO, "decks", "pool_field", DECK + ".csv")

APPLY_SH = r"""#!/usr/bin/env bash
# 自动生成（tools/make_g5_kit.py）：把 G5 需要的文件铺进训练仓库，并跑一次**不花 GPU** 的预检。
#   PTCG_REPO=/root/ptcg_train_kit/port_repo bash apply.sh
#   PTCG_REPO=... bash apply.sh --run      # 预检通过后直接开训
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
PY="${PYTHON:-python3}"
REPO="${PTCG_REPO:-$HOME/work/ptcg_train_kit/port_repo}"
[ -f "$REPO/native/train_native.py" ] || { echo "✗ 找不到训练仓库：$REPO"; echo "  用 PTCG_REPO=/root/ptcg_train_kit/port_repo bash apply.sh"; exit 1; }
echo "repo = $REPO"

# ① 完整性
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

# ③ 铺文件（卡组进池子，位置决定 deck id，所以只补不改）
mkdir -p "$REPO/decks/pool_field" "$REPO/tools" "$HERE/banks_used"
cp -f "decks/pool_field/@DECK@.csv" "$REPO/decks/pool_field/@DECK@.csv"
cp -f tools/g5_db04.sh "$REPO/tools/g5_db04.sh"; chmod +x "$REPO/tools/g5_db04.sh"
cp -f banks/bank_0.bin banks/bank_1.bin "$HERE/banks_used/"
echo "✓ 已铺：decks/pool_field/@DECK@.csv, tools/g5_db04.sh, banks_used/（两个 bank）"

# ④ 预检（不花 GPU）
cd "$REPO"
PTCG_REPO="$REPO" PYTHON="$PY" \
BANKS="$HERE/banks_used/bank_0.bin,$HERE/banks_used/bank_1.bin" \
  bash tools/g5_db04.sh --check
rc=$?
[ $rc -eq 0 ] || { echo; echo "✗ 预检没过 —— 别开训，把上面整段发回来"; exit $rc; }

if [ "${1:-}" = "--run" ]; then
  echo; echo "===== 预检通过，开训 ====="
  PTCG_REPO="$REPO" PYTHON="$PY" \
  BANKS="$HERE/banks_used/bank_0.bin,$HERE/banks_used/bank_1.bin" \
    bash tools/g5_db04.sh
else
  echo
  echo "预检通过。开训命令："
  echo "  cd $REPO"
  echo "  BANKS=$HERE/banks_used/bank_0.bin,$HERE/banks_used/bank_1.bin PYTHON=$PY bash tools/g5_db04.sh"
fi
"""

RUNBOOK = """G5 · db04 卡组线（让 PPO 直接在真实野区最强的那副牌上练）
=================================================================
一句话：**只换牌，不换配方**。G4-B（7e39 + 2 冻结 bank + 真实野区）已经拿到
0.5337 [0.5093, 0.5581]，是本项目**第一次**显著强于端口的候选；
而真实天梯里 db04 的胜率 0.642 vs 7e39 0.568（卡组线的收益是驾驶线的两倍多）。
所以这一支把学习者换成 db04，其余参数与 G4-B 逐项相同（含 SEED=424242）。

## 与 G4-B 的对照（受控）

| 项 | G4-B | G5 |
|---|---|---|
| 学习者卡组 | 7e39 | **db04** |
| 对手卡组分布 | weights_real（185 副） | 同 |
| 冻结 bank | league_g0003/bank_0,1 | **同两个**（包里自带） |
| p_anchor-learner / league-pct | 1.0 / 0.08 | 同 |
| T / AG / MB / LR / SEED / THREADS | 400k / 512 / 512 / 2e-4 / 424242 / 8 | 同 |

## 跑法

```bash
tar xzf g5_db04_kit.tar.gz && cd g5_db04_kit
PTCG_REPO=/root/ptcg_train_kit/port_repo bash apply.sh          # 只预检，不花 GPU
PTCG_REPO=/root/ptcg_train_kit/port_repo bash apply.sh --run    # 预检通过就直接开训
```

预检会打印/校验（这几项如果静默失败，一次训练就白花）：
- `decks/pool_field` 的 185 副与 `native/ptcg_tables_field.bin` **逐行一致** ⇒ deck id 不会错位
- `dunsparce_abra__db04` 在池里的 **deck id（应为 66）**
- `weights_real.txt` version≥3、覆盖 185 副、**anchor 权重非零**（为 0 会被训练器 REJECTED）
- 镜像对局率（学习者也打 db04 ⇒ 对手抽到同牌的比例 ≈ 2.1%）
- 两个 bank 文件、`pt_model_field.bin` 必须是 18,205,408 B（0 字节坏包 ⇒ init.bin 0 字节）
- 训练仓库里有 v2 座位计数补丁（`binding.c` 里有 `n_track`，没有就 `--track-decks` 静默失效）

## 预期

- 耗时：2080 Ti / AG=512 ≈ **6–8 分钟**（400k 步，~1010 steps/s）
- 显存：≈17 GB / 22.5 GB，占卡前先看 `nvidia-smi` 别留残留进程
- 日志里必须出现三行"硬门槛"，缺一行这支就作废：
  1. `anchor decks resolved: {'dunsparce_abra__db04': 66}`
  2. `deck weights -> ... ESS ...`（权重真的被应用了）
  3. `tl0:≈1.00/to0:很小 tl1:... to1:...`（**证据：学习者确实在打 db04**）
     `tl0` = 学习者座位抽到 tracked deck #0（=db04）的比例，应 ≈1.0

## 交回来

```
/root/g5/g5_db04.bin
logs/g5_db04.log
```

我用**与 G4-B 完全相同的冻结评估协议**打分（候选 @db04 vs port @db04，1600 场配对 + 固定评测集），
预注册判据不变：**confirm 的 CI 下界 > 0.50** 才算"驾驶能力有提升"；
同时用真实天梯 db04 的 0.642 作为卡组线的独立证据。

拉回命令（在本机执行，端口换成你的）：
    scp -P <端口> root@<主机>:/root/g5/g5_db04.bin .
    scp -P <端口> root@<主机>:$REPO/logs/g5_db04.log .
"""


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="g5_db04_kit")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--banks", default=None,
                    help="逗号分隔的两个 bank（默认从 dist/league_g0003.tar.gz 里取 bank_0,1）")
    a = ap.parse_args()

    stage = os.path.join(a.outdir, a.tag)
    shutil.rmtree(stage, ignore_errors=True)
    for d in ("tools", "decks/pool_field", "banks", "init", "native"):
        os.makedirs(os.path.join(stage, d), exist_ok=True)

    # ① 运行器
    shutil.copyfile(os.path.join(ROOT, "tools", "g5_db04.sh"),
                    os.path.join(stage, "tools", "g5_db04.sh"))
    os.chmod(os.path.join(stage, "tools", "g5_db04.sh"), 0o755)
    # ② 卡组与权重
    shutil.copyfile(DECK_CSV, os.path.join(stage, "decks", "pool_field", DECK + ".csv"))
    shutil.copyfile(os.path.join(REPO, "decks", "weights_real.txt"),
                    os.path.join(stage, "decks", "weights_real.txt"))
    # ③ bank（受控：与 G4-B 完全相同的两个冻结快照）
    if a.banks:
        for src in [x.strip() for x in a.banks.split(",") if x.strip()]:
            shutil.copyfile(src, os.path.join(stage, "banks", os.path.basename(src)))
    else:
        tgz = os.path.join(ROOT, "dist", "league_g0003.tar.gz")
        if not os.path.exists(tgz):
            raise SystemExit("✗ 找不到 dist/league_g0003.tar.gz（用 --banks 指定两个 .bin）")
        with tarfile.open(tgz) as t:
            for i, member in enumerate(("league_g0003/bank_0.bin", "league_g0003/bank_1.bin")):
                dst = os.path.join(stage, "banks", f"bank_{i}.bin")
                with t.extractfile(member) as fsrc, open(dst, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
    # ④ 父权重与表格 blob
    for rel, dst in ((("native", "pt_model_field.bin"), ("init", "pt_model_field.bin")),
                     (("native", "ptcg_tables_field.bin"), ("native", "ptcg_tables_field.bin"))):
        src = os.path.join(REPO, *rel)
        if not os.path.exists(src):
            raise SystemExit(f"✗ 缺 {src}")
        shutil.copyfile(src, os.path.join(stage, *dst))

    # ⑤ 脚本与文档
    apply_path = os.path.join(stage, "apply.sh")
    with open(apply_path, "w", encoding="utf-8") as f:
        f.write(APPLY_SH.replace("@DECK@", DECK))
    os.chmod(apply_path, 0o755)
    with open(os.path.join(stage, "RUNBOOK_g5.md"), "w", encoding="utf-8") as f:
        f.write(RUNBOOK)

    # ⑥ MANIFEST
    man = []
    for rel in ("tools/g5_db04.sh", f"decks/pool_field/{DECK}.csv", "decks/weights_real.txt",
                "banks/bank_0.bin", "banks/bank_1.bin",
                "init/pt_model_field.bin", "native/ptcg_tables_field.bin"):
        p = os.path.join(stage, rel)
        man.append(f"{sha256(p)}  {rel}  {os.path.getsize(p)}")
    with open(os.path.join(stage, "MANIFEST.txt"), "w", encoding="utf-8") as f:
        f.write("# sha256  path  bytes\n" + "\n".join(man) + "\n")

    tar = os.path.join(a.outdir, f"{a.tag}.tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(stage, arcname=a.tag)
    print(f"✅ {tar}  {os.path.getsize(tar)/1048576:.0f} MB  sha {sha256(tar)[:16]}")
    for line in man:
        print("   " + "  ".join(line.split()[:2]))


if __name__ == "__main__":
    main()
