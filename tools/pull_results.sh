#!/usr/bin/env bash
# **安全提前拉取**：机器随时可能被释放，所以结果不能等全部跑完才收。
#
# 为什么不直接用 collect_eval.sh 的 `rsync --ignore-existing`：
#   正在写的文件会被拷成半截 JSON，而 `--ignore-existing` 之后**永远不会再补**，
#   等于把结果静默弄丢（比没拉更糟）。本脚本先把远端目录拷到临时目录，
#   **逐文件校验可解析**，通过才落到本地（已有同名但样本更小的会被替换）。
#
#   bash tools/pull_results.sh            # 三台机器
#   bash tools/pull_results.sh root@A …   # 指定机器
set -uo pipefail
cd "$(dirname "$0")/.."
HOSTS=("$@")
[ ${#HOSTS[@]} -eq 0 ] && HOSTS=(root@113.44.216.143 root@113.44.139.218 root@1.92.97.36)
R=/root/eval_cpu_kit
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
for h in "${HOSTS[@]}"; do
  echo "== 拉 $h"
  d="$STAGE/$h"
  mkdir -p "$d"
  rsync -az -e "ssh -o StrictHostKeyChecking=no" "$h:$R/data/opp_matrix/" "$d/opp_matrix/" 2>/dev/null
  rsync -az -e "ssh -o StrictHostKeyChecking=no" "$h:$R/logs/meta/" "$d/meta/" 2>/dev/null
  ./tools/py - "$d" <<'PY'
import json, os, shutil, sys
stage = sys.argv[1]
root = os.getcwd()
ok = bad = kept = 0
for kind, dst in (("opp_matrix", "data/opp_matrix"), ("meta", "logs/meta")):
    src = os.path.join(stage, kind)
    for dirpath, _, files in os.walk(src):
        for f in files:
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, src)
            if f.endswith(".json"):
                try:
                    d = json.load(open(p))
                except Exception:
                    bad += 1          # 正在写 / 半截文件 ⇒ 本轮跳过，下轮再拉
                    continue
                n_new = int(((d.get("overall") or {}).get("n") or 0)) if kind == "opp_matrix" else 0
                out = os.path.join(root, dst, rel)
                if kind == "opp_matrix" and os.path.exists(out):
                    try:
                        n_old = int((json.load(open(out)).get("overall") or {}).get("n") or 0)
                    except Exception:
                        n_old = -1
                    if n_old >= n_new:
                        kept += 1
                        continue
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copy2(p, out)
                ok += 1
            elif kind == "meta":
                out = os.path.join(root, dst, rel)
                if os.path.exists(out):
                    continue
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copy2(p, out)
                ok += 1
print(f"   落地 {ok} 个（已更新/新增），跳过半截 {bad} 个，本地已有更全 {kept} 个")
PY
done
