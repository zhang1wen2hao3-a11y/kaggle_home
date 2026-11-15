#!/usr/bin/env bash
# 门禁跑完后**按固定流程收尾**：拉结果 → 刷表 → 汇总 G1/G2/G3/G4/G5 → 提交 git
# （候选是否登记/是否可提交由人工看数后决定，本脚本只负责把证据和表整理好）
set -uo pipefail
cd /home/zhang/code/kaggle/the-pokemon-company || exit 1
CPU=root@113.44.216.143
for i in $(seq 1 40); do
  if ssh -o BatchMode=yes -o ConnectTimeout=10 $CPU 'grep -q GATE_F5F8G_DONE /root/gate_f5f8G.log 2>/dev/null' 2>/dev/null; then break; fi
  sleep 120
done
echo "[$(date +%H:%M)] 门禁结束（或超时），开始收尾"
bash tools/pull_results.sh $CPU 2>&1 | tail -2
bash tools/refresh_tables.sh --quiet 2>&1 | tail -2
python3 - <<'PY'
import json, glob, re, os
def pooled(prefix):
    ps = sorted(glob.glob(f"data/opp_matrix/{prefix}_s95*.json"))
    tot = n = 0
    for p in ps:
        d = json.load(open(p))["overall"]; tot += d["score"]*d["n"]; n += d["n"]
    return (tot/n*100 if n else None), n, len(ps)
print(f"{'格子':22s}{'新候选':>10s}{'n':>6s}{'B0':>8s}{'Δpp':>8s}")
base = {"f5f8G_vs_c_grookey": 46.0, "f5f8G_vs_ph3_2617v1": 42.5,
        "f5f8G_vs_opp_3eaa": 33.5, "f5f8G_vs_shbc": 90.0}
for k, b0 in base.items():
    wr, n, f = pooled(k)
    print(f"{k:22s}{(f'{wr:.1f}%' if wr else 'NA'):>10s}{n:6d}{b0:7.1f}%{(f'{wr-b0:+.1f}' if wr else 'NA'):>8s}")
# 方案A 面板
ps = sorted(glob.glob("logs/meta/candf5f8G_A_s95*/summary.json"))
tot = n = 0
for p in ps:
    d = json.load(open(p)); tot += d["meta_w"]*d["n"]; n += d["n"]
print(f"\n方案A 面板 candf5f8G: {(f'{tot/n*100:.1f}%' if n else 'NA')} (n={n}, {len(ps)} 种子)  B0=61.3%")
PY
bash tools/git_snapshot.sh "candf5f8G（f5f8 补 2617/3eaa 短板，4M 步）门禁结果自动收尾 $(date +%m-%d\ %H:%M)"
echo WRAP_DONE
