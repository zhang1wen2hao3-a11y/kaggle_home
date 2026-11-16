#!/usr/bin/env bash
# **dreepy/mega 专攻线收尾**：等训练出模型 → 校验并拉回 → 转 .pt → **关 GPU** →
#   推三台 CPU → 用中央调度器跑 17 格真实口径面板 + 五格第二种子 → 聚合
#
#   bash tools/finish_dreepy_run.sh
# 判据见 run_f5f8_dreepy.sh 头（五格各 ≥ B0+5pp；老格面板降 ≤2pp）。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# ★ ssh 的端口是 **小写 -p**；大写 -P 在新版 OpenSSH 里是 "-P tag"（连接输出打标签），
#   写成 -P 会把端口参数吃掉、主机名解析成垃圾地址（2026-11-16 踩到，白等 4 小时）。
H="-p 21760 -o ConnectTimeout=15 root@connect.weste.seetacloud.com"
R=/root/ptcg_train_kit/port_repo
RUN=candf5f8G_db04
FINAL=/root/f5f8_dreepy4m/$RUN.bin
NAME=candf5f8D
LOG=logs/finish_dreepy.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "等训练落地（每 60s 探一次；SKIP_WAIT=1 可跳过）"
if [ "${SKIP_WAIT:-0}" = 1 ]; then
  say "跳过等待（假定模型已就绪）"
else
for i in $(seq 1 180); do
  sz=$(timeout 30 ssh -o BatchMode=yes $H "stat -c %s $FINAL 2>/dev/null" 2>/dev/null)
  # ★ 别写 `pgrep -c ... || echo 0`：pgrep 无匹配时退出码 1，`||` 会把输出变成 "0\n0"，
  #   和 "0" 比较永远不等 ⇒ 明明训练跑完了却一直等（2026-11-16 白等了 4 小时，GPU 空转）。
  run=$(timeout 20 ssh -o BatchMode=yes $H 'pgrep -fc "train_[n]ative" 2>/dev/null || true' 2>/dev/null | head -1 | tr -d "[:space:]")
  [ -n "${run:-}" ] || run=0
  say "  探第 $i 次：final=${sz:-0}B 训练进程=${run}"
  if [ "${sz:-0}" = 18205408 ] && [ "$run" = 0 ]; then break; fi
  sleep 60
done
fi
[ -n "${sz:-}" ] || sz=$(timeout 30 ssh -o BatchMode=yes $H "stat -c %s $FINAL 2>/dev/null" 2>/dev/null)
[ "${sz:-0}" = 18205408 ] || { say "✗ 等不到 $FINAL（当前=${sz:-0}B）"; exit 1; }
say "训练结束，开始拉产物并校验"

mkdir -p models/ppo/gpu_$RUN
rsync -az -e "ssh -o BatchMode=yes -p 21760" "root@connect.weste.seetacloud.com:/root/f5f8_dreepy4m/" models/ppo/gpu_$RUN/
rsync -az -e "ssh -o BatchMode=yes -p 21760" "root@connect.weste.seetacloud.com:$R/experiments/native-$RUN/" models/ppo/gpu_$RUN/snapshots/
rsync -az -e "ssh -o BatchMode=yes -p 21760" "root@connect.weste.seetacloud.com:$R/logs/$RUN.log" models/ppo/gpu_$RUN/train.log 2>/dev/null || true

bad=0
for f in $(cd models/ppo/gpu_$RUN/snapshots && ls *.bin 2>/dev/null); do
  l=$(md5sum "models/ppo/gpu_$RUN/snapshots/$f" | cut -d' ' -f1)
  r=$(timeout 30 ssh -o BatchMode=yes $H "md5sum $R/experiments/native-$RUN/$f 2>/dev/null | cut -d' ' -f1")
  [ "$l" = "$r" ] || { say "  ✗ $f md5 不一致"; bad=$((bad+1)); }
done
lf=$(md5sum models/ppo/gpu_$RUN/$RUN.bin | cut -d' ' -f1)
rf=$(timeout 30 ssh -o BatchMode=yes $H "md5sum $FINAL | cut -d' ' -f1")
[ "$lf" = "$rf" ] || { say "✗ final md5 不一致 local=$lf remote=$rf"; exit 1; }
say "✓ final md5 $lf 与快照校验通过（$bad 个不一致）"

cp models/ppo/gpu_$RUN/$RUN.bin models/ppo/$NAME.bin
PATH=/root/miniconda3/bin:$PATH PYTHONPATH=build/port_plain:. ./tools/py tools/native_weights.py import models/ppo/$NAME.bin models/ppo/$NAME.pt 2>&1 | tail -1
say "✓ 转出 models/ppo/$NAME.pt"

if ! grep -q "\"$NAME\"" tools/driver_registry.py; then
  python3 - <<PY
p='tools/driver_registry.py'; s=open(p).read()
add='''    "$NAME": dict(agent="$NAME", weights="models/ppo/$NAME.pt",
                   remote=None, deck="f5f8", fidelity=None,
                   note="f5f8 真实口径补短板版（2026-11-16/17，INIT=candf5f8G，8 bank 里 5 个绑 dreepy/mega 族的端口权重，"
                        "4M 步于 4080S）。真实口径面板基线 57.5%，五格基线 ea39 25.3/07be 39.3/3eaa 39.3/0382 24.7/77a5 50.0"),
'''
anchor='    "candf5f8G": dict('
assert anchor in s
open(p,'w').write(s.replace(anchor, add+anchor))
PY
  say "✓ 已登记 $NAME"
fi

say "═══ 关 GPU（产物已校验）═══"
timeout 30 ssh -o BatchMode=yes $H 'sync; nohup shutdown -h now > /dev/null 2>&1 & sleep 2; echo bye' 2>&1 | tail -1 | sed 's/^/  /' | tee -a "$LOG"

bash tools/make_agent_dir.sh build/agents/$NAME models/ppo/$NAME.pt decks/pool_field/abra_kadabra__f5f8.csv 2>&1 | tail -1 | tee -a "$LOG"

for h in 113.44.216.143 1.92.97.36 113.44.139.218; do
  scp -q models/ppo/$NAME.pt root@$h:/root/eval_cpu_kit/stage_$NAME.pt
  scp -q tools/driver_registry.py root@$h:/root/eval_cpu_kit/tools/
  timeout 120 ssh -o BatchMode=yes root@$h "cd /root/eval_cpu_kit && bash tools/make_agent_dir.sh build/agents/$NAME stage_$NAME.pt decks/pool_field/abra_kadabra__f5f8.csv 2>&1 | tail -1" | sed "s/^/  [$h] /"
done

CELLS="ogerpon_7e39 dunsparce_0bf5 grookey_2617 out_91808263 dreepy_3eaa champ_e1d7 out_07be out_3f45 out_77a5 champ_0904 abra_f5f8 out_ea39 out_9bc9d655 out_0382 dunsparce_9d6e abra_e8ee champ_26c6"
say "跑 17 格真实口径面板（seed 950000）"
./tools/py tools/dispatch_panel.py --cands $NAME --cells $CELLS --seeds 950000 \
   --matches 150 --workers 24 --interval 20 2>&1 | tee -a "$LOG" | tail -5
say "五格补第二种子（n→300）"
./tools/py tools/dispatch_panel.py --cands $NAME \
   --cells out_ea39 out_07be dreepy_3eaa out_0382 out_77a5 --seeds 950001 \
   --matches 150 --workers 24 --interval 20 2>&1 | tee -a "$LOG" | tail -3

for h in 113.44.216.143 1.92.97.36 113.44.139.218; do
  timeout 90 rsync -az -e "ssh -o BatchMode=yes" --include '*/' --include 'vs_*.json' --exclude '*' \
    "root@$h:/root/eval_cpu_kit/logs/meta/" logs/meta/ 2>/dev/null
done
./tools/py tools/real_panel_agg.py $NAME 2>&1 | tee -a "$LOG" | tail -25
bash tools/git_snapshot.sh "$NAME（dreepy/mega 补短板）真实口径面板结果" 2>&1 | tail -1 | tee -a "$LOG"
say "DONE"
