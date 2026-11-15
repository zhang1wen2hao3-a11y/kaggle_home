#!/usr/bin/env bash
# **一台机器"能不能开跑"的一键体检 + 自愈**（跑评测前必跑）
#
#   bash tools/preflight_eval.sh root@HOST [端口]
#
# 检查并自动修复（都是 2026-11-15 真实踩过的坑）：
#   ① numpy/torch 缺失            → 用清华源 + 阿里云 wheel 装（比官方源快 30 倍）
#   ② 没有评估包 / 缺脚本          → 推 dist/eval_cpu_kit.tar.gz 并 setup（**不删已有目录**）
#   ③ 驾驶员 agent 目录没铺        → 按包里的 agents.tsv 逐个补（缺 models_drivers/*.pt 会报出来）
#   ④ 队列里某行的 agent 不存在    → 直接报错退出（不让你跑出一个"静默跑空"的面板）
#   ⑤ runner 起重复了              → 只留一个（重复 runner 会把 32 核抢成 57 负载）
#   ⑥ meta_eval 的 workers 是 auto → 固定成 24（留余量）
#
# 通过则打印 "PREFLIGHT_OK"，可以放心起队列。
set -uo pipefail
cd "$(dirname "$0")/.."
HOST="${1:?用法: bash tools/preflight_eval.sh root@HOST [端口]}"
PORT="${2:-22}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes -p $PORT $HOST"
R=/root/eval_cpu_kit
TAR=dist/eval_cpu_kit.tar.gz
WLF=logs/wishlist.tsv
say() { echo "  $*"; }
die() { echo "✗ $*"; exit 1; }

say "① 连通性"
$SSH "echo alive > /dev/null" || die "连不上 $HOST:$PORT"

say "② 依赖"
st=$($SSH "python3 -c 'import torch,numpy;print(\"DEPS\")' 2>/dev/null")
if [ "$st" != "DEPS" ]; then
  say "装 numpy/torch（清华源 + 阿里云 wheel）…"
  $SSH "pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy filelock fsspec typing-extensions sympy networkx jinja2 markupsafe mpmath > /root/pip.log 2>&1 &&
        pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple 'https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.14.1%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl' >> /root/pip.log 2>&1" \
    || die "装依赖失败（看 /root/pip.log）"
  $SSH "python3 -c 'import torch'" || die "装完仍无 torch"
fi

say "③ 评估包"
# ★ 用 sha 比对而不是"目录在不在"：包更新（新增驾驶员/权重）时必须重推，否则新 agent 永远补不上
LOCAL_SHA=$(sha256sum "$TAR" 2>/dev/null | cut -c1-16)
REMOTE_SHA=$($SSH "test -f $R/.kit_sha && cat $R/.kit_sha || echo none" 2>/dev/null)
if [ "$LOCAL_SHA" != "$REMOTE_SHA" ]; then
  [ -f "$TAR" ] || die "缺 $TAR（先 bash tools/pack_eval_cpu.sh）"
  say "推包 $(du -h "$TAR" | cut -f1) 并解包"
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" "$TAR" "$HOST:/root/" || die "推包失败"
  $SSH "cd /root && tar xzf $(basename $TAR) && cd eval_cpu_kit && bash setup.sh > /root/setup.log 2>&1 && echo $LOCAL_SHA > .kit_sha" \
    || die "setup 失败（看 /root/setup.log）"
fi

say "④ 驾驶员 agent（按 agents.tsv 补）"
$SSH "cd $R && if [ -f agents.tsv ]; then
        while IFS=\$'\t' read -r ag nm deck; do
          [ -n \"\$ag\" ] || continue
          [ -d build/agents/\$ag ] && continue
          w=models_drivers/\$ag.pt; [ -f \"\$w\" ] || { echo \"MISS_W \$ag\"; continue; }
          d=\$(ls decks/pool_field/*__\$deck.csv 2>/dev/null | head -1)
          [ -n \"\$d\" ] || { echo \"MISS_D \$ag\"; continue; }
          bash tools/make_agent_dir.sh build/agents/\$ag \"\$w\" \"\$d\" >/dev/null 2>&1 && echo \"ADD \$ag\"
        done < agents.tsv
      fi" | sed 's/^/    /'

say "⑤ 队列校验（每行的 agent 必须在机器上存在）"
missing=$(ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p "$PORT" "$HOST" \
  "cd $R && cut -f1 logs/wishlist.tsv | sort -u | while read a; do [ -d build/agents/\$a ] || echo \$a; done" 2>/dev/null)
if [ -n "$missing" ]; then
  echo "✗ 这些 agent 在机器上不存在："; echo "$missing" | sed 's/^/    /'
  exit 1
fi
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" "$WLF" "$HOST:$R/logs/" 2>/dev/null

say "⑥ runner 去重 + workers 固定"
$SSH "n=\$(pgrep -cf 'run_wishlis[t]'); if [ \"\$n\" -gt 1 ]; then pkill -f 'run_wishlis[t]'; sleep 2; pkill -f 'eval_pai[r]'; echo '   清掉重复 runner'; fi
      sed -i 's/--workers 0/--workers 24/' $R/tools/meta_eval.sh 2>/dev/null; echo '   workers=24'"

echo "PREFLIGHT_OK  $HOST:$PORT"
