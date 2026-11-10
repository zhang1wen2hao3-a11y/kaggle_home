#!/usr/bin/env bash
# 评估机是**临时实例**（32 核、无 GPU、约 2 小时寿命，关机后 /root 清空）。
# 这个脚本把它从裸机铺到能跑对局：约 5 分钟。
#
#   bash tools/evnode_setup.sh              # 只补齐缺的部分
#   bash tools/evnode_setup.sh --force-kit  # 强制重传 kit（kit 更新后）
#
# 铺完就能用 tools/evnode.sh run ... 评测。
set -uo pipefail
cd "$(dirname "$0")/.."
HOST=$(./tools/py -c "import json;print(json.load(open('ptcg_league/evalnode.json'))['host'])")
ROOT=/root/ptcg_eval
KIT=dist/ptcg_eval_kit.tar.gz
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST")
FORCE_KIT=0; [ "${1:-}" = "--force-kit" ] && FORCE_KIT=1

[ -f "$KIT" ] || { echo "✗ 缺 $KIT（先跑 bash tools/pack_eval_kit.sh）"; exit 1; }

echo "① 主机指纹（临时实例每次重建都会换指纹，不刷新会报 Host key verification failed）"
IP=$(echo "$HOST" | sed "s/.*@//")
ssh-keygen -f "$HOME/.ssh/known_hosts" -R "$IP" >/dev/null 2>&1 || true
ssh-keyscan -T 8 "$IP" >> "$HOME/.ssh/known_hosts" 2>/dev/null || true

echo "② 连通性"
"${SSH[@]}" 'echo OK; nproc' || { echo "✗ 连不上 $HOST"; exit 1; }

echo "③ torch（CPU 版；镜像会瞬时 DNS 失败，失败就重试一次）"
if ! "${SSH[@]}" 'python3 -c "import torch" 2>/dev/null'; then
  for i in 1 2 3; do
    "${SSH[@]}" "pip3 install --no-cache-dir --retries 5 --timeout 60 -q \
      --index-url https://download.pytorch.org/whl/cpu torch numpy" && break
    echo "   第 $i 次失败，等 10 s 重试"; sleep 10
  done
  "${SSH[@]}" 'python3 -c "import torch;print(\"torch\",torch.__version__)"' \
    || { echo "✗ torch 装不上：手动跑 pip3 install torch --index-url https://download.pytorch.org/whl/cpu"; exit 1; }
fi

echo "④ kit（$(( $(stat -c%s "$KIT") / 1048576 )) MB）"
if [ "$FORCE_KIT" = 1 ] || ! "${SSH[@]}" "test -d $ROOT/build/port_plain"; then
  scp -q "$KIT" "$HOST:/root/ptcg_eval_kit.tar.gz" || { echo "✗ scp 失败"; exit 1; }
  local_md5=$(md5sum "$KIT" | awk '{print $1}')
  far=$("${SSH[@]}" "md5sum /root/ptcg_eval_kit.tar.gz | awk '{print \$1}'")
  [ "$local_md5" = "$far" ] || { echo "✗ kit md5 不一致（$local_md5 vs $far）：这条链路会截断大文件，重传"; exit 1; }
  "${SSH[@]}" "cd /root && tar xzf ptcg_eval_kit.tar.gz" || exit 1
else
  echo "   kit 已在远端"
fi

echo "⑤ setup.sh 铺 agent 目录"
"${SSH[@]}" "cd $ROOT && bash setup.sh" 2>&1 | tail -16

echo
echo "✅ 评估机就绪。下一步："
echo "   bash tools/evnode.sh run ag_g6_db04 'f=build/agents/ogerpon_7e39' 300 990001 /tmp/ev.json"
