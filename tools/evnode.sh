#!/usr/bin/env bash
# 评测机的日常操作：推模型 / 跑对局 / 拉结果。一条命令一件小事。
#
# 为什么要有它：评估机是**临时实例**（32 核、无 GPU、只活 2 小时），
# 环境重建一次要 10 分钟。所以流程固定成「大件只传一次，之后只推权重」：
#   · 引擎/面板/评估器 → 打进 kit 传一次（tools/pack_eval_kit.sh）
#   · 之后每轮迭代只推 20 MB 的 model.pt（md5 校验）
#
# 用法：
#   bash tools/evnode.sh push h_26c6_U1 /tmp/h_26c6_U1.pt slowpoke_slowking__26c6
#   bash tools/evnode.sh run  ag_h_26c6_U1 "bc=build/agents/ag_bc_3eaa" 300 990001 /tmp/ev_U1.json
#   bash tools/evnode.sh ls
#
# 对比：本地 12 vCPU 上限 6 面板，300 场约 20 分钟；评估机 32 vCPU/16 面板，约 3 分钟。
set -uo pipefail
cd "$(dirname "$0")/.."
HOST=$(./tools/py -c "import json;print(f\"root@{json.load(open('ptcg_league/evalnode.json'))['host'].split('@')[-1]}\")")
ROOT=/root/ptcg_eval
# 临时实例重建会换主机指纹：先刷新，避免 "Host key verification failed"（密钥本身没问题）
IP=$(echo "$HOST" | sed "s/.*@//")
ssh-keygen -f "$HOME/.ssh/known_hosts" -R "$IP" >/dev/null 2>&1 || true
ssh-keyscan -T 8 "$IP" >> "$HOME/.ssh/known_hosts" 2>/dev/null || true
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST")

cmd=${1:-}; shift || true
case "$cmd" in
  push)
    name=$1; pt=$2; deck=$3
    [ -f "$pt" ] || { echo "✗ 没有 $pt"; exit 1; }
    local_md5=$(md5sum "$pt" | awk '{print $1}')
    scp -q "$pt" "$HOST:$ROOT/models/$name.pt" || { echo "✗ scp 失败"; exit 1; }
    far=$("${SSH[@]}" "md5sum $ROOT/models/$name.pt | awk '{print \$1}'")
    if [ "$local_md5" != "$far" ]; then
      echo "✗ md5 不一致（$local_md5 vs $far）：传输被截断，重传"
      exit 1
    fi
    "${SSH[@]}" "cd $ROOT && bash tools/make_agent_dir.sh build/agents/ag_$name models/$name.pt decks/pool_field/$deck.csv" \
      && echo "✅ 已就位 build/agents/ag_$name ($deck)  md5 $local_md5"
    ;;
  run)
    cand=$1; opp=$2; matches=$3; seed=$4; out=$5
    case "$cand" in /*) ;; *) cand="build/agents/$cand";; esac   # eval_pair 只认目录路径
    tag=$(basename "$out" .json)
    tag=${tag#ev_}                                               # 避免 ev_ev_ 前缀
    # 评估机的 sshd 会在 ~10 分钟处断连 ⇒ 用 nohup 起 + 轮询，ssh 只做短交互
    "${SSH[@]}" "cd $ROOT && rm -rf /root/evlog_$tag && nohup python3 ptcg_league/eval_pair.py \
        --cand $cand --opp '$opp' --matches $matches --seed-base $seed \
        --out /root/$tag.json --logdir /root/log_$tag > /root/$tag.log 2>&1 & echo started" \
      | tail -1
    for _ in $(seq 1 120); do
      sleep 10
      if "${SSH[@]}" "test -f /root/$tag.json" 2>/dev/null; then
        scp -q "$HOST:/root/$tag.json" "$out" && echo "✅ $out"
        "${SSH[@]}" "tail -3 /root/$tag.log"
        exit 0
      fi
    done
    echo "⚠️ 20 分钟未见结果：${SSH[*]} 'tail -20 /root/$tag.log'"
    exit 1
    ;;
  ls)
    "${SSH[@]}" "ls -l $ROOT/models/ | tail -12; echo '--- agents ---'; ls $ROOT/build/agents/ | tr '\n' ' '"
    ;;
  *)
    sed -n '2,20p' "$0"; exit 1;;
esac
