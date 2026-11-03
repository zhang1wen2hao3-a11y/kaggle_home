#!/usr/bin/env bash
# 自动生成：训练联赛 代 3 的 PPO 分支（在 4090 训练机上跑）
#   bash /home/zhang/code/kaggle/the-pokemon-company/ptcg_league/pending/gen_0003_train.sh
set -uo pipefail
REPO="${PTCG_REPO:-$HOME/work/ptcg_train_kit/port_repo}"
cd "$REPO" || { echo "✗ 找不到训练仓库 $REPO"; exit 1; }
PY="${PYTHON:-python3}"
RUN="lg_g0003_7e39_port_plain"
RD="experiments/native-$RUN"
mkdir -p "$RD" logs
cp -f "/home/zhang/code/kaggle/the-pokemon-company/port_repo/native/pt_model_field.bin" "$RD/init.bin"
echo "init.bin = /home/zhang/code/kaggle/the-pokemon-company/port_repo/native/pt_model_field.bin  deck = /home/zhang/code/kaggle/the-pokemon-company/decks/pool_field/teal_mask_ogerpon_ex_applin__7e39.csv"
echo "banks = ['/home/zhang/code/kaggle/the-pokemon-company/bc_pool/agent_sue124_.bin', '/home/zhang/code/kaggle/the-pokemon-company/bc_pool/deck_2617_grookey.bin', '/home/zhang/code/kaggle/the-pokemon-company/bc_pool/deck_f5f8_.bin', '/home/zhang/code/kaggle/the-pokemon-company/ptcg_league/pools/evo/g0002_7e39_port_plain/final.bin']  pct = 0.1"

PYTHONPATH=data:.:native PTCG_TABLES=native/ptcg_tables_field.bin OMP_WAIT_POLICY=PASSIVE \
"$PY" native/train_native.py \
  --run-name "$RUN" \
  --timesteps 1392640 \
  --total-agents 512 \
  --minibatch 512 \
  --lr 0.0002 \
  --mix-self 1.0 \
  --league-ckpts "['/home/zhang/code/kaggle/the-pokemon-company/bc_pool/agent_sue124_.bin', '/home/zhang/code/kaggle/the-pokemon-company/bc_pool/deck_2617_grookey.bin', '/home/zhang/code/kaggle/the-pokemon-company/bc_pool/deck_f5f8_.bin', '/home/zhang/code/kaggle/the-pokemon-company/ptcg_league/pools/evo/g0002_7e39_port_plain/final.bin']" \
  --league-pct 0.1 \
  --deck-pool decks/pool_field \
  --deck-weights-file decks/weights_real.txt \
  --anchor-decks "$(basename /home/zhang/code/kaggle/the-pokemon-company/decks/pool_field/teal_mask_ogerpon_ex_applin__7e39.csv .csv)" \
  --p-anchor-learner 1.0 \
  --p-anchor-self-opp 0.0 \
  --p-anchor-league-opp 0.0 \
  --checkpoint-interval 25 \
  --seed 20284773 \
  2>&1 | tee "logs/$RUN.log"

echo
echo "训练结束。把 final.bin 传回本机："
echo "  scp $RD/final.bin <本机>:/home/zhang/code/kaggle/the-pokemon-company/ptcg_league/pools/evo/g0003_7e39_port_plain/final.bin"
echo "然后在本机跑：./tools/py ptcg_league/run_league.py next --resume"
