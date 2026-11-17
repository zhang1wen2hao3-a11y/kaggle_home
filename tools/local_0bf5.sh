#!/usr/bin/env bash
cd /home/zhang/code/kaggle/the-pokemon-company || exit 1
CELLS="ogerpon_7e39 dunsparce_0bf5 grookey_2617 out_91808263 dreepy_3eaa champ_e1d7 out_07be out_3f45 out_77a5 champ_0904 abra_f5f8 out_ea39 out_9bc9d655 out_0382 dunsparce_9d6e abra_e8ee champ_26c6"
echo "== ① 基线 opp_dunsparce_0bf5（本地补剩余）"
WORKERS=10 bash tools/eval_candidate_local.sh opp_dunsparce_0bf5 "$CELLS" 2>&1 | tail -3
echo "== ② cand0bf5S 17 格"
WORKERS=10 bash tools/eval_candidate_local.sh cand0bf5S "$CELLS" 2>&1 | tail -3
echo "== ③ cand0bf5S 关键格第二种子"
WORKERS=10 bash tools/eval_candidate_local.sh cand0bf5S "" "dunsparce_0bf5 ogerpon_7e39 out_0382 dreepy_3eaa" 2>&1 | tail -2
echo LOCAL_0BF5_DONE
