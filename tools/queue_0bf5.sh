#!/usr/bin/env bash
cd /home/zhang/code/kaggle/the-pokemon-company || exit 1
CELLS="ogerpon_7e39 dunsparce_0bf5 grookey_2617 out_91808263 dreepy_3eaa champ_e1d7 out_07be out_3f45 out_77a5 champ_0904 abra_f5f8 out_ea39 out_9bc9d655 out_0382 dunsparce_9d6e abra_e8ee champ_26c6"
H=root@113.44.139.218
echo "== ① 基线 opp_dunsparce_0bf5 剩余格"
./tools/py tools/dispatch_panel.py --cands opp_dunsparce_0bf5 --cells $CELLS --seeds 950000 \
  --matches 150 --workers 24 --interval 20 --hosts $H 2>&1 | tail -2
echo "== ② cand0bf5S 17 格（门禁）"
./tools/py tools/dispatch_panel.py --cands cand0bf5S --cells $CELLS --seeds 950000 \
  --matches 150 --workers 24 --interval 20 --hosts $H 2>&1 | tail -2
echo "== ③ cand0bf5S 关键格第二种子"
./tools/py tools/dispatch_panel.py --cands cand0bf5S --cells dunsparce_0bf5 ogerpon_7e39 out_0382 dreepy_3eaa \
  --seeds 950001 --matches 150 --workers 24 --interval 20 --hosts $H 2>&1 | tail -2
echo QUEUE_0BF5_DONE
