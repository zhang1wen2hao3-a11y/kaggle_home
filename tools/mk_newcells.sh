#!/usr/bin/env bash
# 铺"真实口径新增 7 格"的对手 agent（端口权重开那副牌）；幂等
cd /root/eval_cpu_kit || exit 1
for pair in "07be:dreepy_drakloak__07be" "3f45:abra_kadabra__3f45" "77a5:mega_lucario_ex_solrock__77a5" \
            "ea39:dreepy_drakloak__ea39" "0382:mega_kangaskhan_ex_dwebble__0382" \
            "91808263:unseen__91808263" "9bc9d655:unseen__9bc9d655"; do
  hex=${pair%%:*}; deck=${pair##*:}
  [ -f "decks/pool_field/$deck.csv" ] || { echo "✗ 缺 $deck.csv"; continue; }
  [ -f "build/agents/port_on_$hex/model.pt" ] && continue
  bash tools/make_agent_dir.sh "build/agents/port_on_$hex" build/port_plain/model.pt "decks/pool_field/$deck.csv" >/dev/null 2>&1
done
echo "new-agents=$(ls -d build/agents/port_on_* 2>/dev/null | wc -l)/7"
