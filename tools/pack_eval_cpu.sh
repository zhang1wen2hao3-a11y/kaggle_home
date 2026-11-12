#!/usr/bin/env bash
# **打一份纯 CPU 评估包**（只装"算 a→b 胜率矩阵 / 12 格面板"需要的最小集合）。
#
#   bash tools/pack_eval_cpu.sh                 # → dist/eval_cpu_kit.tar.gz
#
# 和 tools/pack_eval_kit.sh 的区别：那个是给人评测面板用的完整包（含 panel 对手/候选），
# 这个只求"能在任意 Linux x86_64 + python3 + numpy + torch 上跑 eval_pair"。
#
# 包里不含任何编译产物以外的依赖：
#   · build/port_plain     28 MB（引擎 libcg.so 用 ctypes 加载 → 不绑 Python ABI，3.10/3.12 都能跑）
#   · 根目录 cg/ data/ EN_Card_Data.csv r2e_manifest.json（父进程 import cg.game 用）
#   · 12 个 agent 目录由 setup.sh 现场铺（共用同一个 model.pt，不重复塞 20 MB）
set -uo pipefail
cd "$(dirname "$0")/.."
STAGE="dist/eval_cpu_kit"
OUTTAR="dist/eval_cpu_kit.tar.gz"
rm -rf "$STAGE"; mkdir -p "$STAGE"/{build,decks,ptcg_league,tools,docs,logs/winrate}

echo "① 引擎与推理骨架 build/port_plain（$(du -sh build/port_plain | cut -f1)）"
cp -a build/port_plain "$STAGE/build/port_plain"

echo "② 根目录给父进程用的 cg / data / 表"
for f in cg data EN_Card_Data.csv r2e_manifest.json; do cp -a "build/port_plain/$f" "$STAGE/$f"; done

echo "③ 卡组池"
cp -a decks/pool_field "$STAGE/decks/pool_field"

echo "④ 评估器与工具"
cp -f ptcg_league/eval_pair.py "$STAGE/ptcg_league/"
for f in tools/bo3_eval.py tools/make_agent_dir.sh tools/fill_winrate.sh tools/build_winrate_table.py; do cp -f "$f" "$STAGE/tools/"; done
# tools/py 做成可移植的 shell 壳（目标机不一定是 conda 布局）
cat > "$STAGE/tools/py" <<'EOS'
#!/usr/bin/env bash
exec "${PYTHON:-python3}" "$@"
EOS
chmod +x "$STAGE/tools/py" "$STAGE/tools/"*.sh

echo "⑤ 依赖数据：choice-detail（生成表用）+ 已跑出的结果（跳过重算）"
cp -f docs/the-pockmon-battle-choice-detail.csv "$STAGE/docs/"
n=$(ls logs/winrate/*.json 2>/dev/null | wc -l)
[ "$n" -gt 0 ] && cp -f logs/winrate/*.json "$STAGE/logs/winrate/" 2>/dev/null
echo "   带上已完成的 $n 格"

echo "⑥ setup.sh：铺 12 个 agent 目录 + 自检"
cat > "$STAGE/setup.sh" <<'EOS'
#!/usr/bin/env bash
# 目标机初始化：铺 build/agents/*（12 副牌，全部用同一个端口权重）+ 环境自检。
set -uo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
echo "=== 环境自检 ==="
$PY -V || { echo "✗ 没有 $PY"; exit 1; }
$PY -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__)" || {
  echo "✗ 缺 numpy/torch：pip install numpy torch --index-url https://pypi.org/simple"; exit 1; }
echo "核数 $(nproc)  可用内存 $(free -g | awk '/^Mem:/{print $7}') GB"
echo "=== 引擎自检（ctypes 加载 libcg.so）==="
PYTHONPATH=. $PY -c "import cg.game as g; print('cg 引擎 OK', type(g))" || { echo "✗ 引擎加载失败（glibc/架构不符？）"; exit 1; }

echo "=== 铺 12 个 agent 目录 ==="
# ★ 目录名必须与仓库里 fill_winrate.sh 的 AG 映射**逐字一致**（agent 名 = 卡的短名映射），
#   否则会报 "无法识别的 agent 路径: build/agents/xxx" 静默失败（本地实测踩过）。
declare -A D=(
  [port_0904]=z_dragapult_dreepy_petit__0904        [opp_dunsparce_0bf5]=dunsparce_abra__0bf5
  [port_0d86]=shuppet_dhelmise__0d86                [c_grookey]=grookey_thwackey__2617
  [port_26c6]=slowpoke_slowking__26c6               [opp_3eaa]=dreepy_drakloak__3eaa
  [port_7e39]=teal_mask_ogerpon_ex_applin__7e39     [port_on_9d6e]=dunsparce_abra__9d6e
  [port_db04]=dunsparce_abra__db04                  [port_e1d7]=z_ogerpon_toolbox_miwa__e1d7
  [port_e8ee]=abra_kadabra__e8ee                    [c_abra]=abra_kadabra__f5f8 )
for k in "${!D[@]}"; do
  bash tools/make_agent_dir.sh "build/agents/$k" build/port_plain/model.pt "decks/pool_field/${D[$k]}.csv" >/dev/null \
    || { echo "✗ 铺 $k 失败"; exit 1; }
done
echo "  ✓ $(ls build/agents | wc -l) 个 agent 就绪（共用 model.pt $(md5sum build/port_plain/model.pt | cut -c1-8)）"
echo
echo "=== 开始跑缺格： bash tools/fill_winrate.sh ==="
EOS
chmod +x "$STAGE/setup.sh"

tar czf "$OUTTAR" -C dist eval_cpu_kit
echo
echo "✓ $OUTTAR  $(du -h "$OUTTAR" | cut -f1)"
echo "目标机： tar xzf eval_cpu_kit.tar.gz && cd eval_cpu_kit && bash setup.sh && bash tools/fill_winrate.sh"
