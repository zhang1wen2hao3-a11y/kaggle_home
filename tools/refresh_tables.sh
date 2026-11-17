#!/usr/bin/env bash
# **固定刷新流程**：把 logs/meta 的新结果 → 母表 → 全部交付表 → 校验 → 方案A 测试集。
#
# 固定评测流程的唯一执行入口（规则见 docs/驾驶员与覆盖规则.md）：
#   ① 导出母表（唯一证据源）      ② 重算 b05 口径交付表   ③ 重算顶分段口径交付表
#   ④ 校验表里每个驾驶员都能被注册表解析   ⑤ 生成方案 A 测试集（对手=各牌最佳驾驶员）
#
# 任何一步失败都会**非零退出**并打印原因，不允许"跳过继续"。
# orchestrate_cpu.sh 每轮拉完结果就调用本脚本；手工更新也只调它，不要单步跑。
#
#   bash tools/refresh_tables.sh            # 全流程
#   bash tools/refresh_tables.sh --quiet    # 只打错误
set -uo pipefail
cd "$(dirname "$0")/.."
PY=./tools/py
Q=0
[ "${1:-}" = "--quiet" ] && Q=1
say() { [ "$Q" = 1 ] || echo "$@"; }
die() { echo "✗ $*" >&2; exit 1; }

say "① 母表"
$PY tools/export_master.py 2>&1 | tail -2 || die "export_master 失败"

say "② 交付表（b05 口径）"
$PY tools/build_from_master.py 2>&1 | tail -2 || die "build_from_master(b05) 失败"

say "③ 交付表（顶分段口径）"
$PY tools/build_from_master.py --weights decks/weights_top.txt --suffix _top 2>&1 | tail -1 \
  || die "build_from_master(top) 失败"
$PY tools/build_pilot_table.py 2>&1 | tail -1 || die "build_pilot_table(b05) 失败"
$PY tools/build_pilot_table.py --weights decks/weights_top.txt --suffix _top 2>&1 | tail -1 \
  || die "build_pilot_table(top) 失败"

say "④ 名字校验"
if ! $PY tools/driver_registry.py check 2>&1 | tail -3; then
  die "表里有无法解析的驾驶员名字（见上）—— 先登记 tools/driver_registry.py"
fi

say "⑤ 投资分表（面板补后水平 × 天花板；下一轮补强的排序依据）"
$PY tools/build_investment_table.py 2>&1 | tail -3 || die "build_investment_table 失败"

say "⑥ 方案A 测试集"
$PY tools/make_testset_A.py 2>&1 | tail -2 || die "make_testset_A 失败"

say "✓ 刷新完成"
