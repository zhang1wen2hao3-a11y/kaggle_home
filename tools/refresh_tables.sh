#!/usr/bin/env bash
# **固定刷新流程**：把 logs/meta 的新结果 → 母表 → 全部交付表 → 必须卡组 → 投资分 → 校验 → 方案A 测试集。
#
# 固定评测流程的唯一执行入口（规则见 docs/驾驶员与覆盖规则.md）。
# 任何一步失败都会**非零退出**并打印原因，不允许"跳过继续"。
#
# 2026-11-19 起新增（用户要求）：训练资源要向「高分卡组 + 高覆盖率」倾斜 ⇒
#   ④ 必须卡组清单（17 格 ∪ 母表 ∪ 榜前 N 队用过的牌）
#   ⑤ 补必须卡组的驾驶员（缺模型 ⇒ 优先 BC 权重，没有就端口 init）
#   ⑦ 投资分 = 缺口(远端高分 − 我们) × 覆盖率 × 天花板，三项开三次方
#   ⑧ 覆盖校验：必须卡组**必须**一条不漏地出现在投资分表里，否则非零退出
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

say "⓪ 牌序（统一排序：数字在字母前；不通过就中断）"
$PY tools/deck_order.py check 2>&1 | tail -1 || die "牌序未统一排序 —— 跑 tools/deck_order.py 修（或检查是不是有人手改了 deck_order.txt）"

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

say "④ 必须卡组清单（17格 ∪ 母表 ∪ 榜前N队用过的牌）"
$PY tools/required_decks.py 2>&1 | tail -2 || die "required_decks 失败"

say "⑤ 补必须卡组的驾驶员（缺模型 ⇒ BC 优先，否则端口 init）"
$PY tools/materialize_required_drivers.py 2>&1 | tail -2 || die "materialize_required_drivers 失败"
$PY tools/required_decks.py 2>&1 | tail -1 || die "required_decks(复核) 失败"   # 补完再算，驾驶员列才准

say "⑥ 名字校验"
if ! $PY tools/driver_registry.py check 2>&1 | tail -3; then
  die "表里有无法解析的驾驶员名字（见上）—— 先登记 tools/driver_registry.py"
fi

say "⑦ 投资分表（缺口 × 覆盖率 × 天花板；向高分卡组 + 高覆盖倾斜）"
$PY tools/build_investment_score.py 2>&1 | tail -3 || die "build_investment_score 失败"

say "⑧ 覆盖校验（必须卡组一条不漏）"
$PY tools/required_decks.py --check-covered 2>&1 | tail -3 || die "投资分表没覆盖全部必须卡组"

say "⑨ 淘汰机制（弱 + 无格压力 + 无覆盖率 ⇒ 淘汰；冠军卡组保护）"
$PY tools/prune_decks.py --check 2>&1 | tail -2 || die "有卡组该淘汰却还留在维护表里（见 tools/prune_decks.py 输出）"

say "⑩ 方案A 测试集"
$PY tools/make_testset_A.py 2>&1 | tail -2 || die "make_testset_A 失败"

say "✓ 刷新完成"
