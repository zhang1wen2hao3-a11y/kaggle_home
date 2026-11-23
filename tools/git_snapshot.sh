#!/usr/bin/env bash
# **结果快照提交**：把"证据 + 表 + 工具"按固定白名单提交，给长期跑机用。
#
# 为什么不 `git add -A`：
#   `logs/` 里有大量 per-cell 的 logdir（几十万行日志），`data/` 里有几十 G 的语料，
#   全丢进仓库会把 .git 撑爆、也会把真正的证据（summary.json / vs_*.json / 对位 json）淹掉。
#   ⇒ 只提交**能复现结论的最小集**：母表与全部交付表、evidence json、tools。
#
#   bash tools/git_snapshot.sh "消息"          # 有变化才提交
#   bash tools/git_snapshot.sh "消息" --push   # 顺便推到 origin
set -uo pipefail
cd "$(dirname "$0")/.."
MSG="${1:-例行结果快照}"
PUSH=0
[ "${2:-}" = "--push" ] && PUSH=1

# 白名单（-f 是为了绕过 .gitignore 里的 logs/、data/）
git add -f docs/*.csv tools/*.py tools/*.sh 2>/dev/null
# 训练器源码（奖励/势函数开关等小改动必须入库，否则 GPU 机上永远是旧版）
git add -f port_repo/native/train_native.py port_repo/native/ptcg/env.c port_repo/native/ptcg/ptcg_env.h port_repo/native/ptcg/binding.c 2>/dev/null
git add -f data/opp_matrix/*.json data/lb_readings.tsv data/lb_ours_by_deck.json 2>/dev/null
# 只收证据文件：面板汇总 + 每格对位结果；per-cell logdir 不收
find logs/meta -maxdepth 2 -name summary.json -print0 2>/dev/null | xargs -0 -r git add -f
find logs/meta -maxdepth 2 -name 'vs_*.json' -print0 2>/dev/null | xargs -0 -r git add -f
git add -f LB记录.md docs/*.md 2>/dev/null
# 牌表口径/队列这些小文本是决策依据，必须进仓库（train_queue 记录每副牌的训练状态）
git add -f decks/*.txt decks/*.tsv 2>/dev/null

if git diff --cached --quiet; then
  echo "（无变化，跳过提交）"
  exit 0
fi
n=$(git diff --cached --name-only | wc -l)
git commit -q -m "$MSG" && echo "✓ 提交 $n 个文件：$MSG"
if [ "$PUSH" = 1 ]; then
  git push -q origin HEAD 2>&1 | tail -2 && echo "✓ 已推送 origin"
fi
