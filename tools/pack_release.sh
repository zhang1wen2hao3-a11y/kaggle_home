#!/usr/bin/env bash
# 一次性重打两个交付包，并**证明包内工具与仓库一致**（防版本漂移）。
#
#   bash tools/pack_release.sh
#
# 产出：
#   dist/ptcg_train_kit_v3.tar.gz   57 MB  GPU 侧：清洗过的 port_repo(含两个补丁) + tools + 187 池
#   dist/dual_starter.tar.gz        12 KB  本机侧：只含 3 个工具 + 文档
#
# 为什么要有这个脚本：2026-11-05 出现过 v3 里的 dual_train.py 比仓库旧两版的情况
# （先打了 v3，之后又修了两次 dual_train），用户上手才会踩到。现在统一从这里出包，
# 并且结尾会逐文件比对 md5，不一致直接报错退出。
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
V2="dist/ptcg_train_kit_v2.tar.gz"
[ -f "$V2" ] || { echo "✗ 缺 $V2（v3 以它为基底）"; exit 1; }

TOOLS_IN_KIT=(dual_train.py router_core.py make_agent_dir.sh patch_bank_decks.py db04_line.sh)
STARTER=(tools/dual_train.py tools/router_core.py tools/make_agent_dir.sh
         dist/dual_starter_README.md dist/ONDEMAND_KITS.md)

echo "===== 1) 重打 ptcg_train_kit_v3.tar.gz ====="
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
tar xzf "$V2" -C "$TMP"
STAGE="$TMP/ptcg_train_kit"
mkdir -p "$STAGE/tools"
for t in "${TOOLS_IN_KIT[@]}"; do cp -f "tools/$t" "$STAGE/tools/$t"; done
# 池与 blob 对齐到 187（v2 自带 185，会和线上机器的 187 副不一致）
for d in decks/pool_field/z_dragapult_dreepy_petit__0904.csv \
         decks/pool_field/z_ogerpon_toolbox_miwa__e1d7.csv; do
  cp -f "$d" "$STAGE/port_repo/decks/pool_field/"
done
cp -f port_repo/native/ptcg_tables_field.bin "$STAGE/port_repo/native/"
cp -f dist/ONDEMAND_KITS.md "$STAGE/ONDEMAND_KITS.md" 2>/dev/null || true
cp -f dist/ptcg_train_kit_v3_README.md "$STAGE/README_v3.md" 2>/dev/null || true
( cd "$TMP" && tar czf "$ROOT/dist/ptcg_train_kit_v3.tar.gz" ptcg_train_kit )

echo
echo "===== 2) 重打 dual_starter.tar.gz ====="
tar czf dist/dual_starter.tar.gz -C "$ROOT" "${STARTER[@]}"

echo
echo "===== 3) 一致性验证（包内工具 md5 vs 仓库）====="
bad=0
for t in "${TOOLS_IN_KIT[@]}"; do
  a=$(md5sum "tools/$t" | cut -d' ' -f1)
  b=$(tar xzf dist/ptcg_train_kit_v3.tar.gz -O "ptcg_train_kit/tools/$t" 2>/dev/null | md5sum | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then printf "  ✓ v3/tools/%-24s %s\n" "$t" "${a:0:12}"
  else printf "  ✗ v3/tools/%-24s 包内 %s ≠ 仓库 %s\n" "$t" "${b:0:12}" "${a:0:12}"; bad=1; fi
done
for t in tools/dual_train.py tools/router_core.py tools/make_agent_dir.sh; do
  a=$(md5sum "$t" | cut -d' ' -f1)
  b=$(tar xzf dist/dual_starter.tar.gz -O "$t" 2>/dev/null | md5sum | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then printf "  ✓ starter/%-22s %s\n" "$(basename $t)" "${a:0:12}"
  else printf "  ✗ starter/%-22s 包内 %s ≠ 仓库 %s\n" "$(basename $t)" "${b:0:12}" "${a:0:12}"; bad=1; fi
done

echo
echo "===== 4) 产物 ====="
for f in dist/ptcg_train_kit_v3.tar.gz dist/dual_starter.tar.gz; do
  printf "  %-34s %6.1f MB  sha16 %s\n" "$f" "$(stat -c%s $f | awk '{print $1/1048576}')" "$(sha256sum $f | cut -c1-16)"
done
[ "$bad" = 0 ] && echo "  ✅ 包内工具与仓库完全一致" || { echo "  ✗ 有漂移，禁止交付"; exit 1; }
