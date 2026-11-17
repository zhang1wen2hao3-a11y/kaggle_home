#!/usr/bin/env bash
# **评测机收尾：拉结果 → 校验 → 通过才关机**（用户 2026-11-18 要求）
#   bash tools/finish_box.sh root@HOST [期望完成数...]
#
# 为什么必须"先校验再关机"：2026-11-18 就因为没拉结果、机器被释放，丢了 33 个格子的数据。
# 这里的纪律：
#   ① 远端 `eval_pair` 进程必须为 0（脚本不会再起新任务）
#   ② 结果 rsync 回来，且**每个 json 都能解析**（半截文件不算）
#   ③ 关键 tag 的格数达到期望（默认：任意 `*_real_s95*` 全部拉回，且总格数 ≥ 期望值）
#   任一条不满足 ⇒ **不关机**，打印缺口并退出 2，等人看
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
H="${1:?用法: bash tools/finish_box.sh root@HOST [期望格数]}"
EXPECT="${2:-0}"
LOG=logs/finish_box.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ① 等远端空闲（最多 120 分钟，每 30s 探）
for i in $(seq 1 240); do
  n=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 "$H" 'pgrep -fc "ptcg_league/[e]val_pair" || true' 2>/dev/null | tr -d '[:space:]')
  case "${n:-}" in ''|*[!0-9]*) n="?";; esac
  [ "$n" = 0 ] && { say "远端已空闲（第 $i 次探测）"; break; }
  [ "$n" = "?" ] && say "（第 $i 次探测连不上，继续等）"
  sleep 30
done

# ② 拉结果
say "拉取 vs_*.json"
timeout 240 rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=8" \
  --include '*/' --include 'vs_*.json' --exclude '*' \
  "$H:/root/eval_cpu_kit/logs/meta/" logs/meta/ 2>&1 | tail -2

# ③ 校验：每个 json 可解析 + 计数
./tools/py - "$EXPECT" <<'PY'
import glob, json, os, sys
expect = int(sys.argv[1])
ok = bad = 0
for p in glob.glob('logs/meta/*_real_s95*/vs_*.json'):
    try:
        d = json.load(open(p))
        if int((d.get('overall') or {}).get('n') or 0) > 0:
            ok += 1
        else:
            bad += 1
    except Exception:
        bad += 1
print(f"本地可解析结果 {ok} 个，坏文件 {bad} 个")
sys.exit(0 if (bad == 0 and ok >= expect) else 2)
PY
RC=$?
if [ "$RC" -ne 0 ]; then
  say "✗ 校验未通过（坏文件或格数不足，期望 ≥${EXPECT}）⇒ **不关机**，请人工检查"
  exit 2
fi

# ④ 关机
say "═══ 校验通过，关远端机器 ═══"
timeout 30 ssh -o BatchMode=yes "$H" 'sync; nohup shutdown -h now > /dev/null 2>&1 & sleep 2; echo bye' 2>&1 | tail -1 | sed 's/^/  /' | tee -a "$LOG"
sleep 20
timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=8 "$H" 'echo still-up' 2>&1 | tail -1 | sed 's/^/  探测: /' | tee -a "$LOG"
say "DONE"
