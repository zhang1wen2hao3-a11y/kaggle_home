#!/usr/bin/env bash
# ============================================================================
# 统一解释器入口 —— 解决"装了包但用错 Python"的问题
#
# 本机有两个 Python：
#   base (3.14.6)      零依赖       —— 只能跑评估/对局
#   py310 (3.10.21)    依赖齐全     —— 能跑训练（numpy/lightgbm/torch/...）
#
# 训练脚本必须用 py310，否则 framework.models.available() 里看不到 lightgbm。
# 这个脚本自动挑最合适的解释器，所以所有命令都能统一写成：
#
#     ./run.sh train_value.py --data data/selfplay_v1.jsonl --model lightgbm
#     ./run.sh evaluate.py --agent main.py --opponent agents/random_baseline.py
#     ./run.sh tests/test_memory.py
#
# 其他用法：
#     ./run.sh --check     打印环境诊断（选了哪个解释器、有哪些依赖）
#     ./run.sh --which     只打印选中的解释器路径
#
# 覆盖解释器：PTCG_PYTHON=/path/to/python ./run.sh ...
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ---------------------------------------------------------------- 候选解释器
_candidates() {
    # 1. 显式指定优先
    [ -n "${PTCG_PYTHON:-}" ] && echo "$PTCG_PYTHON"
    # 2. conda 环境（按"依赖齐全程度"排序）
    for env in py310 ptcg kaggle; do
        p="$HOME/miniforge3/envs/$env/bin/python"
        [ -x "$p" ] && echo "$p"
    done
    p="$HOME/anaconda3/envs/py310/bin/python"; [ -x "$p" ] && echo "$p"
    p="$HOME/miniconda3/envs/py310/bin/python"; [ -x "$p" ] && echo "$p"
    # 3. 项目内 venv
    [ -x "$HERE/.venv/bin/python" ] && echo "$HERE/.venv/bin/python"
    [ -x "$HERE/venv/bin/python" ] && echo "$HERE/venv/bin/python"
    # 4. 系统 python
    command -v python3 || true
}

# 该解释器是否具备指定能力：base | ml
_has() {
    local py="$1" level="$2"
    [ -x "$py" ] || command -v "$py" >/dev/null 2>&1 || return 1
    case "$level" in
        base) "$py" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null ;;
        ml)   "$py" -c "import numpy, lightgbm" 2>/dev/null ;;
    esac
}

# ---------------------------------------------------------------- 选解释器
WANT="${PTCG_NEED:-ml}"        # 默认按 ML 需求挑；找不到再降级
PY=""
for cand in $(_candidates); do
    if _has "$cand" "$WANT"; then PY="$cand"; break; fi
done
# 降级：ML 环境找不到就用任意可用的解释器（至少能跑评估/对局）
if [ -z "$PY" ]; then
    for cand in $(_candidates); do
        if _has "$cand" base; then PY="$cand"; break; fi
    done
fi
if [ -z "$PY" ]; then
    echo "错误：找不到可用的 Python 解释器（需要 >= 3.10）" >&2
    echo "提示：用 PTCG_PYTHON=/path/to/python 显式指定" >&2
    exit 1
fi

# ---------------------------------------------------------------- 诊断
_diag() {
    echo "项目目录 : $HERE"
    echo "解释器   : $PY"
    "$PY" - <<'PYEOF'
import sys
print(f"版本     : {sys.version.split()[0]}")
mods = [("numpy", "数据处理"), ("pandas", "数据处理"), ("polars", "数据处理"),
        ("lightgbm", "训练 V/P"), ("sklearn", "基线/指标"), ("scipy", "数值"),
        ("matplotlib", "可视化"), ("torch", "深度模型"), ("kaggle", "提交 CLI")]
for m, why in mods:
    try:
        mod = __import__(m)
        v = getattr(mod, "__version__", "?")
        print(f"  ✅ {m:14s} {v:12s} {why}")
    except ImportError:
        print(f"  ❌ {m:14s} {'':12s} {why}")
PYEOF
    # 引擎与框架
    python_out=$("$PY" - <<'PYEOF' 2>&1 || true
import sys, os
sys.path.insert(0, os.getcwd())
try:
    import cg.game
    print("  ✅ 引擎 cg/ 可加载")
except Exception as e:
    print(f"  ❌ 引擎加载失败: {type(e).__name__}: {e}")
    print("     提示：先跑 bash fetch_data.sh 拉取比赛数据")
try:
    from framework.models import available
    print(f"  ✅ 框架后端: {available()}")
except Exception as e:
    print(f"  ❌ 框架导入失败: {type(e).__name__}: {e}")
PYEOF
)
    echo "$python_out"
}

case "${1:-}" in
    --check|-c)
        _diag
        exit 0
        ;;
    --which|-w)
        echo "$PY"
        exit 0
        ;;
    ""|--help|-h)
        sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        echo
        echo "用法: ./run.sh <脚本> [参数...]"
        echo "      ./run.sh --check    # 环境诊断"
        echo "      ./run.sh --which    # 打印选中的解释器"
        exit 0
        ;;
esac

# ---------------------------------------------------------------- 执行
# 选中的解释器只报到 stderr，避免污染脚本的 stdout（有些脚本输出被解析）
if [ -n "${PTCG_VERBOSE:-}" ]; then
    echo "[run.sh] 解释器: $PY  (需要 $WANT)" >&2
fi

exec "$PY" "$@"
