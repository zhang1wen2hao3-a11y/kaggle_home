#!/usr/bin/env bash
# **GPU 训练机 · 一键环境搭建**（原生 CUDA 训练器）
#
# 从一台**干净**的机器到"能跑 tools/*_kit 里的 apply.sh --run"，全在这一个脚本里。
# 幂等：重复跑不会坏（已编好的会跳过）。
#
#   PTCG_REPO=/root/ptcg_train_kit/port_repo bash tools/setup_gpu_train.sh
#   PTCG_SM=89 PTCG_REPO=... bash tools/setup_gpu_train.sh     # 显式指定架构
#
# 镜像要求（不满足就先换镜像，别硬试）：
#   · **带 CUDA toolkit（要有 nvcc）**，推荐 12.x；只有驱动+torch runtime 的镜像不行
#   · gcc/g++（C++17 + fopenmp）、make
#   · Python 3.10 左右，且 `python3` 能 import torch / numpy
#   · torch 的 CUDA **主版本要等于 nvcc 的主版本**（pufferlib/torch 扩展用系统 nvcc 编）
#
# 本脚本做六件事：
#   0. 把 /usr/local/cuda/bin 塞进 PATH（AutoDL 不塞，`which nvcc` 会误判成"没装"）
#   1. 体检：GPU / VRAM / compute_cap / nvcc / gcc / python+torch
#   2. pybind11（联网装 → 失败则用 torch 自带的 vendor 头文件兜底）
#   3. NCCL（优先 torch 自带那份，避免 "CUDA driver version is insufficient"）
#   4. 编译原生扩展（按实际设备的 sm_XX 编，不猜默认值）
#   5. 三条自检：引擎 import、`make -C native test`、表格 blob 常量核对
set -uo pipefail

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# ── 0. PATH / CUDA_HOME ────────────────────────────────────────────────────
if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
  echo "（已把 /usr/local/cuda/bin 加入 PATH：nvcc 原本不在 PATH 里）"
fi
[ -x /usr/local/cuda/bin/nvcc ] && export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# ── 定位训练仓库根 ─────────────────────────────────────────────────────────
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=""
for cand in "${PTCG_REPO:-}" "$PWD" "$PWD/port_repo" "$HERE/.." "$HERE/../port_repo"; do
  if [ -n "$cand" ] && [ -f "$cand/native/build_native.sh" ]; then ROOT="$(cd "$cand" && pwd)"; break; fi
done
if [ -z "$ROOT" ]; then
  echo "✗ 找不到训练仓库根（应含 native/build_native.sh）"
  echo "  用 PTCG_REPO=<port_repo 绝对路径> 显式指定，例如："
  echo "  PTCG_REPO=/root/ptcg_train_kit/port_repo bash tools/setup_gpu_train.sh"
  exit 1
fi
cd "$ROOT"
echo "仓库根：$ROOT"

# ── 1. 体检 ────────────────────────────────────────────────────────────────
say "1/6 体检"
fail=0
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
  VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  echo "VRAM = ${VRAM} MiB"
  [ "${VRAM:-0}" -lt 20000 ] && echo "⚠️  VRAM 偏小：512 agents 实测占 16.6 GB，显存不够就把 --total-agents 降到 256"
else
  echo "✗ 没有 nvidia-smi（这不是 GPU 机器，或者驱动没装）"; fail=1
fi

if command -v nvcc >/dev/null; then
  echo "nvcc: $(nvcc --version | grep -oE 'release [0-9.]+' | head -1)"
else
  echo "✗ 没有 nvcc ⇒ 镜像选错了（需要带 CUDA toolkit 的镜像，不是只有 runtime 的）"; fail=1
fi
command -v gcc >/dev/null && echo "gcc: $(gcc -dumpversion)" || { echo "✗ 没有 gcc"; fail=1; }
command -v make >/dev/null || { echo "✗ 没有 make"; fail=1; }

PYBIN="${PYTHON:-$(command -v python3 || command -v python)}"   # 支持 PYTHON=... 指定 conda 环境
[ -n "$PYBIN" ] || { echo "✗ 没有 python3"; exit 1; }
export PY="$PYBIN"
"$PY" - <<'EOF' || { echo "✗ python 里 torch/numpy 不全"; exit 1; }
import sys, torch, numpy
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  numpy {numpy.__version__}")
print(f"torch cuda {torch.version.cuda}  cuda_available {torch.cuda.is_available()}")
EOF
[ "$fail" = 0 ] || { echo; echo "✗ 体检没过 —— 先换镜像/装驱动，别急着编"; exit 1; }

# ── 2. pybind11（三条降级路径）─────────────────────────────────────────────
say "2/6 pybind11"
if "$PY" -c "import pybind11" 2>/dev/null; then
  echo "已就绪：$("$PY" -c 'import pybind11;print(pybind11.__version__)')"
else
  echo "缺失 → 试联网安装"
  "$PY" -m pip install --quiet pybind11 -i https://pypi.tuna.tsinghua.edu.cn/simple || true
  "$PY" -m pip install --quiet pybind11 || true
fi
if ! "$PY" -c "import pybind11" 2>/dev/null; then
  echo "联网装不到 → 离线兜底：用 torch 自带的 vendored 头文件"
  SHIM="${HOME}/.ptcg_pybind11"; mkdir -p "$SHIM/pybind11"
  cat > "$SHIM/pybind11/__init__.py" <<'SHIMEOF'
import os


def get_include():
    import torch
    inc = os.path.join(os.path.dirname(torch.__file__), "include")
    if not os.path.isdir(os.path.join(inc, "pybind11")):
        raise RuntimeError("torch 里没有 pybind11 头文件：" + inc)
    return inc


__version__ = "torch-vendored"
SHIMEOF
  export PYTHONPATH="$SHIM:${PYTHONPATH:-}"
  echo "  已建 shim：$SHIM（仅本次编译生效）"
fi

# ── 3. NCCL ────────────────────────────────────────────────────────────────
say "3/6 NCCL"
if [ -z "${NCCL_HOME:-}" ] && "$PY" -c "import nvidia.nccl" 2>/dev/null; then
  export NCCL_HOME="$("$PY" -c 'import nvidia.nccl;print(nvidia.nccl.__path__[0])')"
  echo "NCCL_HOME=$NCCL_HOME（用 torch 自带那份，避免 CUDA runtime 版本不匹配）"
else
  echo "（未设置 NCCL_HOME；单卡一般也能跑，报错再回来看这一节）"
fi

# ── 4. 编译 ────────────────────────────────────────────────────────────────
say "4/6 编译原生扩展"
DETECTED=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id=0 2>/dev/null | tr -d '.' || true)
SM="${PTCG_SM:-$DETECTED}"
if [ -z "$SM" ]; then
  echo "✗ 探测不到 compute_cap，无法确定架构。显式指定：PTCG_SM=86（3090）/89（4090）/80（A100）/90（H100）"
  exit 1
fi
export PTCG_SM="$SM"
echo "设备架构 sm_${PTCG_SM}"
bash native/build_native.sh
ls -la native/puffer_ptcg/_C*.so 2>/dev/null || { echo "✗ 没编出 _C*.so"; exit 1; }

# ── 5. 自检 ────────────────────────────────────────────────────────────────
say "5/6 自检"
PYTHONPATH=data:.:native "$PY" -c "from puffer_ptcg import _C; print('✓ _C 可导入')" \
  || { echo "✗ _C 导入失败"; exit 1; }
if make -C native test >/tmp/ptcg_make_test.log 2>&1; then
  echo "✓ make -C native test 通过"
else
  echo "✗ make -C native test 失败（日志 /tmp/ptcg_make_test.log）"; tail -20 /tmp/ptcg_make_test.log
fi
TBL=native/ptcg_tables_field.bin
if [ -f "$TBL" ]; then
  "$PY" - <<'EOF' || true
# 卡组表必须与 decks/pool_field 的 sorted-glob 逐行一致，否则 anchor 索引会错位
import glob, os, struct
names, rows = [], []
for fn in sorted(glob.glob("decks/pool_field/*.csv")):
    try:
        ids = [int(l) for l in open(fn) if l.strip()]
    except ValueError:
        continue
    if len(ids) == 60:
        names.append(os.path.basename(fn)[:-4]); rows.append(ids)
with open("native/ptcg_tables_field.bin", "rb") as f:
    assert f.read(8) == b"PTCGTAB1", "bad magic"
    nsec = struct.unpack("<15i", f.read(60))[14]
    blob = None
    for _ in range(nsec):
        nm = f.read(24).rstrip(b"\0").decode()
        dt, nd = struct.unpack("<2i", f.read(8))
        shape = struct.unpack("<4q", f.read(32))[:nd]
        n = 1
        for s in shape:
            n *= s
        if nm == "decks":
            r = struct.unpack(f"<{n}i", f.read(n * 4))
            blob = [list(r[i * 60:(i + 1) * 60]) for i in range(shape[0])]
            break
        f.seek(n * {0: 4, 1: 4, 2: 1, 3: 1}[dt], 1)
print(f"卡组池 {len(names)} 副，表格 blob {len(blob)} 行 ⇒ " +
      ("✓ 一致" if rows == blob else "✗ 不一致（先重跑 export_tables.py）"))
EOF
else
  echo "⚠️ 缺 $TBL（训练器要用它；按 native/README.md 生成）"
fi
[ -s native/pt_model_field.bin ] && [ "$(stat -c%s native/pt_model_field.bin)" = "18205408" ] \
  && echo "✓ 父权重模板 native/pt_model_field.bin 18,205,408 B" \
  || echo "⚠️ native/pt_model_field.bin 不是 18,205,408 B（0 字节坏包会让 init.bin 也是 0 字节）"

# ── 6. 下一步 ──────────────────────────────────────────────────────────────
say "6/6 环境就绪"
cat <<NEXT
接下来（以 G8 那类包为例）：
  tar xzf <某个>_kit.tar.gz && cd <某个>_kit
  PTCG_REPO=$ROOT bash apply.sh          # 预检，不花 GPU
  PTCG_REPO=$ROOT bash apply.sh --run    # 预检通过再开训

一条硬规则：**一次只跑一支**。512 agents + 2 bank 实测占 16.6 GB，
22 GB 卡上两份同时跑会 CUDA OOM；9 核机器上 THREADS 默认 8（留一个给系统）。
NEXT
