"""自动 Parquet 缓存 —— 训练前把 JSONL 转成列式格式，加速加载。

## 为什么需要

JSONL 有两个硬伤（都踩过）：
  1. **体积浪费 6~7 倍**：每个 float 写成文本（`"st_turn": 3.0`），二进制只要 4 字节
  2. **加载会内存爆炸**：`[json.loads(l) for l in f]` 把整份数据变成 Python dict，
     1.8 GB 的文件要 10+ GB 内存 —— 本机只有 7.4 GB，直接被 OOM 静默杀掉

## 缓存设计

    data/.cache/
    └── <源文件名>.<kind>.v<N>/
        ├── part-0000.parquet
        ├── part-0001.parquet
        └── meta.json          # 源文件 size/mtime、行列数、特征列名

三条纪律：
  1. **按源文件 size + mtime + 版本号判新鲜度**。源数据重新生成 -> mtime 变 -> 自动重建。
  2. **分块写多个 part**，绝不一次性把全部行读进内存。
  3. **读用 `pl.scan_parquet` 惰性加载**，可以处理超过内存的数据。

## 两种用途

    kind="value"  给 V 用：每行一个决策点，列 = game/label/53 个状态特征
    kind="rank"   给 P（LTR）用：**拍平**成每行一个候选动作，
                  列 = query_id/relevance/状态特征/选项特征，query_id 即 group

## 用法

    from framework import cache
    df = cache.load("data/pool_v4_v.jsonl", kind="value")      # 自动建缓存
    lf = cache.scan("data/selfplay_v1.jsonl", kind="rank")     # 惰性
    cache.ensure("data/pool_v4_v.jsonl", force=True)           # 强制重建

命令行预建：
    ./run.sh -m framework.cache data/pool_v4_v.jsonl --kind value
"""
from __future__ import annotations

import json
import os
import sys
import time

import polars as pl

from framework.dataset import iter_records

CACHE_VERSION = 1
CHUNK_ROWS = 50_000          # 每块行数：控制峰值内存
CACHE_DIRNAME = ".cache"


# ============================================================ 路径与新鲜度
def _cache_root(src: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(src)), CACHE_DIRNAME)


def cache_dir(src: str, kind: str) -> str:
    stem = os.path.splitext(os.path.basename(src))[0]
    return os.path.join(_cache_root(src), f"{stem}.{kind}.v{CACHE_VERSION}")


def _meta_path(d: str) -> str:
    return os.path.join(d, "meta.json")


def _src_stat(src: str) -> dict:
    st = os.stat(src)
    return {"src": os.path.abspath(src), "src_size": st.st_size,
            "src_mtime": int(st.st_mtime)}


def is_fresh(src: str, kind: str) -> bool:
    d = cache_dir(src, kind)
    mp = _meta_path(d)
    if not os.path.exists(mp):
        return False
    try:
        meta = json.load(open(mp))
    except Exception:
        return False
    cur = _src_stat(src)
    return (meta.get("version") == CACHE_VERSION
            and meta.get("kind") == kind
            and meta.get("src") == cur["src"]
            and meta.get("src_size") == cur["src_size"]
            and meta.get("src_mtime") == cur["src_mtime"]
            and meta.get("parts", 0) > 0)


# ============================================================ 构建

# 分块间 schema 必须一致，否则 polars scan_parquet 会报
# "data type mismatch: incoming Int64 != target Float64"。
# 特征值理论上都是 float，但 python 里 max(0.0, int) 这类写法会漏出 int，
# 所以构建时统一兜底。meta 里的 ID/标签列保持原类型。
_ID_LIKE = {"game", "query_id", "opt_index", "n_opts"}


def _coerce(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, int) and k not in _ID_LIKE:
            out[k] = float(v)
        else:
            out[k] = v
    return out

def _flush(rows: list[dict], out_dir: str, part_no: int) -> int:
    if not rows:
        return part_no
    df = pl.DataFrame([_coerce(r) for r in rows])
    df.write_parquet(os.path.join(out_dir, f"part-{part_no:04d}.parquet"),
                     compression="zstd")
    return part_no + 1


def _label_of(r: dict) -> float | None:
    res = r.get("result")
    if res is None:
        return None
    if res == 2:
        return 0.5
    return 1.0 if res == r["player"] else 0.0


def _build_value(src: str, out_dir: str) -> dict:
    """kind=value：每行一个决策点。"""
    # 第一遍：收集全部特征列名（避免分块间 schema 不一致）
    keys: set[str] = set()
    n_scan = 0
    for r in iter_records(src):
        keys.update(r.get("features", {}).keys())
        n_scan += 1
    keys = sorted(keys)

    # 第二遍：分块写
    part = 0
    buf: list[dict] = []
    n_rows = skipped = 0
    for r in iter_records(src):
        lab = _label_of(r)
        if lab is None:
            skipped += 1
            continue
        row = {"game": int(r["game"]), "label": lab,
               "player": int(r["player"]), "turn": int(r.get("turn", 0))}
        row.update(r.get("features") or {})
        buf.append(row)
        n_rows += 1
        if len(buf) >= CHUNK_ROWS:
            part = _flush(buf, out_dir, part)
            buf.clear()
    part = _flush(buf, out_dir, part)
    return {"parts": part, "rows": n_rows, "skipped": skipped,
            "scanned": n_scan, "columns": keys}


def _build_rank(src: str, out_dir: str) -> dict:
    """kind=rank：拍平成每行一个候选动作（LTR 的 group 就是 query_id）。"""
    state_keys: set[str] = set()
    opt_keys: set[str] = set()
    n_scan = 0
    for r in iter_records(src):
        state_keys.update(r.get("features", {}).keys())
        for of in (r.get("options") or []):
            opt_keys.update(of.keys())
        n_scan += 1
    state_keys, opt_keys = sorted(state_keys), sorted(opt_keys)
    if not opt_keys:
        raise SystemExit(f"{src} 里没有任何 options 字段，无法构建 rank 数据。"
                         f"请用 gen_data.py 去掉 --no-options 重新生成。")

    part = 0
    buf: list[dict] = []
    qid = -1
    n_rows = n_opts = skipped = 0
    for r in iter_records(src):
        opts = r.get("options") or []
        if not opts:
            skipped += 1
            continue
        lab = _label_of(r)
        if lab is None:
            skipped += 1
            continue
        qid += 1
        n_opts += len(opts)
        chosen = set(r.get("action") or [])
        base = {"query_id": qid, "game": int(r["game"]), "label": lab,
                "n_opts": len(opts)}
        base.update(r.get("features") or {})
        for i, of in enumerate(opts):
            row = dict(base)
            row.update(of)
            row["chosen"] = 1.0 if i in chosen else 0.0
            row["opt_index"] = i
            buf.append(row)
            n_rows += 1
            if len(buf) >= CHUNK_ROWS:
                part = _flush(buf, out_dir, part)
                buf.clear()
    part = _flush(buf, out_dir, part)
    return {"parts": part, "rows": n_rows, "queries": qid + 1,
            "option_rows": n_opts, "skipped": skipped, "scanned": n_scan,
            "state_columns": state_keys, "option_columns": opt_keys}


_BUILDERS = {"value": _build_value, "rank": _build_rank}


def ensure(src: str, kind: str = "value", force: bool = False,
           verbose: bool = True) -> str:
    """确保缓存存在且新鲜；返回缓存目录路径。"""
    if kind not in _BUILDERS:
        raise ValueError(f"未知 kind={kind}，可选 {list(_BUILDERS)}")
    if not os.path.exists(src):
        raise FileNotFoundError(src)

    d = cache_dir(src, kind)
    if not force and is_fresh(src, kind):
        if verbose:
            meta = json.load(open(_meta_path(d)))
            print(f"  [cache] 命中 {os.path.basename(d)}  "
                  f"({meta.get('rows', 0):,} 行, mtime 匹配)")
        return d

    if verbose:
        sz = os.path.getsize(src) / 1e6
        print(f"  [cache] 构建 {os.path.basename(d)}  (源 {sz:.0f} MB)...")
    t0 = time.time()
    os.makedirs(d, exist_ok=True)
    # 清掉旧 part，避免残留文件混进扫描
    for f in os.listdir(d):
        if f.endswith(".parquet"):
            os.remove(os.path.join(d, f))

    info = _BUILDERS[kind](src, d)
    meta = {"version": CACHE_VERSION, "kind": kind, **_src_stat(src), **info,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "build_seconds": round(time.time() - t0, 1)}
    with open(_meta_path(d), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    if verbose:
        print(f"  [cache] 完成 {info['rows']:,} 行 / {info['parts']} part "
              f"({time.time()-t0:.1f}s)")
    return d


# ============================================================ 读取
def scan(src: str, kind: str = "value", force: bool = False) -> pl.LazyFrame:
    """惰性加载（内存友好，可以处理超过内存的数据）。"""
    d = ensure(src, kind, force=force)
    return pl.scan_parquet(os.path.join(d, "*.parquet"))


def load(src: str, kind: str = "value", force: bool = False) -> pl.DataFrame:
    """一次性加载成 DataFrame（数据量大时优先用 scan）。"""
    return scan(src, kind, force=force).collect()


def meta(src: str, kind: str = "value") -> dict:
    d = cache_dir(src, kind)
    return json.load(open(_meta_path(d)))


def clear(src: str, kind: str | None = None) -> None:
    """删除缓存（kind=None 则删掉该源的所有 kind）。"""
    kinds = [kind] if kind else list(_BUILDERS)
    for k in kinds:
        d = cache_dir(src, k)
        if os.path.isdir(d):
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
            os.rmdir(d)
            print(f"已删除 {d}")


# ============================================================ CLI
def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="构建/查看 parquet 缓存")
    ap.add_argument("src", nargs="+", help="一个或多个 JSONL 路径")
    ap.add_argument("--kind", default="value", choices=list(_BUILDERS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--show", action="store_true", help="打印 schema 与样例")
    args = ap.parse_args()

    for src in args.src:
        if not os.path.isabs(src):
            src = os.path.join(os.getcwd(), src)
        print(f"\n=== {os.path.basename(src)} ({args.kind}) ===")
        if args.clear:
            clear(src, args.kind)
            continue
        d = ensure(src, args.kind, force=args.force)
        m = meta(src, args.kind)
        print(f"  行数 {m.get('rows', 0):,} | part {m.get('parts', 0)} | "
              f"构建 {m.get('build_seconds', '?')}s")
        if args.show:
            df = scan(src, args.kind).head(3).collect()
            print(f"  列数 {df.width}")
            print(f"  前 12 列: {df.columns[:12]}")
            print(df.head(2))


if __name__ == "__main__":
    _main()
