#!/usr/bin/env python3
"""**重建 `bc_pool/registry.tsv`**：每个 BC 脑子一行，带帧数、基线一致率、最佳一致率。

为什么要这个脚本：`tools/bc_pool.sh` 早期版本把每轮的输出用 `tail -12` 截断了，
登记表里只剩名字和 sha（帧数/一致率全是空列）。这些数字是"这个脑子值不值得进联赛"的
唯一依据，不能靠人眼记。所以统一由本脚本从两个可靠来源重建：

1. `<pool>/<name>.pt.json`（`tools/bc_train.py` 新版本自己写的摘要，权威）；
2. 老运行留下的聚合日志 `logs/bc_pool_run.log`（按 `####[name]` 分段解析，兜底）；
3. 帧数从 `data/bc_all.npz` 的元数据直接数（不信任日志）。

用法：
    ./tools/py tools/bc_pool_registry.py                 # 重建并打印
    ./tools/py tools/bc_pool_registry.py --check         # 只校验，不写
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

COLS = ["name", "filter", "frames", "val_frames", "base_val_acc", "val_acc",
        "train_acc", "val_loss", "bin", "bin_sha16", "date", "source"]


def frames_from_npz(npz):
    """{filter_spec: (frames, val_frames)}。filter_spec 用 `deck:xxxx` / `agent:NAME`。"""
    import collections
    import numpy as np
    z = np.load(npz, allow_pickle=True)
    arch, agent, split = z["arch"], z["agent"], z["split"]
    out = {}
    c = collections.Counter(arch)
    cv = collections.Counter(arch[split == "val"])
    for k, n in c.items():
        out[f"deck:{k}"] = (int(n), int(cv.get(k, 0)))
    ca = collections.Counter(agent)
    cav = collections.Counter(agent[split == "val"])
    for k, n in ca.items():
        out[f"agent:{k}"] = (int(n), int(cav.get(k, 0)))
    return out


def parse_aggregate_log(path):
    """从 `logs/bc_pool_run.log` 抓：baseline val 一致率 / 每轮 va_acc / 最后最佳。"""
    if not os.path.exists(path):
        return {}
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    for blk in re.split(r"^#{16} \[(?P<name>[^\]]+)\] (?P<spec>\S+) .*?$", txt, flags=re.M):
        pass
    # 按 "#### [name] spec" 切段
    parts = re.split(r"^#{8,} \[([^\]]+)\] (\S+) .*$", txt, flags=re.M)
    for i in range(1, len(parts), 3):
        name, spec, body = parts[i], parts[i + 1], parts[i + 2]
        d = dict(spec=spec)
        m = re.search(r"val\s+loss ([\d.]+)\s+一致率 ([\d.]+)%", body)
        if m:
            d["base_val_loss"], d["base_val_acc"] = float(m.group(1)), float(m.group(2)) / 100
        eps = re.findall(r"va_loss ([\d.]+)\s+va_acc ([\d.]+)%", body)
        if eps:
            last = max(eps, key=lambda kv: float(kv[1]))
            d["val_loss"], d["val_acc"] = float(last[0]), float(last[1]) / 100
        eps2 = re.findall(r"tr_acc ([\d.]+)%.*?va_acc ([\d.]+)%", body)
        if eps2:
            last = max(eps2, key=lambda kv: float(kv[1]))
            d["train_acc"] = float(last[0]) / 100
        d["failed"] = ("Killed" in body) or ("⚠️ 无 .pt" in body)
        out[name] = d
    return out


def build(pool="bc_pool", npz="data/bc_all.npz", log="logs/bc_pool_run.log"):
    npz = os.path.join(ROOT, npz)
    frames = frames_from_npz(npz) if os.path.exists(npz) else {}
    agg = parse_aggregate_log(os.path.join(ROOT, log))
    rows = []
    for pt in sorted(glob.glob(os.path.join(ROOT, pool, "*.pt"))):
        name = os.path.splitext(os.path.basename(pt))[0]
        jp = pt + ".json"
        d = json.load(open(jp, encoding="utf-8")) if os.path.exists(jp) else {}
        if not d:
            a = agg.get(name, {})
            d = dict(filter=dict(deck=(a.get("spec", "").split(":", 1)[1]
                                       if a.get("spec", "").startswith("deck:") else None),
                                 agent=(a.get("spec", "").split(":", 1)[1]
                                        if a.get("spec", "").startswith("agent:") else None)),
                     baseline=dict(val_acc=a.get("base_val_acc")),
                     best=dict(val_acc=a.get("val_acc"), train_acc=a.get("train_acc"),
                               val_loss=a.get("val_loss")),
                     _src="aggregate_log")
        f = d.get("filter") or {}
        spec = (f"deck:{f['deck']}" if f.get("deck") else
                f"agent:{f['agent']}" if f.get("agent") else "?")
        if spec in frames:
            n, nv = frames[spec]
        else:
            # `--only-agent Raihan` 是**子串**匹配（实际名字是 "Raihan Ramadistra"），
            # 所以这里也要按子串把命中的行加起来，否则帧数会显示 0。
            key = spec.split(":", 1)[-1]
            hits = [(v, frames[k]) for k, v in [("", spec)]  # placeholder
                    ] if False else [(k, frames[k]) for k in frames
                                     if k.startswith(spec.split(":", 1)[0] + ":")
                                     and key.lower() in k.lower()]
            n = sum(x[0] for _, x in hits) if hits else d.get("n_train")
            nv = sum(x[1] for _, x in hits) if hits else d.get("n_val")
        base = (d.get("baseline") or {}).get("val_acc")
        bst = (d.get("best") or {})
        bin_ = os.path.join(ROOT, pool, name + ".bin")
        sha = ""
        if os.path.exists(bin_):
            import hashlib
            sha = hashlib.sha256(open(bin_, "rb").read()).hexdigest()[:16]
        rows.append(dict(
            name=name, filter=spec,
            frames=n if n is not None else (d.get("n_train") or 0) + (d.get("n_val") or 0),
            val_frames=nv if nv is not None else (d.get("n_val") or 0),
            base_val_acc="" if base is None else f"{base:.3f}",
            val_acc="" if bst.get("val_acc") is None else f"{bst['val_acc']:.3f}",
            train_acc="" if bst.get("train_acc") is None else f"{bst['train_acc']:.3f}",
            val_loss="" if bst.get("val_loss") is None else f"{bst['val_loss']:.4f}",
            bin=os.path.basename(bin_) if os.path.exists(bin_) else "",
            bin_sha16=sha,
            date=__import__("time").strftime("%Y-%m-%d", __import__("time").localtime(
                os.path.getmtime(pt))),
            source=d.get("_src", "pt.json" if os.path.exists(jp) else "unknown"),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="bc_pool")
    ap.add_argument("--npz", default="data/bc_all.npz")
    ap.add_argument("--out", default=None, help="默认 <pool>/registry.tsv")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rows = build(a.pool, a.npz)
    if not rows:
        raise SystemExit("bc_pool 里没有 .pt")
    out = a.out or os.path.join(ROOT, a.pool, "registry.tsv")
    lines = ["\t".join(COLS)]
    for r in rows:
        lines.append("\t".join(str(r[c]) for c in COLS))
    txt = "\n".join(lines) + "\n"
    w = max(len(r["name"]) for r in rows) + 2
    print(f"{'name':<{w}}{'frames':>8}{'val':>8}{'base':>7}{'Δpp':>7}  filter")
    for r in rows:
        d = ""
        try:
            d = f"{(float(r['val_acc'])-float(r['base_val_acc']))*100:+.1f}"
        except (ValueError, KeyError):
            pass
        fv, fv_b = r["val_acc"] or "-", r["base_val_acc"] or "-"
        print(f"{r['name']:<{w}}{r['frames']:>8}{fv:>8}{fv_b:>7}{d:>7}  {r['filter']}")
    if a.check:
        print("\n(--check，未写入)")
        return
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"\n→ {out}（{len(rows)} 行）")


if __name__ == "__main__":
    main()
