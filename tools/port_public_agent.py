#!/usr/bin/env python3
"""在本地复现 busyaprime 的 Playground 移植包（公开榜读数 1132.4）。

构成：
  1. Dipam Chakraborty 的第 3 名 agent（MIT，`submission/`，已按 sha256 校验 14/14）
  2. busyaprime 的补丁（座位→先后手、先手固定答 first、round 变化重置追踪器、表扩容、新卡行）
  3. 竞赛引擎 `cg/` 与卡表 `EN_Card_Data_R2_full.csv`（来自竞赛数据）
  4. 新卡行以字节增量形式应用（model.pt / data/embed_pca.npz）

最后用 busyaprime 记录在 notebook 里的 `V3_MANIFEST` 做逐文件 sha256 校验 ——
**PASS 就意味着我们本地构建的包与他真正提交的 56908917 字节一致**。

用法：./run.sh /tmp/port/build.py
"""
import base64
import hashlib
import io
import json
import lzma
import pathlib
import random
import re
import shutil
import sys
import tarfile

PROJ = pathlib.Path("/home/zhang/code/kaggle/the-pokemon-company")
DIPAM = pathlib.Path("/tmp/dipam_ptcg/submission")
CELLS = pathlib.Path("/tmp/port")
OUT = pathlib.Path("/tmp/port/out")
sha = lambda b: hashlib.sha256(b).hexdigest()      # noqa: E731
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- ① Dipam 的 14 个文件 + sha256 校验
ns = {"re": re, "hashlib": hashlib, "sha": sha}
src4 = (CELLS / "cell4.py").read_text()
DIPAM_SHA = eval(re.search(r"DIPAM_SHA = (\{.*?\n\})", src4, re.S).group(1), {})
# ⚠️ 只取 DIPAM_SHA 里列出的 14 个文件：仓库 submission/ 下还有 README.md，
#    真实提交包里没有它（他们的 manifest 明确排除），带上就与真实包不一致。
dipam = {p.relative_to(DIPAM).as_posix(): p.read_bytes()
         for p in DIPAM.rglob("*")
         if p.is_file() and p.relative_to(DIPAM).as_posix() in DIPAM_SHA}
bad = [k for k, v in DIPAM_SHA.items() if sha(dipam.get(k, b"")) != v]
assert not bad, f"Dipam 仓库文件不匹配: {bad}"
print(f"① Dipam 第 3 名 agent：{len(DIPAM_SHA)}/{len(DIPAM_SHA)} 个文件 sha256 通过")

# ---------------- ② busyaprime 的补丁
ns["dipam"] = dipam
exec((CELLS / "cell5.py").read_text(), ns)
patched, R2E_MANIFEST = ns["patched"], ns["R2E_MANIFEST"]
print(f"② 补丁应用成功：{len(patched)} 个文件 → {', '.join(sorted(patched))}")

# ---------------- ③ 新卡行的字节增量
exec((CELLS / "cell6.py").read_text(), ns)
MODEL_DELTA, NPZ_DELTA = ns["MODEL_DELTA"], ns["NPZ_DELTA"]


def apply_delta(payload_b64, sources):
    head, lit = lzma.decompress(base64.b64decode(payload_b64)).split(b"\n", 1)
    src, out, li = b"".join(sources), bytearray(), 0
    for op in json.loads(head):
        if op[0] == 0:
            out += lit[li:li + op[1]]; li += op[1]
        else:
            out += src[op[1]:op[1] + op[2]]
    return bytes(out)


model_pt = apply_delta(MODEL_DELTA, [dipam["model.pt"]])
embed_npz = apply_delta(NPZ_DELTA, [dipam["data/embed_pca.npz"], model_pt])
print(f"③ model.pt {len(model_pt):,} 字节 / embed_pca {len(embed_npz):,} 字节")

# ---------------- ④ 组装
AG = OUT / "agent"
shutil.rmtree(AG, ignore_errors=True)
for rel, b in dipam.items():
    (AG / rel).parent.mkdir(parents=True, exist_ok=True)
    (AG / rel).write_bytes(b)
for rel, text in patched.items():
    (AG / rel).parent.mkdir(parents=True, exist_ok=True)
    (AG / rel).write_bytes(text.encode())
(AG / "r2e_manifest.json").write_bytes(R2E_MANIFEST.encode())
(AG / "model.pt").write_bytes(model_pt)
(AG / "data" / "embed_pca.npz").write_bytes(embed_npz)
shutil.copytree(PROJ / "cg", AG / "cg", ignore=shutil.ignore_patterns("__pycache__"))
cards = (PROJ / "data" / "EN_Card_Data_R2_full.csv").read_bytes()
(AG / "EN_Card_Data.csv").write_bytes(cards)
(AG / "data" / "EN_Card_Data.csv").write_bytes(cards)
deck = [l.strip() for l in dipam["deck.csv"].decode().splitlines() if l.strip()]
assert len(deck) == 60, len(deck)
random.Random(33).shuffle(deck)
(AG / "deck.csv").write_bytes(("\n".join(deck) + "\n").encode())
print(f"④ 组装完成：{len(list(AG.rglob('*')))} 个条目")

# ---------------- ⑤ 打包
out = OUT / "submission.tar.gz"
with tarfile.open(out, "w:gz") as t:
    for p in sorted(AG.rglob("*")):
        if p.is_file():
            t.add(p, arcname=str(p.relative_to(AG)))
print(f"⑤ 已打包 {out}: {out.stat().st_size / 2**20:.1f} MiB")

# ---------------- ⑥ 用他们记录的 manifest 逐文件校验
src8 = (CELLS / "cell8.py").read_text()
V3 = eval(re.search(r"V3_MANIFEST = (\{.*?\n\})", src8, re.S).group(1), {})
with tarfile.open(out) as t:
    built = {m.name: t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
ok = 0
rows = []
for rel in sorted(set(V3) | set(built)):
    b = built.get(rel)
    if b is None:
        got = "missing"
    elif rel == "deck.csv":
        got = "multiset:" + sha("\n".join(sorted(
            l.strip() for l in b.decode().splitlines() if l.strip())).encode())
    else:
        got = sha(b)
    same = got == V3.get(rel)
    ok += same
    rows.append(f"  {'ok ' if same else 'BAD'} {rel:32s} {len(b) if b else 0:>11,}  {got[-16:]}")
print("\n⑥ 与真实提交 56908917 的 manifest 比对：")
print("\n".join(rows))
n = len(set(V3) | set(built))
verdict = "PASS" if ok == n == len(V3) else "FAIL"
print(f"\nMANIFEST {verdict} {ok}/{len(V3)}")
sys.exit(0 if verdict == "PASS" else 1)
