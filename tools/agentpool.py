#!/usr/bin/env python3
"""按卡组哈希取得一个可调用的「我们端口 + 指定卡组」agent，带 staging 与加载缓存。

为什么需要：做决策级对比/蒸馏数据抽取时，回放里每个座位用的卡组都不同。端口的行为
依赖 `deck.csv`（tracker 要按自己的 60 张维护奖品/牌库状态），所以每遇到一副新的卡组
就得 staging 一个新目录。这个模块把这件事封装掉，避免各工具各写一份。

⚠️ 内存：每个 agent 实例约 300–400 MB（torch + 模型）。回放语料里可能有上百副不同卡组，
   所以缓存的是 staging **目录**（便宜），agent 实例只保留最近 `_LRU` 个。
"""
import hashlib
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from check_port_fidelity import load_agent, prepare   # noqa: E402

_STAGE_ROOT = "/tmp/agentpool"
_LRU = 3
_cache = {}


def deck_key(deck_ids):
    return hashlib.sha256(",".join(str(x) for x in sorted(deck_ids)).encode()).hexdigest()[:12]


def stage_dir(deck_ids, bundle="build/port_plain"):
    """staging 一个「端口代码 + 指定 deck.csv」目录（软链，幂等）。"""
    hh = deck_key(deck_ids)
    out = os.path.join(_STAGE_ROOT, hh)
    if not os.path.exists(os.path.join(out, "model.pt")):
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out, exist_ok=True)
        b = os.path.join(ROOT, bundle)
        for e in os.listdir(b):
            if e in ("model.pt", "deck.csv"):
                continue
            os.symlink(os.path.abspath(os.path.join(b, e)), os.path.join(out, e))
        os.symlink(os.path.join(b, "model.pt"), os.path.join(out, "model.pt"))
        with open(os.path.join(out, "deck.csv"), "w") as f:
            f.write("\n".join(str(x) for x in deck_ids) + "\n")
    return out


def get_agent(deck_ids, bundle="build/port_plain"):
    """取该卡组对应的 agent 可调用对象（LRU 缓存）。"""
    hh = deck_key(deck_ids)
    if hh in _cache:
        _cache[hh]["used"] = True
        return _cache[hh]["fn"]
    fn = load_agent(prepare(stage_dir(deck_ids, bundle)))
    # 简单的 LRU：超了就淘汰最久未用的
    if len(_cache) >= _LRU:
        victim = min(_cache, key=lambda k: _cache[k]["n"])
        del _cache[victim]
    _cache[hh] = {"fn": fn, "n": 0, "used": True}
    for k, v in _cache.items():
        v["n"] += 1
    return fn
