"""数据集读写与切分。

格式：JSONL，一行一个决策点。

    {"game": 3, "step": 12, "player": 1,
     "features": {...},                # 状态特征（V 用）
     "options": [{...}, {...}],        # 每个候选动作的特征（P 用）
     "action": [2],                    # 实际选了哪些下标
     "result": 0,                      # 0/1 = 胜者下标, 2 = 平局
     "reward": -1.0}                   # 从 player 视角看的回报

两条纪律：
    1. **按对局切分**（group split）。同一局的不同步骤高度相关，
       随机切会让验证集泄漏，把泛化能力估高。这里强制按 game 分组。
    2. **标签可以有隐藏信息，特征不能有**。特征来自 framework.features，
       那里只放公开信息。
"""
from __future__ import annotations

import json
from collections import defaultdict


# ---------------------------------------------------------------- 读写
def write_records(path: str, records: list[dict]) -> int:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def read_records(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_records(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------- 切分
def split_by_game(records: list[dict], val_ratio: float = 0.2,
                  seed: int = 0) -> tuple[list[dict], list[dict]]:
    """按【对局】切分训练/验证，杜绝同局泄漏。"""
    import random
    games = sorted({r["game"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(games)
    n_val = max(1, int(len(games) * val_ratio))
    val_games = set(games[:n_val])
    train = [r for r in records if r["game"] not in val_games]
    val = [r for r in records if r["game"] in val_games]
    return train, val


# ---------------------------------------------------------------- 目标构造
def to_value_dataset(records: list[dict], flip_on_loss: bool = True
                     ) -> tuple[list[dict], list[float]]:
    """给 V 用的 (特征, 标签)。

    标签 = 从"当前行动方"视角看的最终胜负（1=赢, 0=输, 0.5=平）。
    视角约定与 features.state_features 一致 —— 这是 V 能正确评估局面的前提。
    """
    X, y = [], []
    for r in records:
        res = r.get("result")
        if res is None:
            continue
        if res == 2:
            label = 0.5
        else:
            label = 1.0 if res == r["player"] else 0.0
        X.append(r["features"])
        y.append(label)
    _ = flip_on_loss
    return X, y


def to_rank_dataset(records: list[dict]) -> tuple[list[dict], list[float], list[int]]:
    """给 P / LTR 用的 (特征, 相关性标签, group)。

    group 是每个决策点的候选数 —— 这是 Learning-to-Rank 的关键输入，
    LGBMRanker 靠它知道"哪些行属于同一个 query"。

    标签来源：搜索给出的动作排序（软标签）或实际动作（硬标签）。
    这里先用**实际动作**做一版基线，等搜索跑通后替换成搜索分布。
    """
    X, y, groups = [], [], []
    for r in records:
        opts = r.get("options") or []
        if not opts:
            continue
        chosen = set(r.get("action") or [])
        for i, of in enumerate(opts):
            X.append(of)
            y.append(1.0 if i in chosen else 0.0)
        groups.append(len(opts))
    return X, y, groups


# ---------------------------------------------------------------- 统计
def describe(records: list[dict]) -> dict:
    games = defaultdict(int)
    for r in records:
        games[r["game"]] += 1
    results = defaultdict(int)
    for r in records:
        results[r.get("result")] += 1
    steps = list(games.values())
    return {
        "records": len(records),
        "games": len(games),
        "steps_per_game_avg": (sum(steps) / len(steps)) if steps else 0,
        "steps_per_game_max": max(steps) if steps else 0,
        "result_dist": dict(results),
        "feature_count": len(records[0]["features"]) if records else 0,
    }
