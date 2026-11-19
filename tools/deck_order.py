#!/usr/bin/env python3
"""**牌序的唯一来源与唯一写入口**（2026-11-20 用户要求：从"生成"这一步就把排序做掉）

用户在 `the-pockmon-battle-choice.csv` 上要求：行/列名字**统一排序** —— 数字在字母前，
内部按字符顺序。只在生成后去排序没用，因为下一轮重新生成又乱；所以把排序收进这个模块，
**所有读牌序的地方都走 `load()`，所有写牌序的地方都走 `save()`**。

排序规则：短名是 4 位十六进制（`0-9a-f`），ASCII 序本身就把数字排在字母前，
所以 `sorted()` 即可满足"数字>字母、内部按顺序"。为防止以后出现大小写混用，统一转小写后再排。

    ./tools/py tools/deck_order.py            # 打印当前牌序
    ./tools/py tools/deck_order.py check      # 校验文件是否已排序（未排序则非零退出）
"""
from __future__ import annotations
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "decks", "deck_order.txt")
HEADER = ("# 维护表牌序（唯一来源；build_from_master / build_in_table / build_winrate_table 都读它）\n"
          "# 排序规则（2026-11-20 用户要求）：统一按短名 ASCII 序 —— 数字在字母前，内部按字符顺序\n"
          "# ⚠️ 不要手改顺序；增删牌请用 tools/add_highscore_decks.py / tools/merge_similar_decks.py，\n"
          "#    它们都走 deck_order.save() 自动排序。\n")
_LEGACY = ["0904", "0bf5", "0d86", "2617", "26c6", "3eaa", "7e39", "9d6e", "db04", "e1d7", "e8ee", "f5f8"]


def sort_key(name: str) -> str:
    return name.strip().lower()


def normalize(ids):
    """去重 + 去空 + 统一小写 + 排序。"""
    return sorted({i.strip().lower() for i in ids if i and i.strip() and not i.startswith("#")}, key=sort_key)


def load():
    try:
        ids = [ln.strip() for ln in open(PATH, encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    except OSError:
        ids = []
    return normalize(ids) or sorted(_LEGACY, key=sort_key)


def save(ids):
    """写回牌序文件（自动排序）。返回写入后的列表。"""
    out = normalize(ids)
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "w", encoding="utf-8") as fh:
        fh.write(HEADER)
        for h in out:
            fh.write(h + "\n")
    return out


def main(argv=None):
    argv = argv or sys.argv[1:]
    ids = load()
    if argv[:1] == ["check"]:
        raw = [ln.strip() for ln in open(PATH, encoding="utf-8") if ln.strip() and not ln.startswith("#")]
        if raw != ids:
            print(f"✗ 牌序文件未按统一规则排序：{raw} → 应为 {ids}")
            return 1
        print(f"✓ 牌序已统一排序（{len(ids)} 副）")
        return 0
    print(f"牌序（{len(ids)} 副）：{' '.join(ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
