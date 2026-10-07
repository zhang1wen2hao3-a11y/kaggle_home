#!/usr/bin/env python3
"""本地对局框架：让两个 agent 打完整一局。

整个工程的地基 —— 没有可靠的本地对局，胜率就无从谈起。

设计要点：
  * 对手 agent 的签名与提交版完全一致：agent(obs_dict) -> list[int]。
    本地验证过的 agent 可直接提交，不存在"本地/线上两套代码"的偏差。
  * 非法返回值会被 sanitize（越界/重复/数量不符），而不是让引擎崩溃。
    Kaggle 线上对非法返回如何处理未知，本地先把这类 bug 变可见。
  * 座位可交换：同一对 agent 跑两次（互换先手），消除先后手偏差。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

MAX_STEPS = 20000
DECK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")


def load_deck(path: str = DECK_PATH) -> list[int]:
    with open(path) as f:
        deck = [int(x) for x in f.read().split() if x.strip()]
    if len(deck) != 60:
        raise ValueError(f"卡组必须是 60 张，当前 {len(deck)} 张")
    return deck


def sanitize(sel, select_data) -> tuple[list[int], str | None]:
    """把 agent 的返回值修正成引擎能接受的合法选择。

    返回 (修正后的选择, 问题描述或 None)。
    """
    opts = select_data["option"]
    n = len(opts)
    lo, hi = select_data["minCount"], min(select_data["maxCount"], n)
    note = None

    if not isinstance(sel, list) or not all(isinstance(i, int) for i in sel):
        return list(range(lo)), "not list[int]"
    if len(set(sel)) != len(sel):
        sel = sorted(set(sel))
        note = "duplicates"
    bad = [i for i in sel if i < 0 or i >= n]
    if bad:
        sel = [i for i in sel if 0 <= i < n]
        note = "out of range"
    if len(sel) < lo:
        sel = sorted(set(sel) | set(range(n)))[:lo]
        note = "too few"
    elif len(sel) > hi:
        sel = sel[:hi]
        note = "too many"
    return sel, note


def play_match(agent0, agent1, deck0=None, deck1=None, max_steps=MAX_STEPS):
    """打完整一局。

    Returns:
        dict: {result, winner, steps, error, warnings}
            result: 0/1 = 该索引玩家获胜, 2 = 平局, None = 未正常结束
    """
    deck0 = deck0 or load_deck()
    deck1 = deck1 or load_deck()
    agents = {0: agent0, 1: agent1}
    warnings = []

    obs, sd = battle_start(deck0, deck1)
    if obs is None:
        return {"result": None, "winner": None, "steps": 0,
                "error": f"开局失败 errorPlayer={sd.errorPlayer} errorType={sd.errorType}",
                "warnings": warnings}

    steps = 0
    result = None
    try:
        while steps < max_steps:
            cur = obs.get("current")
            sel_data = obs.get("select")
            if cur is None or sel_data is None:
                break
            result = cur.get("result", -1)
            if result != -1:
                break

            me = cur["yourIndex"]
            try:
                raw = agents[me](obs)
            except Exception as e:  # agent 崩了视作该方失利
                return {"result": 1 - me, "winner": 1 - me, "steps": steps,
                        "error": f"agent[{me}] 抛异常: {type(e).__name__}: {e}",
                        "warnings": warnings}

            choice, note = sanitize(raw, sel_data)
            if note:
                warnings.append(f"step {steps} player {me}: {note}")
            obs = battle_select(choice)
            steps += 1
    finally:
        battle_finish()

    if result == -1 or result is None:
        return {"result": None, "winner": None, "steps": steps,
                "error": f"未在 {max_steps} 步内结束", "warnings": warnings}
    return {"result": result, "winner": None if result == 2 else result,
            "steps": steps, "error": None, "warnings": warnings}
