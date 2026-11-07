"""多模型路由提交件用的 main.py（自包含，不 import tools/）。

为什么要单独一份：Kaggle 用 `exec(compile(main.py))` 加载，**没有 __file__**，也不能
import 仓库里的 `tools/router_core.py`。所以把「识别 + 切换 + 上下文交接」全部内联。

设计（每条都来自实测）：

1. **冷启动一律泛用模型**：开局对手明置区为空，识别没有信息量。
2. **只在回合边界切换**：中途接手会让新策略面对"别人下出来的局面"；且最多切 MAX_SWITCHES 次。
3. **上下文交接用共享 tracker**：`InfoTracker` 只依赖观测（不依赖策略），实测可以安全共享，
   零成本；切换时把 `forced_run` 也同步过去。
4. **记录识别窗口**：每次判定、每次切换、以及整局的首次识别回合都打到 stderr，
   评测日志里能直接看到「第几手识别出什么、第几手切到了谁」。
   拉回放复现时用同一套规则再跑一遍，就能核对"规则模型当时判对了没有"。
5. **任何异常都退回泛用模型**，绝不让路由层 forfeit。
"""
import json
import os
import random
import sys
import time

MAX_SWITCHES = 1
MIN_CARDS = 3              # 至少看到这么多张不同牌才判定
MIN_COVERAGE = 0.80        # 第一名覆盖率
MIN_MARGIN = 0.15          # 与第二名覆盖率之差


def _agent_dir():
    try:
        d = os.path.dirname(os.path.abspath(__file__))  # noqa: F821
        if os.path.exists(os.path.join(d, "deck.csv")):
            return d
    except NameError:
        pass
    if os.path.exists("deck.csv"):
        return os.getcwd()
    return "/kaggle_simulations/agent"


AGENT_DIR = _agent_dir()
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

import torch

torch.set_num_threads(2)

from ptcg.rl.search_agent import SearchAgent
from ptcg.tracker import InfoTracker

HEADS = 8
_ROUTE_PATH = os.path.join(AGENT_DIR, "routes.json")
with open(_ROUTE_PATH, encoding="utf-8") as _f:
    _R = json.load(_f)
ROUTE = {k: int(v) for k, v in (_R.get("route") or {}).items()}   # family -> 模型序号
FAMILY_CARDS = {k: set(v) for k, v in (_R.get("family_cards") or {}).items()}
MODEL_NAMES = list(_R.get("models") or [])

_DECK = os.path.join(AGENT_DIR, "deck.csv")
AGENTS = []
for _m in MODEL_NAMES:
    AGENTS.append(SearchAgent(os.path.join(AGENT_DIR, "models", _m),
                              _DECK, cfg=None, seed=0, device="cpu",
                              data_dir=AGENT_DIR, heads=HEADS, temp=1.0))
DECK = list(AGENTS[0].deck)

# 共享 tracker：识别与上下文都与"用哪个模型"无关
for _a in AGENTS[1:]:
    _a.tracker = AGENTS[0].tracker

ST = {"cur": 0, "switches": 0, "visible": set(), "turn": None, "decisions": 0,
      "first_turn": None, "first_family": None, "switch_turn": None,
      "switch_to": None, "tried": 0}


def _log(msg):
    print(f"[router] {msg}", file=sys.stderr, flush=True)


def _obs_ids(obs, seat):
    """对手明置区（前场/后备/弃牌/竞技场）里出现过的卡 id。"""
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    oi = 1 - seat
    if oi >= len(players):
        return set()
    p = players[oi]
    out = set()
    for zone in ("active", "bench", "discard"):
        for c in (p.get(zone) or []):
            if c and c.get("id"):
                out.add(int(c["id"]))
    for c in (cur.get("stadium") or []):
        if c and c.get("id"):
            out.add(int(c["id"]))
    return out


def _guess():
    v = ST["visible"]
    if len(v) < MIN_CARDS:
        return None, 0.0, 0.0
    scored = []
    for fam, cards in FAMILY_CARDS.items():
        scored.append((len(v & cards) / len(v), fam))
    scored.sort(reverse=True)
    top_cov, top_fam = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    ok = top_cov >= MIN_COVERAGE and (top_cov - second) >= MIN_MARGIN
    return (top_fam if ok else None), top_cov, (top_cov - second)


def _reset_episode():
    ST.update(cur=0, switches=0, visible=set(), turn=None, decisions=0,
              first_turn=None, first_family=None, switch_turn=None,
              switch_to=None, tried=0)
    base = AGENTS[0]
    base.tracker = InfoTracker(base.deck)
    base.forced_run = 0
    for a in AGENTS[1:]:
        a.tracker = base.tracker
        a.forced_run = 0


def _share_tracker(src_idx):
    """把当前模型的上下文交给目标模型（共享 tracker + forced_run 同步）。"""
    src = AGENTS[src_idx]
    for a in AGENTS:
        a.tracker = src.tracker
        a.forced_run = src.forced_run


def _decide(obs, seat):
    """返回本手该用的模型序号；需要时才切换。"""
    cur = obs.get("current") or {}
    turn = cur.get("turn")
    if turn is not None:
        if ST["turn"] is None:
            ST["turn"] = turn
        elif turn != ST["turn"]:
            ST["turn"] = turn           # 回合边界：唯一允许切换的时机
        else:
            return ST["cur"]
    else:
        return ST["cur"]
    ST["visible"] |= _obs_ids(obs, seat)
    fam, cov, margin = _guess()
    if fam and ST["first_turn"] is None:
        ST["first_turn"], ST["first_family"] = turn, fam
        _log(f"首次识别 turn={turn} → {fam} (cov={cov:.2f} margin={margin:.2f} "
             f"可见{len(ST['visible'])}张 第{ST['decisions']}手)")
    if ST["switches"] >= MAX_SWITCHES or not fam:
        return ST["cur"]
    tgt = ROUTE.get(fam, 0)
    if tgt != ST["cur"]:
        _share_tracker(ST["cur"])
        _log(f"切换 turn={turn} {ST['cur']}→{tgt}（{fam}）")
        ST["cur"] = tgt
        ST["switches"] += 1
        ST["switch_turn"] = turn
        ST["switch_to"] = MODEL_NAMES[tgt]
    return ST["cur"]


def _legal_random(sel):
    n = len(sel["option"])
    k = random.randint(sel["minCount"], min(sel["maxCount"], n))
    return sorted(random.sample(range(n), k))


def _episode_summary():
    _log(f"SUMMARY first_turn={ST['first_turn']} first_family={ST['first_family']} "
         f"switches={ST['switches']} switch_turn={ST['switch_turn']} "
         f"switch_to={ST['switch_to']} decisions={ST['decisions']}")


def agent(obs_dict: dict) -> list[int]:
    ST["decisions"] += 1
    if obs_dict.get("select") is None:                 # 开局：交卡组
        _reset_episode()
        _log(f"episode start deck={len(DECK)} 张 models={MODEL_NAMES} route={ROUTE}")
        return list(AGENTS[0].agent(obs_dict))
    try:
        rd = (obs_dict.get("current") or {}).get("round")
        if rd != ST.get("round"):
            if ST.get("round") is not None:            # Bo3 的第 2/3 局
                _reset_episode()
                _log(f"new round={rd} 重置 tracker")
            ST["round"] = rd
        seat = int((obs_dict.get("current") or {}).get("yourIndex") or 0)
        idx = _decide(obs_dict, seat)
        return list(AGENTS[idx].agent(obs_dict))
    except Exception as e:                             # noqa: BLE001
        _log(f"决策异常 → 泛用模型: {e}")
        try:
            return list(AGENTS[0].agent(obs_dict))
        except Exception:
            try:
                return _legal_random(obs_dict["select"])
            except Exception:
                return [0] if obs_dict["select"].get("option") else []


try:
    import atexit
    atexit.register(_episode_summary)
except Exception:
    pass
