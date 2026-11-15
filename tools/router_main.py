"""Kaggle-format **路由版** agent：一个提交包里放 N 个模型，按对手原型规则切换。

包内布局（由 tools/make_sub_router.sh 生成）：

    main.py            本文件
    route.json         {"general": 0, "routes": {"<原型族>": <模型号>}, "max_switches": 2}
    families.json      原型族 → [[deck_hash, [卡的 id...]], ...]（离线打包时算好，运行时不再 glob）
    models/model_0.pt  泛用模型（general）
    models/model_k.pt  专家模型（按原型族路由）
    deck.csv / ptcg/ / cg/ / data/ / EN_Card_Data.csv / r2e_manifest.json   与单模型包一致

三条硬规则（来自 tools/router_core.py 的实测结论）：
  ① 冷启动/识别不确定 → 一律泛用模型；
  ② 只在回合边界切换，最多 max_switches 次；
  ③ 所有模型共用同一个 InfoTracker（状态与策略无关），零成本交接。

安全阀：任何异常退回当前/泛用模型；模型再异常给合法随机答案 —— 绝不让路由层 forfeit。
"""
import json
import os
import random
import sys
import time

_LOAD_T0 = time.time()


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

HEADS = 8   # 打包时由 make_sub_router.sh 按模型覆盖（HEADS=<n>）

# ── 路由表与原型族表 ─────────────────────────────────────────────
with open(os.path.join(AGENT_DIR, "route.json"), encoding="utf-8") as f:
    ROUTE_CFG = json.load(f)
with open(os.path.join(AGENT_DIR, "families.json"), encoding="utf-8") as f:
    _FAM_RAW = json.load(f)
FAMILIES = {fam: [dict(hash=h, cards=frozenset(c)) for h, c in decks] for fam, decks in _FAM_RAW.items()}

DECIDE_MIN_COVERAGE = 0.80
DECIDE_MIN_MARGIN = 0.15
DECIDE_MIN_CARDS = 3


class Classifier:
    def __init__(self):
        self.visible = set()

    def observe(self, obs, opp_index):
        cur = obs.get("current") or {}
        players = cur.get("players") or []
        if opp_index >= len(players):
            return
        p = players[opp_index]
        for zone in ("active", "bench", "discard"):
            for c in (p.get(zone) or []):
                if c and c.get("id"):
                    self.visible.add(int(c["id"]))
        for c in (cur.get("stadium") or []):
            if c and c.get("id"):
                self.visible.add(int(c["id"]))

    def guess(self):
        v = self.visible
        if len(v) < DECIDE_MIN_CARDS:
            return None, 0.0, 0.0, len(v)
        scored = []
        for fam, decks in FAMILIES.items():
            best = 0.0
            for d in decks:
                best = max(best, len(v & d["cards"]) / len(v))
            scored.append((best, fam))
        scored.sort(reverse=True)
        top_cov, top_fam = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        margin = top_cov - second
        ok = top_cov >= DECIDE_MIN_COVERAGE and margin >= DECIDE_MIN_MARGIN
        return (top_fam if ok else None), top_cov, margin, len(v)


# ── 载入 N 个模型 ────────────────────────────────────────────────
DECK_PATH = os.path.join(AGENT_DIR, "deck.csv")
MODELS = []
GENERAL = int(ROUTE_CFG.get("general", 0))
ROUTES = {str(k): int(v) for k, v in (ROUTE_CFG.get("routes") or {}).items()}
MAX_SWITCHES = int(ROUTE_CFG.get("max_switches", 2))
for rel in ROUTE_CFG["models"]:
    p = os.path.join(AGENT_DIR, rel)
    MODELS.append(SearchAgent(p, DECK_PATH, cfg=None, seed=0, device="cpu",
                              data_dir=AGENT_DIR, heads=HEADS, temp=0.0))
# 共用 general 的 tracker（规则③）
_TRACKER = MODELS[GENERAL].tracker
for a in MODELS:
    a.tracker = _TRACKER
DECK = list(MODELS[GENERAL].deck)

RT = {"cur": GENERAL, "switches": 0, "last_turn": None, "clf": None, "round": None}
ROT = {"est": 600.0, "boot": False}
REJ = {"sig": None, "tried": []}
LAST = {"sig": None, "logs": None}


def _legal_random(sel):
    n = len(sel["option"])
    k = random.randint(sel["minCount"], min(sel["maxCount"], n))
    return sorted(random.sample(range(n), k))


def _sel_sig(obs):
    sel = obs["select"]
    cur = obs["current"]
    return (cur.get("turn"), cur.get("yourIndex"), sel.get("type"),
            sel.get("context"), len(sel.get("option") or []),
            sel.get("minCount"), sel.get("maxCount"),
            json.dumps(sel.get("option"), sort_keys=True)[:600])


def _recover(sel):
    n = len(sel["option"])
    lo = max(sel["minCount"], 1)
    for k in range(lo, min(sel["maxCount"], n) + 1):
        for start in range(n):
            cand = sorted((start + j) % n for j in range(k))
            if len(set(cand)) == k and cand not in REJ["tried"]:
                return cand
    return _legal_random(sel)


def _pick_model(obs):
    """返回本手该用的 SearchAgent；任何异常都回退泛用模型。"""
    seat = (obs.get("current") or {}).get("yourIndex", 0)
    try:
        if RT["clf"] is None:
            RT["clf"] = Classifier()
        RT["clf"].observe(obs, 1 - seat)
        turn = (obs.get("current") or {}).get("turn")
        if turn is None:
            return MODELS[RT["cur"]]
        if RT["last_turn"] is None:
            RT["last_turn"] = turn
            return MODELS[RT["cur"]]
        if turn == RT["last_turn"]:
            return MODELS[RT["cur"]]
        RT["last_turn"] = turn
        if RT["switches"] >= MAX_SWITCHES:
            return MODELS[RT["cur"]]
        fam, cov, margin, n = RT["clf"].guess()
        if os.environ.get("PTCG_ROUTER_DEBUG"):
            print(f"[router] turn={turn} 可见{n}张 → {fam or '未判定'} "
                  f"(cov={cov:.2f} m={margin:.2f})", file=sys.stderr)
        tgt = ROUTES.get(fam, GENERAL) if fam else GENERAL
        if tgt != RT["cur"]:
            RT["cur"] = tgt
            RT["switches"] += 1
        return MODELS[RT["cur"]]
    except Exception:
        return MODELS[GENERAL]


def agent(obs_dict: dict) -> list[int]:
    t0 = time.time()
    if obs_dict.get("select") is None:
        REJ["sig"], REJ["tried"] = None, []
        LAST["sig"], LAST["logs"] = None, None
        ROT["est"] = 600.0
        RT["cur"], RT["switches"], RT["last_turn"] = GENERAL, 0, None
        if RT["clf"] is not None:
            RT["clf"].visible.clear()
        if not ROT["boot"]:
            ROT["est"] -= time.time() - _LOAD_T0
            ROT["boot"] = True
        ans = MODELS[GENERAL].agent(obs_dict)
        ROT["est"] -= time.time() - t0
        return list(ans)

    sel = obs_dict["select"]
    rd = (obs_dict.get("current") or {}).get("round")
    if rd != RT["round"]:
        if RT["round"] is not None:
            _TRACKER.__init__(MODELS[GENERAL].deck)
            for a in MODELS:
                a.forced_run = 0
            LAST["sig"], LAST["logs"] = None, None
            REJ["sig"], REJ["tried"] = None, []
        RT["round"] = rd

    try:
        rot = obs_dict.get("remainingOverageTime")
        if rot is None:
            rot = ROT["est"]
        sig = _sel_sig(obs_dict)
        logs_j = json.dumps(obs_dict.get("logs") or [], sort_keys=True)
        represented = sig == LAST["sig"]
        advanced = represented and logs_j != LAST["logs"]
        LAST["sig"], LAST["logs"] = sig, logs_j
        if not represented or advanced:
            REJ["sig"], REJ["tried"] = sig, []
        ambiguous = represented and not advanced and bool(REJ["tried"])
        if rot < 30:
            ans = _recover(sel) if ambiguous else _legal_random(sel)
        elif not represented:
            ans = _pick_model(obs_dict).agent(obs_dict)
        else:
            ans = _pick_model(obs_dict).agent(obs_dict)
            if ambiguous and ans in REJ["tried"]:
                ans = _recover(sel)
        if sel.get("context") == 41:            # IS_FIRST: 先手
            idx = [i for i, o in enumerate(sel.get("option") or []) if o.get("type") == 1]
            if idx:
                ans = [idx[0]]
        REJ["tried"].append(ans)
        ROT["est"] -= time.time() - t0
        return ans
    except Exception:
        if os.environ.get("PTCG_DEBUG"):
            import traceback
            traceback.print_exc()
        try:
            return _legal_random(sel)
        except Exception:
            return [0] if sel.get("option") else []
