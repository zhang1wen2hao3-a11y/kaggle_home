#!/usr/bin/env python3
"""**规则路由核心**：决定"这一手用哪个模型"。

## 定位

提交包里会有 N 个模型（上限实测 ~10 个：每个 +17.1 MiB / +22 MB RAM）。
本模块不训练、不评估，只负责**在线选择 + 上下文交接 + 兜底**，可被 main.py 直接 import。

## 三条硬规则（对应已实测的事实）

**① 冷启动一律用泛用模型。** 开局看不到对手任何牌，信息量为 0；识别不确定时也回退泛用
（`DECIDE_MIN_COVERAGE` / `DECIDE_MIN_MARGIN` 没过就不切）。

**② 只在回合边界切换。** 用 `obs["current"]["turn"]` + `yourIndex` 判定；最多切 `MAX_SWITCHES`
次（默认 2）。越早切越好——中途接手等于让新策略面对"别人下出来的局面"，漂移会变大。

**③ 上下文必须交接。** 编码行依赖 `InfoTracker`（对手已揭示手牌/牌库、我方奖品推断、残留效果），
实测：带历史 vs 不带历史，同一帧的编码行差 10~32 个位置（随对局推进变大）。
两种做法都支持：
  - `shared_tracker=True`：所有模型共用同一个 tracker（状态与策略无关 ⇒ 可行、零成本）
  - `shared_tracker=False`：每个模型独立 tracker，切换时用缓存的历史重放重建（1~4 ms）

## 恢复/兜底（宁可打弱，不能崩）

任何一步异常（分类器抛错、重建失败、时间不足、观测异常）都退回当前/泛用模型；
模型调用本身再异常则用合法随机。**绝不允许因为路由层而 forfeit 一局。**

## 用法（离线自检，不需要 GPU）

    ./tools/py tools/router_core.py --replay epsoide_miwa/episode-XXX.json --trace
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 判定阈值：覆盖率 = 对手已见牌中有多少能被这副牌解释；margin = 第一名与第二名的覆盖率差
DECIDE_MIN_COVERAGE = 0.80
DECIDE_MIN_MARGIN = 0.15
DECIDE_MIN_CARDS = 3          # 至少看到这么多张不同牌才尝试判定
MAX_SWITCHES = 2


def deck_hash(ids):
    return hashlib.sha256(",".join(str(x) for x in sorted(ids)).encode()).hexdigest()[:8]


def load_families(extra_dirs=()):
    """卡组 → (family, 卡表集合)。

    family 取池内文件名的原型前缀（`dreepy_drakloak__3eaa` → `dreepy_drakloak`），
    这样同一原型的不同构筑会归并到一族——路由表按**族**配模型，不按单一构筑。
    """
    fam = collections.defaultdict(list)
    # 只用 pool_field（命名规范：<archetype>__<hash4>）。decks/top/ 的文件名是
    # top__<hash>，会被误并成一个巨大的 "top" 族，反而破坏分类。
    pats = ["decks/pool_field/*.csv"]
    for d in extra_dirs:
        pats.append(os.path.join(d, "*.csv"))
    for pat in pats:
        for f in glob.glob(os.path.join(ROOT, pat)):
            stem = os.path.basename(f)[:-4]
            family = stem.split("__")[0] if "__" in stem else stem
            try:
                ids = [int(x) for x in open(f) if x.strip()]
            except (OSError, ValueError):
                continue
            if len(ids) == 60:
                # ⚠️ 必须同时保留**多重集**（用于算 deck hash）与**去重集合**（用于覆盖率）。
                #    只用 frozenset 会把 11 张基本能量压成 1 张，hash 就对不上真实的 8 位码。
                fam[family].append(dict(hash=deck_hash(ids), cards=frozenset(ids)))
    return dict(fam)


class ArchetypeClassifier:
    """从对手**可见**卡集合推断原型族：覆盖率 + 领先幅度双阈值。"""

    def __init__(self, families):
        self.families = families
        self.visible = set()

    def observe(self, obs, opp_index):
        """把对手当前可见的卡收集起来（明置区）。"""
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
        """→ (family|None, coverage, margin, n_visible)"""
        v = self.visible
        if len(v) < DECIDE_MIN_CARDS:
            return None, 0.0, 0.0, len(v)
        scored = []
        for fam, decks in self.families.items():
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


class Router:
    """持有 N 个 SearchAgent，按规则切换。模型和奖励都不在这里——只做选择与交接。"""

    def __init__(self, agents, route_table, general_index=0,
                 shared_tracker=True, max_switches=MAX_SWITCHES, log=None):
        """
        agents: list[SearchAgent]，索引即模型编号
        route_table: {family: 模型编号}，缺省回退 general_index
        """
        self.agents = agents
        self.route = dict(route_table or {})
        self.general = int(general_index)
        self.shared = bool(shared_tracker)
        self.max_switches = int(max_switches)
        self.log = log if log is not None else (lambda *a, **k: None)
        self.history = []                 # 全部 obs（实测 ~1.74 MB/局）
        self.clf = None
        self._fam = None
        self.cur = self.general
        self.switches = 0
        self.last_turn = None
        if self.shared:
            base = agents[self.general].tracker
            for a in agents:
                a.tracker = base
        else:
            self.clf_needs = True

    # ---- 上下文交接 -------------------------------------------------
    def _rebuild(self, idx):
        """独立 tracker 模式：用缓存历史把该模型的上下文重放出来（1~4 ms）。"""
        ag = self.agents[idx]
        if self.shared or not self.history:
            return True
        try:
            from ptcg.tracker import InfoTracker
            ag.tracker = InfoTracker(ag.deck)
            fr = 0
            for (obs, seat) in self.history:
                c = dict(obs); c.setdefault("logs", []); c["player"] = seat
                ag.tracker.update(c)
                sel = c.get("select") or {}
                fr = fr + 1 if ag._forced(sel) is not None else 0
            ag.forced_run = fr
            return True
        except Exception as e:                      # noqa: BLE001
            self.log(f"[router] rebuild 失败 → 留在当前模型: {e}")
            return False

    # ---- 决策 -------------------------------------------------------
    def _is_switch_point(self, obs):
        cur = obs.get("current") or {}
        turn = cur.get("turn")
        if turn is None:
            return False
        if self.last_turn is None:
            self.last_turn = turn
            return False
        changed = turn != self.last_turn
        self.last_turn = turn
        return changed

    def agent(self, obs, seat):
        """返回本手该用的 SearchAgent；调用方负责 `ag.agent(obs)`。"""
        self.history.append((obs, seat))
        try:
            if obs.get("select") is None:            # 开局：返回卡组
                return self.agents[self.general]
            if self.clf is None:
                self.clf = ArchetypeClassifier(load_families())
            self.clf.observe(obs, 1 - seat)
            if self.switches >= self.max_switches or not self._is_switch_point(obs):
                return self.agents[self.cur]
            fam, cov, margin, n = self.clf.guess()
            self.log(f"[router] turn={obs['current'].get('turn')} 可见{n}张 "
                     f"→ {fam or '未判定'} (cov={cov:.2f} margin={margin:.2f})")
            tgt = self.route.get(fam, self.general) if fam else self.general
            if tgt != self.cur and self._rebuild(tgt):
                self.log(f"[router] 切换 {self.cur} → {tgt}（{fam}）")
                self.cur = tgt
                self.switches += 1
            return self.agents[self.cur]
        except Exception as e:                       # noqa: BLE001
            self.log(f"[router] 决策异常 → 泛用模型: {e}")
            return self.agents[self.general]


def _trace(path):
    """离线自检：打印每一手时分类器的判定（用来量"识别窗口"有多长）。"""
    import sys
    sys.path.insert(0, ROOT)
    from framework.replay import iter_decisions
    fam = load_families()
    d = json.loads(open(path).read())
    steps = d.get("steps") or []
    names = [a.get("Name", "?") for a in (d.get("info") or {}).get("Agents", [])]
    clf = ArchetypeClassifier(fam)
    print(f"回放 {os.path.basename(path)}  双方: {names}")
    first_hit = None
    for t, j, obs, act in iter_decisions(steps):
        clf.observe(obs, 1 - j)
        f, cov, margin, n = clf.guess()
        turn = (obs.get("current") or {}).get("turn")
        if f and first_hit is None:
            first_hit = (t, turn, f)
        if t % 9 == 0 or f:
            print(f"  t={t:3d} turn={turn} 可见={n:2d} → {str(f):28s} cov={cov:.2f} m={margin:.2f}")
        if first_hit and t > first_hit[0] + 12:
            break
    if first_hit:
        print(f"\n首次成功判定：第 {first_hit[0]} 个决策（turn {first_hit[1]}）→ {first_hit[2]}")
    else:
        print("\n整局都没能判定")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default="")
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args()
    if a.replay:
        _trace(a.replay)
    else:
        fam = load_families()
        print(f"卡组池 {sum(len(v) for v in fam.values())} 副 → {len(fam)} 个原型族")
        for k in sorted(fam, key=lambda x: -len(fam[x]))[:8]:
            print(f"  {k:40s} {len(fam[k])} 副")
