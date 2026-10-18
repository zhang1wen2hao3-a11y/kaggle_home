"""卡组表示、合法性校验、遗传算子。

## 合法性规则（已实测确认，见 tests/）
    len(deck) == 60
    同名卡 <= 4 张，**基础能量豁免**
    至少 1 张基础宝可梦
    ACE SPEC 全场 <= 1 张（不同名也算超）
    所有 cardId 必须在卡池里

引擎的报错码：2=同名超限 / 3=无基础宝可梦 / 4=ACE SPEC 超限。

## 为什么必须校验能量匹配

之前手工造卡组时踩过坑：给 Mega Dragonite ex（招式要 WATER+LIGHTNING）
只配了 WATER 能量，结果**招式永远打不出来**，胜率 0.04。
所以 random_deck 会**先读招式的能量需求再配能量**，避免造出"能开局但打不动"的废组。
"""
from __future__ import annotations

import random
from collections import Counter

# EnergyType 枚举值（与 cg.api 对齐，避免框架层强依赖引擎）
ENERGY_COLORLESS = 0


class CardPool:
    """卡池索引：按用途分类，并提供进化链/能量需求查询。"""

    def __init__(self, load_engine: bool = True):
        self.by_id: dict[int, object] = {}
        self.basic_pokemon: list[int] = []
        self.stage1: list[int] = []
        self.stage2: list[int] = []
        self.basic_energy: list[int] = []
        self.special_energy: list[int] = []
        self.items: list[int] = []
        self.supporters: list[int] = []
        self.stadiums: list[int] = []
        self.tools: list[int] = []
        self.ace_spec: set[int] = set()
        self._name_to_ids: dict[str, list[int]] = {}
        self._energy_card_by_type: dict[int, int] = {}
        self._base_cache: dict[int, str | None] = {}
        self._attack_cache: dict[int, object] = {}
        if load_engine:
            self._load()

    # ---------------------------------------------------------- 加载
    def _load(self) -> None:
        from cg.api import all_card_data, all_attack
        for c in all_card_data():
            self.by_id[c.cardId] = c
        for a in all_attack():
            self._attack_cache[a.attackId] = a

        for cid, c in self.by_id.items():
            ct = int(c.cardType)
            self._name_to_ids.setdefault(c.name, []).append(cid)
            if getattr(c, "aceSpec", False):
                self.ace_spec.add(cid)
            if ct == 0:                       # 宝可梦
                if getattr(c, "basic", False):
                    self.basic_pokemon.append(cid)
                elif getattr(c, "stage1", False):
                    self.stage1.append(cid)
                elif getattr(c, "stage2", False):
                    self.stage2.append(cid)
            elif ct == 5:                     # 基础能量
                self.basic_energy.append(cid)
                et = int(c.energyType)
                self._energy_card_by_type.setdefault(et, cid)
            elif ct == 6:
                self.special_energy.append(cid)
            elif ct == 1:
                self.items.append(cid)
            elif ct == 2:
                self.tools.append(cid)
            elif ct == 3:
                self.supporters.append(cid)
            elif ct == 4:
                self.stadiums.append(cid)

        self.basic_pokemon.sort()
        self.basic_energy.sort()
        self.trainers = sorted(self.items + self.supporters + self.stadiums + self.tools)
        self.all_ids = sorted(self.by_id)
        self._build_power()

    # ---------------------------------------------------------- 强度先验
    def _build_power(self) -> None:
        """给每张宝可梦算一个粗粒度"强度分" = HP + 最高招式伤害。

        动机：卡池里强卡和弱卡差距极大。实测随机卡组对样例卡组只有 ~0.05，
        因为随机抽到的攻击手（如 Tangrowth）打不过 Mega Abomasnow ex（350HP/200伤）。
        所以随机生成时应当**偏向强卡**，否则 GA 从极低起点开始爬，浪费大量代。
        """
        self._power: dict[int, float] = {}
        for cid, c in self.by_id.items():
            if int(c.cardType) != 0:
                continue
            best_dmg = 0
            for aid in (getattr(c, "attacks", None) or []):
                atk = self._attack_cache.get(aid)
                if atk is not None:
                    best_dmg = max(best_dmg, int(atk.damage or 0))
            self._power[cid] = float((c.hp or 0) + best_dmg)

        ranked = sorted((self._power.get(c, 0.0), c) for c in self.basic_pokemon)
        ranked.sort(key=lambda x: -x[0])
        self.pokemon_by_power = [c for _, c in ranked]
        # 只保留有进化潜力或本身够强的基础宝可梦，作为"强卡池"
        n_top = max(30, len(self.pokemon_by_power) // 4)
        self.strong_basics = self.pokemon_by_power[:n_top]

    def power(self, cid: int) -> float:
        return self._power.get(cid, 0.0)

    # ---------------------------------------------------------- 查询
    def is_basic_energy(self, cid: int) -> bool:
        c = self.by_id.get(cid)
        return bool(c is not None and int(c.cardType) == 5)

    def is_basic_pokemon(self, cid: int) -> bool:
        c = self.by_id.get(cid)
        return bool(c is not None and int(c.cardType) == 0 and getattr(c, "basic", False))

    def evolution_chain(self, cid: int) -> list[int]:
        """返回从基础到最高阶的进化链（含自身）。

        ⚠️ 注意：`evolvesFrom` 存的是**名字**，而同一名字可能对应多张不同 cardId
        （不同扩展包的同名卡）。所以正向按名字找"下一阶"会挑到错误的卡
        —— 实测 Snover(722) 会走到 Abomasnow(419)，而卡组里其实是
        Mega Abomasnow ex(723)。因此这里改为**沿名字回溯找基础形态**，
        再收集所有以该名字为根的同族卡。
        """
        base = self.base_name(cid)
        if base is None:
            return [cid]
        family = [x for x, c in self.by_id.items()
                  if int(c.cardType) == 0 and self.base_name(x) == base]
        # 按"进化深度"排序，保证基础在前的顺序稳定
        family.sort(key=lambda x: (self.evolve_depth(x), x))
        return family if family else [cid]

    def base_name(self, cid: int) -> str | None:
        """沿 evolvesFrom 一路回溯到基础形态的名字（带缓存）。"""
        if cid in self._base_cache:
            return self._base_cache[cid]
        cur_id = cid
        seen: set[int] = set()
        result = None
        while cur_id is not None and cur_id not in seen:
            seen.add(cur_id)
            cur = self.by_id.get(cur_id)
            if cur is None:
                break
            prev = getattr(cur, "evolvesFrom", None)
            if not prev:
                result = getattr(cur, "name", None)
                break
            ids = self._name_to_ids.get(prev)
            if not ids:
                result = prev
                break
            cur_id = ids[0]
        if result is None:
            cur = self.by_id.get(cur_id) if cur_id is not None else None
            result = getattr(cur, "name", None) if cur else None
        self._base_cache[cid] = result
        return result

    def evolve_depth(self, cid: int) -> int:
        d = 0
        cur_id = cid
        seen: set[int] = set()
        while cur_id is not None and cur_id not in seen:
            seen.add(cur_id)
            cur = self.by_id.get(cur_id)
            if cur is None or not getattr(cur, "evolvesFrom", None):
                break
            ids = self._name_to_ids.get(cur.evolvesFrom)
            if not ids:
                break
            cur_id = ids[0]
            d += 1
        return d

    def line_in_deck(self, deck: list[int], cid: int) -> set[int]:
        """卡组里与 cid 同族的全部 cardId（按族谱名判定，不受同名不同 ID 影响）。"""
        base = self.base_name(cid)
        if base is None:
            return {cid}
        return {c for c in deck
                if c in self.by_id and int(self.by_id[c].cardType) == 0
                and self.base_name(c) == base}

    def energy_types_needed(self, cid: int) -> set[int]:
        """该卡所有招式需要的能量属性（忽略无色，任何能量都能付）。"""
        need: set[int] = set()
        c = self.by_id.get(cid)
        if c is None:
            return need
        for aid in (getattr(c, "attacks", None) or []):
            atk = self._attack_cache.get(aid)
            if atk is None:
                continue
            for e in (atk.energies or []):
                ei = int(e)
                if ei != ENERGY_COLORLESS:
                    need.add(ei)
        return need

    def energy_card_for_type(self, etype: int) -> int | None:
        return self._energy_card_by_type.get(etype)

    def deck_from_file(self, path: str) -> list[int]:
        with open(path) as f:
            return [int(x) for x in f.read().split() if x.strip()][:60]

    # ---------------------------------------------------------- 摘要
    def describe(self, deck: list[int], top: int = 6) -> str:
        cnt = Counter(deck)
        parts = []
        for cid, n in cnt.most_common(top):
            c = self.by_id.get(cid)
            parts.append(f"{getattr(c, 'name', cid)}x{n}")
        return ", ".join(parts)

    def deck_stats(self, deck: list[int]) -> dict:
        cnt = Counter(deck)
        bp = sum(1 for cid in cnt if self.is_basic_pokemon(cid))
        be = sum(n for cid, n in cnt.items() if self.is_basic_energy(cid))
        pk = sum(n for cid, n in cnt.items()
                 if cid in self.by_id and int(self.by_id[cid].cardType) == 0)
        return {"basic_pokemon_kinds": bp, "basic_energy": be,
                "pokemon": pk, "trainers": 60 - pk - be,
                "unique": len(cnt)}



def energy_cards_for(pool: CardPool, deck: list[int]) -> list[int]:
    """为一副卡组里【所有】攻击手挑选能量卡。

    ⚠️ 必须是"所有攻击手"，不能只看新换上的那个。
       实测踩过：变异出 Kyogre(水) + Drampa(恶) 却只配了恶能量，
       Kyogre 的招式永远打不出来，整副牌废掉一半。
    """
    need: set[int] = set()
    for cid in set(deck):
        c = pool.by_id.get(cid)
        if c is not None and int(c.cardType) == 0 and (getattr(c, "attacks", None) or []):
            need |= pool.energy_types_needed(cid)
    cards = [pool.energy_card_for_type(t) for t in need]
    cards = [c for c in cards if c is not None]
    if not cards:
        cards = [pool.basic_energy[0]] if pool.basic_energy else []
    return cards


# ============================================================ 合法性
def _count_by_name(pool: CardPool, deck: list[int]) -> Counter:
    """按【卡名】计数 —— PTCG 的 4 张上限是按名字算的，不是按 cardId。

    ⚠️ 这是实测出来的关键规则：卡池里同名不同卡（不同扩展包）很多，
       两个不同 ID 各放 4 张 = 同名 8 张，引擎直接以 errorType=2 拒绝。
       按 cardId 计数会误判为合法，然后整局游戏开不起来 —— 表现是"胜率 0.000"。
    """
    c: Counter = Counter()
    for cid in deck:
        card = pool.by_id.get(cid)
        if card is not None:
            c[card.name] += 1
    return c


def is_legal(pool: CardPool, deck: list[int]) -> tuple[bool, str]:
    if len(deck) != 60:
        return False, f"必须 60 张（当前 {len(deck)}）"
    for cid in deck:
        if cid not in pool.by_id:
            return False, f"未知 cardId={cid}"

    # 4 张上限：按名字算，基础能量豁免
    by_name: Counter = Counter()
    for cid in deck:
        card = pool.by_id[cid]
        if pool.is_basic_energy(cid):
            continue
        by_name[card.name] += 1
    for name, n in by_name.items():
        if n > 4:
            return False, f"「{name}」同名 {n} 张（上限 4，按卡名计）"

    if not any(pool.is_basic_pokemon(cid) for cid in set(deck)):
        return False, "没有基础宝可梦"

    ace = [cid for cid in set(deck) if cid in pool.ace_spec]
    ace_count = sum(deck.count(c) for c in ace)
    if ace_count > 1:
        return False, f"ACE SPEC 共 {ace_count} 张（上限 1）"
    return True, ""


def repair(pool: CardPool, deck: list[int], rng: random.Random,
           fill: list[int] | None = None) -> list[int]:
    """把任意卡组修成合法。

    按【卡名】限 4 张；基础能量不限；ACE SPEC 全场最多 1 张；
    必须含基础宝可梦；不足 60 张用基础能量补齐。
    """
    cnt_id: Counter = Counter()
    by_name: Counter = Counter()
    ace_used = 0

    for cid in deck:
        card = pool.by_id.get(cid)
        if card is None:
            continue
        if cid in pool.ace_spec:
            if ace_used >= 1:
                continue
            ace_used += 1
            cnt_id[cid] += 1
            by_name[card.name] += 1
        elif pool.is_basic_energy(cid):
            cnt_id[cid] += 1                 # 基础能量不限张数
        else:
            if by_name[card.name] >= 4:
                continue
            by_name[card.name] += 1
            cnt_id[cid] += 1
        if sum(cnt_id.values()) >= 60:
            break

    if not any(pool.is_basic_pokemon(cid) for cid in cnt_id):
        # 挑一个和已有名字不冲突的基础宝可梦
        cands = [c for c in pool.strong_basics
                 if by_name.get(pool.by_id[c].name, 0) < 4]
        cid = rng.choice(cands or pool.basic_pokemon)
        cnt_id[cid] = max(cnt_id.get(cid, 0), 1)

    out = [cid for cid, n in cnt_id.items() for _ in range(n)]
    # fill 允许调用方指定补牌池（例如只补指定属性的能量）。
    # 不指定时用全部基础能量 —— 但那会破坏属性一致性，导致卡组里塞满用不上的能量。
    fill = fill or (pool.basic_energy or pool.all_ids)
    while len(out) < 60:
        out.append(rng.choice(fill))
    while len(out) > 60:
        out.pop()
    return out


# ============================================================ 遗传算子
def random_deck(pool: CardPool, rng: random.Random,
                n_lines: int | None = None, n_trainer_kinds: int = 8,
                power_bias: bool = True) -> list[int]:
    """生成一副【可打、且不太弱】的随机卡组。

    关键两点：
      1. 先选攻击手，再按它的招式能量需求配能量
         —— 避免造出"能开局但招式永远打不出来"的废组（踩过这个坑）
      2. power_bias=True 时从**强卡池**里选攻击手
         —— 否则随机抽到的弱卡会被样例卡组碾压，GA 起点过低
    """
    n_lines = n_lines or rng.choice([1, 1, 2])
    base_pool = pool.strong_basics if power_bias else pool.basic_pokemon
    # 1) 选 1~2 条含进化链的基础宝可梦（优先强卡）
    basics_with_evo = [cid for cid in base_pool if len(pool.evolution_chain(cid)) > 1]
    can_use = basics_with_evo or base_pool or pool.basic_pokemon
    k = min(n_lines, len(can_use))
    line_starts = rng.sample(can_use, k=k)

    deck: list[int] = []
    attackers: list[int] = []
    for b in line_starts:
        chain = pool.evolution_chain(b)
        per = 4 if len(chain) <= 2 else 3
        for cid in chain:
            deck += [cid] * per
        attackers.append(chain[-1])

    # 2) 按【全部攻击手】的能量需求挑基本能量
    energy_cards = energy_cards_for(pool, deck) or list(pool.basic_energy)

    # 3) 配比：宝可梦 ~14~18 / 训练师 ~18~24 / 能量 ~20~26
    target_energy = rng.randint(18, 26)
    trainer_budget = 60 - len(deck) - target_energy
    if trainer_budget > 0:
        kinds = rng.sample(pool.trainers, k=min(n_trainer_kinds, len(pool.trainers)))
        # ACE SPEC 最多一张
        picked_ace = False
        for cid in kinds:
            if cid in pool.ace_spec:
                if picked_ace:
                    continue
                deck += [cid] * 1
                picked_ace = True
            else:
                n = rng.randint(2, 4)
                deck += [cid] * n
            if len(deck) >= 60 - target_energy:
                break
    # 4) 能量填满
    while len(deck) < 60:
        deck.append(rng.choice(energy_cards))
    deck = deck[:60]
    return repair(pool, deck, rng)


def mutate(pool: CardPool, deck: list[int], rng: random.Random,
           n_swaps: int = 2, keep_energy: bool = True) -> list[int]:
    """点变异：换掉若干个卡位。

    keep_energy=True 时，被换掉的若是能量就换成能量，避免能量比例剧烈漂移。
    """
    out = list(deck)
    for _ in range(n_swaps):
        i = rng.randrange(60)
        old = out[i]
        if keep_energy and pool.is_basic_energy(old):
            cand = pool.basic_energy
        elif rng.random() < 0.5:
            cand = pool.basic_energy
        elif rng.random() < 0.5:
            cand = pool.trainers
        else:
            # 换宝可梦时也偏向强卡 —— 否则变异出来的多是被碾压的弱攻击手
            cand = pool.strong_basics + pool.stage1
        new = rng.choice(cand)
        if new in pool.ace_spec and any(a in pool.ace_spec for a in out):
            continue
        out[i] = new
    return repair(pool, out, rng)


def crossover(pool: CardPool, a: list[int], b: list[int],
              rng: random.Random) -> list[int]:
    """单点交叉：取 A 的前半 + B 的后半，再修复合法性。"""
    cut = rng.randrange(10, 50)
    child = list(a[:cut]) + list(b[cut:])
    return repair(pool, child[:60], rng)


def structural_mutate(pool: CardPool, deck: list[int], rng: random.Random) -> list[int]:
    """结构变异：整条进化线替换，并同步换掉不匹配的能量。

    点变异只能微调配比，无法改变"这是什么卡组"。而卡组强度的主要差别
    恰恰来自攻击手 —— 实测随机卡组对样例卡组只有 0.05，就是因为攻击手太弱。

    ⚠️ 必须用 pool.line_in_deck() 来定位同族卡：`evolvesFrom` 是名字，
       同名可能对应多个 cardId，按名字正向匹配会挑错（踩过这个 bug）。
    """
    cnt = Counter(deck)
    pokemon_in_deck = [cid for cid in cnt if cid in pool.by_id
                       and int(pool.by_id[cid].cardType) == 0]
    if not pokemon_in_deck:
        return repair(pool, deck, rng)

    # 丢掉"最弱的那一族"（同族卡全部移除）
    weakest = min(pokemon_in_deck, key=lambda c: pool.power(c))
    old_line = pool.line_in_deck(deck, weakest)
    kept = [cid for cid in deck if cid not in old_line]

    # 选新族：不能和剩下的族重复，也不能还是刚被丢掉的那一族
    old_base = pool.base_name(weakest)
    existing_bases = {pool.base_name(c) for c in kept
                      if c in pool.by_id and int(pool.by_id[c].cardType) == 0}
    cands = [c for c in pool.strong_basics
             if pool.base_name(c) not in existing_bases and pool.base_name(c) != old_base]
    if not cands:
        return repair(pool, deck, rng)
    new_basic = rng.choice(cands)
    new_chain = pool.evolution_chain(new_basic)
    per = 4 if len(new_chain) <= 2 else 3
    for cid in new_chain:
        kept += [cid] * per

    # 按【卡组里所有攻击手】的需求重配能量（不能只看新换上的那条线）
    kept = [cid for cid in kept if not pool.is_basic_energy(cid)]
    energy_cards = energy_cards_for(pool, kept) or [rng.choice(pool.basic_energy)]
    while len(kept) < 60:
        kept.append(rng.choice(energy_cards))
    return repair(pool, kept[:60], rng)


def mutate_deck(pool: CardPool, deck: list[int], rng: random.Random,
                n_swaps: int = 2, structural_p: float = 0.25) -> list[int]:
    """混合变异：以一定概率做结构变异，否则做点变异。"""
    if rng.random() < structural_p:
        return structural_mutate(pool, deck, rng)
    return mutate(pool, deck, rng, n_swaps=n_swaps)
