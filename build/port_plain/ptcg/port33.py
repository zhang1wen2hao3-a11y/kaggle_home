"""33 port of the Featured agent to the Playground pool: proxies for new cards and attacks.

Built once at import from the engine's own card table. A new card (id beyond the trained pool) maps to the old
card closest to it: same card type first, then stage, rule box, energy type, HP, attack count and damage,
retreat. A new attack maps to the attack in the same slot of the proxy card.
"""
import copy

from cg.api import all_card_data, all_attack

from .rl.buffers import N_CARDS, N_ATTACKS

OLD_MAX_CARD = 1267          # the Featured pool ended at 1267; ids above are R2 additions
OLD_MAX_ATTACK = 1556        # 33 r2e: not used at run time (no remap); the build reads its proxies


def _feat(c):
    att = [ATT[a] for a in c.attacks if a in ATT]
    dmg = max([a.damage for a in att], default=0)
    cost = min([len(a.energies) for a in att], default=0)
    return (c.cardType, c.basic, c.stage1, c.stage2, c.ex, c.megaEx, c.aceSpec, c.energyType,
            c.hp, len(c.attacks), dmg, cost, c.retreatCost, len(c.skills))


def _dist(f, g):
    if f[0] != g[0]:
        return 1e9
    d = 50.0 * sum(a != b for a, b in zip(f[1:7], g[1:7]))      # stage, ex, mega, ace spec
    d += 8.0 * (f[7] != g[7])                                     # energy type
    d += abs(f[8] - g[8]) / 20.0 + 3.0 * abs(f[9] - g[9])        # hp, attack count
    d += abs(f[10] - g[10]) / 20.0 + 2.0 * abs(f[11] - g[11])    # best damage, cheapest cost
    d += abs(f[12] - g[12]) + 3.0 * abs(f[13] - g[13])           # retreat, abilities
    return d


CARDS = {c.cardId: c for c in all_card_data()}
ATT = {a.attackId: a for a in all_attack()}
OLD = [c for i, c in CARDS.items() if i <= OLD_MAX_CARD]
CARD_PROXY, ATTACK_PROXY = {}, {}
for i, c in CARDS.items():
    if i <= OLD_MAX_CARD:
        continue
    f = _feat(c)
    best = min(OLD, key=lambda o: (_dist(f, _feat(o)), o.cardId))
    CARD_PROXY[i] = best.cardId
    for k, aid in enumerate(c.attacks):
        if aid > OLD_MAX_ATTACK:
            pa = best.attacks[min(k, len(best.attacks) - 1)] if best.attacks else 0
            ATTACK_PROXY[aid] = pa
for aid in ATT:                       # any new attack not reached through a new card
    if aid > OLD_MAX_ATTACK and aid not in ATTACK_PROXY:
        ATTACK_PROXY[aid] = 0


def _walk(x):
    if isinstance(x, dict):
        if "serial" in x and isinstance(x.get("id"), int) and x["id"] in CARD_PROXY:
            x["id"] = CARD_PROXY[x["id"]]
        if isinstance(x.get("cardId"), int) and x["cardId"] in CARD_PROXY:
            x["cardId"] = CARD_PROXY[x["cardId"]]
        if isinstance(x.get("attackId"), int) and x["attackId"] in ATTACK_PROXY:
            x["attackId"] = ATTACK_PROXY[x["attackId"]]
        for v in x.values():
            if isinstance(v, (dict, list)):
                _walk(v)
    elif isinstance(x, list):
        for v in x:
            if isinstance(v, (dict, list)):
                _walk(v)


def remap_obs(obs):
    """A deep copy of the observation with new cards and attacks replaced by their proxies."""
    o = copy.deepcopy({k: v for k, v in obs.items() if k != "search_begin_input"})
    _walk(o)
    o["search_begin_input"] = obs.get("search_begin_input")
    return o
