#!/usr/bin/env python3
"""填充 docs/the-pockmon-battle-{cards,match,players}.csv 三张表（GBK 编码，与模板一致）。

数据来源与口径：
  cards   卡组宇宙    ← data/ladder_games.tsv（1756 局真人天梯档案，双座位都计入）
  match   对位记录    ← ①真人 vs 真人（同档案）②我们 vs 对手（data/our_games.tsv，640 局）
                        ③端口 vs 端口（data/scheduler/matrix_full.csv，49 格）
  players 选手/克隆体 ← 档案里的 (卡组, 选手) 组合 + 我们本地各 agent 目录（端口/BC/训练）

字段（模板原样）：
  cards   : DeckID, 卡组, Meta 占比, 样本局数
  match   : DeckID, 玩家卡组, 玩家, DeckID, 对手卡组, 对手, 胜局, 败局, 平局
  players : DeckID, 玩家, 类型, 父玩家, 卡组, Checkpoint, 数据来源
"""
import collections, csv, json, os, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
ENC = "gb18030"   # GBK 的超集：模板是 GBK，但选手名含越南语/土耳其语字符，GBK 编不下

# ---------- 卡组哈希 → 全名 ----------
H2NAME = {}
for p in glob.glob(os.path.join(ROOT, "decks", "pool_field", "*.csv")):
    base = os.path.basename(p)[:-4]
    if "__" in base:
        H2NAME.setdefault(base.split("__")[-1][:4], base)
# 档案里的 8 位哈希也注册进来
for r in csv.DictReader(open(os.path.join(ROOT, "data", "ladder_games.tsv")), delimiter="\t"):
    for k in ("seat0_deck", "seat1_deck"):
        h = (r.get(k) or "")
        if len(h) >= 4:
            H2NAME.setdefault(h[:4], h)

def name_of(h):
    return H2NAME.get(h[:4], h[:4])


def hid(deck_str):
    """从「全名」或「哈希」取 4 位 DeckID；认不出（'?'/空）返回 ''。

    our_games.tsv 里存的是全名（如 dunsparce_abra__9d6e），
    直接取 [:4] 会得到 'duns' 这种假 ID —— 早期版本就踩了这个坑。
    """
    t = (deck_str or "").strip()
    if not t or t == "?":
        return ""
    if "__" in t:
        return t.split("__")[-1][:4]
    return t[:4] if all(c in "0123456789abcdef" for c in t[:4].lower()) else ""

LB = json.load(open(os.path.join(ROOT, "data", "lb_scores.json")))

# ---------- 卡组名聚合：archetype → 简洁主体名 ----------
# 规则：取牌组"主体"（通常是主攻手/最终进化）作为名字，主体基本一致就归同一个名字。
# 多线牌组按"卡数最多的那条线"判（例：9d6e/0bf5/db04 的 60 张里 Abra+Kadabra+Alakazam
# 共 12 张、Dunsparce+Dudunsparce 只有 7 张 ⇒ 主体是 alakazam，dunsparce 只是抽牌引擎）。
ARCH2SHORT = {
    # 厄鬼椪家族
    "teal_mask_ogerpon_ex_applin": "ogerpon",
    "teal_mask_ogerpon_ex_hero_s_cape": "ogerpon",
    "cornerstone_mask_ogerpon_ex_dwebble": "ogerpon",
    "raging_bolt_ex_teal_mask_ogerpon_ex": "ogerpon",
    "z_ogerpon_toolbox_miwa": "ogerpon",
    # 胡地家族（Dunsparce 系列是抽牌引擎，主体是 Alakazam）
    "dunsparce_abra": "alakazam",
    "abra_kadabra": "alakazam",
    "dudunsparce_abra": "alakazam",
    "kadabra_duskull": "alakazam",
    "duskull_kadabra": "alakazam",
    # 土龙节节（纯引擎向，没有胡地线）
    "dunsparce_dudunsparce": "dudunsparce",
    "dudunsparce_dunsparce": "dudunsparce",
    # 其他单线
    "grookey_thwackey": "rillaboom",
    "dreepy_drakloak": "dragapult",
    "z_dragapult_dreepy_petit": "dragapult",
    "shuppet_dhelmise": "shuppet",
    "shuppet_banette": "shuppet",
    "slowpoke_slowking": "slowking",
    "slowpoke_mega_kangaskhan_ex": "kangaskhan",
    "mega_kangaskhan_ex_dwebble": "kangaskhan",
    "cinderace_mega_kangaskhan_ex": "kangaskhan",
    "mega_lucario_ex_solrock": "lucario",
    "riolu_mega_lucario_ex": "lucario",
    "mega_lucario_ex_riolu": "lucario",
    "mega_lucario_ex_makuhita": "lucario",
    "riolu_lunatone": "lucario",
    "beldum_metang": "metagross",
    "steven_s_beldum_steven_s_metagross_ex": "metagross",
    "dwebble_crustle": "crustle",
    "dwebble_bunnelby": "dwebble",
    "duraludon_archaludon_ex": "archaludon",
    "snover_mega_abomasnow_ex": "abomasnow",
    "munkidori_marnie_s_impidimp": "munkidori",
    "munkidori_dwebble": "munkidori",
    "iono_s_voltorb_iono_s_tadbulb": "voltorb",
    "girafarig_enamorus": "girafarig",
    "staryu_mega_starmie_ex": "starmie",
    "chandelure_litwick": "chandelure",
    "larvitar_tyranitar": "tyranitar",
    "great_tusk_stonjourner": "great_tusk",
    "iron_thorns_ex_tapu_koko_ex": "iron_thorns",
    "iron_thorns_ex_pikachu_ex": "iron_thorns",
    "charcadet_ceruledge_ex": "ceruledge",
    "gastly_mega_gengar_ex": "gengar",
    "cynthia_s_gible_cynthia_s_gabite": "garchomp",
    "hop_s_snorlax_hop_s_cramorant": "snorlax",
}


def short_of(deckname):
    """全名/哈希 → 简洁主体名（认不出就当单线处理：取 archetype 首段）。"""
    if "__" not in deckname:
        return deckname
    arch = deckname.split("__")[0]
    return ARCH2SHORT.get(arch, arch.split("_")[0])


# ---------- cards：真人天梯档案的卡组分布 ----------
arch = list(csv.DictReader(open(os.path.join(ROOT, "data", "ladder_games.tsv")), delimiter="\t"))
appear = collections.Counter()
for r in arch:
    for k in ("seat0_deck", "seat1_deck"):
        h = hid(r.get(k))
        if h: appear[h] += 1
tot = sum(appear.values())
cards = []
for h, n in appear.most_common():
    cards.append(dict(DeckID=h, 卡组=short_of(name_of(h)), Meta占比=n / tot * 100, 样本局数=n))

# 主体名 → 代表哈希（样本最多的那个 decklist），match 表合并后用它填 DeckID
REP = {}
for c in cards:
    k = c["卡组"]
    if k not in REP or c["样本局数"] > REP[k][0]:
        REP[k] = (c["样本局数"], c["DeckID"])

# cards 按主体名聚合：同一主体的不同 decklist 合并成一行（样本多者当代表哈希）
_agg = {}
for c in cards:
    k = c["卡组"]
    if k not in _agg:
        _agg[k] = dict(DeckID=c["DeckID"], 卡组=k, Meta占比=0.0, 样本局数=0, _mx=0)
    _agg[k]["Meta占比"] += c["Meta占比"]
    _agg[k]["样本局数"] += c["样本局数"]
    if c["样本局数"] > _agg[k]["_mx"]:
        _agg[k]["DeckID"], _agg[k]["_mx"] = c["DeckID"], c["样本局数"]
cards = sorted(_agg.values(), key=lambda r: -r["样本局数"])
for r in cards:
    r.pop("_mx", None); r["Meta占比"] = round(r["Meta占比"], 2)

# ---------- match ----------
matches = []
# ① 真人 vs 真人
pair = collections.defaultdict(lambda: [0, 0])   # (d0,a0,d1,a1) -> [胜, 负]
for r in arch:
    d0, a0 = hid(r.get("seat0_deck")), r.get("seat0_agent") or ""
    d1, a1 = hid(r.get("seat1_deck")), r.get("seat1_agent") or ""
    w = r.get("winner")
    if not (d0 and d1 and w in ("0", "1")):
        continue
    if w == "0": pair[(d0, a0, d1, a1)][0] += 1
    else:        pair[(d0, a0, d1, a1)][1] += 1
for (d0, a0, d1, a1), (win, lose) in sorted(pair.items(), key=lambda t: -(t[1][0] + t[1][1])):
    if win + lose < 3:          # 少于 3 局的组合噪声太大
        continue
    matches.append(dict(DeckID=d0, 玩家卡组=short_of(name_of(d0)), 玩家=a0, DeckID2=d1,
                        对手卡组=short_of(name_of(d1)), 对手=a1, 胜局=win, 败局=lose, 平局=0))
n_human = len(matches)

# ② 我们 vs 对手
ours = list(csv.DictReader(open(os.path.join(ROOT, "data", "our_games.tsv")), delimiter="\t"))
pm = collections.defaultdict(lambda: [0, 0])
for r in ours:
    d0 = hid(r.get("our_deck")); d1 = hid(r.get("opp_deck"))
    if not (d0 and d1) or r.get("result") not in ("0", "1"):
        continue        # 对手卡组未识别（'?'）的局不进对位表——归不到具体卡组
    sub = r.get("submission") or "ours"
    key = (d0, sub, d1, r.get("opponent") or "")
    if r["result"] == "1": pm[key][0] += 1
    else:                  pm[key][1] += 1
for (d0, sub, d1, opp), (w, l) in sorted(pm.items(), key=lambda t: -(t[1][0] + t[1][1])):
    matches.append(dict(DeckID=d0, 玩家卡组=short_of(name_of(d0)), 玩家=sub, DeckID2=d1,
                        对手卡组=short_of(name_of(d1)), 对手=opp, 胜局=w, 败局=l, 平局=0))
n_ours = len(matches) - n_human

# ③ 端口 vs 端口（49 格矩阵）
mp = os.path.join(ROOT, "data", "scheduler", "matrix_full.csv")
n_matrix = 0
if os.path.exists(mp):
    for r in csv.DictReader(open(mp)):
        n = int(float(r["n_games"])); wr = float(r["cand_win"])
        win = round(wr * n); lose = n - win
        matches.append(dict(DeckID=r["cand_deck"], 玩家卡组=short_of(name_of(r["cand_deck"])),
                            玩家="端口权重", DeckID2=r["opp_deck"],
                            对手卡组=short_of(name_of(r["opp_deck"])), 对手="端口权重",
                            胜局=win, 败局=lose, 平局=0))
        n_matrix += 1

# ---------- match：同对合并 + 胜率高者置左 + 按前列排序 ----------
try:
    from pypinyin import lazy_pinyin          # 有就用真拼音
    def sort_key(txt):
        return lazy_pinyin(txt)
except ImportError:                            # 没装就用码位（ASCII 名仍是字母序）
    def sort_key(txt):
        return [txt]

# 以「(卡组, 玩家) 实体对」为唯一键；同一对只保留一条，两个方向相加
pair_tot = {}          # frozenset({实体A, 实体B}) → {(实体): [胜, 负, 平]}
for m in matches:
    a = (m["玩家卡组"], m["玩家"])
    b = (m["对手卡组"], m["对手"])
    key = tuple(sorted([a, b]))
    cur = pair_tot.setdefault(key, {a: [0, 0, 0], b: [0, 0, 0]})
    cur.setdefault(a, [0, 0, 0]); cur.setdefault(b, [0, 0, 0])
    cur[a][0] += m["胜局"]; cur[a][1] += m["败局"]; cur[a][2] += m["平局"]
    cur[b][0] += m["败局"]; cur[b][1] += m["胜局"]

merged = []
for (e1, e2), per in pair_tot.items():
    (w1, l1, d1), (w2, l2, d2) = per[e1], per[e2]
    if (w1 + l1) == 0 or (w2 + l2) == 0:
        continue
    if w1 / (w1 + l1) < w2 / (w2 + l2):      # 胜率高的放左边
        e1, e2, w1, l1, d1, w2, l2, d2 = e2, e1, w2, l2, d2, w1, l1, d1
    merged.append(dict(DeckID=REP.get(e1[0], ("", ""))[1], 玩家卡组=e1[0], 玩家=e1[1],
                       DeckID2=REP.get(e2[0], ("", ""))[1], 对手卡组=e2[0], 对手=e2[1],
                       胜局=w1, 败局=l1, 平局=d1))
# 排序：优先按前几列（左卡组名 → 左玩家 → 右卡组名 → 右玩家）
merged.sort(key=lambda m: (sort_key(m["玩家卡组"]), sort_key(m["玩家"]),
                           sort_key(m["对手卡组"]), sort_key(m["对手"])))
n_before = len(matches)
matches = merged

# ---------- players ----------
players = []
# 真人：每个(卡组,选手)，≥2 局
pg = collections.Counter()
for r in arch:
    for dk, ag in (("seat0_deck", "seat0_agent"), ("seat1_deck", "seat1_agent")):
        h = (r.get(dk) or "")[:4]; a = r.get(ag) or ""
        if h and a: pg[(h, a)] += 1
for (h, a), n in sorted(pg.items(), key=lambda t: (-t[1], t[0])):
    if n < 2: continue
    players.append(dict(DeckID=h, 玩家=a, 类型="真人", 父玩家="", 卡组=short_of(name_of(h)),
                        Checkpoint="", 数据来源=f"Kaggle 回放（{n} 局）"))

# 本地 agent：从 deck.csv 反查卡组
def deck_of(agent_dir):
    """用 deck.csv 内容比对 pool_field，认出这个 agent 打哪副牌。"""
    p = os.path.join(agent_dir, "deck.csv")
    if not os.path.exists(p): return ""
    try:
        a = open(p).read()
    except Exception:
        return ""
    for q in glob.glob(os.path.join(ROOT, "decks", "pool_field", "*.csv")):
        try:
            if open(q).read() == a:
                return os.path.basename(q)[:-4]
        except Exception:
            pass
    return ""

# 类型按**权重身份**判定（比目录名前缀可靠）：
#   端口权重 md5 8b5fcda672ac → 端口基线；BC 父权重 6007d6f083 → BC克隆；其余 → PPO训练
import hashlib
def md5_of(path):
    try:
        return hashlib.md5(open(path, "rb").read()).hexdigest()[:12]
    except Exception:
        return ""
PORT_MD5 = "8b5fcda672ac"
BC_MD5 = "6007d6f083"        # g8_bc_parent / opp_shuppet_human 同一份 BC 父权重
for d in sorted(glob.glob(os.path.join(ROOT, "build", "agents", "*"))):
    if not os.path.isdir(d): continue
    deckname = deck_of(d)
    if not deckname: continue
    base = os.path.basename(d)
    m = md5_of(os.path.join(d, "model.pt"))
    if m == PORT_MD5:
        kind = "端口基线"
    elif m.startswith(BC_MD5):
        kind = "BC克隆"
    else:
        kind = "PPO训练"
    parent = ""
    if kind == "BC克隆":
        # 找该卡组语料最好的真人教师（帧数最多者）
        best = max(((n, a) for (h, a), n in pg.items() if h == deckname.split("__")[-1][:4]),
                   default=(0, ""))
        parent = best[1]
    players.append(dict(DeckID=deckname.split("__")[-1][:4], 玩家=base, 类型=kind,
                        父玩家=parent, 卡组=short_of(deckname),
                        Checkpoint=f"build/agents/{base}/model.pt",
                        数据来源="我们训练" if kind != "端口基线" else "端口仓库权重"))

# ---------- 写盘 ----------
def write(path, header, rows, fields):
    with open(path, "w", newline="", encoding=ENC, errors="replace") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(k, "") for k in fields])

write(os.path.join(DOCS, "the-pockmon-battle-cards.csv"),
      ["DeckID", "卡组", "Meta 占比", "样本局数"], cards, ["DeckID", "卡组", "Meta占比", "样本局数"])
write(os.path.join(DOCS, "the-pockmon-battle-match.csv"),
      ["DeckID", "玩家卡组", "玩家", "DeckID", "对手卡组", "对手", "胜局", "败局", "平局"],
      matches, ["DeckID", "玩家卡组", "玩家", "DeckID2", "对手卡组", "对手", "胜局", "败局", "平局"])
write(os.path.join(DOCS, "the-pockmon-battle-players.csv"),
      ["DeckID", "玩家", "类型", "父玩家", "卡组", "Checkpoint", "数据来源"],
      players, ["DeckID", "玩家", "类型", "父玩家", "卡组", "Checkpoint", "数据来源"])

print(f"cards   : {len(cards)} 行（档案 {tot} 个座位出现）")
print(f"match   : {len(matches)} 行（原始 {n_before} 行：真人 {n_human} + 我们 {n_ours} + 端口矩阵 {n_matrix}，同对合并后，胜率高者置左）")
print(f"players : {len(players)} 行 = 真人 {sum(1 for p in players if p['类型']=='真人')} "
      f"+ 本地 agent {sum(1 for p in players if p['类型']!='真人')}")
