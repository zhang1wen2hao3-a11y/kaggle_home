#!/usr/bin/env python3
"""**必须卡组清单**：训练资源往哪倾斜、表里必须出现哪些牌。

用户 2026-11-19 要求：
  "训练资源要向高分卡组和高覆盖率两个方向倾斜 …… 刷新脚本也要支持这些必须卡组能出现在我们维护的 csv 文件里，
   初始如果没有模型，可以选择 BC 或者 init。"

本脚本定义"必须卡组"（三路来源取并集），并给出每副牌的：
  · 覆盖率（我们的对局里，这副牌作为**对手**出现的份额）——"高覆盖率"
  · 榜前 N 队的局数与胜率（他们用这副牌打得怎么样）      ——"高分卡组"的现场证据
  · 我们现有的驾驶员（没有则标 缺司机 ⇒ 由 materialize_required_drivers.sh 补 BC/init）

三路来源：
  A. `decks/testset_meta_real.tsv` 的 17 个真实份额格（我们评测口径必须覆盖的对手）
  B. `epsoide_top*/` 回放语料里，榜前 N 队用过且局数 ≥ MIN_TEAM_GAMES 的牌
  C. `docs/the-pockmon-battle-choice-in-table.csv`（母表）里已有的挑战者/被挑战者

输出：`docs/必须卡组清单.csv`（排序按 覆盖率×对局数），并打印缺司机的牌。

    ./tools/py tools/required_decks.py                     # 生成并打印
    ./tools/py tools/required_decks.py --missing-only      # 只打缺司机的
"""
from __future__ import annotations
import argparse, collections, csv, glob, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = os.path.join(ROOT, "decks", "pool_field")
TS = os.path.join(ROOT, "decks", "testset_meta_real.tsv")
MASTER = os.path.join(ROOT, "docs", "the-pockmon-battle-choice-in-table.csv")
SCAN = os.path.join(ROOT, "logs", "sub_feedback", "scan.jsonl")
OUT = os.path.join(ROOT, "docs", "必须卡组清单.csv")
MIN_TEAM_GAMES = 10          # 榜前 N 队至少用这么多局才算"他们在用的牌"
DECK_IDS: dict[str, tuple] = {}   # 4hex -> 排序后的 60 个卡号（用来把 agent 目录认亲；
#                                   ★ 不要用 deck.csv 的 md5：同样的卡换个行序 md5 就变，会大面积误报"缺司机"）


_CACHE_DIR = os.path.join(ROOT, "data", "cache")


def _load_cache(name):
    try:
        return json.load(open(os.path.join(_CACHE_DIR, name)))
    except (OSError, ValueError):
        return None


def _save_cache(name, obj):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tmp = os.path.join(_CACHE_DIR, name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, os.path.join(_CACHE_DIR, name))


def _md5_cached(path, cache):
    """按 (mtime, size) 命中缓存；miss 才真算 md5。agent 的 model.pt 都是 20 MB，
    28 副牌 × 每个 agent 各算一遍就是几百 MB 的重复哈希 —— 这是 120 秒里的主要开销。"""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = path
    ent = cache.get(key)
    if ent and ent[0] == st.st_mtime and ent[1] == st.st_size:
        return ent[2]
    try:
        h = hashlib.md5(open(path, "rb").read()).hexdigest()
    except OSError:
        return None
    cache[key] = [st.st_mtime, st.st_size, h]
    return h


def pool_index():
    """4hex/8hex → (池子文件名, 原型, 张数)"""
    idx = {}
    for p in glob.glob(os.path.join(POOL, "*.csv")):
        ids = sorted(int(x) for x in open(p) if x.strip())
        if not ids:
            continue
        h8 = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:8]
        name = os.path.basename(p)[:-4]
        arch = name.split("__")[0]
        idx[h8[:4]] = (name, arch, len(ids), h8)
        DECK_IDS[h8[:4]] = tuple(ids)
    return idx


def testset_decks():
    out = set()
    for ln in open(TS):
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.rstrip("\n").split("\t")
        if f[0] == "tail_mix":
            continue
        out.add(f[0].split("_")[-1])
    return out


def scan_coverage():
    """覆盖率口径 = 该牌在所有"真实对局"里被使用的比例。

    两个来源合起来算（只用我们自己的 241 局会把 0d86 之类判成 0%，但那只是"我们还没碰上"）：
      · `logs/sub_feedback/scan.jsonl`：我们提交的 agent 的对局，取**对手**那一边；
      · `epsoide_top*/`：榜前 N 队的回放，**双方都算**（这些就是真实场里的牌）。
    """
    n = collections.Counter(); tot = 0
    own = collections.Counter(); own_tot = 0
    try:
        for ln in open(SCAN):
            r = json.loads(ln)
            h = r.get("opp_hash")
            if not h or r.get("win") is None or r.get("selfmirror"):
                continue
            own_tot += 1
            own[h[:4]] += 1
            n[h[:4]] += 1
            tot += 1
    except OSError:
        pass
    global _REPLAY_COV
    for h4, c in _REPLAY_COV.items():
        n[h4] += c; tot += c
    cov = {k: v / tot for k, v in n.items()} if tot else {}
    own_cov = {k: v / own_tot for k, v in own.items()} if own_tot else {}
    return cov, tot, own_cov


_REPLAY_COV: dict[str, int] = {}
_REPLAY_BY_PLAYER: dict[tuple, list] = {}   # (4hex, 玩家) -> [局数, 胜场]


def lb_by_player():
    """最新榜单 CSV（`kaggle competitions leaderboard <slug> -d` 解出来的那张）→ {队名: 分}。
    回放里的 `info.Agents[].Name` 用的就是队名，可以直接 join。"""
    import glob as _g
    cands = sorted(_g.glob(os.path.join(ROOT, "data", "leaderboard", "*.csv")), key=os.path.getmtime)
    out = {}
    if cands:
        for r in csv.DictReader(open(cands[-1], encoding="utf-8-sig", errors="replace")):
            nm = (r.get("TeamName") or "").strip().strip('"')
            try:
                out[nm] = float(r.get("Score") or 0)
            except ValueError:
                pass
    if not out:
        try:
            out = {k: float(v) for k, v in json.load(open(os.path.join(ROOT, "data", "lb_scores.json"))).items()}
        except OSError:
            pass
    return out


def _team_usage_fingerprint(paths):
    h = hashlib.sha256()
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            continue
        h.update(f"{p}|{int(st.st_mtime)}|{st.st_size}\n".encode())
    return h.hexdigest()[:20]


def _one_episode(p):
    """解析一个回放 → [(4hex, 玩家名, 是否该玩家赢)]；整个回放在函数内解析完就释放
    （第一版把所有 ep 收进 list，700 个回放同时驻留 ⇒ OOM 被 Killed）。"""
    import sys as _sys
    _sys.path.insert(0, ROOT)
    from framework.replay import decks_of
    try:
        ep = json.load(open(p))
        dk = decks_of(ep.get("steps") or [])
    except Exception:
        return None
    names = [(x.get("Name") or "").strip().strip('"')
             for x in ((ep.get("info") or {}).get("Agents") or [])]
    rw = ep.get("rewards") or []
    w = None
    if len(rw) >= 2 and rw[0] is not None and rw[0] != rw[1]:
        w = 0 if rw[0] > rw[1] else 1
    out = []
    for i, d in dk.items():
        if not d:
            continue
        h4 = hashlib.sha256(",".join(map(str, sorted(d))).encode()).hexdigest()[:4]
        out.append((h4, names[i] if i < len(names) else "", w is not None and w == i))
    return out


def team_usage():
    """榜前 N 队回放语料：{4hex: [局数, 胜场]}

    ★ 2026-11-22 性能：这里要解析 700+ 个回放 json（每局几百 KB~几 MB），原来整条
      `refresh_tables.sh` 的 240 秒几乎全花在这一步（它被调用两次）。
      现在：① 按"文件清单+mtime+size"的指纹做磁盘缓存，没变就直接读结果；
            ② 变的时候用 8 线程并发解析。
    """
    use = collections.Counter(); win = collections.Counter(); files = 0
    sys.path.insert(0, ROOT)
    try:
        from framework.replay import decks_of
    except Exception:
        return {}, {}, 0
    paths = sorted(glob.glob(os.path.join(ROOT, "epsoide_top*", "*.json")))
    sig = _team_usage_fingerprint(paths)
    cached = _load_cache("team_usage.json")
    if cached and cached.get("sig") == sig:
        global _REPLAY_COV, _REPLAY_BY_PLAYER
        _REPLAY_COV = dict(cached["cov"])
        _REPLAY_BY_PLAYER = {tuple(k.split("\t", 1)): v for k, v in cached["by_player"].items()}
        return (collections.Counter(cached["use"]), collections.Counter(cached["win"]),
                int(cached["files"]))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for rec in ex.map(_one_episode, paths, chunksize=4):
            if not rec:
                continue
            files += 1
            for h4, nm, won in rec:
                _REPLAY_COV[h4] = _REPLAY_COV.get(h4, 0) + 1
                use[h4] += 1
                if nm:
                    r = _REPLAY_BY_PLAYER.setdefault((h4, nm), [0, 0])
                    r[0] += 1
                    if won:
                        r[1] += 1
                if won:
                    win[h4] += 1
    _save_cache("team_usage.json", dict(
        sig=sig, files=files, use=dict(use), win=dict(win), cov=dict(_REPLAY_COV),
        by_player={f"{k[0]}\t{k[1]}": v for k, v in _REPLAY_BY_PLAYER.items()}))
    return use, win, files


def master_decks():
    out = set()
    try:
        txt = open(MASTER, "rb").read().decode("gb18030", errors="replace")
        for r in csv.DictReader(txt.splitlines()):
            for k in ("挑战者", "被挑战者"):
                v = (r.get(k) or "").strip()
                if v:
                    out.add(v.split("_")[-1].lower()[:4])
    except OSError:
        pass
    return out


_AGENT_INDEX = None


def agent_index(port_md5):
    """{卡表 ids: ([全部驾驶员], [训练过的驾驶员])} —— 只算一次，供 driver_for 查表。"""
    global _AGENT_INDEX
    if _AGENT_INDEX is not None:
        return _AGENT_INDEX
    cache = _load_cache("md5_agents.json") or {}
    idx = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "build", "agents", "*", "deck.csv"))):
        try:
            ids = tuple(sorted(int(x) for x in open(d) if x.strip()))
        except (OSError, ValueError):
            continue
        name = d.split(os.sep)[-2]
        all_d, trained = idx.setdefault(ids, ([], []))
        all_d.append(name)
        md5 = _md5_cached(os.path.join(os.path.dirname(d), "model.pt"), cache)
        if md5 and md5 != port_md5:
            trained.append(name)
    _save_cache("md5_agents.json", cache)
    _AGENT_INDEX = idx
    return idx


def driver_for(h4, port_md5):
    """该卡组现在有哪些驾驶员 —— 直接按 **agent 目录里 deck.csv 的 md5** 认亲，
    不信注册表（注册表只登记了少数训练代，端口/BC 那一批从来没进去，用它判"缺司机"会大面积误报）。

    返回 (全部驾驶员, 有训练驾驶员的?)：model.pt 与 port_plain 同 md5 ⇒ 那是 init/端口，不算"训练过"。
    """
    ids = DECK_IDS.get(h4)
    if not ids:
        return [], []
    all_d, trained = agent_index(port_md5).get(ids, ([], []))
    return list(all_d), list(trained)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing-only", action="store_true")
    ap.add_argument("--check-covered", action="store_true",
                    help="校验框：投资分表里必须出现全部必须卡组，缺一条就非零退出")
    ap.add_argument("--min-team-games", type=int, default=MIN_TEAM_GAMES)
    a = ap.parse_args()

    idx = pool_index()
    port_md5 = hashlib.md5(open(os.path.join(ROOT, "build", "port_plain", "model.pt"), "rb").read()).hexdigest()
    ts = {h[:4] for h in testset_decks()}
    use, win, nfiles = team_usage()      # ← 必须先跑：它顺手填 _REPLAY_COV，覆盖率要用
    LB = lb_by_player()
    ext_replay, ext_player = {}, {}
    for (h4, nm), (g, w) in _REPLAY_BY_PLAYER.items():
        if g < MIN_TEAM_GAMES or (w / g) < 0.50:      # 同一套"宁缺毋滥"规则：打得少或一直输不算
            continue
        sc = LB.get(nm, 0.0)
        if sc > ext_replay.get(h4, 0.0):
            ext_replay[h4], ext_player[h4] = sc, nm
    cov, covtot, own_cov = scan_coverage()
    mas = {h[:4] for h in master_decks()}

    decks = set(ts) | mas | {h[:4] for h, g in use.items() if g >= a.min_team_games}
    rows = []
    for h4 in sorted(decks):
        name, arch, ncard, h8 = idx.get(h4, (f"未入库:{h4}", "?", 0, h4))
        g = use.get(h4, 0); w = win.get(h4, 0)
        all_d, trained = driver_for(h4, port_md5)
        rows.append(dict(
            h4=h4, name=name, arch=arch, ncard=ncard, h8=h8,
            ts="✓" if h4 in ts else "",
            games=g, wr=(w / g if g else None),
            cov=cov.get(h4, 0.0), owncov=own_cov.get(h4, 0.0),
            ext_replay=ext_replay.get(h4), ext_player=ext_player.get(h4, ""),
            drivers=",".join(all_d),
            trained=",".join(trained),
            missing="" if trained else ("仅 init/BC，优先排训练" if all_d else "缺司机"),
        ))
    rows.sort(key=lambda r: (-r["cov"], -r["games"]))
    with open(OUT, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["4hex", "8hex", "卡组文件", "原型", "17格", "榜前N队局数", "榜前N队胜率",
                     "对手覆盖率", "覆盖率(我方对局)", "回放口径天花板", "回放口径最高分玩家",
                     "已训练驾驶员", "现有驾驶员", "建议"])
        for r in rows:
            wr.writerow([r["h4"], r["h8"], r["name"], r["arch"], r["ts"], r["games"],
                         (f"{r['wr']*100:.1f}%" if r["wr"] is not None else ""),
                         f"{r['cov']*100:.1f}%", f"{r['owncov']*100:.1f}%",
                         r["ext_replay"] or "", r["ext_player"], r["trained"], r["drivers"], r["missing"]])
    if a.check_covered:
        invp = os.path.join(ROOT, "docs", "卡组天花板与投资分.csv")
        if not os.path.exists(invp):
            print("✗ 还没有 docs/卡组天花板与投资分.csv（先跑 build_investment_score.py）")
            return 2
        have = {(r.get("卡组") or "").strip() for r in csv.DictReader(open(invp, encoding="utf-8-sig"))}
        # 近似同构变体（0bf5→9d6e 等）在投资分表里已并入代表 ⇒ 校验时按代表名比对
        merge = {}
        try:
            for ln in open(os.path.join(ROOT, "decks", "deck_merge.txt"), encoding="utf-8"):
                ln = ln.strip()
                if ln and not ln.startswith("#") and len(ln.split()) == 2:
                    merge[ln.split()[0]] = ln.split()[1]
        except OSError:
            pass
        missing = [r["h4"] for r in rows if merge.get(r["h4"], r["h4"]) not in have]
        if missing:
            print(f"✗ 投资分表缺 {len(missing)} 副必须卡组：{', '.join(missing)}")
            for r in rows:
                if r["h4"] in missing:
                    print(f"   {r['h4']}  {r['name']}  覆盖率 {r['cov']*100:.1f}%  驾驶员 {r['drivers'] or '无'}")
            return 3
        print(f"✓ 投资分表已覆盖全部 {len(rows)} 副必须卡组")
        return 0
    miss = [r for r in rows if r["missing"]]
    print(f"必须卡组 {len(rows)} 副（17格 {len(ts)} / 母表 {len(mas)} / 榜前N队≥{a.min_team_games}局 {sum(1 for h,g in use.items() if g>=a.min_team_games)}）"
          f"，覆盖率样本 {covtot} 局，回放语料 {nfiles} 局")
    n_train = sum(1 for r in rows if not r["trained"])
    n_none = sum(1 for r in rows if not r["drivers"])
    print(f"→ {os.path.relpath(OUT, ROOT)}；无驾驶员 {n_none} 副 / 无**训练过**的驾驶员 {n_train} 副")
    if a.missing_only:
        for r in miss:
            print(f"   {r['h4']}  {r['name']:42s} 覆盖率 {r['cov']*100:5.1f}%  榜前N队 {r['games']:3d} 局")
    return 0


if __name__ == "__main__":
    sys.exit(main())
