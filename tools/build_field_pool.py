#!/usr/bin/env python3
"""Step A：把**真实 Playground 场**的卡组建成训练卡组池（含新卡），并给出按档位份额的权重。

## 为什么必须做这一步

公开 repo 的训练池是**上一届（Featured 轮）的 616 副**：实测最高卡 id = 1267、**含新卡的卡组 0 副**。
而 Playground 真实顶段有 **45%+ 在打含新卡的卡组**（`e8eef8d8` 5 张、`a47a7971` 8 张、`0d86f3a5` 21 张）。
直接拿旧池训练 = 对着一个"不存在新卡"的场训练 ⇒ 训完也用不上。

所以本工具做三件事：
1. 从回放里抽出**所有真实出现过的卡组**（去重、按 hash）；
2. **合法性校验**（规则 + **真引擎开局**，最权威）；
3. 按档位输出**权重文件**（`version 3` 格式，用名字而非索引），只统计**对手侧**的分布。

## 合法性判据（任一不过就丢弃并计数）

恰好 60 张 / 卡 id 在卡表内 / 同名 ≤4（基本能量除外）/ ACE SPEC ≤1 / 至少 1 张基础宝可梦 /
**`cg.game.battle_start()` 能开局**（引擎认可）。

## 用法

    ./run.sh tools/build_field_pool.py                  # 建池 + 三套权重
    ./run.sh tools/build_field_pool.py --dry-run        # 只统计不落盘
    ./run.sh tools/build_field_pool.py --no-engine-check
"""
import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from framework.replay import decks_of  # noqa: E402


def h(ids):
    return hashlib.sha256(",".join(str(x) for x in sorted(ids)).encode()).hexdigest()[:8]


def load_card_sheet():
    """id → (名字, 阶段/类型, Rule)。"""
    path = os.path.join(ROOT, "data", "EN_Card_Data_R2_full.csv")
    info = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                cid = int(r["Card ID"])
            except Exception:
                continue
            info.setdefault(cid, (
                (r.get("Card Name") or "").strip(),
                (r.get("Stage (Pokémon)/Type (Energy and Trainer)") or "").strip(),
                (r.get("Rule") or "").strip(),
            ))
    return info


def is_basic_energy(name: str) -> bool:
    return name.startswith("Basic {") and name.endswith("Energy")


def check_legality(ids, sheet):
    if len(ids) != 60:
        return False, f"张数 {len(ids)}"
    for c in ids:
        if c not in sheet:
            return False, f"未知卡 id"
    cnt = collections.Counter(ids)
    for c, n in cnt.items():
        nm, _stage, rule = sheet[c]
        if not is_basic_energy(nm) and n > 4:
            return False, "同名超过 4 张"
        if "ACE SPEC" in rule and n > 1:
            return False, "ACE SPEC 超过 1 张"
    if sum(n for c, n in cnt.items() if "Basic Pokémon" in sheet[c][1]) < 1:
        return False, "没有基础宝可梦"
    return True, ""


def engine_ok(ids):
    """用真引擎开一局（最权威的合法性检查）。"""
    try:
        from cg.game import battle_start, battle_finish
        obs, st = battle_start(list(ids), list(ids), reverse_player=False)
        ok = bool(st.battlePtr) and st.errorPlayer < 0
        if ok:
            battle_finish()
            return True, ""
        return False, f"引擎拒绝 {st.errorPlayer}/{st.errorType}"
    except Exception as e:
        return False, f"引擎异常 {type(e).__name__}"


def slug(ids, sheet):
    """取出现最多的两条宝可梦线做 archetype 名。"""
    cnt = collections.Counter(ids)
    mons = [(c, n) for c, n in cnt.items() if "Pokémon" in sheet.get(c, ("", "", ""))[1]]
    mons.sort(key=lambda x: (-x[1], x[0]))
    parts = []
    for c, _n in mons[:2]:
        nm = re.sub(r"[^A-Za-z0-9]+", "_", sheet[c][0] or str(c)).strip("_").lower()
        if nm and nm not in parts:
            parts.append(nm)
    return ("_".join(parts) or "deck")[:48]


def collect(dirs):
    """收集卡组。返回 seen / occ_all / occ_field 三份。

    ⚠️ 必须区分"两侧都算"与"只算对手侧"：`epsoide_port` 是我们自己那份提交的回放，
    两侧都算会把自己重复计一次（实测该档一副占到 77.6%）；而训练要的权重是**对手场**分布。
    """
    seen = {}
    occ_all = collections.defaultdict(collections.Counter)
    occ_field = collections.defaultdict(collections.Counter)
    files = []
    for d in dirs:
        files += sorted(glob.glob(os.path.join(ROOT, d.strip(), "*.json")))
    print(f"扫描 {len(files)} 局回放")

    # ⚠️ 不能把 595 局的 JSON 全留在内存里（每局约 5 MB，解析后更大 → 会 OOM）。
    #    先用每目录前 3 个文件判定"这个目录是谁的对局"，再单遍处理。
    dir_names = collections.defaultdict(collections.Counter)
    for d in {os.path.basename(os.path.dirname(f)) for f in files}:
        for f in [x for x in files if os.path.basename(os.path.dirname(x)) == d][:3]:
            try:
                names = [a.get("Name") or "" for a in
                         ((json.load(open(f)).get("info") or {}).get("Agents") or [])]
            except Exception:
                continue
            for n in names:
                dir_names[d][n] += 1
    owner_of = {}
    for d, c in dir_names.items():
        top, n = c.most_common(1)[0]
        owner_of[d] = top if n > sum(c.values()) * 0.5 else None
    print(f"目录归属（>50% 同一 agent 才算'我们的局'）：{dict(sorted(owner_of.items()))}")

    for f in files:
        try:
            d_raw = json.load(open(f))
            ds = decks_of(d_raw.get("steps") or [])
        except Exception:
            continue
        band = os.path.basename(os.path.dirname(f))
        names = [a.get("Name") or "" for a in ((d_raw.get("info") or {}).get("Agents") or [])]
        owner = owner_of.get(band)
        for seat, ids in ds.items():
            if len(ids) != 60:
                continue
            hh = h(ids)
            seen.setdefault(hh, ids)
            occ_all[hh][band] += 1
            nm = names[seat] if seat < len(names) else ""
            if owner is None or owner not in nm:          # ★ 只算对手侧
                occ_field[hh][band] += 1
        del d_raw, ds

    # 语料表补成员（`data/deck_corpus.csv` 覆盖了 epsoide_teams2 等 1756 局的卡组，
    # 但**不含对手侧归属**，所以只用于"建池成员"，不参与权重）
    corpus = os.path.join(ROOT, "data", "deck_corpus.csv")
    if os.path.exists(corpus):
        n_add = 0
        with open(corpus) as fh:
            for r in csv.DictReader(fh):
                ids = [int(x) for x in r["deck"].split()]
                if len(ids) == 60 and r["deck_hash"] not in seen:
                    seen[r["deck_hash"]] = ids
                    n_add += 1
        print(f"语料表补充卡组 {n_add} 副（仅用于建池成员）")
    return seen, occ_all, occ_field


# 分档 = 采样框。每档只用"对手侧"出镜，各档互不混用（混用会造出一个
# 谁都没见过的假分布）。选档依据见 下一步.md §9.4：
BANDS = (
    # 我们自己的提交（~1000 分）：镜像 54.5%。最近一次定量 59%（n≈78，SE 5.6pp）。
    ("ours", {"epsoide_port"}),
    # 我们更早/更低分段的局：snover_mega_abomasnow 29.4%，长尾 90 副。
    ("mid", {"epsoide"}),
    # LB 域快照（另一分段，21 副，shuppet_dhelmise 23.9%）。
    ("lbfeed", {"epsoide_field"}),
    # 1465 档：Dipam 的 108 局，对手侧 107 次。与 EXP-0127 记录的六个原型份额
    # 逐位吻合（29.0/15.9/11.2/9.3/7.5/5.6 vs 28.7/15.7/11.1/9.3/7.4/5.6）。
    # ⚠️ 不要把 `epsoide_teams2` 并进来：它是"2 个 agent × 1000 局"的镜像堆
    #    （26c6e843 与 0d86f3a5 恰好各 1000 次，且 26c6e843 的 1024 次出镜
    #    100% 来自同一个 agent），两边都算会把那两人的卡组当成天梯场，
    #    与实测差 20pp。它只用于**建池成员**。
    ("top", {"epsoide_dipam"}),
)
EPS = 1e-4          # 未在该档出现的卡组的下限份额（不是 0：探索 + 新卡组）


def band_share(occ_field, good, srcs):
    tot = collections.Counter()
    for hh in good:
        for b, n in occ_field.get(hh, {}).items():
            if b in srcs:
                tot[hh] += n
    s = sum(tot.values())
    return {hh: v / s for hh, v in tot.items()} if s else {}


def write_weights(names, occ_field, good, manifest=None):
    """把某一档的实测分布写成 `version 3` 权重文件。

    必须**逐个写全池子里的每一副**：训练器 `weight_vector()` 用 `default`
    填满所有未列出的名字（native/train_native.py:178-180），所以只列出现过的
    那 20~30 副时，剩下 150+ 副会各拿 `default=1.0`，把"这一档 59% 是镜像"
    这类真实集中度直接稀释掉。反过来，文件里出现池子里没有的名字会让**整个
    请求被拒绝**（:175-177），所以名字只能取自同一份池子。
    """
    out = {}
    for tag, srcs in BANDS:
        share = band_share(occ_field, good, srcs)
        if not share:
            print(f"⚠️ {tag}: 该档没有样本，跳过")
            continue
        full = {hh: max(share.get(hh, 0.0), EPS) for hh in good}
        tot = sum(full.values())
        full = {k: v / tot for k, v in full.items()}
        path = os.path.join(ROOT, "decks", f"weights_{tag}.txt")
        with open(path, "w") as f:
            f.write("version 3\n")
            f.write(f"default {EPS}\n")
            for hh in sorted(full, key=lambda x: names[x]):
                f.write(f"{names[hh]} {full[hh]:.6f}\n")
        nz = sum(1 for hh in good if share.get(hh, 0.0) > 0)
        top = max(share.values())
        ess = 1.0 / sum(v * v for v in full.values())
        print(f"  weights_{tag}.txt：全池 {len(full)} 副（其中实测出现 {nz} 副），"
              f"最大份额 {top:.1%}，ESS {ess:.1f}")
        out[tag] = {"n_seen": nz, "n_written": len(full), "top_share": top, "ess": ess}
        if manifest is not None:
            manifest["weights"][tag] = out[tag]
    return out


def weights_from_manifest(path=None):
    """`--weights-only`：manifest 里已存了每副的 occ_field，不必再扫回放。"""
    path = path or os.path.join(ROOT, "decks", "pool_field", "_manifest.json")
    m = json.load(open(path, encoding="utf-8"))
    names = {d["hash"]: d["name"] for d in m["decks"]}
    occ_field = {d["hash"]: d["occ_field"] for d in m["decks"]}
    good = {d["hash"]: [0] * 60 for d in m["decks"]}
    # 名字按哈希稳定生成；manifest 里存的旧名可能已过期，这里顺带把
    # 目录里的 .csv 改成新名（deck id = 排序位置，所以必须整体一致）。
    sheet = load_card_sheet()
    newnames, ids_of = {}, {}
    for d in m["decks"]:
        cur = os.path.join(ROOT, "decks", "pool_field", d["name"] + ".csv")
        ids = [int(x) for x in open(cur).read().split()]
        ids_of[d["hash"]] = ids
        newnames[d["hash"]] = f"{slug(ids, sheet)}__{d['hash'][:4]}"
    renamed = sum(1 for d in m["decks"] if newnames[d["hash"]] != d["name"])
    for d in m["decks"]:
        cur = os.path.join(ROOT, "decks", "pool_field", d["name"] + ".csv")
        new = os.path.join(ROOT, "decks", "pool_field", newnames[d["hash"]] + ".csv")
        if cur != new:
            os.replace(cur, new)
        elif not os.path.exists(cur):
            open(cur, "w").write("\n".join(str(x) for x in ids_of[d["hash"]]) + "\n")
        d["name"] = newnames[d["hash"]]
    if renamed:
        print(f"按哈希稳定规则改名 {renamed} 副")
    good = {d["hash"]: ids_of[d["hash"]] for d in m["decks"]}
    names = {d["hash"]: d["name"] for d in m["decks"]}
    m["weights"] = {}
    write_weights(names, occ_field, good, m)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    print("已更新 _manifest.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="epsoide_port,epsoide_dipam,epsoide_field,epsoide")
    ap.add_argument("--out", default="decks/pool_field")
    ap.add_argument("--no-engine-check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--weights-only", action="store_true",
                    help="只按 _manifest.json 重算权重文件（不扫回放）")
    args = ap.parse_args()

    if args.weights_only:
        weights_from_manifest()
        return

    sheet = load_card_sheet()
    print(f"卡表 {len(sheet)} 张")
    seen, occ_all, occ_field = collect(args.dirs.split(","))
    print(f"去重后候选卡组 {len(seen)} 副")

    bad = collections.Counter()
    good = {}
    for hh, ids in seen.items():
        ok, why = check_legality(ids, sheet)
        if not ok:
            bad[why] += 1
            continue
        if not args.no_engine_check:
            ok, why = engine_ok(ids)
            if not ok:
                bad[why] += 1
                continue
        good[hh] = ids
    print(f"合法卡组 {len(good)} 副；丢弃 {sum(bad.values())} 副：{dict(bad)}")

    NEW_MIN = 1268
    with_new = sum(1 for ids in good.values() if max(ids) >= NEW_MIN)
    print(f"其中含新卡（id ≥ {NEW_MIN}）：{with_new} 副（{with_new / max(len(good), 1):.1%}）")

    # ⚠️ 后缀用**卡组哈希**，不用出镜排名：排名会随新数据加入而变化，
    #    改名会让已有权重文件里的名字全部失效（训练器对未知名字整份拒绝），
    #    而且 deck id = 排序后的位置，改名等于静默重排整个 id 空间。
    names = {hh: f"{slug(good[hh], sheet)}__{hh[:4]}" for hh in good}

    if args.dry_run:
        for tag, srcs in (("ours", {"epsoide_port"}), ("mid", {"epsoide", "epsoide_field"}),
                          ("top", {"epsoide_dipam", "epsoide_field"})):
            sh = band_share(occ_field, good, srcs)
            print(f"  [dry] weights_{tag}: {len(sh)} 副，最大份额 "
                  f"{(max(sh.values()) if sh else 0):.1%}")
        return

    outdir = os.path.join(ROOT, args.out)
    os.makedirs(outdir, exist_ok=True)
    for f in glob.glob(os.path.join(outdir, "*.csv")):
        os.remove(f)
    for hh, ids in good.items():
        with open(os.path.join(outdir, names[hh] + ".csv"), "w") as f:
            f.write("\n".join(str(x) for x in ids) + "\n")
    print(f"已写出 {len(good)} 副到 {args.out}/")

    manifest = {"decks": [], "weights": {}}
    write_weights(names, occ_field, good, manifest)
    for hh, ids in good.items():
        manifest["decks"].append({
            "name": names[hh], "hash": hh, "has_new_cards": int(max(ids) >= NEW_MIN),
            "occ_all": dict(occ_all[hh]), "occ_field": dict(occ_field[hh]),
        })
    with open(os.path.join(outdir, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("已写出 _manifest.json")


if __name__ == "__main__":
    main()
