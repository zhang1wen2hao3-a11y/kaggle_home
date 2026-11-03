#!/usr/bin/env python3
"""**补丁 v2**：把"观测哪些卡组"从"过采样哪些卡组"里拆开 + 按座位输出实际卡组计数。

## 为什么 v1 不够（实测教训）

v1 把计数器挂在 `--anchor-decks` 上。但 `draw_deck()` 在 `p_anchor > 0` 时是这么挑的：

    int a = (int)rng_below(&g->rng, (uint32_t)env->n_anchors);   // 从 anchor 列表里随机挑

⇒ 为了"观测 7 副牌"把 7 副塞进 `--anchor-decks`，**学习者就被打散到 7 副牌上**
（`--p-anchor-learner 1.0` 不是"锁死某一副"，而是"100% 从 anchor 列表里抽一副"）。
实测日志 `d0 1.00 d1 0.67 d2 0.00 d3 0.67` 就是这个后果。

## v2 的语义

| 参数 | 作用 | 影响采样吗 |
|---|---|---|
| `--anchor-decks` | 过采样 + 逐 anchor 胜率 `d<i>` | ✅ 影响 |
| **`--track-decks`（新）** | **只观测**：按座位统计这些卡组各出现多少次 | ❌ **不影响** |

G4 的正确配法：

    --anchor-decks teal_mask_ogerpon_ex_applin__7e39                  # 学习者锁死这一副
    --track-decks shuppet_banette__f2af,shuppet_dhelmise__0d86,\
shuppet_dhelmise__fe82,teal_mask_ogerpon_ex_applin__7e39              # 只观测

期望日志：`tl3:1.00`（学习者永远 7e39）、`to0/to1/to2 ≈ 0.33`（对手 99% 打 shuppet）、`to3 ≈ 0.00`。

## 用法

    python3 tools/patch_deck_seat_counters.py --check   # 只看要不要打
    python3 tools/patch_deck_seat_counters.py            # 打（幂等，留 .bak_deckseat）
    bash native/build_native.sh
"""
import argparse
import os
import shutil
import sys

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = next((d for d in (_here, os.path.join(_here, "port_repo"))
             if os.path.isdir(os.path.join(d, "native/ptcg"))), _here)
PN = os.path.join(ROOT, "native", "ptcg")
H, E, B = (os.path.join(PN, x) for x in ("ptcg_env.h", "env.c", "binding.c"))
T = os.path.join(ROOT, "native", "train_native.py")

# ───────────────────────────── ptcg_env.h ─────────────────────────────
H_LOG_OLD = "    float deck_games[PT_MAX_ANCHORS], deck_wins[PT_MAX_ANCHORS];"
H_LOG_NEW = """    float deck_games[PT_MAX_ANCHORS], deck_wins[PT_MAX_ANCHORS];
    // TRACKED decks (--track-decks, independent of --anchor-decks): games each
    // tracked deck was actually dealt to each SEAT. Separate from deck_games
    // above because:
    //   * anchor_idx controls SAMPLING (draw_deck picks a random entry from it
    //     when p_anchor > 0), so stuffing decks in there to observe them also
    //     spreads the learner across all of them;
    //   * a league bank is a policy, NOT a deck (the opponent deck comes from
    //     the global weights), so only the opponent column can answer "what did
    //     the other seat actually play?" -- the G4 hard gate.
    float deck_games_trk[PT_MAX_ANCHORS], deck_games_opp[PT_MAX_ANCHORS];"""

H_CFG_OLD = """    int n_anchors;
    int anchor_idx[PT_MAX_ANCHORS];"""
H_CFG_NEW = """    int n_anchors;
    int anchor_idx[PT_MAX_ANCHORS];
    // Observation-only deck list (no sampling effect). 0 = fall back to the
    // anchor list so runs without --track-decks keep their old behaviour.
    int n_track;
    int track_idx[PT_MAX_ANCHORS];"""

# ─────────────────────────────── env.c ───────────────────────────────
# ⚠️ 计数必须放在 game_start（**每局无条件**），不能放在"已决局"分支里：
#    实测过放在已决分支会得到 to* 合计 = 0.00→0.02→0.15→0.61 随 ep_len 单调上升的假数据
#    （每局必有两副牌，to*/tl* 应恒为 ≈1）。已决口径会漏掉静默 reset 的局。
E_OLD = """    pt_decks_unlock();
    g->battle = pt_battle_start(g->deck_pair[0], g->deck_pair[1]);"""
E_NEW = """    pt_decks_unlock();
    // TRACK_COUNTERS: per-seat ACTUAL deck counts over --track-decks, counted
    // ONCE PER EPISODE (not per decided game): every game deals two decks, so
    // counting only decided games makes the ratio drift with game length --
    // measured 0.00/0.02/0.15/0.61 across four epochs while the learner side
    // stayed pinned at 1.00. Observation only; no sampling effect.
    {
        Log* trk_log = &env->log;
        int idl = g->deck_idx[g->learner_seat];
        int ido = g->deck_idx[1 - g->learner_seat];
        int nt = env->n_track > 0 ? env->n_track : env->n_anchors;
        const int* trk = env->n_track > 0 ? env->track_idx : env->anchor_idx;
        for (int i = 0; i < nt; i++) {
            if (trk[i] == idl) trk_log->deck_games_trk[i] += 1;
            if (trk[i] == ido) trk_log->deck_games_opp[i] += 1;
        }
    }
    g->battle = pt_battle_start(g->deck_pair[0], g->deck_pair[1]);"""

# ───────────────────────────── binding.c ─────────────────────────────
B_OLD = """    for (int i = 0; i < PT_MAX_ANCHORS; i++) {
        if (log->deck_games[i] > 0.0f) {
            dict_set(out, DECK_WR_KEYS[i], log->deck_wins[i] / log->deck_games[i]);
            dict_set(out, DECK_N_KEYS[i], log->deck_games[i]);
        }
    }"""
B_NEW = """    for (int i = 0; i < PT_MAX_ANCHORS; i++) {
        if (log->deck_games[i] > 0.0f) {
            dict_set(out, DECK_WR_KEYS[i], log->deck_wins[i] / log->deck_games[i]);
            dict_set(out, DECK_N_KEYS[i], log->deck_games[i]);
        }
    }
    // Per-seat counts over --track-decks (G4 hard gate).
    // dict_set stores the key POINTER, so these MUST be static literals.
    static const char* DECK_TRKL_KEYS[PT_MAX_ANCHORS] = {
        "deck_n_learner_0","deck_n_learner_1","deck_n_learner_2","deck_n_learner_3",
        "deck_n_learner_4","deck_n_learner_5","deck_n_learner_6","deck_n_learner_7",
        "deck_n_learner_8","deck_n_learner_9","deck_n_learner_10","deck_n_learner_11",
        "deck_n_learner_12","deck_n_learner_13","deck_n_learner_14","deck_n_learner_15"};
    static const char* DECK_TRKO_KEYS[PT_MAX_ANCHORS] = {
        "deck_n_opp_0","deck_n_opp_1","deck_n_opp_2","deck_n_opp_3",
        "deck_n_opp_4","deck_n_opp_5","deck_n_opp_6","deck_n_opp_7",
        "deck_n_opp_8","deck_n_opp_9","deck_n_opp_10","deck_n_opp_11",
        "deck_n_opp_12","deck_n_opp_13","deck_n_opp_14","deck_n_opp_15"};
    for (int i = 0; i < PT_MAX_ANCHORS; i++) {
        if (log->deck_games_trk[i] > 0.0f)
            dict_set(out, DECK_TRKL_KEYS[i], log->deck_games_trk[i]);
        if (log->deck_games_opp[i] > 0.0f)
            dict_set(out, DECK_TRKO_KEYS[i], log->deck_games_opp[i]);
    }"""

# ───────────────────── binding.c（解析 kwargs，v2 补漏） ─────────────────────
B_PARSE_OLD = """    env->p_anchor_learner =
        (it = dict_get_unsafe(kwargs, "p_anchor_learner")) ? (float)it->value : 0.0f;"""
B_PARSE_NEW = """    // TRACK_PARSE: observation-only deck list. 0 => fall back to the anchor
    // list (keeps old runs numerically identical). Without this parse the
    // counter loop would ALWAYS fall back -- which is exactly what the first
    // v2 verification run showed (tl0=1.00 for the anchor slot, all to*=0).
    env->n_track = (it = dict_get_unsafe(kwargs, "n_track")) ? (int)it->value : 0;
    if (env->n_track > PT_MAX_ANCHORS) env->n_track = PT_MAX_ANCHORS;
    if (env->n_track < 0) env->n_track = 0;
    for (int i = 0; i < env->n_track; i++) {
        char tkey[24];
        snprintf(tkey, sizeof(tkey), "track_%d", i);
        env->track_idx[i] = (it = dict_get_unsafe(kwargs, tkey)) ? (int)it->value : 0;
    }
    env->p_anchor_learner =
        (it = dict_get_unsafe(kwargs, "p_anchor_learner")) ? (float)it->value : 0.0f;"""

# ─────────────────────────── train_native.py ───────────────────────────
T_CLI_OLD = """    p.add_argument("--anchor-decks", default=ANCHOR_DECKS,
                   help="comma list of deck names to oversample")"""
T_CLI_NEW = """    p.add_argument("--anchor-decks", default=ANCHOR_DECKS,
                   help="comma list of deck names to oversample")
    p.add_argument("--track-decks", default="",
                   help="comma list of deck names to COUNT per seat (observation only, "
                        "no sampling effect). Empty = fall back to --anchor-decks.")"""

T_RES_OLD = """    a.deck_short = [n.split("__")[0][:24] for n in names]"""
T_RES_NEW = """    _tn = [s.strip() for s in (getattr(a, "track_decks", "") or "").split(",") if s.strip()]
    if _tn:
        _tab = os.environ.get("PTCG_TABLES", "native/ptcg_tables.bin")
        a.track_idx = resolve_anchor_indices(_tn, _tab, a.deck_pool)
        print(f"track decks (observation only): {dict(zip(_tn, a.track_idx))}", flush=True)
    else:
        a.track_idx = list(a.anchor_idx)
    a.deck_short = [n.split("__")[0][:24] for n in names]"""

T_KW_OLD = """            "n_anchors": len(a.anchor_idx),
            **{f"anchor_{i}": v for i, v in enumerate(a.anchor_idx)},"""
T_KW_NEW = """            "n_anchors": len(a.anchor_idx),
            **{f"anchor_{i}": v for i, v in enumerate(a.anchor_idx)},
            "n_track": len(getattr(a, "track_idx", [])),
            **{f"track_{i}": v for i, v in enumerate(getattr(a, "track_idx", []))},"""

T_LOG_ANCHOR = '''                  + "".join(f"d{b} {env.get(f'deck_wr_{b}', -1):.2f} "'''
T_LOG_INSERT = '''                  + "".join(f" tl{b}:{env.get(f'deck_n_learner_{b}', 0):.2f}"
                            f"/to{b}:{env.get(f'deck_n_opp_{b}', 0):.2f}"
                            for b in range(min(len(getattr(a, 'track_idx', [])), 7)))
'''

PATCHES = [
    (H, H_LOG_OLD, H_LOG_NEW, "deck_games_trk", "ptcg_env.h(Log)"),
    (H, H_CFG_OLD, H_CFG_NEW, "track_idx", "ptcg_env.h(cfg)"),
    (E, E_OLD, E_NEW, "TRACK_COUNTERS", "env.c"),
    (B, B_OLD, B_NEW, "deck_n_opp_0", "binding.c"),
    (B, B_PARSE_OLD, B_PARSE_NEW, "TRACK_PARSE", "binding.c(parse)"),
    (T, T_CLI_OLD, T_CLI_NEW, "--track-decks", "train_native.py(cli)"),
        (T, T_RES_OLD, T_RES_NEW, "a.track_idx = list(a.anchor_idx)", "train_native.py(resolve)"),
    (T, T_KW_OLD, T_KW_NEW, '"n_track"', "train_native.py(kwargs)"),
    (T, T_LOG_ANCHOR, T_LOG_INSERT + T_LOG_ANCHOR, "tl{b}:", "train_native.py(log)"),
]


def apply_one(path, old, new, marker, label, check):
    src = open(path, encoding="utf-8").read()
    if marker in src:
        print(f"  = {label}: 已打过")
        return 0
    if old not in src:
        print(f"  ✗ {label}: 找不到锚点，**未改动**（版本不同，请把该文件对应片段发我）")
        return 1
    if check:
        print(f"  ! {label}: 需要打")
        return 0
    bak = path + ".bak_deckseat"
    if not os.path.exists(bak):
        shutil.copyfile(path, bak)
    open(path, "w", encoding="utf-8").write(src.replace(old, new, 1))
    print(f"  ✓ {label}: 已打")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    print(f"补丁 v2（观测/过采样分离 + 按座位实际卡组计数）  root={ROOT}")
    for f in (H, E, B, T):
        if not os.path.exists(f):
            raise SystemExit(f"✗ 缺文件 {f}")
    bad = 0
    for p in PATCHES:
        bad += apply_one(*p, a.check)
    if bad:
        print("\n有文件没打上 ⇒ 先别编译，把 ✗ 的那份文件对应片段发我。")
        return 1
    if a.check:
        return 0
    print("\n下一步：")
    print("  1) bash native/build_native.sh")
    print("  2) 验证（学习者锁一副，只观测）：")
    print("       --anchor-decks teal_mask_ogerpon_ex_applin__7e39")
    print("       --track-decks shuppet_banette__f2af,shuppet_dhelmise__0d86,"
          "shuppet_dhelmise__fe82,teal_mask_ogerpon_ex_applin__7e39")
    print("       --timesteps 120000")
    print("     期望日志出现 tl3:1.00  to0:0.33 to1:0.33 to2:0.33 to3:0.00（量级）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
