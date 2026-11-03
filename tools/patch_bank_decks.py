#!/usr/bin/env python3
"""给训练器加 **`--bank-decks`**：让每个冻结 bank **绑定自己的卡组**。

## 为什么需要

`env.c` 的 `game_start()` 里，冻结 bank 座位的卡组也是从**全局权重表** `deck_cw` 抽的：

```c
p_opp = (opp_kind == 2) ? p_anchor_league_opp : p_anchor_self_opp;   // 我们配 0.0
iO    = draw_deck(g, T, env, p_opp, n_opp_pool);                     // 纯按权重抽
```

⇒ 「bank={G8(shuppet)}」时，G8 **只有 shuppet 份额（8.3%）的局在打自己擅长的牌**。
审计（`下一步.md` §31.2）算过：`league_pct=0.08 × 8.3% ≈ 0.66%`，训练里几乎碰不到强对手。

有了这个补丁，可以写成「bank 0 = G8 且只打 `shuppet_dhelmise__0d86`」，
这才是 League Training（多卡组、每副牌一个强者）能落地的前提。

## 补丁内容（四处，幂等）

1. `native/ptcg/ptcg_env.h`：Env 结构加 `n_bank_decks` + `bank_deck[PT_MAX_BANKS]`
2. `native/ptcg/binding.c`：解析 `n_bank_decks` / `bank_deck_<i>`（沿用 `n_track` 的既有模式）
3. `native/ptcg/env.c`：`game_start()` 里若 `opp_kind==2` 且有绑定 ⇒ 覆盖对手座位卡组
4. `native/train_native.py`：新增 `--bank-decks`（与 `--league-ckpts` 一一对应，空位用 `-`）

用法（在 port_repo 根目录）：
    ./tools/py tools/patch_bank_decks.py --check     # 只看会改什么
    ./tools/py tools/patch_bank_decks.py             # 应用（自带 .bak_bankdecks 备份）
默认行为不变：不给 `--bank-decks` 时 `n_bank_decks=0`，走原来的全局权重抽样。
"""
import argparse
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(
    os.path.join(__file__, "..", "..")))) if False else os.getcwd()
REPO = os.path.join(ROOT, "port_repo") if os.path.isdir(os.path.join(ROOT, "port_repo")) else ROOT

H = os.path.join(REPO, "native/ptcg/ptcg_env.h")
B = os.path.join(REPO, "native/ptcg/binding.c")
E = os.path.join(REPO, "native/ptcg/env.c")
T = os.path.join(REPO, "native/train_native.py")

MARK_H = "n_bank_decks"          # 头文件里的判据
MARK_B = "n_bank_decks"          # binding.c 里的判据
MARK_E = "bank_deck override"    # env.c 里的判据
MARK_T = "bank-decks"            # train_native.py 里的判据


def backup(path):
    bak = path + ".bak_bankdecks"
    if not os.path.exists(bak):
        shutil.copyfile(path, bak)
    return bak


def patch_header(s):
    if MARK_H in s:
        return s, "已打过"
    anchor = "    int tag;\n"
    if anchor not in s:
        raise SystemExit("✗ ptcg_env.h: 找不到 `int tag;` 锚点")
    add = (
        "    // BANK_DECKS: bank index (1..n) -> deck id, -1 = free draw from deck_cw.\n"
        "    // n_bank_decks == 0 keeps the old behaviour (global weight draw) exactly.\n"
        "    int n_bank_decks;\n"
        "    int bank_deck[PT_MAX_BANKS + 1];\n"
    )
    return s.replace(anchor, anchor + add, 1), "已插入 n_bank_decks / bank_deck[]"


def patch_binding(s):
    if MARK_B in s:
        return s, "已打过"
    anchor = ('        env->track_idx[i] = (it = dict_get_unsafe(kwargs, tkey)) ? (int)it->value : 0;\n'
              '    }\n')
    if anchor not in s:
        raise SystemExit("✗ binding.c: 找不到 track_idx 解析块锚点")
    add = (
        "    // BANK_DECKS_PARSE: per-bank deck binding (see tools/patch_bank_decks.py).\n"
        "    // Absent => 0 => the old global-weight draw, numerically identical.\n"
        "    env->n_bank_decks = (it = dict_get_unsafe(kwargs, \"n_bank_decks\")) ? (int)it->value : 0;\n"
        "    if (env->n_bank_decks > PT_MAX_BANKS) env->n_bank_decks = PT_MAX_BANKS;\n"
        "    if (env->n_bank_decks < 0) env->n_bank_decks = 0;\n"
        "    for (int i = 0; i <= PT_MAX_BANKS; i++) env->bank_deck[i] = -1;\n"
        "    for (int i = 0; i < env->n_bank_decks; i++) {\n"
        "        char bkey[28];\n"
        "        snprintf(bkey, sizeof(bkey), \"bank_deck_%d\", i);\n"
        "        env->bank_deck[i] = (it = dict_get_unsafe(kwargs, bkey)) ? (int)it->value : -1;\n"
        "    }\n"
    )
    return s.replace(anchor, anchor + add, 1), "已插入 n_bank_decks 解析"


def patch_env(s):
    if MARK_E in s:
        return s, "已打过"
    anchor = "    int iO = draw_deck(g, T, env, p_opp, n_opp_pool);\n"
    if anchor not in s:
        raise SystemExit("✗ env.c: 找不到 `int iO = draw_deck(...)` 锚点")
    add = (
        "    // bank_deck override: a frozen bank may be PINNED to one deck (League\n"
        "    // Training needs \"this opponent is a specialist of deck X\"). Only the\n"
        "    // league seat is affected; tag 0 (self-play/random) never takes this path.\n"
        "    if (g->opp_kind == 2 && env->tag > 0 && env->tag <= env->n_bank_decks) {\n"
        "        int bd = env->bank_deck[env->tag - 1];\n"
        "        if (bd >= 0 && bd < T->n_decks) iO = bd;\n"
        "    }\n"
    )
    return s.replace(anchor, anchor + add, 1), "已插入 bank_deck override"


def patch_trainer(s):
    if MARK_T in s and "--bank-decks" in s:
        return s, "已打过"
    # ① argparse
    anchor_arg = '    p.add_argument("--track-decks", default="",'
    if anchor_arg not in s:
        raise SystemExit("✗ train_native.py: 找不到 --track-decks 参数锚点")
    add_arg = (
        '    p.add_argument("--bank-decks", default="",\n'
        '                   help="comma list aligned with --league-ckpts; each entry is a deck\\n'
        '                        name, or \'-\' to keep the global weight draw. PINS that frozen\\n'
        '                        bank to one deck (League Training: per-deck specialists).")\n'
    )
    s = s.replace(anchor_arg, add_arg + anchor_arg, 1)
    # ② 解析：放在 track 解析之后
    anchor_py = '    a.deck_short = [n.split("__")[0][:24] for n in names]\n'
    if anchor_py not in s:
        raise SystemExit("✗ train_native.py: 找不到 a.deck_short 锚点")
    add_py = (
        '    # BANK_DECKS: 与 --league-ckpts 逐位对应，"-" 表示不绑定（保持全局权重抽样）\n'
        '    _bd = [x.strip() for x in (getattr(a, "bank_decks", "") or "").split(",")]\n'
        '    _bd = [x for x in _bd if x]\n'
        '    a.bank_deck_idx = []\n'
        '    if _bd:\n'
        '        _names = [x for x in _bd if x != "-"]\n'
        '        _tab2 = os.environ.get("PTCG_TABLES", "native/ptcg_tables.bin")\n'
        '        _resolved = resolve_anchor_indices(_names, _tab2, a.deck_pool) if _names else []\n'
        '        _it = iter(_resolved)\n'
        '        a.bank_deck_idx = [next(_it) if x != "-" else -1 for x in _bd]\n'
        '        print(f"bank decks (PINNED): {dict(zip(_bd, a.bank_deck_idx))}", flush=True)\n'
    )
    s = s.replace(anchor_py, anchor_py + add_py, 1)
    # ③ 传给 env
    anchor_env = '            "n_track": len(getattr(a, "track_idx", [])),\n'
    if anchor_env not in s:
        raise SystemExit("✗ train_native.py: 找不到 env kwargs 的 n_track 锚点")
    add_env = (
        '            "n_bank_decks": len(getattr(a, "bank_deck_idx", []) or []),\n'
        '            **{f"bank_deck_{i}": v for i, v in enumerate(getattr(a, "bank_deck_idx", []) or [])},\n'
    )
    s = s.replace(anchor_env, add_env + anchor_env, 1)
    return s, "已加 --bank-decks（argparse + 解析 + env kwargs）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只报告会改什么")
    ap.add_argument("--repo", default=None, help="port_repo 根目录（默认自动探测）")
    a = ap.parse_args()
    repo = a.repo or REPO
    files = {
        H: patch_header, B: patch_binding, E: patch_env, T: patch_trainer,
    }
    if a.repo:
        try:
            sys.argv = sys.argv  # no-op
        except Exception:
            pass
    ok = True
    for path, fn in files.items():
        p2 = path
        if a.repo:
            p2 = os.path.join(a.repo, os.path.relpath(path, REPO))
        if not os.path.exists(p2):
            print(f"✗ 缺 {p2}")
            ok = False
            continue
        src = open(p2, encoding="utf-8").read()
        try:
            new, msg = fn(src)
        except SystemExit as e:
            print(f"✗ {os.path.basename(p2)}: {e}")
            ok = False
            continue
        print(f"{'（跳过）' if new == src else '待改：'}{os.path.basename(p2)}  {msg}")
        if not a.check and new != src:
            backup(p2)
            open(p2, "w", encoding="utf-8").write(new)
    if a.check:
        print("\n[--check] 未写入")
    elif ok:
        print("\n✅ 打完。下一步：重新编译原生扩展（`bash native/build_native.sh`），"
              "再用 `make -C native test` 与 `02_preflight.sh` 验证。")


if __name__ == "__main__":
    main()
