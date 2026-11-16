#!/usr/bin/env python3
"""**驾驶员注册表（唯一的名字 → 实体映射）**

为什么要有这个文件
------------------
`docs/the-pockmon-battle-choice*.csv` 里的"驾驶员"列是**字符串名字**，训练流水线要拿它
去生成对手（bank / 测试集 agent 目录）。历史上名字散落在三处硬编码：

* `tools/make_testset_A.py`  的 `DRIVER_AGENT`（名字 → agent 目录名）
* `tools/choice_to_banks.py` 的 `DEFAULT_PILOT_MAP`（名字 → GPU 机上的 .bin）
* 各种一次性脚本里的手工字典

三处不同步 ⇒ 表里出现脚本不认识的名字时，脚本会**静默回退到 port**，
于是"用最强驾驶员当对手"的评测其实一直在用弱端口，结果却记成了强驾驶员。
（2026-11-14 的 98% vs 8.5% 事故就是这一类。）

规则
----
1. 表里出现的每个非 `init` 名字，都必须在这里登记；`tools/check_drivers.py` 会扫表校验。
2. 未知名字 ⇒ 调用方必须**报错退出**，不许静默回退。
3. **一个实体只有一个名字**，不登记别名（历史上 `shuppet_specialist`/`g8_bc_parent` 是同一权重，已统一为 `g8_bc_parent`，旧名出现即报错）。
4. 权重丢了的名字登记在 `RETIRED`，check 时只提示、不报错（历史表里仍在引用）。

用法
----
    ./tools/py tools/driver_registry.py list
    ./tools/py tools/driver_registry.py resolve r1A
    ./tools/py tools/driver_registry.py check        # 扫 docs/ 下的交付表
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 每个驾驶员：agent = 评估机上的 build/agents/<agent>；weights = 本地权重；remote = GPU 机上的纯权重 .bin
# deck = 该驾驶员是"针对哪副牌"训出来的（None = 跨牌能通用）；fidelity = docs/opponent_library.md 的 A 一致率
DRIVERS = {
    "init": dict(agent=None, weights=None, remote=None, deck=None,
                 note="端口权重；所有卡组通用（对手侧就是目前 12 panel 的默认对手）"),

    # ── shuppet 0d86 ─────────────────────────────────────────────
    # ★ 唯一名字：g8_bc_parent（旧名 shuppet_specialist 已弃用，不要再用/不要再加别名）
    "g8_bc_parent": dict(agent="g8_bc_parent", weights="build/agents/g8_bc_parent/model.pt",
                         remote="/root/kl/g8_bc_parent.bin", deck="0d86", fidelity=0.739,
                         note="ミワハルキ BC 克隆；全帧 171k；对 g6_db04 0.9675、同牌镜像 vs 端口 0.9325 ⇒ 唯一合格的强对手"),
    "g8_shuppet": dict(agent="g8_shuppet", weights="build/agents/g8_shuppet/model.pt",
                       remote="/root/kl/g8_shuppet_bc.bin", deck="0d86", fidelity=None,
                       note="PPO 版 shuppet 驾驶员（vs BC 克隆 0.60，vs g6_db04 0.945）；可当 league bank"),

    # ── slowking 26c6 ───────────────────────────────────────────
    "r1A": dict(agent="r1A_26c6", weights="models/ppo/r1A.pt",
                remote="/root/kl/r1A_slowking.bin", deck="26c6", fidelity=None,
                note="slowking 10M 专才（26c6 行在顶分段口径下 +6.8pp）"),
    "bc_26c6": dict(agent="bc_26c6", weights="models/ppo/bc_26c6.pt",
                    remote="/root/kl/bc_26c6.bin", deck="26c6", fidelity=0.895,
                    note="ミワハルキ slowking BC；保真最高但 B 标准不强（同牌镜像 0.2625 < 端口）"),

    # ── db04 ────────────────────────────────────────────────────
    "g6_db04": dict(agent="g6_db04", weights="models/ppo/g6_db04.pt",
                    remote="/root/kl/g6_db04.bin", deck="db04", fidelity=None,
                    note="G6 代 db04 权重（天梯 1082.1 那次提交的模型）；行级为负收益，别再用它当驾驶员"),
    "g6_db04_on_2617": dict(agent="g6_db04_on_2617", weights="models/ppo/g6_db04.pt",
                            remote=None, deck="26c6", fidelity=None,
                            note="同一权重换开 26c6（md5 与 g6_db04 相同）"),

    # ── 2617 / 9d6e / 7e39 / e1d7 ───────────────────────────────
    "ph3_2617_v1": dict(agent="ph3_2617_v1", weights="build/agents/ph3_2617_v1/model.pt",
                        remote="/root/kl/ph3_2617_v1.bin", deck="2617", fidelity=None,
                        note="第 3 期 2617 候选 v1（顶分段口径 2617 行最佳）"),
    "ph3_2617_v2": dict(agent="ph3_2617_v2", weights="build/agents/ph3_2617_v2/model.pt",
                        remote="/root/kl/ph3_2617_v2.bin", deck="2617", fidelity=None, note="第 3 期 2617 候选 v2"),
    "ph3_9d6e_v2": dict(agent="ph3_9d6e_v2", weights="build/agents/ph3_9d6e_v2/model.pt",
                        remote="/root/kl/ph3_9d6e_v2.bin", deck="9d6e", fidelity=None,
                        note="第 3 期 9d6e 候选 v2；行级负收益"),
    "ab9d6e_r1A": dict(agent="ab9d6e_r1A", weights="models/ppo/ab9d6e_r1A.pt",
                       remote="/root/kl/ab9d6e_r1A.bin", deck="9d6e", fidelity=None, note="第 1 轮 A 臂"),
    "ab9d6e_r1B": dict(agent="ab9d6e_r1B", weights="models/ppo/ab9d6e_r1B.pt",
                       remote="/root/kl/ab9d6e_r1B.bin", deck="9d6e", fidelity=None, note="第 1 轮 B 臂"),
    "ab9d6e_r2A": dict(agent="ab9d6e_r2A", weights="models/ppo/ab9d6e_r2A.pt",
                       remote="/root/kl/ab9d6e_r2A.bin", deck="9d6e", fidelity=None,
                       note="第 2 轮 A 臂（9d6e 行唯一过线：vs e8ee +15.7pp、vs f5f8 +14.0pp，n=600）"),
    "ab9d6e_r2B": dict(agent="ab9d6e_r2B", weights="models/ppo/ab9d6e_r2B.pt",
                       remote="/root/kl/ab9d6e_r2B.bin", deck="9d6e", fidelity=None, note="第 2 轮 B 臂"),
    "bc_3eaa": dict(agent="bc_3eaa", weights="build/agents/bc_3eaa/model.pt",
                    remote=None, deck="3eaa", fidelity=None,
                    note="dreepy_drakloak__3eaa 的 BC 克隆（2026-11-08，Dipam Chakraborty 的 146,938 帧，"
                         "data/bc_3eaa.npz）。长期只当对手/陪练存在，没登记也没量过强度；"
                         "2026-11-16 复盘才确认它是 dreepy 家族唯一的『人怎么打』信号"),
    "cand7e39S": dict(agent="cand7e39S", weights="models/ppo/cand7e39S.pt",
                      remote="/root/kl/cand7e39S.bin", deck="7e39", fidelity=None,
                      note="7e39 舰牌候选；行级负收益"),
    # ── 特攻训练出来的专家（命名规则：cand<卡组4位>S，S=specialist）───────
    #    注意：名字里带卡组号但**不等于**卡组号，避免和 26c6 / f5f8 这些卡组名混淆
    "cand26c6S": dict(agent="cand26c6S", weights="models/ppo/cand26c6S.pt",
                      remote="/root/kl/cand26c6S.bin", deck="26c6", fidelity=None,
                      note="slowking 特攻（2026-11-15，BC 起点+4 强陪练+4M 步，LR 1e-4）；"
                           "训练内原名 deck_26c6__db04 / 输出 slowking_shup4m"),
    "cand26c6M": dict(agent="cand26c6M", weights="models/ppo/cand26c6M.pt",
                      remote="/root/kl/cand26c6M.bin", deck="26c6", fidelity=None,
                      note="slowking 特攻 · 镜像强化版（2026-11-15，INIT=cand26c6S，陪练换成 2 个 slowking "
                           "镜像 + shuppet + f5f8，4M 步/43 min 于 3090）。"
                           "镜像 vs r1A 39.9%→55.8%、H2H vs cand26c6S 59.5%，但整体弱（全梯场 49.7%）"),
    "candf5f8G": dict(agent="candf5f8G", weights="models/ppo/candf5f8G.pt",
                      remote=None, deck="f5f8", fidelity=None,
                      note="f5f8 补短板版（2026-11-16，INIT=candf5f8S，8 bank 里 3 个绑 2617 / 2 个绑 3eaa，"
                           "4M 步/43 min 于 3090）。实测（n=600/格，3 种子）：2617 格 46.0%→54.2%（+8.2pp ✓）"
                           "、强 2617 驾驶员 42.5%→47.7%（+5.2pp ✓）、3eaa 格 33.5%→35.5%（+2.0pp ✗ 未达标）；"
                           "方案A 面板 61.3%→62.7%（+1.4pp，n=450）；直接对打 vs candf5f8S 52.7%（n=600，未显著）；"
                           "vs shuppet 90.0% 持平。⇒ 2617 修好、3eaa 没修好，整体略优于 candf5f8S"),
    "candf5f8S": dict(agent="candf5f8S", weights="models/ppo/candf5f8S.pt",
                      remote="/root/kl/candf5f8S.bin", deck="f5f8", fidelity=None,
                      note="f5f8 特攻（2026-11-15，4,030,464 步）；训练内原名 deck_f5f8__db04，"
                           "快照曾用临时名 f5f8s_<步数>，正式名只用 candf5f8S"),

    # ── BC 克隆（人类替身，同时用作陪练/考官）───────────────────
    "deck_0d86_": dict(agent="deck_0d86_", weights="bc_pool/deck_0d86_.pt",
                       remote="/root/kl/deck_0d86_.bin", deck="0d86", fidelity=0.731,
                       note="全玩家 shuppet BC（171k 帧）；陪练强度 85%，有梯度"),
    "deck_f5f8_": dict(agent="deck_f5f8_", weights="bc_pool/deck_f5f8_.pt",
                       remote="/root/kl/deck_f5f8_.bin", deck="f5f8", fidelity=0.696,
                       note="f5f8 的 BC 克隆（76.8k 帧）；天梯上它是 slowking 的克星，必进 slowking 陪练池"),
    "agent_sue124_": dict(agent="agent_sue124_", weights="bc_pool/agent_sue124_.pt",
                          remote="/root/kl/agent_sue124_.bin", deck="0d86", fidelity=0.668,
                          note="sue124 的 shuppet BC；实测被 f5f8 BC 打 100% ⇒ **无梯度，别当陪练**"),
    "candE1D7S": dict(agent="candE1D7S", weights="models/ppo/candE1D7S.pt",
                      remote="/root/kl/candE1D7S.bin", deck="e1d7", fidelity=None, note="e1d7 舰牌候选"),
}

# ★ 一个实体只能有一个名字：这里**不保留任何别名**。
#   历史名 `shuppet_specialist` 已彻底弃用（= 本文件里的 `g8_bc_parent`，同一权重）；
#   任何表/脚本里再出现旧名，`driver_registry.py check` 会直接报错。
ALIASES = {}

# 权重已丢失的历史名字：历史表里仍可能出现，check 只提示不报错
RETIRED = {
    "bc_grookey3": "2617 的 BC 克隆，权重未留存（只在 -choice-all.csv 历史档里）",
    "ph3_db04_v1": "db04 第 3 期 v1，权重未留存",
    "ph3_9d6e_v1": "9d6e 第 3 期 v1，权重未留存",
}


# 旧名提示：**不是别名**，只是报错时给一句人话（不会把它解析成别的名字）
HINTS = {
    "shuppet_specialist": "该名字已弃用，唯一名字是 g8_bc_parent",
    "port": "端口权重请写 init",
    "port_w": "端口权重请写 init",
}


class UnknownDriver(KeyError):
    pass


def canonical(name: str) -> str:
    """名字规范化：只做去空格/大小写不动，不认别名（别名一律当未知处理）。"""
    return (name or "init").strip()


def info(name: str) -> dict:
    n = canonical(name)
    if n not in DRIVERS:
        if n in RETIRED:
            return dict(retired=True, note=RETIRED[n], agent=None, weights=None, remote=None, deck=None)
        hint = HINTS.get(n)
        raise UnknownDriver(f"未登记的驾驶员：{name!r}"
                            + (f"（{hint}）" if hint else "（请在 tools/driver_registry.py 里登记）"))
    return dict(DRIVERS[n], retired=False, name=n)


def agent_dir(name: str):
    """评估机上的 agent 目录（None = 用该牌原有的 port agent）。"""
    d = info(name)
    return f"build/agents/{d['agent']}" if d.get("agent") else None


def weights_path(name: str):
    d = info(name)
    w = d.get("weights")
    return os.path.join(ROOT, w) if w else None


def remote_bin(name: str):
    return info(name).get("remote")


def deck_of(name: str):
    return info(name).get("deck")


DECK_IDS = {"0904", "0bf5", "0d86", "2617", "26c6", "3eaa", "7e39", "9d6e", "db04", "e1d7", "e8ee", "f5f8"}


def check_names() -> int:
    """护栏：① 驾驶员名不许等于卡组号（否则表里"驾驶员"和"被挑战者"会撞）
             ② 名字唯一（不设别名，一个实体一个名字）"""
    bad = 0
    for n in DRIVERS:
        if n.lower() in DECK_IDS:
            print(f"  ✗ 驾驶员名 {n!r} 与卡组号相同，容易混淆"); bad += 1
    dup = [k for k, v in ALIASES.items() if v in ALIASES]
    if dup:
        print(f"  ✗ 别名链 {dup}（不允许）"); bad += len(dup)
    print(f"驾驶员名检查：{len(DRIVERS)} 个，错误 {bad} 条")
    return bad


def check_tables(paths=None) -> int:
    """扫交付表里出现的驾驶员名字，校验是否都能解析。返回错误数。"""
    import csv, glob, io
    if paths is None:
        pats = ["the-pockmon-battle-choice*.csv", "the-pockmon-battle-choices-win-rate*.csv"]
        paths = sorted(p for pat in pats for p in glob.glob(os.path.join(ROOT, "docs", pat)))
    bad, seen = [], {}
    for p in paths:
        try:
            rows = [r for r in csv.reader(io.StringIO(open(p, "rb").read().decode("gb18030", "replace"))) if r]
        except OSError:
            continue
        if not rows:
            continue
        h = rows[0]
        cols = [i for i, c in enumerate(h) if "驾驶员" in c]
        for r in rows[1:]:
            for i in cols:
                if i >= len(r):
                    continue
                name = r[i].strip()
                if name in ("", "/", "待测", "init"):
                    continue
                seen.setdefault(name, set()).add(os.path.basename(p))
    for name, files in sorted(seen.items()):
        try:
            d = info(name)
        except UnknownDriver as e:
            bad.append((name, f"未登记：{e}"))
            continue
        if d["retired"]:
            print(f"  ⚠️ {name:24} 历史名字，权重已丢失（{d['note']}）  出现在 {sorted(files)}")
            continue
        if d["weights"] and not os.path.exists(weights_path(name)):
            bad.append((name, f"权重文件缺失：{d['weights']}"))
        if d["agent"] and not os.path.isdir(os.path.join(ROOT, "build", "agents", d["agent"])):
            print(f"  ⚠️ {name:24} 本地无 agent 目录 build/agents/{d['agent']}（评估机上需要有）")
    print(f"\n表中出现的非 init 驾驶员 {len(seen)} 个；错误 {len(bad)} 条")
    for n, why in bad:
        print(f"  ✗ {n}: {why}")
    return len(bad)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for n, d in DRIVERS.items():
            w = d.get("weights") or "-"
            ok = "✓" if (d.get("weights") is None or os.path.exists(os.path.join(ROOT, w))) else "✗"
            print(f"{n:22} {ok} deck={d.get('deck') or '-':6} agent={d.get('agent') or '-':16} {w}")
        print("\n别名:", ", ".join(f"{k}→{v}" for k, v in ALIASES.items()))
        print("已退役:", ", ".join(RETIRED))
        return 0
    if cmd == "resolve":
        d = info(sys.argv[2])
        print(d)
        return 0
    if cmd == "check":
        bad = check_names()
        bad += check_tables()
        return 1 if bad else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
