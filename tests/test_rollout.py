"""rollout 模块的回归测试。

重点保护三件事：
  1. `rule_action` 必须与 main.py / agents/greedy.py **逐状态等价**
     —— 它是 rollout 的"大脑"，不一致就等于在评估另一个策略
  2. rollout 必须真的打到终局并返回合法结果（不能静默返回 None 蒙混过关）
  3. `choose_action` 的安全阀：所有候选打平时必须保持规则选择不动

运行：./run.sh tests/test_rollout.py
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "agents"))

from cg.api import (  # noqa: E402
    OptionType, SelectType, search_begin, search_step, search_end,
    to_observation_class,
)
from cg.game import battle_start, battle_select  # noqa: E402
from framework.filler import make_filler  # noqa: E402
from framework.rollout import (  # noqa: E402
    choose_action, evaluate_actions, has_type, index_of_type, is_main,
    rule_action, rollout_to_end,
)
from local_match import load_deck  # noqa: E402

OK = "\033[32m✅\033[0m"
NG = "\033[31m❌\033[0m"
_fail = 0


def check(cond, msg):
    global _fail
    print(f"  {OK if cond else NG} {msg}")
    if not cond:
        _fail += 1


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ============================================================ 1. 等价性
def test_equivalence():
    print("\n[1] rule_action 与 main.py / agents/greedy.py 逐状态等价")
    main = load_module(os.path.join(HERE, "main.py"), "t_main")
    greedy = load_module(os.path.join(HERE, "agents/greedy.py"), "t_greedy")

    # 用 10 副卡组而不是 6 副：修好"RETREAT 排在 ATTACK 之前"这个 bug 之后，
    # 对局明显变短（不再无限撤退），同样局数下的状态数少了一半。
    files = sorted(glob.glob(os.path.join(HERE, "decks", "*.csv")))[:10]
    decks = [load_deck(f) for f in files]
    total = mismatch = 0
    for i, dA in enumerate(decks):
        for j, dB in enumerate(decks):
            if i >= j:
                continue
            for rev in (False, True):
                obs = battle_start(dA, dB, reverse_player=rev)[0]
                for _ in range(250):
                    sel, cur = obs.get("select"), obs.get("current")
                    if cur["result"] != -1 or not sel:
                        break
                    total += 1
                    a_rule = rule_action(sel)
                    a_main = main.agent(obs)
                    a_greedy = greedy.agent(obs)
                    if not (a_rule == a_main == a_greedy):
                        mismatch += 1
                    obs = battle_select(a_main)
    check(total > 5000, f"覆盖 {total:,} 个状态（10 副卡组 × 45 对 × 2 种 reverse）")
    check(mismatch == 0, f"不一致 = {mismatch}")


# ============================================================ 2. rollout 终止性
def take_state_with_attack(deckA, deckB, limit=400):
    """走到一个 MAIN 且 ATTACK 可选的局面。"""
    obs = battle_start(deckA, deckB)[0]
    for _ in range(limit):
        sel, cur = obs.get("select"), obs.get("current")
        if cur["result"] != -1 or not sel or not sel.get("option"):
            return None
        if is_main(sel) and has_type(sel, OptionType.ATTACK):
            return obs
        obs = battle_select(rule_action(sel))
    return None


def test_rollout_runs_to_end():
    print("\n[2] rollout 真的打到终局")
    A = load_deck(os.path.join(HERE, "decks/sample_official.csv"))
    B = load_deck(os.path.join(HERE, "decks/fire_gouging.csv"))
    obs = take_state_with_attack(A, B)
    if obs is None:
        check(False, "找不到含 ATTACK 的局面（环境异常）")
        return
    my_index = obs["current"]["yourIndex"]
    filler = make_filler("uniform", my_deck=A)

    results = []
    for _ in range(4):
        world = filler.fill(obs, None, A)
        root = search_begin(agent_observation=to_observation_class(obs), **world).searchId
        st = search_step(root, [index_of_type(obs["select"], OptionType.ATTACK)])
        results.append(rollout_to_end(st.searchId, st.observation, my_index))
        search_end()

    check(all(r is not None for r in results),
          f"4 次 rollout 全部返回结果（None 数 = {sum(r is None for r in results)}）")
    check(all(r in (0.0, 0.5, 1.0) for r in results if r is not None),
          f"结果取值合法：{results}")


# ============================================================ 3. CRN 结构
def test_crn_structure():
    print("\n[3] evaluate_actions 的 CRN 结构")
    A = load_deck(os.path.join(HERE, "decks/sample_official.csv"))
    B = load_deck(os.path.join(HERE, "decks/fire_gouging.csv"))
    obs = take_state_with_attack(A, B)
    if obs is None:
        check(False, "找不到局面")
        return
    sel = obs["select"]
    base = rule_action(sel)[0]
    cands = [base]
    for want in (OptionType.ATTACK, OptionType.END):
        i = index_of_type(sel, want)
        if i is not None and i not in cands:
            cands.append(i)

    filler = make_filler("uniform", my_deck=A)
    out, info = evaluate_actions(obs, cands, filler=filler, my_deck=A, n_worlds=5)
    lens = {a: len(v) for a, v in out.items()}
    check(info["worlds"] == 5, f"成功采样 5 个 world（实际 {info['worlds']}）")
    check(all(n == 5 for n in lens.values()),
          f"每个候选都有 5 个结果（配对比较的前提）：{lens}")


# ============================================================ 4. 安全阀
def test_neutral_keeps_rule():
    print("\n[4] 候选打平时必须保持规则选择（安全阀）")
    A = load_deck(os.path.join(HERE, "decks/sample_official.csv"))
    B = load_deck(os.path.join(HERE, "decks/fire_gouging.csv"))
    obs = take_state_with_attack(A, B)
    if obs is None:
        check(False, "找不到局面")
        return
    base = rule_action(obs["select"])
    filler = make_filler("uniform", my_deck=A)
    # margin 设成无穷大 => 无论信号多强都不许偏离
    action, info = choose_action(obs, filler=filler, my_deck=A, n_worlds=4, margin=1e9)
    check(action == base, f"margin=1e9 时保持规则选择 {base}（实际 {action}）")
    check(info["reason"] != "", f"给出理由: {info['reason']}")


def test_non_main_untouched():
    print("\n[5] 非 MAIN 决策一律走规则（不去评估）")
    A = load_deck(os.path.join(HERE, "decks/sample_official.csv"))
    B = load_deck(os.path.join(HERE, "decks/fire_gouging.csv"))
    obs = battle_start(A, B)[0]
    filler = make_filler("uniform", my_deck=A)
    checked = 0
    for _ in range(120):
        sel, cur = obs.get("select"), obs.get("current")
        if cur["result"] != -1 or not sel:
            break
        if not is_main(sel):
            base = rule_action(sel)
            action, info = choose_action(obs, filler=filler, my_deck=A, n_worlds=3)
            if action != base:
                check(False, f"非 MAIN 决策被改动了: {base} -> {action}")
                return
            checked += 1
        obs = battle_select(rule_action(sel))
    check(checked > 0, f"覆盖 {checked} 个非 MAIN 决策，均未被改动")


def main():
    print("=" * 64)
    print("rollout 模块回归测试")
    print("=" * 64)
    test_equivalence()
    test_rollout_runs_to_end()
    test_crn_structure()
    test_neutral_keeps_rule()
    test_non_main_untouched()
    print("\n" + "=" * 64)
    if _fail:
        print(f"{NG} {_fail} 项失败")
        sys.exit(1)
    print(f"{OK} 全部通过")
    print("=" * 64)


if __name__ == "__main__":
    main()
