#!/usr/bin/env python3
"""生成**评估愿望单**：所有还想要的 (agent, tag, 种子, 协议)，两台上机按 shard 分工跑。

    ./tools/py tools/gen_wishlist.py > /tmp/wishlist.tsv

每行：`agent_dir<TAB>tag_base<TAB>seed<TAB>proto<TAB>shard`
- proto = v1（对手=端口，原 12 格测试集）或 A（对手=各自最佳驾驶员）
- shard = 0/1，轮流分配；两台机器各跑一个 shard，互不重复
"""
import sys

# (agent 目录, tag 基名)  —— tag 会再加 _s<seed>（v1）或 _A_s<seed>（A 协议）
PAIRS = [
    ("cand7e39S", "cand7e39S"), ("port_7e39", "port_7e39"),
    ("port_0d86", "port_0d86"), ("g8_bc_parent", "g8_bc_parent"), ("g8_shuppet", "g8_shuppet"),
    ("candE1D7S", "candE1D7S"), ("port_e1d7", "port_e1d7"),
    ("port_db04", "port_db04"), ("port_0904", "port_0904"), ("opp_dunsparce_0bf5", "port_0bf5"),
    ("opp_3eaa", "port_3eaa"), ("port_e8ee", "port_e8ee"), ("c_abra", "port_f5f8"),
    ("port_26c6", "port_26c6"), ("r1A_26c6", "r1A_26c6"),
    ("c_grookey", "port_2617"), ("ph3_2617_v1", "ph3_2617_v1"), ("ph3_2617_v2", "ph3_2617_v2"),
    ("ab9d6e_r1A", "ab9d6e_r1A"), ("ab9d6e_r2A", "ab9d6e_r2A"),
    ("ab9d6e_r1B", "ab9d6e_r1B"), ("ab9d6e_r2B", "ab9d6e_r2B"),
    ("bc_26c6", "bc_26c6"), ("ph3_9d6e_v2", "ph3_9d6e_v2"), ("g6_db04", "meta_g6_db04"),
]
# A 协议（对手=各自最佳驾驶员）只跑"我方会被选中的那些行"
A_ROWS = ["cand7e39S", "port_7e39", "g6_db04", "port_on_9d6e", "ab9d6e_r2A",
          "g8_bc_parent", "r1A_26c6", "ph3_2617_v1", "candE1D7S", "port_e1d7"]
SEEDS = [950000, 950001, 950002]

rows = []
for ag, tag in PAIRS:
    for sd in SEEDS:
        rows.append((ag, tag, sd, "v1"))
for ag, tag in [(x, x) for x in A_ROWS]:
    for sd in SEEDS:
        rows.append((ag, tag, sd, "A"))
for i, (ag, tag, sd, proto) in enumerate(rows):
    print(f"{ag}\t{tag}\t{sd}\t{proto}\t{i % 2}")
