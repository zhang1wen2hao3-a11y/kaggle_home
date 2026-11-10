# models/archive —— 退役 agent 仓（Tier B/C）

由 `python3 tools/agent_tier.py --apply` 生成，索引见 `index.tsv`。

## 分级判据（固化在 tools/agent_tier.py）

| 判据 | 内容 | 结果 |
|---|---|---|
| ① 角色必需 | 面板 7 对手（覆盖评估基线）+ 当前代工作模型 | Tier A，留 `build/agents/` |
| ② 活代码引用 | 被 `tools/`・`ptcg_league/`・`decks/` 的 `.sh/.py/.tsv` 引用（`logs/`、`docs/` 的提及**不算**） | Tier B |
| ③ 真实天梯 | `data/lb_ours.json` 记录的我们自己提交分数 | ≥1010 → Tier A；有分数但被超越 → Tier B |

## 现状

- Tier A 13 个（266 MB）留在 `build/agents/`：面板 7 + `g6_db04`/`port_db04`（天梯 1060.3/1042.3）+ 当前代 `g8_bc_parent`/`g8_shuppet`/`bc_3eaa`/`bc_26c6`
- Tier B 21 个、Tier C 28 个 → 本目录（49 个，944 MB 账面）
- `build/agents/router_db01` 是指向 `dist/router_router_db01` 的符号链接，保持原位

## 使用约定

- 归档模型**仍可被评估**：直接写路径，如 `--cand models/archive/g6_7e39`
- `tools/eval_specialist.sh`、`tools/cross_matrix.sh` 已内置回退解析（先 `build/agents/`，再 `models/archive/`）
- 其余历史分析脚本（`arena_log.py`/`exp_registry.py`/`specialist_value.py`/`bc_screen.sh`/`bc_category_acc.py`）里硬编码的 B 类名字需改成 `models/archive/<name>`
- `head_26c6_A1/B1`、`head_3eaa_C1` 的 `model.pt` 是指向 `/tmp` 的失效符号链接，本体已丢失（各 0.4 MB）
- 复原办法：`dist/ptcg_sub_*.tar.gz` 里含完整权重（`g6_7e39`/`g6_9d6e`/`g14a_e1d7` 均已验证）
