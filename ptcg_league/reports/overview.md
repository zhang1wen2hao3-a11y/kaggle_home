# 训练联赛 · 总览

- 规则 v2 sha `f69b708097c3ce1c`   代 2
- 当前 best：`lg_boot:port_plain`
- 池：init 7 / evo 0 / league 2（上限 16）

## 正式联赛池

| agent | 状态 | 卡组 | h2h 确认 | 固定评测集 | w_i |
|---|---|---|---|---|---|
| `sp:g0002_7e39_port_plain` | specialist | `teal_mask_ogerpon_ex_applin__7e39` | 0.5025 | — | 0.143 |
| `lg_boot:port_plain` | best | `teal_mask_ogerpon_ex_applin__7e39` | — | — | 0.143 |

## 最近 12 条事件

| 代 | 动作 | 对象 | 说明 |
|---|---|---|---|
| 1 | reject | `g0001_7e39_port_plain` | screen_score=0.4750<0.52 |
| 2 | specialist | `sp:g0002_7e39_port_plain` |  |

## 每代判定

| 代 | 分支 | 候选 h2h | CI | 判定 | 原因 |
|---|---|---|---|---|---|
| 1 | `g0001_7e39_port_plain` | — | — | ❌ 淘汰 | screen_score=0.4750<0.52 |
| 2 | `g0002_7e39_port_plain` | — | — | ⭐ 特色对手 | 总体 0.5025<0.52，但shuppet 对位配对提升 +0.0525 CI 下界 +0.0071>0（n=400 |
