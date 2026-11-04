# 卡组 ID 映射（阶段 D0，2026-11-04）

## 这份文件解决什么

训练器的 **deck id = `decks/pool_field/*.csv` 按文件名 sorted-glob 的位置**（0 起）。
往池里加卡组时，如果新名字排在有卡组中间，后面每一个 id 都会平移，
历史实验里的 `deck_id`、anchor、对位矩阵全部会指向错的牌 —— 而且是**静默**的。

阶段 D 要把两副头部卡组（0904 / e1d7）加进训练池，所以必须把这件事钉死。

## 结论：**既有 185 个 id 一个都没动**

新增的两副用 `z_` 前缀命名，保证它们在 sorted-glob 里排最后，作为 **id 185 / 186 追加**：

| new_id | 名字 | hash8 | 来源 |
|---|---|---|---|
| 185 | `z_dragapult_dreepy_petit__0904` | `0904e46f` | Petit Canard（#2，1519.9） |
| 186 | `z_ogerpon_toolbox_miwa__e1d7` | `e1d754a7` | やる気元気ミワハルキ 2nd（#1，1560.7） |

⇒ **所有历史实验结果（G5–G12、各对位矩阵、`models/ppo/*.json` 里的 `deck_id`）全都继续有效**，
例如 `dunsparce_abra__db04` 仍是 `deck_id=66`、`teal_mask_ogerpon_ex_applin__7e39` 仍是 `177`、
`shuppet_dhelmise__0d86` 仍是 `143`。

## 证据（不是口头声明）

对照对象是**打进历史训练包里的旧 blob**，不是我自己记的：

```bash
tar xzf dist/ptcg_train_kit_v2.tar.gz -C /tmp/oldkit \
    --wildcards '*/native/ptcg_tables_field.bin' '*/decks/pool_field/*.csv'
# 旧 blob: 185 行；新 blob: 187 行
# 逐行比对结果：名字相同 185/185，卡表相同 185/185，新增 2 行在末尾
```

逐项映射表在 **`decks/pool_field_id_map.tsv`**（列：`old_id / new_id / name / hash8 / status`，
`status ∈ {unchanged, added}`）。任何工具要用"旧 id → 新 id"请直接读这张表，不要靠文档推断。

## tables blob 怎么重建

blob 是构建产物（`port_repo/native/*.bin` 被 git 忽略），但从池子可以确定性重建：

```bash
cd port_repo
PYTHONPATH=data:. python native/tools/export_tables.py \
    native/ptcg_tables_field.bin --pool decks/pool_field
```

重建后必须校验（`pool` 的 sorted-glob 行与 blob 行逐行一致，否则禁止开训）：

```bash
cd port_repo && bash ../tools/db04_line.sh --check   # ① 池 187 = blob 187 ✓ ② anchor → deck id
```

## 训练包侧注意

`tools/make_line_kit.py` 生成的 `apply.sh` 现在会把包里的
`native/ptcg_tables_field.bin` 原样装到 `$REPO/native/`。
否则 GPU 机器上那份旧 185 行 blob 会在 `resolve_anchor_indices` 处直接拒训 ——
这是 2026-11-04 修掉的缺口。
