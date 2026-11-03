# port_repo 补丁（2026-11-04 备份）

`port_repo/` 是上游仓库 `github.com/dipamc/kaggle-ptcg-ai-battle` 的克隆（109 MB，含自己的 `.git`），
它被主仓库 `.gitignore` 排除，所以**它不在 GitHub 上**。本目录把我们对它唯一的改动存下来，
换机器时可以原样重建。

## 内容

| 文件 | 说明 |
|---|---|
| `port_repo_1c7c475_seat_and_bank_decks.patch` | `git format-patch -1` 导出，3.4 MB（含 200 个卡组文件与 embed_pca.npz 的二进制 diff） |

改动内容（提交信息全文见 patch 头部）：

- `native/ptcg/{env.c,ptcg_env.h,binding.c}`：v2 座位计数器（`n_track`/`track_i`/`tl*`/`to*`）
  与 **PINNED bank decks**（`n_bank_decks`/`bank_deck[]`/`--bank-decks`，启动打印
  `bank decks (PINNED): {...}`）
- `native/train_native.py`：把 `--bank-decks` / `--league-pct` 透传到 C 侧
- `native/model/ptcg_model.cu`、`native/tools/native_weights.py`、`ptcg/rl/{buffers,cards}.py`、
  `tools/deck_matrix.py`：加宽/导出兼容与口径修正
- `decks/pool_field/` + `decks/weights_*.txt`：185 条 ladder 派生卡组与采样权重
- `tools/export_embed_from_ckpt.py`、`tools/extend_embed_pca.py`：embedding PCA 扩展

## 重建

```bash
git clone https://github.com/dipamc/kaggle-ptcg-ai-battle.git port_repo
cd port_repo && git am ../patch/port_repo_1c7c475_seat_and_bank_decks.patch
# 若 git am 因二进制/上下文冲突失败，退回手动三路合并：
#   git apply -3 ../patch/port_repo_1c7c475_seat_and_bank_decks.patch
```

## 等价物

`dist/ptcg_train_kit_v2.tar.gz`（57 MB，sha `9b3872f01fb6abd1`）里的 `port_repo/` 已经是打好补丁的版本，
且自带 `setup_gpu_train.sh` / `tools/db04_line.sh` / `tools/patch_bank_decks.py`，
比打 patch 更省事 —— 但它在 `.gitignore` 的 `dist/` 里，也不在 GitHub 上，需要单独传输。
