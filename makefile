.PHONY: check list md data clean test gen train eval sweep

# 所有命令都走 ./run.sh —— 它会自动挑到依赖齐全的解释器（py310），
# 避免"装了 lightgbm 却 import 不到"这类问题。

# ---------------------------------------------------------------- 环境
check:           ## 环境诊断：解释器、依赖、引擎、框架后端
	./run.sh --check

# ---------------------------------------------------------------- 实验记录
list:            ## 列出实验记录（按胜率降序）
	./run.sh exp_log.py list

md:              ## 重新生成 实验记录.md
	./run.sh exp_log.py md

# ---------------------------------------------------------------- 数据与评估
data:            ## 拉取比赛数据到 data/（换机器时先跑这个）
	bash fetch_data.sh

eval:            ## 评估提交入口 vs random 基线（200 局）
	./run.sh evaluate.py --agent main.py --opponent agents/random_baseline.py --games 200

sweep:           ## 扫描策略配置（每配置 60 局）
	./run.sh sweep.py 60

# ---------------------------------------------------------------- 训练
gen:             ## 生成自对弈数据（两个对手各 300 局并合并）
	./run.sh gen_data.py --games 300 --opponent agents/random_baseline.py \
		--out data/sp_vs_random.jsonl --game-offset 0
	./run.sh gen_data.py --games 300 --opponent agents/greedy.py \
		--out data/sp_vs_greedy.jsonl --game-offset 100000
	cat data/sp_vs_random.jsonl data/sp_vs_greedy.jsonl > data/selfplay_v1.jsonl

train:           ## 训练价值模型 V（MODEL=lightgbm 可换后端）
	./run.sh train_value.py --data data/selfplay_v1.jsonl --model $(or $(MODEL),lightgbm)

# ---------------------------------------------------------------- 测试
test:            ## 跑回归测试
	./run.sh tests/test_memory.py

clean:           ## 清理缓存
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
