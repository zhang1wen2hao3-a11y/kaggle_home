.PHONY: list md data clean

list:            ## 列出实验记录
	python exp_log.py list

md:              ## 重新生成 实验记录.md
	python exp_log.py md

data:            ## 拉取比赛数据到 data/
	bash fetch_data.sh

clean:           ## 清理缓存
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
