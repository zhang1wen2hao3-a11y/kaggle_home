#!/usr/bin/env bash
# 拉取 PTCG 比赛数据到 data/。数据不在仓库里，换机器时先跑这个。
set -euo pipefail
SLUG="the-pokemon-company-ptcg-ai-battle-challenge-playground"
HERE="$(cd "$(dirname "$0")" && pwd)"
CRED="${KAGGLE_JSON:-$HOME/.kaggle/kaggle.json}"

U=$(python3 -c "import json;print(json.load(open('$CRED'))['username'])")
K=$(python3 -c "import json;print(json.load(open('$CRED'))['key'])")

echo "== 下载 $SLUG =="
curl -s -u "$U:$K" -L -o /tmp/ptcg_data.zip \
  "https://www.kaggle.com/api/v1/competitions/data/download-all/$SLUG"
mkdir -p "$HERE/data"
python3 -c "
import zipfile
zipfile.ZipFile('/tmp/ptcg_data.zip').extractall('$HERE/data')
"
rm -f /tmp/ptcg_data.zip
echo "完成：$(find "$HERE/data" -type f | wc -l) 个文件已就位到 $HERE/data"
