#!/bin/bash
C=/mnt/d/penguin_data/code/Web3_Auditor/tools/colab_qwen_audit/.cli-venv/bin/colab
D=/mnt/d/penguin_data/code/quantitative_trading_course/ML4T_Midterm/artifacts/deep
$C exec -s ml4t-deep -f $D/colab/check.py 2>&1 | grep -v -i "update\|colab cli is available\|silence this"
for f in bundle.tar.gz meta.parquet static.npy mask.npy X.npy; do
  $C upload -s ml4t-deep $D/$f /content/deep/$f 2>&1 | grep -v -i "update\|available\|silence" | tail -2
done
$C exec -s ml4t-deep -f $D/colab/check.py 2>&1 | grep -v -i "update\|colab cli is available\|silence this"
