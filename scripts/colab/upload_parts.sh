#!/bin/bash
C=/mnt/d/penguin_data/code/Web3_Auditor/tools/colab_qwen_audit/.cli-venv/bin/colab
D=/mnt/d/penguin_data/code/quantitative_trading_course/ML4T_Midterm/artifacts/deep
$C exec -s ml4t-deep -f $D/colab/mkdirs.py 2>&1 | grep "parts dir"
for f in $D/parts/X.part*; do
  for attempt in 1 2 3; do
    out=$($C upload -s ml4t-deep $f /content/deep/parts/$(basename $f) 2>&1 | grep "Upload")
    echo "$(basename $f) try $attempt: ${out:0:60}"
    echo "$out" | grep -q "Uploaded" && break
  done
done
$C exec -s ml4t-deep -f $D/colab/assemble.py 2>&1 | grep -v -i "update\|available\|silence"
echo "local sha256 $(cat $D/colab/X.sha256)"
