#!/bin/bash
# Pack /content/deep/out on the VM, download it into artifacts/deep/, extract, stop the VM.
C=/mnt/d/penguin_data/code/Web3_Auditor/tools/colab_qwen_audit/.cli-venv/bin/colab
R=/mnt/d/penguin_data/code/quantitative_trading_course/ML4T_Midterm
bash $R/scripts/colab/run.sh fetch.py
$C download -s ml4t-deep /content/deep/out.tar.gz $R/artifacts/deep/out.tar.gz 2>&1 | grep -i "download"
tar -xzf $R/artifacts/deep/out.tar.gz -C $R/artifacts/deep/ && ls -la $R/artifacts/deep/out
$C stop -s ml4t-deep 2>&1 | grep -v -i "update\|available\|silence"
$C sessions 2>&1 | grep -v -i "update\|available\|silence" | tail -2
