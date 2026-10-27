#!/bin/bash
# usage: run.sh <script.py>   (runs a local helper on the ml4t-deep VM)
C=/mnt/d/penguin_data/code/Web3_Auditor/tools/colab_qwen_audit/.cli-venv/bin/colab
D=/mnt/d/penguin_data/code/quantitative_trading_course/ML4T_Midterm/artifacts/deep/colab
$C exec -s ml4t-deep -f $D/$1 2>&1 | grep -v -i "update\|available\|silence"
