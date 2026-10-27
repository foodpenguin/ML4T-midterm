#!/bin/bash
# New A100 VM with all rung-4 inputs.
source "$(dirname "$0")/env.sh"
"$C" new -s "$S" --gpu A100 2>&1 | grep -i "session"
bash "$R/scripts/colab/upload.sh"
bash "$R/scripts/colab/upload_parts.sh"
