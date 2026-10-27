#!/bin/bash
# Download all run outputs and stop the VM ONLY if the download, extraction and every DONE
# marker check succeed; otherwise leave the VM running so nothing is lost.
set -euo pipefail
source "$(dirname "$0")/env.sh"
D="$R/artifacts/deep"
bash "$R/scripts/colab/run.sh" fetch.py
"$C" download -s "$S" /content/deep/out_all.tar.gz "$D/out_all.tar.gz" 2>&1 | grep -q "Downloaded" \
  || { echo "download failed; VM left running" >&2; exit 1; }
tar -xzf "$D/out_all.tar.gz" -C "$D"
for d in out out_v2 out_ablation; do
  [ -f "$D/$d/DONE" ] && [ -f "$D/$d/predictions.parquet" ] || { echo "$d incomplete; VM left running" >&2; exit 1; }
done
ls -la "$D/out" "$D/out_v2" "$D/out_ablation"
"$C" stop -s "$S" 2>&1 | quiet
"$C" sessions 2>&1 | quiet | tail -2
