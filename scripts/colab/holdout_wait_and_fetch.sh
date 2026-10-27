#!/bin/bash
# Wait for the holdout deep_v2 fit, download it, and stop the VM only after a verified download.
set -uo pipefail
source "$(dirname "$0")/env.sh"
D="$R/artifacts/holdout"
for i in $(seq 1 120); do
  out=$(bash "$R/scripts/colab/run.sh" holdout_poll.py)
  line=$(echo "$out" | grep "alive:")
  echo "$(date +%H:%M:%S) $line"
  echo "$line" | grep -q "DONE: True" && break
  echo "$line" | grep -q "alive: False" && { echo "$out"; echo "training died; VM left running" >&2; exit 2; }
  sleep 60
done
bash "$R/scripts/colab/run.sh" holdout_fetch.py
"$C" download -s "$S" /content/hold/out.tar.gz "$D/deep_out.tar.gz" 2>&1 | grep -q "Downloaded" \
  || { echo "download failed; VM left running" >&2; exit 1; }
tar -xzf "$D/deep_out.tar.gz" -C "$D" || { echo "extract failed; VM left running" >&2; exit 1; }
[ -f "$D/deep_out/DONE" ] && [ -f "$D/deep_out/predictions.parquet" ] || { echo "incomplete; VM left running" >&2; exit 1; }
ls -la "$D/deep_out"
"$C" stop -s "$S" 2>&1 | quiet
"$C" sessions 2>&1 | quiet | tail -1
