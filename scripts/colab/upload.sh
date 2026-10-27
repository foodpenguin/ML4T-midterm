#!/bin/bash
# Upload the small rung-4 inputs.
source "$(dirname "$0")/env.sh"
D="$R/artifacts/deep"
bash "$R/scripts/colab/run.sh" check.py
for f in bundle.tar.gz meta.parquet static.npy mask.npy; do
  "$C" upload -s "$S" "$D/$f" "/content/deep/$f" 2>&1 | grep "Upload"
done
