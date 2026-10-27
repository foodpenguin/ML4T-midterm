#!/bin/bash
# New A100 VM with the holdout deep_v2 inputs (features only; no evaluation labels).
source "$(dirname "$0")/env.sh"
D="$R/artifacts/holdout/deep"
"$C" new -s "$S" --gpu A100 2>&1 | grep -i "session"
bash "$R/scripts/colab/run.sh" holdout_mkdirs.py
for f in bundle.tar.gz meta.parquet static.npy mask.npy rows.npz config.json; do
  "$C" upload -s "$S" "$D/$f" "/content/hold/$f" 2>&1 | grep "Upload"
done
for f in "$D"/parts/X.part*; do
  for attempt in 1 2 3; do
    out=$("$C" upload -s "$S" "$f" "/content/hold/parts/$(basename "$f")" 2>&1 | grep "Upload")
    echo "$(basename "$f") try $attempt: ${out:0:40}"
    echo "$out" | grep -q "Uploaded" && break
  done
done
bash "$R/scripts/colab/run.sh" holdout_assemble.py
echo "local sha256 $(sha256sum "$D/X.npy" | cut -d' ' -f1)"
