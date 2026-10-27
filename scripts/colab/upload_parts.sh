#!/bin/bash
# X.npy exceeds the upload limit: upload 30 MB parts, reassemble on the VM, compare SHA-256.
source "$(dirname "$0")/env.sh"
D="$R/artifacts/deep"
bash "$R/scripts/colab/run.sh" mkdirs.py
for f in "$D"/parts/X.part*; do
  for attempt in 1 2 3; do
    out=$("$C" upload -s "$S" "$f" "/content/deep/parts/$(basename "$f")" 2>&1 | grep "Upload")
    echo "$(basename "$f") try $attempt: ${out:0:40}"
    echo "$out" | grep -q "Uploaded" && break
  done
done
bash "$R/scripts/colab/run.sh" assemble.py
echo "local sha256 $(sha256sum "$D/X.npy" | cut -d' ' -f1)"
