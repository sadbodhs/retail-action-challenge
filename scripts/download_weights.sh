#!/usr/bin/env bash
# Download the trained weights into weights/ and check every file of the downloaded sets against weights/SHA256SUMS.
#   scripts/download_weights.sh                 # v10 set (also covers VERSION=v7 and v8)
#   SETS=v9 scripts/download_weights.sh         # the all-data v9 set (SETS="v10 v9" for both)
# The archives retail-action-weights-<set>.tar come from huggingface.co/Sadbodh/retail-action-weights
# (WEIGHTS_URL overrides the base URL); archives already in dist/ are used as they are.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BASE="${WEIGHTS_URL:-https://huggingface.co/Sadbodh/retail-action-weights/resolve/main}"
mkdir -p dist
for s in ${SETS:-v10}; do
    f="retail-action-weights-$s.tar"
    if [ ! -f "dist/$f" ]; then
        curl -L --fail -o "dist/$f.part" "$BASE/$f"
        mv "dist/$f.part" "dist/$f"
    fi
    tar -xf "dist/$f"
done
# every file of the requested sets must be present and match (weights/v9/ lines belong to the v9 set)
for s in ${SETS:-v10}; do
    if [ "$s" = v9 ]; then grep ' weights/v9/' weights/SHA256SUMS; else grep -v ' weights/v9/' weights/SHA256SUMS; fi
done | sha256sum --quiet -c -
echo "weights OK"
