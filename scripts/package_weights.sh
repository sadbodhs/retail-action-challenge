#!/usr/bin/env bash
# Pack the shipped weights into dist/retail-action-weights-{v10,v9}.tar and write weights/SHA256SUMS.
#   scripts/package_weights.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
V10="weights/v3/pose_mx_s0.pt weights/v3/pose_mx_s1.pt weights/v3/pose_mx_s2.pt
     weights/v4/pose_L12_s0.pt weights/v4/pose_L12_s1.pt weights/v4/pose_L12_s2.pt
     weights/v7/pose_gcn_s0.pt weights/v7/pose_gcn_s1.pt
     weights/v6/refiner5_cnxs.pt weights/v7/refiner5_cnxb.pt
     weights/v8/refinerz_cnxs.pt weights/v8/refinerf_cnxs.pt
     weights/v10/refinerz_v2b.pt weights/v10/refinerf_v2b.pt"
V9="$(ls weights/v9/*.pt)"
mkdir -p dist
tar -cf dist/retail-action-weights-v10.tar $V10
tar -cf dist/retail-action-weights-v9.tar $V9
sha256sum $V10 $V9 > weights/SHA256SUMS
ls -l dist/*.tar
