#!/usr/bin/env bash
# One-command inference on the challenge test set -> submissions/submission.csv
#   RA_DATA=/data/extracted scripts/infer.sh              # v10 (final): v8 + ConvNeXt-V2-B zoom and full-frame models
#   VERSION=v9 RA_DATA=/data/extracted scripts/infer.sh   # v9: v8 retrained on train + test + validation
#   VERSION=v8 RA_DATA=/data/extracted scripts/infer.sh   # v8: v7 + zoom refiner + full-frame fusion + confidence rescoring
#   VERSION=v7 RA_DATA=/data/extracted scripts/infer.sh   # v7: 6 transformers + 2 skeleton GCNs, 5-frame ConvNeXt-S/B refiners
# With the default WEIGHTS=weights, missing weights are downloaded first (scripts/download_weights.sh: the v10 archive,
# which also covers v7 and v8, or the v9 archive for VERSION=v9).
# WEIGHTS=weights_trained (a directory inside the repository) uses the output of scripts/train_all.sh (v10) or
# scripts/retrain_all.sh (v9) instead of weights/.
# RA_DATA is the folder where the Kaggle zip was extracted (default data/extracted inside the clone; a relative
# path is taken from the current directory). It must contain challenge_test_set/ (metadata.json + both mp4s per
# sample) and sample_submission.csv. The crop caches (about 40 GB in cache_infer/) are deleted when the run ends.
set -euo pipefail
if [ -n "${RA_DATA:-}" ]; then
    RA_DATA="$(cd -- "$RA_DATA" 2>/dev/null && pwd)" || { echo "RA_DATA is not a directory" >&2; exit 1; }
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
VERSION="${VERSION:-v10}"
case "$VERSION" in v7|v8|v9|v10) ;; *) echo "VERSION must be v7, v8, v9 or v10 (got $VERSION)" >&2; exit 1 ;; esac
WDIR="${WEIGHTS:-weights}"
case "$WDIR" in /*|../*) echo "WEIGHTS must be a directory inside the repository (e.g. weights_trained)" >&2; exit 1 ;; esac
if [ "$WDIR" = weights ] && ! ls weights/"$VERSION"/*.pt >/dev/null 2>&1; then
    SETS=v10; [ "$VERSION" = v9 ] && SETS=v9
    SETS=$SETS scripts/download_weights.sh
fi
ls "$WDIR/$VERSION"/*.pt >/dev/null 2>&1 || { echo "no $VERSION weights in $WDIR/$VERSION/" >&2; exit 1; }
export RA_DATA="${RA_DATA:-$PWD/data/extracted}"
[ -d "$RA_DATA/challenge_test_set" ] && [ -f "$RA_DATA/sample_submission.csv" ] || {
    echo "RA_DATA=$RA_DATA must contain challenge_test_set/ and sample_submission.csv" >&2; exit 1; }
rm -rf cache_infer submissions/submission.csv
export RA_IMAGE="${RA_IMAGE:-sadbodh-retail-action:latest}"
echo "building $RA_IMAGE (the first build pulls the base image, about 9 GB)" >&2
docker build -t "$RA_IMAGE" . >&2
export RA_OFFLINE=1   # the container needs no network, and the data is mounted read-only
COMMON=(--root /data --cache /workspace/cache_infer --split challenge --sample-sub /data/sample_submission.csv --csv /workspace/submissions/submission.csv)
if [ "$VERSION" = v9 ]; then
    W=/workspace/$WDIR/v9
    ZB=""; [ -f "$WDIR/v9/refz_cnxb.pt" ] && ZB="$W/refz_cnxb.pt"
    scripts/docker_run.sh python -m ra.refine submit "${COMMON[@]}" --crops /workspace/cache_infer/crops \
        --pose $W/pose_L12_s0.pt $W/pose_L12_s1.pt $W/pose_L12_s2.pt $W/pose_L6_s0.pt $W/pose_L6_s1.pt $W/pose_L6_s2.pt $W/pose_gcn_s0.pt $W/pose_gcn_s1.pt \
        --ckpt $W/ref5_cnxs.pt $W/ref5_cnxb.pt --zoom-ckpt $W/refz_cnxs.pt $ZB --full-ckpt $W/reff_cnxs.pt \
        --rescore-tau 0.15 --alpha 1.0 --sec 0.5
else
    W=/workspace/$WDIR
    POSE="$W/v4/pose_L12_s0.pt $W/v4/pose_L12_s1.pt $W/v4/pose_L12_s2.pt $W/v3/pose_mx_s0.pt $W/v3/pose_mx_s1.pt $W/v3/pose_mx_s2.pt $W/v7/pose_gcn_s0.pt $W/v7/pose_gcn_s1.pt"
    REF="$W/v6/refiner5_cnxs.pt $W/v7/refiner5_cnxb.pt"
    EXTRA=""
    [ "$VERSION" = v8 ] && EXTRA="--zoom-ckpt $W/v8/refinerz_cnxs.pt --full-ckpt $W/v8/refinerf_cnxs.pt --rescore-tau 0.15"
    [ "$VERSION" = v10 ] && EXTRA="--zoom-ckpt $W/v8/refinerz_cnxs.pt $W/v10/refinerz_v2b.pt --full-ckpt $W/v8/refinerf_cnxs.pt $W/v10/refinerf_v2b.pt --rescore-tau 0.15"
    scripts/docker_run.sh python -m ra.refine submit "${COMMON[@]}" --crops /workspace/cache_infer/crops \
        --pose $POSE --ckpt $REF --alpha 1.0 --sec 0.5 $EXTRA
fi
rm -rf cache_infer
[ -s submissions/submission.csv ] || { echo "no submissions/submission.csv was written" >&2; exit 1; }
echo "wrote submissions/submission.csv" >&2
