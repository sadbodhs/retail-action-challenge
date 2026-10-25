#!/usr/bin/env bash
# Retrain every v8 component on all labelled data (train + test + validation) -> runs/all_* -> weights_trained/v9
# (WEIGHTS=weights_trained VERSION=v9 scripts/infer.sh); the downloaded weights/v9 stays untouched.
# No split is held out, so there is no local score; v9 = v8's recipe with ~6 % more data.
#   scripts/retrain_all.sh            (about 3 h on one RTX 3090; two jobs at a time)
# Needs the GT crops for train, test and validation in /data/crops5, /data/crops5z and /data/crops5f, as built by
# step 3 of scripts/train_all.sh; ZOOM_B=1 also trains a ConvNeXt-B zoom refiner. Stops at the first failed job.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
export RA_DATA="$PWD/data"   # mounted as /data; the ra.* modules read /data/extracted
export RA_IMAGE="${RA_IMAGE:-sadbodh-retail-action:latest}"
docker build -q -t "$RA_IMAGE" . >/dev/null
ALL="train test validation"
POSE="--train $ALL --val '' --drop 0.2 --epochs 60 --metres 1 --extra 1"
run() { echo "[$(date +%H:%M)] $1"; eval "scripts/docker_run.sh python -m $2" > "logs/all_$1.log" 2>&1; }
# "wait $p" returns the background job's exit status, so set -e stops on its failure

# pose models: 3 x 12-layer + 3 x 6-layer transformers, 2 skeleton GCNs (two at a time)
for s in 0 1 2; do
    run "L12_s$s" "ra.train $POSE --layers 12 --seed $s --out runs/all_L12_s$s" & p=$!
    run "L6_s$s" "ra.train $POSE --seed $s --out runs/all_L6_s$s"
    wait $p
done
run gcn_s0 "ra.train $POSE --enc gcn --layers 6 --seed 0 --out runs/all_gcn_s0" & p=$!
run gcn_s1 "ra.train $POSE --enc gcn --layers 6 --seed 1 --out runs/all_gcn_s1"
wait $p

# refiners (two at a time): 5-frame ConvNeXt-S + B, zoom ConvNeXt-S, full-frame ConvNeXt-S
REF="ra.refine train --train $ALL --val '' --epochs 12"
run ref5_cnxs "$REF --arch convnext_s --lr 2e-4 --crops /data/crops5 --out runs/all_ref5_cnxs" & p=$!
run ref5_cnxb "$REF --arch convnext_b --lr 1.5e-4 --crops /data/crops5 --out runs/all_ref5_cnxb"
wait $p
run refz_cnxs "$REF --arch convnext_s --lr 2e-4 --zoom 2 --crops /data/crops5z --out runs/all_refz_cnxs" & p=$!
run reff_cnxs "$REF --arch convnext_s --lr 2e-4 --s4 1 --in-size 288 --crops /data/crops5f --out runs/all_reff_cnxs"
wait $p
if [ "${ZOOM_B:-0}" = 1 ]; then
    run refz_cnxb "$REF --arch convnext_b --lr 1.5e-4 --zoom 2 --crops /data/crops5z --out runs/all_refz_cnxb"
fi

# package as weights_trained/v9 (refiners in fp16)
mkdir -p weights_trained/v9
for n in L12_s0 L12_s1 L12_s2 L6_s0 L6_s1 L6_s2 gcn_s0 gcn_s1; do cp "runs/all_$n/model.pt" "weights_trained/v9/pose_$n.pt"; done
scripts/docker_run.sh python - <<'EOF'
import glob, os, torch
for src in sorted(glob.glob("runs/all_ref*/refiner.pt")):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    ck["state"] = {k: v.half() if v.is_floating_point() else v for k, v in ck["state"].items()}
    dst = "weights_trained/v9/" + os.path.basename(os.path.dirname(src)).replace("all_", "") + ".pt"
    torch.save(ck, dst)
    print(dst)
EOF
echo "[$(date +%H:%M)] RETRAIN_DONE"
