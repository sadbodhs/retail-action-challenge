#!/usr/bin/env bash
# Train every v10 component from scratch on train + test (validation held out) -> weights_trained/,
# laid out like weights/ so that  WEIGHTS=weights_trained VERSION=v10 scripts/infer.sh  uses them.
#   scripts/train_all.sh          (about 8.5 h on one RTX 3090, plus ~120 GB of cached crops under data/)
# Needs the competition zip extracted to data/extracted inside the clone (see README). Same flags as the shipped
# checkpoints (each stores its training args); seeds and cuDNN make retrained weights close, not identical.
# Stops at the first failed job; each job logs to logs/train_<name>.log.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
export RA_DATA="$PWD/data"   # mounted as /data; the ra.* modules read /data/extracted
[ -d "$RA_DATA/extracted/train" ] || { echo "$RA_DATA/extracted/train not found: extract the competition zip to data/extracted (see README)" >&2; exit 1; }
export RA_IMAGE="${RA_IMAGE:-sadbodh-retail-action:latest}"
docker build -q -t "$RA_IMAGE" . >/dev/null
run() { echo "[$(date +%H:%M)] $1"; eval "scripts/docker_run.sh python -m $2" > "logs/train_$1.log" 2>&1; }
# jobs run two at a time; "wait $p" returns the background job's exit status, so set -e stops on its failure

# 1. per-split feature caches
run cache "ra.data"

# 2. pose models: 3 x 6-layer + 3 x 12-layer transformers, 2 skeleton GCNs (two at a time)
POSE="ra.train --train train test --drop 0.2 --metres 1 --extra 1 --epochs 60 --eval-every 60"
for s in 0 1 2; do
    run "L6_s$s" "$POSE --layers 6 --seed $s --out runs/v10_L6_s$s" & p=$!
    run "L12_s$s" "$POSE --layers 12 --seed $s --out runs/v10_L12_s$s"
    wait $p
done
run gcn_s0 "$POSE --enc gcn --layers 6 --seed 0 --out runs/v10_gcn_s0" & p=$!
run gcn_s1 "$POSE --enc gcn --layers 6 --seed 1 --out runs/v10_gcn_s1"
wait $p

# 3. training crops around GT points (validation only for per-epoch monitoring)
for sp in train test validation; do
    run "crops5_$sp" "ra.crops --split $sp --mode gt --nf 5 --out /data/crops5"
    run "crops5z_$sp" "ra.crops --split $sp --mode gt --nf 5 --crop 144 --noise zoom --out /data/crops5z"
    run "crops5f_$sp" "ra.crops --split $sp --mode gt --nf 5 --full 320 --out /data/crops5f"
done

# 4. refiners: 5-frame ConvNeXt-S + B and zoom + full-frame ConvNeXt-S two at a time; the two ConvNeXt-V2-B models
#    one after the other, as the shipped ones were trained
REF="ra.refine train --train train test --epochs 12"
V2B="--arch timm:convnextv2_base.fcmae_ft_in22k_in1k --bs 48 --lr 1e-4"
run ref5_cnxs "$REF --arch convnext_s --lr 2e-4 --crops /data/crops5 --out runs/v10_ref5_cnxs" & p=$!
run ref5_cnxb "$REF --arch convnext_b --lr 1.5e-4 --crops /data/crops5 --out runs/v10_ref5_cnxb"
wait $p
run refz_cnxs "$REF --arch convnext_s --lr 2e-4 --zoom 2 --crops /data/crops5z --out runs/v10_refz_cnxs" & p=$!
run reff_cnxs "$REF --arch convnext_s --lr 2e-4 --s4 1 --in-size 288 --crops /data/crops5f --out runs/v10_reff_cnxs"
wait $p
run refz_v2b "$REF $V2B --zoom 2 --crops /data/crops5z --out runs/v10_refz_v2b"
run reff_v2b "$REF $V2B --s4 1 --in-size 288 --crops /data/crops5f --out runs/v10_reff_v2b"

# 5. package under the names infer.sh expects (refiners in fp16, as shipped)
scripts/docker_run.sh python - <<'EOF'
import os, shutil, torch
names = {"L6_s0": "v3/pose_mx_s0", "L6_s1": "v3/pose_mx_s1", "L6_s2": "v3/pose_mx_s2",
         "L12_s0": "v4/pose_L12_s0", "L12_s1": "v4/pose_L12_s1", "L12_s2": "v4/pose_L12_s2",
         "gcn_s0": "v7/pose_gcn_s0", "gcn_s1": "v7/pose_gcn_s1",
         "ref5_cnxs": "v6/refiner5_cnxs", "ref5_cnxb": "v7/refiner5_cnxb",
         "refz_cnxs": "v8/refinerz_cnxs", "reff_cnxs": "v8/refinerf_cnxs",
         "refz_v2b": "v10/refinerz_v2b", "reff_v2b": "v10/refinerf_v2b"}
for run, dst in names.items():
    # log.json is written only after the last epoch: refuse to package an unfinished run
    assert os.path.exists(f"runs/v10_{run}/log.json"), f"runs/v10_{run} did not finish"
    dst = f"weights_trained/{dst}.pt"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if run.startswith("ref"):
        ck = torch.load(f"runs/v10_{run}/refiner.pt", map_location="cpu", weights_only=False)
        ck["state"] = {k: v.half() if v.is_floating_point() else v for k, v in ck["state"].items()}
        torch.save(ck, dst)
    else:
        shutil.copy(f"runs/v10_{run}/model.pt", dst)
    print(dst)
EOF
echo "[$(date +%H:%M)] TRAIN_DONE"
