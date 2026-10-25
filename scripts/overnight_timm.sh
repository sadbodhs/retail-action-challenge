#!/usr/bin/env bash
# Overnight queue: timm backbones (ImageNet-22k / ConvNeXt-V2) for the touch-point refiners.
# Trained on train + test, validation held out; two streams in parallel.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
R="ra.refine train --train train test --val validation --epochs 12"
V2B=timm:convnextv2_base.fcmae_ft_in22k_in1k
V2T=timm:convnextv2_tiny.fcmae_ft_in22k_in1k
B22=timm:convnext_base.fb_in22k_ft_in1k
mkdir -p logs
job() { echo "[$(date +%H:%M)] start $1"; scripts/docker_run.sh python -m $2 > "logs/tm_$1.log" 2>&1; local rc=$?; echo "[$(date +%H:%M)] end $1 ($rc)"; }
(
  job v2b_5f "$R --arch $V2B --lr 1e-4 --bs 48 --crops /data/crops5 --out runs/tm_v2b_5f"
  job v2b_zoom "$R --arch $V2B --lr 1e-4 --bs 48 --zoom 2 --crops /data/crops5z --out runs/tm_v2b_zoom"
  job v2b_full "$R --arch $V2B --lr 1e-4 --bs 48 --s4 1 --in-size 288 --crops /data/crops5f --out runs/tm_v2b_full"
) &
(
  job b22_5f "$R --arch $B22 --lr 1.5e-4 --bs 48 --crops /data/crops5 --out runs/tm_b22_5f"
  job v2t_5f "$R --arch $V2T --lr 2e-4 --crops /data/crops5 --out runs/tm_v2t_5f"
  job b22_zoom "$R --arch $B22 --lr 1.5e-4 --bs 48 --zoom 2 --crops /data/crops5z --out runs/tm_b22_zoom"
) &
wait
echo "[$(date +%H:%M)] OVERNIGHT_DONE"
