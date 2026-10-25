# RetailAction Challenge (PRAW @ WACV 2027): team sadbodh

Entry for the [RetailAction Challenge](https://www.kaggle.com/competitions/retail-action-challenge-praw-workshop-wacv-2027):
detect shopper–shelf interactions in dual-view top-view retail clips, localise each one in time
(start/end in [0, 1]) and in space (the touch point in both camera views).

**Final submission (v10): public leaderboard mAP Global 0.9327**, local validation 0.901.

Trained weights: [huggingface.co/Sadbodh/retail-action-weights](https://huggingface.co/Sadbodh/retail-action-weights) (fetched by `scripts/download_weights.sh`).

## Quick start: reproduce the submission

Requirements:
- Linux, Docker with the NVIDIA Container Toolkit.
- An NVIDIA GPU with ≥ 16 GB memory and compute capability 7.5 or newer (Turing or later), and NVIDIA driver 580
  or newer. The image's PyTorch 2.13 (CUDA 13.0) has kernels for sm_75, sm_80, sm_86, sm_90, sm_100 and sm_120 only,
  so V100, P100 and older GPUs stop with "no kernel image is available". We used an RTX 3090 (compute capability
  8.6), driver 595.99.
- About 75 GB free disk: the Docker image takes 22 GB unpacked (31 GB with Docker's containerd image store), the
  crop caches that `scripts/infer.sh` writes to `cache_infer/` during a run take about 40 GB (deleted when the
  run ends), and the weights plus the downloaded archive in `dist/` take 2 GB.

```bash
git clone https://github.com/sadbodhs/retail-action-challenge.git && cd retail-action-challenge
RA_DATA=/data/extracted scripts/infer.sh                 # fetches the weights, builds the image, writes submissions/submission.csv
cmp submissions/submission.csv submissions/v10_v2b.csv   # no output: identical to the submitted file
```

`RA_DATA` is the folder where the competition zip was extracted (`/data/extracted` above; a relative path is
taken from the directory you run the command in; default `data/extracted` inside the clone). It must contain
`challenge_test_set/` (each sample's `metadata.json` and both mp4s) and `sample_submission.csv`; the script
checks this before it starts. `scripts/infer.sh` builds the Docker image (`Dockerfile`: public
`vllm/vllm-openai:v0.30.0` base, pinned by digest, with PyTorch 2.13 / CUDA 13, plus `av`, `kaggle` and `timm`;
the first build pulls about 9 GB) as `sadbodh-retail-action:latest` and runs the whole pipeline inside it, with no
network and the data mounted read-only. On the first run it also fetches the
weights (about 1 GB, from Hugging Face via `scripts/download_weights.sh`, every file checked against
`weights/SHA256SUMS`). Inference takes about 15 min on one RTX 3090 for the 2,500 challenge clips, plus the first
image build and weights download. `RA_GPUS=device=1` picks another GPU. Tested on a machine with 16 CPU cores and
32 GB RAM; `RA_PRED_BS` (default 128) lowers the refiners' batch size if host memory is short, and the byte-identical
check used the default.

Tested from a fresh clone of this repository with the image rebuilt from scratch: the output is byte-identical to `submissions/v10_v2b.csv` (and `VERSION=v9` to `v9_alldata.csv`), and two runs give the same file.
On another GPU model the file will not be byte-identical: the refiners run in bfloat16, so touch points and, through
the confidence rescoring, scores can differ in their trailing digits. Compare with a numeric tolerance there.

Other versions: `VERSION=v9` (v8 retrained on all labelled data; fetches the 0.67 GB v9 archive; compare with
`submissions/v9_alldata.csv`), `VERSION=v8`, `VERSION=v7`. These four are the versions with published
weights; `scripts/infer.sh` rejects any other value.

## Method

Two stages plus a rescoring step.

1. **Pose models** (`ra/data.py`, `ra/model.py`, `ra/train.py`), metadata only, no pixels. Per frame and view:
   the subject's 19 pose keypoints and scores, the face position, the sampling score and timing features
   (gaps to neighbouring frames, the dense sampling-score curve). Two dataset properties matter a lot:
   - pose keypoints are stored as **(row, col)**; swapping to (x, y) puts the hand next to the GT point;
   - labels are **frame-index fractions** (`start = k / T`), so the model's time axis is `i / T`.

   Encoders: 3 × 6-layer and 3 × 12-layer transformers (d = 256) and 2 skeleton graph networks (ST-GCN style,
   both views' skeletons in one graph). Heads (1D CenterNet): action-centre heatmap, start/end offsets, and a
   **pointer head** for the touch point (attention over hand/wrist/elbow keypoint candidates plus a residual),
   trained on the error in metres. Decoding: up to 5 heatmap peaks per clip, flip TTA, ensemble average.
2. **Touch-point refiners** (`ra/crops.py`, `ra/refine.py`), pixels at the predicted actions only.
   - Crop refiners: 5 full-resolution 224 px crops (10/30/50/70/90 % of the predicted span) around the
     stage-1 point, stacked to 15 channels; ConvNeXt-S and ConvNeXt-B with an FPN to stride 8 and a heatmap
     soft-argmax, averaged.
   - Zoom refiners: ConvNeXt-S and ConvNeXt-V2-B on a 112 px window around the refined point, enlarged 2×;
     blended 75 / 25 with the crop refiners' point.
   - Full-frame models: ConvNeXt-S and ConvNeXt-V2-B on whole frames resized to 320 px (a random 288 px window in
     training, the central one at test time), never shown the stage-1 point; blended in at 30 %.
   - The crop and zoom refiners are trained on crops around GT points plus noise shaped like the previous stage's
     error, the full-frame models on whole frames; the loss is in metres.
3. **Confidence rescoring**: spread = disagreement (m) of the refiners' shifted/flipped estimates + 0.5 × the
   full-frame disagreement; score × exp(−spread / 0.15 m).

Median touch-point error on validation (v8): 0.094 m after stage 1, 0.045 m after stage 2.

**Metric** (`ra/metric.py`): our re-implementation of mAP Global (tIoU 0.05–0.5 × distance 1.0–0.05 m,
metres per pixel from bone lengths, 101 confidence thresholds). The official scorer is not public.

## Train from scratch

```bash
# competition data into data/ inside the clone (Kaggle API token in ~/.kaggle)
mkdir -p data
docker build -t sadbodh-retail-action:latest .
scripts/docker_run.sh kaggle competitions download -c retail-action-challenge-praw-workshop-wacv-2027 -p /data
scripts/docker_run.sh python -c "import zipfile; zipfile.ZipFile('/data/retail-action-challenge-praw-workshop-wacv-2027.zip').extractall('/data/extracted')"
scripts/train_all.sh                                     # all v10 components -> weights_trained/
WEIGHTS=weights_trained RA_DATA=$PWD/data/extracted scripts/infer.sh
```

`scripts/train_all.sh` uses the flags stored in each shipped checkpoint: train + test splits, validation held
out. It reads the data from `data/extracted` inside the clone and writes about 120 GB of cached crops under
`data/` (crops5 40 GB, crops5z 13 GB, crops5f 63 GB), plus 62 GB for the zip and its extraction. It takes about
8.5 h on one RTX 3090, estimated from the run times of its parts. We trained the shipped models component by
component during development; the script chains the same commands but has not been run end to end in one go.
It stops at the first failed job; each job's log is in `logs/train_<name>.log`.
`scripts/retrain_all.sh` is the v9 variant (all three labelled splits, about 3 h); run it after
`scripts/train_all.sh`, whose step 3 builds the crops it reads. It writes `weights_trained/v9`, used with
`WEIGHTS=weights_trained VERSION=v9 scripts/infer.sh`.

## Results

Local mAP Global on the labelled `validation` split (never trained on, except v9). Published weights cover v7,
v8, v9 and v10; the `weights/v2` to `weights/v6` sets named below are not published.

| Run | Train data | Val mAP | Submission |
|---|---|---|---|
| base: timestamp time axis, (row,col) pose read as (x,y), direct xy regression | train | 0.678 | `submissions/base_v1.csv` |
| frame-index time axis + (row,col) fix | train | 0.756 | – |
| + pointer head for the touch point | train | 0.777 | – |
| + dropout 0.2 | train | 0.783 | – |
| **v2**: 3-seed ensemble + flip TTA (`weights/v2`) | train + test | 0.801 | `submissions/ens3_v2.csv`: **public LB 0.8623** |
| pose: + metre loss + timing features (3-seed mean 0.785 vs 0.782 on train only) | train + test | 0.809 | – |
| **v3**: + ResNet-50 crop refiner, secondary peaks × 0.5 (`weights/v3`) | train + test | 0.862 | `submissions/v3_refine.csv` (not submitted) |
| pose: 12-layer transformer (3 seeds) / + v3's 3 pose models | train + test | 0.816 / 0.817 (pose stage) | – |
| **v4**: 6 pose models + ConvNeXt-S & ConvNeXt-T refiners averaged (`weights/v4`) | train + test | **0.872** | `submissions/v4_cnx_ens.csv`: **public LB 0.9154** (4th of 10 at submission) |
| **v5**: v4 + 2 BiGRU pose models (pose stage mixes transformer and recurrent encoders, `weights/v5`) | train + test | **0.876** | `submissions/v5_gru_ens.csv`: **public LB 0.9170** (4th of 10 at submission) |
| **v6**: v5 pose models + 5-frame ConvNeXt-S/T refiners (crops at 10/30/50/70/90 % of the span, `weights/v6`) | train + test | **0.880** | `submissions/v6_5frame.csv`: **public LB 0.9218** (4th of 10 at submission) |
| **v7**: 6 transformers + 2 skeleton GCNs (ST-GCN style) → 5-frame ConvNeXt-S + ConvNeXt-B refiners (`weights/v7`) | train + test | 0.883 | `submissions/v7_gcn_cnxb.csv` (not submitted) |
| **v8**: v7 + zoom refiner + independent full-frame model (fused) + localisation-confidence rescoring (`weights/v8`) | train + test | **0.899** | `submissions/v8_fusion.csv`: **public LB 0.9307** |
| **v9**: v8 recipe retrained on train + test + validation (`weights/v9`, `scripts/retrain_all.sh`) | train + test + val | – (no held-out split) | `submissions/v9_alldata.csv`: public LB 0.9298 |
| **v10**: v8 + ConvNeXt-V2-B (timm, ImageNet-22k) as a second zoom refiner and a second full-frame model, averaged with the v8 ones (`weights/v10`) | train + test | **0.901** | `submissions/v10_v2b.csv`: **public LB 0.9327** (4th of 10 at submission) |

Ranks are the team's position among 10 teams on the public leaderboard on the day each file was submitted; the
final ranking is decided on the private leaderboard.

Organisers' MViT-b DETR baseline: 0.417 (paper, test split) / 0.391 (public leaderboard).

## Tried and not kept

| Idea | Result |
|---|---|
| Second refinement pass (re-crop around the refined point) | +0.001–0.002 |
| Refiners on v4 pose crops: ResNet-50 / ConvNeXt-T / ConvNeXt-S / S+T / S+T+R50 | 0.866 / 0.869 / 0.870 / **0.872** / 0.870 |
| Wider pose model: d384×8 / d512×8 layers (deeper is what helps: 10/12/14 layers 0.794/0.796/0.797 vs 0.785) | 0.788 / 0.782 |
| Frozen MViTv2-S (K400) tokens as extra pose-model input | 0.70 (vs 0.785): 1536-d input overfits |
| Fine-tuned MViTv2-S temporal localizer (3 epochs), heatmap fused with the pose model | alone 0.687; fused +0.0014 at best: dropped |
| Other model families as single pose models: TCN (12 dilated-conv layers) / BiGRU (3 layers) | 0.783 / 0.775 (transformers: 0.796–0.802) |
| Pose ensemble: 6 transformers / + 2 TCN / + 2 BiGRU / + both (pose stage) | 0.817 / 0.820 / **0.821** / 0.820 |
| Other refiner families, single: EfficientNetV2-S / Swin-T / DINOv2-B (median error) | 0.048 / 0.050 / 0.065 m (ConvNeXt-S: 0.046 m) |
| Refiner ensembles with them: S+T / S+T+EffNet / S+T+EffNet+Swin / S+EffNet | 0.872 / 0.871 / 0.872 / 0.870: ConvNeXt pair kept |
| Skeleton graph network (ST-GCN style, both views' skeletons + cross-view edges, learned adjacency) as single pose model | 0.791 / 0.787 (2 seeds): best non-transformer family |
| Pose ensemble: 6 transformers + 2 BiGRU / + 2 GCN instead / + both (pose stage) | 0.821 / **0.825** / 0.825: GCN kept, BiGRU dropped |
| 5-frame refiners on v7 pose crops: S+T / B / **S+B** / S+T+B | 0.881 / 0.880 / **0.883** / 0.882 |
| **A. Localisation-confidence rescoring**: score × exp(−spread/τ), spread = disagreement (m) of the refiners' shifted/flipped estimates | 0.883 → **0.895** (τ 0.05–0.2 all 0.893–0.895): kept |
| A'. Shifted-crop TTA for the points themselves | 0.882 (vs 0.883): not kept |
| **B. Zoom refiner**: second ConvNeXt-S on a 112 px window around the refined point, enlarged 2×; median 0.035 m on its crops | +0.002 at a 0.75 blend (0.897): kept |
| C. Stride-4 heatmap level (ConvNeXt-S) | 0.0424 vs 0.0422 m alone; 0.894 as an extra member: dropped |
| **E. Independent full-frame model**: whole 600 px frames (5, both views) at 320 px, no point estimate; median 0.053 m alone | fused 30 % + its disagreement in the rescoring: **0.899** (+0.002): kept |
| F. Learned rescorer (MLP on 12 per-detection signals, 5-fold CV on validation) | 0.898 alone, 0.900 blended with the v8 score (v8 0.899): not kept; a perfect ranking would give 0.931 |
| G. Each refiner's heatmap spread as an extra rescoring signal | 0.900 at best: not kept |
| ConvNeXt-B zoom refiner (0.0351 m vs 0.0349 m for ConvNeXt-S), alone or with it | 0.899 / 0.899: not kept |
| 5-frame vs 3-frame refiner crops, median error (ConvNeXt-S / T) | 0.042 / 0.043 m vs 0.046 / 0.047 m: kept (v6) |
| Blending 3-frame and 5-frame refiners | 0.880 at best vs 0.880 for 5-frame alone: not worth the second crop pass |
| Qwen3-VL-8B pointing at the touch point (50 crops) | 0.143 m median (refiner 0.038 m on the same crops) |
| Fingertips from RTMW whole-body pose (133 kp) | best-of-42 hand keypoints is 0.136 m from GT (our 5-frame crop refiner: 0.042 m on GT-centred validation crops) |
| Points averaged over neighbouring frames; higher min score | no gain |
| H. timm backbones as extra 5-frame refiners: ConvNeXt-V2-B / ConvNeXt-V2-T / ConvNeXt-B (22k), median error | 0.0422 / 0.0435 / 0.0441 m; added to the crop ensemble 0.8989–0.8994 (v8: 0.8991): not kept |
| **H'. ConvNeXt-V2-B as a second zoom refiner / second full-frame model / both** | 0.9004 / 0.9002 / **0.9012**: both kept (v10) |

## Repository layout

| Path | Contents |
|---|---|
| `ra/data.py` | metadata parsing, per-frame features, split caches |
| `ra/model.py`, `ra/train.py` | pose models (transformer, GRU, TCN, skeleton GCN) and their training |
| `ra/predict.py` | pose-model inference and decoding |
| `ra/crops.py`, `ra/refine.py` | crop extraction, refiners, rescoring, the full submission pipeline |
| `ra/metric.py` | local mAP Global |
| `scripts/infer.sh` | one-command inference |
| `scripts/docker_run.sh` | runs a command in the project container (GPU, repo, data mounted) |
| `scripts/download_weights.sh`, `scripts/package_weights.sh` | fetch / pack the weights |
| `scripts/train_all.sh`, `scripts/retrain_all.sh` | training recipes for v10 / v9 |
| `scripts/overnight_timm.sh` | development queue that trained the timm backbones (experiment H, including the shipped ConvNeXt-V2-B refiners) |
| `scripts/exp_*.py`, `scripts/probe_*.py`, `scripts/error_analysis.py` | development experiments behind "Tried and not kept", on the validation split. They first need the v7 validation crops: `scripts/docker_run.sh python -m ra.crops --split validation --mode pred --nf 5 --out /data/crops5 --ckpt` followed by the eight v7 pose models (`weights/v4/pose_L12_s{0,1,2}.pt`, `weights/v3/pose_mx_s{0,1,2}.pt`, `weights/v7/pose_gcn_s{0,1}.pt`); then `exp_a_tta_rescore.py`, `exp_b_zoom_val_crops.py` and `exp_e_fullframe_val_crops.py` before the others. `exp_bc_eval.py`'s stride-4 model, `exp_h_backbones.py`'s other timm models and the v3 ResNet-50 refiner used by `probe_vlm.py` are not published; `probe_fingertips.py` needs `pip install rtmlib onnxruntime` (its docstring has the command) |
| `ra/vidfeat.py`, `ra/vidtal.py` | MViTv2-S video experiments in "Tried and not kept" (not used by any submission) |
| `scripts/scoreboard.sh` | development helper: local run scores and our Kaggle submissions |
| `submissions/` | every file we generated, including those submitted |
| `LICENSE` | MIT licence for the code |

## Licence

Code: MIT (see `LICENSE`).

Weights ([huggingface.co/Sadbodh/retail-action-weights](https://huggingface.co/Sadbodh/retail-action-weights)):
CC-BY-NC-4.0. The two ConvNeXt-V2-B refiners (`weights/v10/refinerz_v2b.pt`, `weights/v10/refinerf_v2b.pt`) are
fine-tuned from timm's `convnextv2_base.fcmae_ft_in22k_in1k`, which is CC-BY-NC-4.0, so the published weights are
non-commercial. The ConvNeXt-S and ConvNeXt-B refiners start from torchvision ImageNet-1k weights, and the pose models
are trained from scratch. All weights are trained on the
[RetailAction dataset](https://huggingface.co/datasets/standard-cognition/RetailAction), whose own terms also apply.
