"""Probe: do whole-body hand keypoints (RTMW-x, COCO-WholeBody, via rtmlib/ONNX) locate the touch point well?

For 50 validation actions (rank0 view, middle frame of the GT span): person box from our pose
keypoints -> RTMW-x whole-body -> its 42 hand keypoints (indices 91-132). Compare, in metres, to the GT point:
  best_fingertip : closest confident hand keypoint (oracle choice, an upper bound for this cue)
  mean_hand      : mean of the more confident hand's keypoints
  pose_hand_kp   : closer of the dataset's own right_hand / left_hand keypoints

rtmlib and onnxruntime are not in the project image, and the RTMW model is downloaded from
download.openmmlab.com on first use, so run it with network access:
  scripts/docker_run.sh bash -c "pip install --user rtmlib onnxruntime && python scripts/probe_fingertips.py"
"""
import json

import av
import numpy as np
from rtmlib import RTMPose

from ra.crops import frame_ids, video_path
from ra.data import build_cache
from ra.metric import m_per_px

N = 50
# RTMW-x whole-body (133 keypoints; 91-132 are the two hands), from rtmlib's model zoo
MODEL = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"

va = [s for s in build_cache("/data/extracted", "validation", "/data/cache") if s["labels"]]
va = [va[i] for i in np.linspace(0, len(va) - 1, N).astype(int)]
model = RTMPose(MODEL, model_input_size=(288, 384), backend="onnxruntime", device="cpu")

res = {"best_fingertip": [], "mean_hand": [], "pose_hand_kp": []}
for s in va:
    a = s["labels"][0]
    if not np.isfinite(a["pts"][0][0]):
        continue
    T = len(s["t"])
    with av.open(video_path("/data/extracted", "validation", s["id"], "rank0")) as f:
        frames = [fr.to_ndarray(format="rgb24") for fr in f.decode(video=0)]
    fi = frame_ids(a["start"], a["end"], T, len(frames))[1]
    kp = s["kp"][0][min(fi, T - 1)]
    ok = kp[:, 2] > 0
    if ok.sum() < 3:
        continue
    xy = kp[ok, :2] * 600
    x0, y0 = xy.min(0) - 60
    x1, y1 = xy.max(0) + 60
    box = [max(x0, 0), max(y0, 0), min(x1, 600), min(y1, 600)]  # xyxy
    kps, scs = model(frames[fi][:, :, ::-1].copy(), bboxes=[box])  # rtmlib expects BGR
    pts, sc = kps[0], scs[0]
    hand, hsc = pts[91:133], sc[91:133]
    g = np.array(a["pts"][0]) * 600
    mpp = m_per_px(s["kp"][0], np.array([600, 600]))
    conf = hsc > 0.3
    d = np.linalg.norm(hand - g, axis=1) * mpp
    res["best_fingertip"].append(d[conf].min() if conf.any() else d.min())
    side = 0 if hsc[:21].mean() >= hsc[21:].mean() else 1
    res["mean_hand"].append(np.linalg.norm(hand[side * 21 : side * 21 + 21].mean(0) - g) * mpp)
    from ra.data import JOINTS
    hk = [kp[JOINTS.index(n)] for n in ("right_hand", "left_hand") if kp[JOINTS.index(n), 2] > 0]
    if hk:
        res["pose_hand_kp"].append(min(np.linalg.norm(h[:2] * 600 - g) * mpp for h in hk))
print(json.dumps({k: {"n": len(v), "median_m": round(float(np.median(v)), 4)} for k, v in res.items()}))
