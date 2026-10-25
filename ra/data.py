"""Parse RetailAction samples (metadata.json) into fixed per-frame feature arrays.

Each sample dir holds rank0/rank1 videos and a metadata.json with, per view and per frame:
pose keypoints of the subject (normalised, stored as (row, col) and swapped to (x, y) here, + joint score),
face position and a
motion-aware sampling score. We turn that into a [T, F] float array (both views
concatenated, aligned to rank0 timestamps) and cache all samples of a split in one .pkl.
"""
import glob
import json
import os
import pickle
from datetime import datetime
from multiprocessing import Pool

import numpy as np

JOINTS = [
    "top_of_head", "neck", "right_shoulder", "right_elbow", "right_wrist", "right_hand",
    "left_shoulder", "left_elbow", "left_wrist", "left_hand", "middle_of_waist",
    "right_hip", "right_knee", "right_ankle", "right_foot",
    "left_hip", "left_knee", "left_ankle", "left_foot",
]
J = len(JOINTS)
VIEWS = ("rank0", "rank1")
# per view: J*(x, y, score, present) + face (col, row, present) + sampling score
VIEW_DIM = J * 4 + 3 + 1
FEAT_DIM = 2 * VIEW_DIM + 1  # + normalised time
# optional timing features (x2): gaps to prev/next frame (normalised + seconds), and the dense
# sampling-score curve summarised over each frame's interval (count, max, mean)
EXTRA_DIM = 7

# Split name -> glob of sample dirs inside the extracted Kaggle zip.
SPLIT_GLOBS = {
    "train": "train/train/*",
    "validation": "validation/validation/*",
    "test": "test/test/*",
    "challenge": "challenge_test_set/*/data/*",
}


def _ts(s):
    return datetime.fromisoformat(s).timestamp()


def _frame_ts(cam, t0, t1):
    """Frame timestamps; some samples lack them, then use face timestamps or uniform spacing."""
    if cam.get("frame_timestamps"):
        return np.array([_ts(s) for s in cam["frame_timestamps"]])
    n = len(cam.get("poses") or [])
    faces = cam.get("face_positions") or []
    if n and len(faces) == n and all(faces):
        return np.array([_ts(f["sampled_at"]) for f in faces])
    return np.linspace(t0, t1, n, endpoint=False) if n else np.zeros(0)


def _view_feats(cam, ref_t, t0, t1):
    """[len(ref_t), VIEW_DIM] features for one camera, nearest-matched to the ref timestamps."""
    ts = _frame_ts(cam, t0, t1)
    out = np.zeros((len(ref_t), VIEW_DIM), np.float32)
    if len(ts) == 0:
        return out, np.zeros((len(ref_t), J, 3), np.float32)
    idx = np.abs(ts[None, :] - ref_t[:, None]).argmin(1)
    poses = cam.get("poses") or []
    kp = np.zeros((len(ts), J, 3), np.float32)  # x, y, score (score 0 = missing)
    for i, p in enumerate(poses[: len(ts)]):
        if not p:
            continue
        pose, sc = p.get("pose") or {}, p.get("joint_scores") or {}
        for j, name in enumerate(JOINTS):
            if name in pose and pose[name] is not None:
                kp[i, j, :2] = pose[name][1], pose[name][0]  # stored as (row, col)
                kp[i, j, 2] = max(float(sc.get(name, 1.0)), 1e-3)
    face = np.zeros((len(ts), 3), np.float32)
    fts = [(_ts(f["sampled_at"]), f["col"], f["row"]) for f in cam.get("face_positions") or [] if f]
    if fts:
        ft = np.array([f[0] for f in fts])
        fi = np.abs(ft[None, :] - ts[:, None]).argmin(1)
        face[:, 0] = [fts[k][1] for k in fi]
        face[:, 1] = [fts[k][2] for k in fi]
        face[:, 2] = 1.0
    samp = np.zeros(len(ts), np.float32)
    ss = [x for x in cam.get("sampling_scores") or [] if x and x[1] is not None]
    if ss:
        st = np.array([_ts(s[0]) for s in ss])
        sv = np.array([s[1] for s in ss], np.float32)
        si = np.abs(st[None, :] - ts[:, None]).argmin(1)
        samp = sv[si]
    kp_r, face_r, samp_r = kp[idx], face[idx], samp[idx]
    present = (kp_r[..., 2] > 0).astype(np.float32)
    out[:, : J * 4] = np.concatenate([kp_r, present[..., None]], -1).reshape(len(ref_t), -1)
    out[:, J * 4 : J * 4 + 3] = face_r
    out[:, J * 4 + 3] = samp_r
    return out, kp_r


def _timing_feats(cams, ref_t, t0, dur):
    T = len(ref_t)
    out = np.zeros((T, EXTRA_DIM), np.float32)
    if T == 0:
        return out
    edges = np.concatenate([[t0], ref_t, [t0 + dur]])
    gp, gn = ref_t - edges[:-2], edges[2:] - ref_t
    out[:, 0], out[:, 1] = gp / dur, gn / dur
    out[:, 2], out[:, 3] = np.clip(gp, 0, 5), np.clip(gn, 0, 5)
    ss = []
    for v in VIEWS:
        ss = [x for x in (cams.get(v) or {}).get("sampling_scores") or [] if x and x[1] is not None]
        if ss:
            break
    if ss:
        st = np.array([_ts(x[0]) for x in ss])
        sv = np.array([x[1] for x in ss], np.float32)
        lo, hi = ref_t - gp / 2, ref_t + gn / 2
        for i in range(T):
            m = (st >= lo[i]) & (st < hi[i])
            if m.any():
                out[i, 4], out[i, 5], out[i, 6] = m.sum() / 10, sv[m].max(), sv[m].mean()
    return out


VID_DIM = 2 * 768  # frozen MViTv2-S tokens, both views (ra/vidfeat.py)


def sample_x(s, extra=False, vid=False):
    """Model input for a cached sample: base features, plus timing / video features if the model uses them."""
    parts = [s["x"]] + ([s["x2"]] if extra else []) + ([s["vf"]] if vid else [])
    return np.concatenate(parts, 1) if len(parts) > 1 else s["x"]


def attach_vid(samples, path):
    """Per-frame video features from vidfeat's {id: [2, 16, 768]} -> s["vf"] [T, VID_DIM]."""
    from .vidfeat import token_index
    with open(path, "rb") as f:
        feats = pickle.load(f)
    for s in samples:
        T = len(s["t"])
        f = feats.get(s["id"])
        if f is None:
            s["vf"] = np.zeros((T, VID_DIM), np.float32)
            continue
        ti = token_index(T)
        s["vf"] = np.concatenate([f[0][ti], f[1][ti]], 1).astype(np.float32)


def _labels(meta):
    labels = meta.get("content", {}).get("labels")
    if labels is None:
        return None
    if isinstance(labels, dict):  # {"action": [...]}
        labels = labels.get("action", [])
    acts = []
    for a in labels:
        sp = a["spatial"]["action_cam"]
        acts.append({
            "start": float(a["start"]), "end": float(a["end"]),
            "pts": [[sp[v]["x"], sp[v]["y"]] if v in sp else [np.nan, np.nan] for v in VIEWS],
            "cls": a.get("label", a.get("action_class")),
        })
    return acts


def load_sample(sample_dir):
    with open(os.path.join(sample_dir, "metadata.json")) as f:
        meta = json.load(f)
    content = meta["content"]
    seg = content["segment_info"]
    t0, t1 = _ts(seg["sampled_at_start"] + "+00:00"), _ts(seg["sampled_at_end"] + "+00:00")
    cams = content["action_cam"]
    ref_t = _frame_ts(cams.get("rank0") or {}, t0, t1)
    if len(ref_t) == 0:
        ref_t = _frame_ts(cams["rank1"], t0, t1)
    dur = max(t1 - t0, 1e-6)
    tn = np.clip((ref_t - t0) / dur, 0, 1).astype(np.float32)
    feats, kps = [], []
    for v in VIEWS:
        fv, kv = _view_feats(cams.get(v) or {}, ref_t, t0, t1)
        feats.append(fv)
        kps.append(kv)
    x = np.concatenate(feats + [tn[:, None]], 1)
    return {
        "id": os.path.basename(sample_dir.rstrip("/")),
        # labels are frame-index fractions (k / T), so that is the time axis we predict on
        "t": (np.arange(len(tn)) / len(tn)).astype(np.float32), "x": x, "kp": np.stack(kps),  # kp: [2, T, J, 3] for the metric's m/px factor
        "x2": _timing_feats(cams, ref_t, t0, dur),
        "dur": dur, "labels": _labels(meta),
    }


def _safe_load(d):
    try:
        return load_sample(d)
    except Exception as e:  # keep going; report broken samples
        print(f"[warn] {d}: {e!r}")
        return None


def build_cache(root, split, cache_dir, workers=14):
    path = os.path.join(cache_dir, f"{split}.pkl")
    vid = os.path.join(cache_dir, f"vid_{split}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            samples = pickle.load(f)
        if os.path.exists(vid):
            attach_vid(samples, vid)
        return samples
    dirs = sorted(glob.glob(os.path.join(root, SPLIT_GLOBS[split])))
    print(f"{split}: {len(dirs)} samples")
    if not dirs:
        raise FileNotFoundError(f"no samples for {split} under {root}")
    with Pool(workers) as pool:
        samples = [s for s in pool.map(_safe_load, dirs, chunksize=32) if s is not None]
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(samples, f)
    return samples


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/extracted")
    ap.add_argument("--cache", default="/data/cache")
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_GLOBS))
    a = ap.parse_args()
    for s in a.splits:
        ss = build_cache(a.root, s, a.cache)
        T = [len(x["t"]) for x in ss]
        n_lab = [len(x["labels"]) for x in ss if x["labels"] is not None]
        print(s, len(ss), "T min/mean/max", min(T), np.mean(T), max(T),
              "labelled", len(n_lab), "acts/sample", np.bincount(n_lab) if n_lab else None)
