"""Image crops around touch points, for the crop refiner (ra/refine.py).

A query = (sample, view, action span, centre point). For each query we cut, from that view's
video, nf full-resolution crop x crop patches centred on the point (--crop, default 256, 144 for the zoom
refiners), at fixed fractions of the frame range from start*T to end*T - 1: nf=3 (default) at 25/50/75 %,
nf=5 at 10/30/50/70/90 % (every shipped refiner uses nf=5). With --full S each stack is instead the whole
600x600 frame resized to S x S. Kinds of queries:
  gt   : labelled splits; centre = GT point + noise shaped like the pose model's error (training)
  pred : centre = pose-model prediction for every predicted action (evaluation / inference)
  zoom / full : built by ra.refine submit around the refined points / for the full-frame models
Crops go to a uint8 memmap [N, nf, size, size, 3] (size = --crop, or S with --full S); metadata to a .npz next to it.
In every mode but gt, a video that cannot be decoded stops the build with an error.

  scripts/docker_run.sh python -m ra.crops --split train --mode gt --nf 5 --out /data/crops5
  scripts/docker_run.sh python -m ra.crops --split validation --mode pred --nf 5 --out /data/crops5 \
      --ckpt weights/v4/pose_L12_s0.pt weights/v4/pose_L12_s1.pt weights/v4/pose_L12_s2.pt \
      weights/v3/pose_mx_s0.pt weights/v3/pose_mx_s1.pt weights/v3/pose_mx_s2.pt weights/v7/pose_gcn_s0.pt weights/v7/pose_gcn_s1.pt
"""
import argparse
import glob
import os
import pickle
from multiprocessing import Pool

import av
import numpy as np
from PIL import Image

from .data import SPLIT_GLOBS, VIEWS, build_cache
from .metric import m_per_px

CROP = 256
WH = 600
FRACS = (0.25, 0.5, 0.75)                 # default 3 frames
FRACS_BY_NF = {3: FRACS, 5: (0.1, 0.3, 0.5, 0.7, 0.9)}


def video_path(root, split, sid, view):
    pattern = os.path.join(root, os.path.dirname(SPLIT_GLOBS[split]), sid, f"{view}_video.mp4")
    hits = glob.glob(pattern)  # the challenge split's dir pattern has a wildcard level
    return hits[0] if hits else pattern


def frame_ids(start, end, T, n_video, fracs=FRACS):
    """Metadata frame indices at `fracs` of the span, mapped onto the video's frame count."""
    a, b = start * T, max(end * T - 1, start * T)
    idx = np.clip(np.round(a + np.array(fracs) * (b - a)), 0, T - 1)
    if n_video != T and T > 1:
        idx = np.round(idx * (n_video - 1) / (T - 1))
    return np.clip(idx, 0, n_video - 1).astype(int)


def cut(frame, cx, cy, size=CROP):
    """size x size patch of an HxWx3 frame centred at pixel (cx, cy), zero-padded at borders."""
    out = np.zeros((size, size, 3), np.uint8)
    x0, y0 = int(round(cx)) - size // 2, int(round(cy)) - size // 2
    H, W = frame.shape[:2]
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + size, W), min(y0 + size, H)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return out, x0, y0


def _work(job):
    root, split, sid, T, qs, fracs, size, resize = job
    crops, metas = [], []
    for v, view in enumerate(VIEWS):
        vq = [q for q in qs if q["view"] == v]
        if not vq:
            continue
        p = video_path(root, split, sid, view)
        try:
            with av.open(p) as f:
                frames = [fr.to_ndarray(format="rgb24") for fr in f.decode(video=0)]
        except Exception as e:
            print(f"[warn] {p}: {e!r}")
            frames = []
        for q in vq:
            if not frames:
                crops.append(np.zeros((len(fracs), resize or size, resize or size, 3), np.uint8))
                metas.append({**q, "origin": (0, 0), "ok": False})
                continue
            fi = frame_ids(q["start"], q["end"], T, len(frames), fracs)
            stack, origin = [], None
            for i in fi:
                c, x0, y0 = cut(frames[i], q["cx"] * WH, q["cy"] * WH, size)
                if resize:  # full-frame mode: the whole frame, downscaled
                    c = np.asarray(Image.fromarray(c).resize((resize, resize), Image.BILINEAR))
                stack.append(c)
                origin = (x0, y0)
            crops.append(np.stack(stack))
            metas.append({**q, "origin": origin, "ok": True})
    return crops, metas


# training-crop centre noise (px): (probability, sigma) mixtures shaped like the error of the stage feeding the crops
NOISE = {"stage1": ((0.8, 15), (0.2, 40)),        # pose-model points: ~18 px median, heavy tail
         "zoom": ((0.7, 6), (0.25, 15), (0.05, 30))}  # refined points: ~8 px median


def gt_queries(samples, rng, noise="stage1"):
    out = []
    for s in samples:
        for a_i, a in enumerate(s["labels"] or []):
            for v in range(2):
                gx, gy = a["pts"][v]
                if not np.isfinite(gx):
                    continue
                # pose-model error: ~18 px median, heavy tail (p95 ~74 px) -> Gaussian mixture
                probs, sigmas = zip(*NOISE[noise])
                sig = sigmas[rng.choice(len(sigmas), p=probs)] / WH
                cx, cy = np.clip([gx + rng.normal(0, sig), gy + rng.normal(0, sig)], 0, 1)
                out.append({"sid": s["id"], "act": a_i, "view": v, "start": a["start"], "end": a["end"],
                            "cx": cx, "cy": cy, "tx": gx, "ty": gy})
    return out


def pred_queries(samples, preds):
    out = []
    for s in samples:
        for a_i, a in enumerate(preds.get(s["id"], [])):
            for v in range(2):
                out.append({"sid": s["id"], "act": a_i, "view": v, "start": a["start"], "end": a["end"],
                            "cx": a["pts"][v][0], "cy": a["pts"][v][1], "tx": np.nan, "ty": np.nan})
    return out


def build(root, cache, split, mode, out_dir, ckpts=None, workers=14, seed=0, preds=None, nf=3, crop=CROP, noise="stage1",
          full=0):
    """mode "gt": GT-centred training crops. Otherwise crops around `preds` (or the pose ensemble's).
    full > 0: whole-frame stacks resized to full x full px, independent of any point estimate."""
    samples = build_cache(root, split, cache)
    by_id = {s["id"]: s for s in samples}
    os.makedirs(out_dir, exist_ok=True)
    if mode == "gt":
        qs = gt_queries(samples, np.random.default_rng(seed), noise)
    else:
        if preds is None:
            from .predict import decode, load_models, run_model
            preds = decode(run_model(load_models(ckpts), samples), samples)
        with open(os.path.join(out_dir, f"{split}_{mode}_preds.pkl"), "wb") as f:
            pickle.dump(preds, f)
        qs = pred_queries(samples, preds)
    jobs = {}
    for q in qs:
        jobs.setdefault(q["sid"], []).append(q)
    fracs = FRACS_BY_NF[nf]
    if full:
        crop = WH
        for q in qs:
            q["cx"] = q["cy"] = 0.5
    job_list = [(root, split, sid, len(by_id[sid]["t"]), q, fracs, crop, full) for sid, q in jobs.items()]
    N = len(qs)
    print(f"{split}/{mode}: {N} queries from {len(job_list)} samples")
    mm = np.lib.format.open_memmap(os.path.join(out_dir, f"{split}_{mode}.npy"), "w+", np.uint8, (N, len(fracs), full or crop, full or crop, 3))
    metas, k = [], 0
    with Pool(workers) as pool:
        for crops, ms in pool.imap(_work, job_list, chunksize=8):
            for c, m in zip(crops, ms):
                mm[k] = c
                metas.append(m)
                k += 1
    mm.flush()
    bad = sum(not m["ok"] for m in metas)
    if bad and mode != "gt":  # a missing or broken mp4 would otherwise give a meaningless point
        raise RuntimeError(f"{split}/{mode}: {bad} of {k} crop queries have no decodable video "
                           "(each sample needs rank0 and rank1 mp4s): check RA_DATA")
    mpu = {sid: [m_per_px(by_id[sid]["kp"][v], np.array([WH, WH])) for v in range(2)] for sid in jobs}
    np.savez(os.path.join(out_dir, f"{split}_{mode}_meta.npz"),
             sid=np.array([m["sid"] for m in metas]), act=np.array([m["act"] for m in metas]),
             view=np.array([m["view"] for m in metas]), ok=np.array([m["ok"] for m in metas]),
             centre=np.array([[m["cx"], m["cy"]] for m in metas], np.float32),
             target=np.array([[m["tx"], m["ty"]] for m in metas], np.float32),
             origin=np.array([m["origin"] for m in metas], np.float32),
             mpp=np.array([mpu[m["sid"]][m["view"]] for m in metas], np.float32),
             scale=np.float32(full / crop if full else 1.0))
    print("done", k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/extracted")
    ap.add_argument("--cache", default="/data/cache")
    ap.add_argument("--out", default="/data/crops")
    ap.add_argument("--split", required=True)
    ap.add_argument("--mode", choices=["gt", "pred"], required=True)
    ap.add_argument("--ckpt", nargs="*")
    ap.add_argument("--nf", type=int, default=3, choices=sorted(FRACS_BY_NF), help="frames per crop stack")
    ap.add_argument("--crop", type=int, default=CROP, help="cached crop size (zoom refiner: 144)")
    ap.add_argument("--noise", default="stage1", choices=sorted(NOISE), help="training-centre noise model")
    ap.add_argument("--full", type=int, default=0, help="whole-frame stacks resized to this size (0 = off)")
    a = ap.parse_args()
    build(a.root, a.cache, a.split, a.mode, a.out, a.ckpt, nf=a.nf, crop=a.crop, noise=a.noise, full=a.full)
