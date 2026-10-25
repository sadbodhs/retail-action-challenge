"""Frozen MViTv2-S (Kinetics-400) temporal features per view, for the pose transformer.

Each view's clip is resampled to 32 frames, cropped (CROP_PX square around the subject's median
pose position) and resized to 224, then run as two 16-frame clips. Tokens are averaged over
space -> 2 x 8 = 16 temporal features of 768 dims per view, saved as fp16 {id: [2, 16, 768]}.

  scripts/docker_run.sh python -m ra.vidfeat --split train
"""
import argparse
import os
import pickle
from multiprocessing import Pool

import av
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from .crops import video_path
from .data import VIEWS, build_cache

NF = 32          # frames per view fed to MViT (2 clips x 16)
TOK = 16         # temporal tokens per view
CROP_PX = 352
SIZE = 224
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)


def frame_select(T):
    """Indices of the NF frames (of T) fed to MViT; frame i maps to token sel_pos(i) // 2."""
    return np.round(np.linspace(0, T - 1, NF)).astype(int)


def token_index(T):
    """[T] token index for every metadata frame."""
    sel = frame_select(T)
    return np.array([int(np.abs(sel - i).argmin()) // 2 for i in range(T)])


def _load(job):
    root, split, sid, centres = job
    out = np.zeros((2, NF, SIZE, SIZE, 3), np.uint8)
    for v, view in enumerate(VIEWS):
        try:
            with av.open(video_path(root, split, sid, view)) as f:
                frames = [fr.to_ndarray(format="rgb24") for fr in f.decode(video=0)]
        except Exception:
            continue
        if not frames:
            continue
        H, W = frames[0].shape[:2]
        cx, cy = centres[v][0] * W, centres[v][1] * H
        x0 = int(np.clip(cx - CROP_PX / 2, 0, max(W - CROP_PX, 0)))
        y0 = int(np.clip(cy - CROP_PX / 2, 0, max(H - CROP_PX, 0)))
        for k, i in enumerate(frame_select(len(frames))):
            crop = torch.from_numpy(frames[i][y0 : y0 + CROP_PX, x0 : x0 + CROP_PX]).permute(2, 0, 1)[None].float()
            out[v, k] = F.interpolate(crop, size=(SIZE, SIZE), mode="bilinear", antialias=True)[0].permute(1, 2, 0).clamp(0, 255).byte().numpy()
    return sid, out


def subject_centres(s):
    """Per view: median position of the subject's visible keypoints (normalised), else image centre."""
    out = []
    for v in range(2):
        kp = s["kp"][v].reshape(-1, 3)
        ok = kp[:, 2] > 0
        out.append(np.median(kp[ok, :2], 0) if ok.any() else np.array([0.5, 0.5]))
    return out


@torch.no_grad()
def mvit_tokens(model, clips):
    """clips: [B, 3, 16, 224, 224] normalised -> [B, 8, 768] space-averaged tokens."""
    x = model.conv_proj(clips)
    x = x.flatten(2).transpose(1, 2)
    x = model.pos_encoding(x)
    thw = (model.pos_encoding.temporal_size,) + model.pos_encoding.spatial_size
    for blk in model.blocks:
        x, thw = blk(x, thw)
    x = model.norm(x)[:, 1:]
    return x.reshape(x.shape[0], thw[0], thw[1] * thw[2], -1).mean(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/extracted")
    ap.add_argument("--cache", default="/data/cache")
    ap.add_argument("--out", default="/data/cache", help="the pose models' --cache dir, read by --vid 1")
    ap.add_argument("--split", required=True)
    ap.add_argument("--bs", type=int, default=8, help="samples per GPU batch (x2 views x2 clips)")
    a = ap.parse_args()
    samples = build_cache(a.root, a.split, a.cache)
    model = torchvision.models.video.mvit_v2_s(weights="KINETICS400_V1").cuda().eval()
    mean, std = MEAN.cuda(), STD.cuda()
    jobs = [(a.root, a.split, s["id"], subject_centres(s)) for s in samples]
    feats, buf = {}, []

    def flush():
        ids = [b[0] for b in buf]
        x = torch.from_numpy(np.stack([b[1] for b in buf])).cuda()        # [B, 2, NF, H, W, 3]
        B = x.shape[0]
        x = x.view(B * 2 * 2, NF // 2, SIZE, SIZE, 3).permute(0, 4, 1, 2, 3).float() / 255
        x = (x - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = mvit_tokens(model, x).float()                              # [B*4, 8, 768]
        t = t.view(B, 2, TOK, -1).half().cpu().numpy()
        for i, sid in enumerate(ids):
            feats[sid] = t[i]
        buf.clear()

    with Pool(12, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        for c in range(0, len(jobs), 256):  # bounded chunks: imap would buffer decoded clips without limit
            for item in pool.imap(_load, jobs[c : c + 256], chunksize=4):
                buf.append(item)
                if len(buf) == a.bs:
                    flush()
            print(c, flush=True)
    if buf:
        flush()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"vid_{a.split}.pkl"), "wb") as f:
        pickle.dump(feats, f)
    print("done", len(feats))


if __name__ == "__main__":
    main()
