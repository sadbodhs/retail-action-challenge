"""Fine-tuned MViTv2-S temporal localizer (video stream), fused with the pose model's heatmap.

Each view: 16 frames (uniform over the clip) cropped around the subject, 224x224 -> MViTv2-S
(Kinetics-400 init) -> 8 space-averaged temporal tokens. Both views are concatenated per token,
a 2-layer transformer follows, and per token we predict an action-centre heatmap logit and
start/end offsets (same targets as the pose model, at token resolution).

  scripts/docker_run.sh python -m ra.vidtal train --train train test --out runs/vtal
  scripts/docker_run.sh python -m ra.vidtal dump --ckpt runs/vtal/vtal.pt --split validation   # -> cache/vtal_<split>.pkl
"""
import argparse
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from . import vidfeat
from .data import build_cache
from .vidfeat import MEAN, STD, SIZE, mvit_tokens, subject_centres

NF = 16    # frames per view
NT = 8     # temporal tokens per view (MViT temporal stride 2)


def frame_select(T):
    return np.round(np.linspace(0, T - 1, NF)).astype(int)


def token_times(T):
    """Normalised time (frame-index axis, like the labels) of each token's centre."""
    sel = frame_select(T)
    return (0.5 * (sel[0::2] + sel[1::2]) / T).astype(np.float32)


def _load(job):
    vidfeat.NF = NF  # decode NF frames per view
    sid, out = vidfeat._load(job)
    return sid, out


class VidTAL(nn.Module):
    def __init__(self, pretrained=True, d=384):
        super().__init__()
        self.mvit = torchvision.models.video.mvit_v2_s(weights="KINETICS400_V1" if pretrained else None)
        self.proj = nn.Linear(2 * 768, d)
        self.pos = nn.Parameter(torch.zeros(1, NT, d))
        enc = nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, 2)
        self.hm = nn.Linear(d, 1)
        self.off = nn.Linear(d, 2)
        nn.init.constant_(self.hm.bias, -2.0)
        self.register_buffer("mean", MEAN)
        self.register_buffer("std", STD)

    def forward(self, clips):
        """clips: [B, 2, NF, H, W, 3] uint8 -> heatmap logits [B, NT], offsets [B, NT, 2]."""
        B = clips.shape[0]
        x = clips.view(B * 2, NF, SIZE, SIZE, 3).permute(0, 4, 1, 2, 3).float() / 255
        x = (x - self.mean) / self.std
        t = mvit_tokens(self.mvit, x).view(B, 2, NT, 768)
        h = self.proj(torch.cat([t[:, 0], t[:, 1]], -1)) + self.pos
        h = self.enc(h)
        return self.hm(h).squeeze(-1), self.off(h) * 0.1


def targets(s):
    tt = token_times(len(s["t"]))
    hm = np.zeros(NT, np.float32)
    off = np.zeros((NT, 2), np.float32)
    best = np.full(NT, np.inf)
    for a in s["labels"] or []:
        c = 0.5 * (a["start"] + a["end"])
        sig = max((a["end"] - a["start"]) / 2, 0.04)  # coarser than the pose model: ~4 frames per token
        g = np.exp(-0.5 * ((tt - c) / sig) ** 2)
        g[np.abs(tt - c).argmin()] = 1.0
        hm = np.maximum(hm, g)
        d = np.abs(tt - c)
        own = (d < best) & (g > 0.05)
        best = np.where(own, d, best)
        off[own] = np.stack([a["start"] - tt[own], a["end"] - tt[own]], 1)
    return hm, off, np.isfinite(best)


def batches(samples, split, root, bs, shuffle, pool):
    """Yields (samples, clips [B, 2, NF, H, W, 3]); `split` names the split, or None to use s["_split"]."""
    order = np.random.permutation(len(samples)) if shuffle else np.arange(len(samples))
    for b in range(0, len(order) - (bs - 1 if shuffle else 0), bs):
        chunk = [samples[i] for i in order[b : b + bs]]
        jobs = [(root, split or s["_split"], s["id"], subject_centres(s)) for s in chunk]
        clips = np.stack([c for _, c in pool.map(_load, jobs)])
        yield chunk, torch.from_numpy(clips)


def focal(lh, hm):
    p = torch.sigmoid(lh).clamp(1e-4, 1 - 1e-4)
    pos = hm >= 0.999
    l = -((1 - p) ** 2 * torch.log(p))[pos].sum() - ((1 - hm) ** 4 * p ** 2 * torch.log(1 - p))[~pos].sum()
    return l / pos.sum().clamp(min=1)


def train(a):
    tr = []
    for sp in a.train:  # sample ids repeat across splits, so remember each sample's split
        for s in build_cache(a.root, sp, a.cache):
            if s["labels"] is not None:
                s["_split"] = sp
                tr.append(s)
    model = VidTAL().cuda()
    back = list(model.mvit.parameters())
    head = [p for n, p in model.named_parameters() if not n.startswith("mvit.")]
    opt = torch.optim.AdamW([{"params": back, "lr": a.lr * 0.1}, {"params": head, "lr": a.lr}], weight_decay=0.05)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr * 0.1, a.lr], total_steps=steps, pct_start=0.05)
    os.makedirs(a.out, exist_ok=True)
    with Pool(12, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        for ep in range(a.epochs):
            model.train()
            t0, tot, n = time.time(), 0.0, 0
            for chunk, clips in batches(tr, None, a.root, a.bs, True, pool):
                T = [targets(s) for s in chunk]
                hm = torch.from_numpy(np.stack([t[0] for t in T])).cuda()
                off = torch.from_numpy(np.stack([t[1] for t in T])).cuda()
                rm = torch.from_numpy(np.stack([t[2] for t in T])).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lh, lo = model(clips.cuda(non_blocking=True))
                lh, lo = lh.float(), lo.float()
                loss = focal(lh, hm) + (F.smooth_l1_loss(lo[rm] * 10, off[rm] * 10, beta=0.1) if rm.any() else 0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item()
                n += 1
                if n % 200 == 0:
                    print(json.dumps({"ep": ep, "it": n, "loss": round(tot / n, 4), "s": round(time.time() - t0)}), flush=True)
            print(json.dumps({"ep": ep, "loss": round(tot / max(n, 1), 4), "s": round(time.time() - t0)}), flush=True)
            torch.save({"state": model.state_dict(), "args": vars(a)}, os.path.join(a.out, "vtal.pt"))


@torch.no_grad()
def dump(a):
    """Per-sample token heatmap probs + offsets -> cache/vtal_<split>.pkl {id: (times, prob, off)}."""
    samples = build_cache(a.root, a.split, a.cache)
    ck = torch.load(a.ckpt, map_location="cuda")
    model = VidTAL(pretrained=False).cuda().eval()
    model.load_state_dict(ck["state"])
    out = {}
    with Pool(12, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        for chunk, clips in batches(samples, a.split, a.root, 16, False, pool):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lh, lo = model(clips.cuda())
            p, o = torch.sigmoid(lh.float()).cpu().numpy(), lo.float().cpu().numpy()
            for i, s in enumerate(chunk):
                out[s["id"]] = (token_times(len(s["t"])), p[i], o[i])
    with open(os.path.join(a.cache, f"vtal_{a.split}.pkl"), "wb") as f:
        pickle.dump(out, f)
    print("done", len(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "dump"])
    ap.add_argument("--root", default="/data/extracted")
    ap.add_argument("--cache", default="/data/cache")
    ap.add_argument("--train", nargs="+", default=["train"])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--ckpt")
    ap.add_argument("--out", default="runs/vtal")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    train(a) if a.cmd == "train" else dump(a)
