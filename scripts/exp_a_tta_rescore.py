"""Experiment A: shifted-crop TTA and localisation-confidence rescoring (no training).

For every v7 validation query: 2 refiners x 5 crop offsets x {plain, flipped} = 20 point estimates.
  - shift TTA: average all 20 instead of the 4 centre-crop ones (v7 baseline)
  - rescoring: members that disagree (spread in metres) mark an uncertain point; scale the
    action's score by exp(-spread / tau) so well-placed actions rank first
  scripts/docker_run.sh python scripts/exp_a_tta_rescore.py
"""
import json
import pickle

import numpy as np
import torch

from ra.crops import CROP, WH
from ra.data import build_cache
from ra.metric import map_global
from ra.refine import IN, apply_points, load_refiner

REF = ["weights/v6/refiner5_cnxs.pt", "weights/v7/refiner5_cnxb.pt"]
OFFSETS = [(16, 16), (0, 0), (32, 0), (0, 32), (32, 32)]  # first = the centre crop used so far
CROPS = "/data/crops5"

va = build_cache("/data/extracted", "validation", "/data/cache")
preds = pickle.load(open(f"{CROPS}/validation_pred_preds.pkl", "rb"))
meta = np.load(f"{CROPS}/validation_pred_meta.npz")
X = np.load(f"{CROPS}/validation_pred.npy", mmap_mode="r")
N = len(X)


@torch.no_grad()
def members(model):
    """[N, len(OFFSETS), 2 (plain, flip), 2 (x, y)] full-frame normalised points."""
    out = np.zeros((N, len(OFFSETS), 2, 2), np.float32)
    model.eval()
    for b in range(0, N, 96):
        chunk = np.asarray(X[b : b + 96])                                   # [B, nf, CROP, CROP, 3]
        for oi, (ox, oy) in enumerate(OFFSETS):
            x = torch.from_numpy(np.ascontiguousarray(chunk[:, :, oy : oy + IN, ox : ox + IN]))
            x = x.permute(0, 1, 4, 2, 3).reshape(len(chunk), -1, IN, IN).float().cuda().to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(x).float()
                pf = model(torch.flip(x, dims=[-1])).float()
            pf = torch.stack([IN - pf[:, 0], pf[:, 1]], -1)
            org = torch.from_numpy(meta["origin"][b : b + len(chunk)] + np.array([ox, oy], np.float32)).cuda()
            out[b : b + len(chunk), oi, 0] = ((p + org) / WH).cpu().numpy()
            out[b : b + len(chunk), oi, 1] = ((pf + org) / WH).cpu().numpy()
    return out


M = np.stack([members(load_refiner(p)) for p in REF], 1)                    # [N, models, offsets, flip, 2]
np.save("/data/crops5/validation_pred_members.npy", M)


def score(pts, spread=None, tau=None):
    new = apply_points(preds, pts, meta)
    if spread is not None:  # per action: mean spread (metres) over its two views
        per = {}
        for q in range(N):
            per.setdefault((str(meta["sid"][q]), int(meta["act"][q])), []).append(spread[q])
        for (sid, a), v in per.items():
            new[sid][a]["score"] *= float(np.exp(-np.mean(v) / tau))
    for acts in new.values():
        for a in acts[1:]:
            a["score"] *= 0.5
    return round(map_global(new, va, (WH, WH)), 4)


res = {}
centre = M[:, :, 0].reshape(N, -1, 2).mean(1)
allm = M.reshape(N, -1, 2).mean(1)
res["v7_baseline_centre_crop"] = score(centre)
res["shift_tta_5_offsets"] = score(allm)
for name, pts, mem in (("centre", centre, M[:, :, :1]), ("shift", allm, M)):
    flat = mem.reshape(N, -1, 2)
    spread = (np.linalg.norm((flat - flat.mean(1, keepdims=True)) * WH, axis=-1).mean(1) * meta["mpp"])
    for tau in (0.01, 0.02, 0.05, 0.1):
        res[f"{name}+rescore_tau{tau}"] = score(pts, spread, tau)
print(json.dumps(res, indent=1))
