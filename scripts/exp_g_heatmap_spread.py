"""Experiment G: each refiner's heatmap spread (its own uncertainty) as a rescoring signal (validation).

  scripts/docker_run.sh python scripts/exp_g_heatmap_spread.py
"""
import json
import pickle

import numpy as np
import torch

from ra.crops import WH
from ra.data import build_cache
from ra.metric import map_global
from ra.refine import CropSet, apply_points, ensemble_points, load_refiner, rescore

va = build_cache("/data/extracted", "validation", "/data/cache")
preds = pickle.load(open("/data/crops5/validation_pred_preds.pkl", "rb"))
meta = np.load("/data/crops5/validation_pred_meta.npz")


@torch.no_grad()
def heat_spread(path, crop_dir):
    """Per query: heatmap spread in metres (frame px x mpp), averaged with the flipped copy."""
    model = load_refiner(path).eval()
    model.return_spread = True
    ds = CropSet(crop_dir, "validation", "pred", False, zoom=model.zoom, in_size=model.in_size)
    out = np.zeros(len(ds), np.float32)
    for x, _, _, mpp, k in torch.utils.data.DataLoader(ds, 128, num_workers=8):
        x = x.cuda().to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, sp = model(x)
            _, spf = model(torch.flip(x, dims=[-1]))
        out[k.numpy()] = (0.5 * (sp + spf).float().cpu() / model.zoom / ds.scale * mpp).numpy()
    return out


H = {n: heat_spread(p, c) for n, p, c in [("S", "weights/v6/refiner5_cnxs.pt", "/data/crops5"), ("B", "weights/v7/refiner5_cnxb.pt", "/data/crops5"),
                                         ("Z", "weights/v8/refinerz_cnxs.pt", "/data/crops5z"), ("E", "weights/v8/refinerf_cnxs.pt", "/data/crops5f")]}
np.savez("/data/crops5/validation_pred_heatspread.npz", **H)
M = np.load("/data/crops5/validation_pred_members.npy"); N = len(M); f = M.reshape(N, -1, 2)
spread_a = np.linalg.norm((f - f.mean(1, keepdims=True)) * WH, axis=-1).mean(1) * meta["mpp"]
st1 = ensemble_points(["weights/v6/refiner5_cnxs.pt", "weights/v7/refiner5_cnxb.pt"], "/data/crops5", "validation", "pred")
zp = ensemble_points(["weights/v8/refinerz_cnxs.pt"], "/data/crops5z", "validation", "pred")
pe = ensemble_points(["weights/v8/refinerf_cnxs.pt"], "/data/crops5f", "validation", "pred")
p1 = 0.75 * zp + 0.25 * st1; fin = 0.7 * p1 + 0.3 * pe
d_full = np.linalg.norm((p1 - pe) * WH, axis=-1) * meta["mpp"]
v8_spread = spread_a + 0.5 * d_full


def score(spread, tau):
    p = apply_points(preds, fin, meta)
    rescore(p, spread, meta, tau)
    for acts in p.values():
        for a in acts[1:]:
            a["score"] *= 0.5
    return round(map_global(p, va, (WH, WH)), 4)


res = {"v8": score(v8_spread, 0.15)}
print("median heat spread (m):", {k: round(float(np.median(v)), 3) for k, v in H.items()})
hs = {"Z": H["Z"], "S+B": 0.5 * (H["S"] + H["B"]), "all4": np.mean([H[k] for k in H], 0)}
for name, h in hs.items():
    for lam in (0.25, 0.5, 1.0):
        for tau in (0.15, 0.2):
            res[f"v8 spread + {lam} x heat[{name}], tau {tau}"] = score(v8_spread + lam * h, tau)
print(json.dumps(res, indent=1))
