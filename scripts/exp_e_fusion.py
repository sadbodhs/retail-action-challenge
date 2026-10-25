"""Experiment E: fuse the independent full-frame model with the crop pipeline (validation).

P1 = crop pipeline points (stage-1 S+B refiners, zoom refiner blended at 0.75); PE = full-frame model.
Fusions tried: weighted blend; switch to PE when the two disagree by more than delta; disagreement
as an extra confidence signal for A's rescoring.
  scripts/docker_run.sh python scripts/exp_e_fusion.py
"""
import json
import pickle

import numpy as np

from ra.crops import WH
from ra.data import build_cache
from ra.metric import map_global
from ra.refine import apply_points, ensemble_points, rescore

S, B, Z, E = "weights/v6/refiner5_cnxs.pt", "weights/v7/refiner5_cnxb.pt", "weights/v8/refinerz_cnxs.pt", "weights/v8/refinerf_cnxs.pt"
va = build_cache("/data/extracted", "validation", "/data/cache")
preds = pickle.load(open("/data/crops5/validation_pred_preds.pkl", "rb"))
meta = np.load("/data/crops5/validation_pred_meta.npz")
metaf = np.load("/data/crops5f/validation_pred_meta.npz")
assert all((meta[k] == metaf[k]).all() for k in ("sid", "act", "view"))
M = np.load("/data/crops5/validation_pred_members.npy")
N = len(M)
f = M.reshape(N, -1, 2)
spread_a = np.linalg.norm((f - f.mean(1, keepdims=True)) * WH, axis=-1).mean(1) * meta["mpp"]

st1 = ensemble_points([S, B], "/data/crops5", "validation", "pred")
zp = ensemble_points([Z], "/data/crops5z", "validation", "pred")
P1 = 0.75 * zp + 0.25 * st1
PE = ensemble_points([E], "/data/crops5f", "validation", "pred")
d = np.linalg.norm((P1 - PE) * WH, axis=-1) * meta["mpp"]                  # disagreement, metres


def score(pts, spread, tau=0.1):
    p = apply_points(preds, pts, meta)
    rescore(p, spread, meta, tau)
    for acts in p.values():
        for a in acts[1:]:
            a["score"] *= 0.5
    return round(map_global(p, va, (WH, WH)), 4)


res = {"current_best (P1 + rescore)": score(P1, spread_a), "full-frame alone (PE + rescore)": score(PE, spread_a)}
print("disagreement median %.3f m, share > 0.15 m %.3f" % (np.median(d), (d > 0.15).mean()))
for w in (0.15, 0.3, 0.5):
    res[f"blend w_E={w}"] = score((1 - w) * P1 + w * PE, spread_a)
for delta in (0.15, 0.25):
    sw = np.where((d > delta)[:, None], PE, P1)
    res[f"switch to E if d>{delta}"] = score(sw, spread_a)
for lam in (0.25, 0.5, 1.0):
    for tau in (0.1, 0.15):
        res[f"P1, rescore spread_a + {lam} x d, tau {tau}"] = score(P1, spread_a + lam * d, tau)
for lam in (0.5,):
    res[f"blend 0.3 + rescore spread_a + {lam} x d"] = score(0.7 * P1 + 0.3 * PE, spread_a + lam * d, 0.15)
print(json.dumps(res, indent=1))
