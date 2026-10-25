"""Experiments B (zoom second stage) and C (stride-4 refiner), end to end on validation, with A's rescoring.

  scripts/docker_run.sh python scripts/exp_bc_eval.py
"""
import json
import pickle

import numpy as np

from ra.crops import WH
from ra.data import build_cache
from ra.metric import map_global
from ra.refine import apply_points, ensemble_points, rescore

S, B = "weights/v6/refiner5_cnxs.pt", "weights/v7/refiner5_cnxb.pt"
C4, Z = "runs/ref5s4_cnxs/refiner.pt", "weights/v8/refinerz_cnxs.pt"
va = build_cache("/data/extracted", "validation", "/data/cache")
preds = pickle.load(open("/data/crops5/validation_pred_preds.pkl", "rb"))
meta = np.load("/data/crops5/validation_pred_meta.npz")
metaz = np.load("/data/crops5z/validation_pred_meta.npz")
assert all((meta[k] == metaz[k]).all() for k in ("sid", "act", "view"))
M = np.load("/data/crops5/validation_pred_members.npy")                  # experiment A: S, B x shifts x flip
N = len(M)
f = M.reshape(N, -1, 2)
spread = np.linalg.norm((f - f.mean(1, keepdims=True)) * WH, axis=-1).mean(1) * meta["mpp"]


def score(p, tau=0.1):
    p = {k: [dict(a) for a in v] for k, v in p.items()}
    if tau:
        rescore(p, spread, meta, tau)
    for acts in p.values():
        for a in acts[1:]:
            a["score"] *= 0.5
    return round(map_global(p, va, (WH, WH)), 4)


st1 = ensemble_points([S, B], "/data/crops5", "validation", "pred")
ref = apply_points(preds, st1, meta)
zp = ensemble_points([Z], "/data/crops5z", "validation", "pred")
res = {"v7": score(ref, 0), "v7+rescore": score(ref)}
for alpha in (0.5, 0.75, 1.0):
    res[f"+zoom alpha{alpha} +rescore"] = score(apply_points(ref, zp, metaz, alpha))
st1c = ensemble_points([S, B, C4], "/data/crops5", "validation", "pred")
res["S+B+stride4 +rescore"] = score(apply_points(preds, st1c, meta))
# zoom-vs-stage-1 disagreement as an extra confidence signal
dz = np.linalg.norm((zp - st1) * WH, axis=-1) * meta["mpp"]
for w in (0.5, 1.0):
    spread_bak = spread.copy()
    spread = spread_bak + w * dz
    res[f"+zoom alpha1.0 +rescore(spread + {w} x zoom shift)"] = score(apply_points(ref, zp, metaz, 1.0))
    spread = spread_bak
print(json.dumps(res, indent=1))
