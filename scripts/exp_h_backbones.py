"""Experiment H: timm-backbone refiners as stage-1 / zoom / full-frame members, end to end on validation (v8 recipe).

  scripts/docker_run.sh env RA_PRED_BS=16 python scripts/exp_h_backbones.py stage1=S,B,runs/tm_b22_5f/refiner.pt
  (each arg: stage1=..., zoom=..., full=... ; comma-separated checkpoints; S/B/Z/E = the v8 models)
"""
import json
import pickle
import sys

import numpy as np

from ra.crops import WH
from ra.data import build_cache
from ra.metric import map_global
from ra.refine import apply_points, ensemble_points, rescore

V8 = {"S": "weights/v6/refiner5_cnxs.pt", "B": "weights/v7/refiner5_cnxb.pt", "Z": "weights/v8/refinerz_cnxs.pt", "E": "weights/v8/refinerf_cnxs.pt"}
cfg = {"stage1": "S,B", "zoom": "Z", "full": "E"}
for arg in sys.argv[1:]:
    k, v = arg.split("=", 1)
    cfg[k] = v
paths = {k: [V8.get(x, x) for x in v.split(",")] for k, v in cfg.items()}
va = build_cache("/data/extracted", "validation", "/data/cache")
preds = pickle.load(open("/data/crops5/validation_pred_preds.pkl", "rb"))
meta = np.load("/data/crops5/validation_pred_meta.npz")
M = np.load("/data/crops5/validation_pred_members.npy")
N = len(M)
f = M.reshape(N, -1, 2)
spread_a = np.linalg.norm((f - f.mean(1, keepdims=True)) * WH, axis=-1).mean(1) * meta["mpp"]
st1 = ensemble_points(paths["stage1"], "/data/crops5", "validation", "pred")
zp = ensemble_points(paths["zoom"], "/data/crops5z", "validation", "pred")
pe = ensemble_points(paths["full"], "/data/crops5f", "validation", "pred")
p1 = 0.75 * zp + 0.25 * st1
fin = 0.7 * p1 + 0.3 * pe
p = apply_points(preds, fin, meta)
rescore(p, spread_a + 0.5 * np.linalg.norm((p1 - pe) * WH, axis=-1) * meta["mpp"], meta, 0.15)
for acts in p.values():
    for a in acts[1:]:
        a["score"] *= 0.5
print(json.dumps({"config": cfg, "map_global": round(map_global(p, va, (WH, WH)), 4)}))
