"""Experiment B helper: validation zoom crops around v7's refined points (stage-1 crops in /data/crops5).

  scripts/docker_run.sh python scripts/exp_b_zoom_val_crops.py
"""
import os
import pickle

import numpy as np

from ra.crops import build
from ra.refine import apply_points, ensemble_points

REF = ["weights/v6/refiner5_cnxs.pt", "weights/v7/refiner5_cnxb.pt"]
preds = pickle.load(open("/data/crops5/validation_pred_preds.pkl", "rb"))
meta = np.load("/data/crops5/validation_pred_meta.npz")
refined = apply_points(preds, ensemble_points(REF, "/data/crops5", "validation", "pred"), meta)
os.makedirs("/data/crops5z", exist_ok=True)
build("/data/extracted", "/data/cache", "validation", "pred", "/data/crops5z", preds=refined, nf=5, crop=144)
