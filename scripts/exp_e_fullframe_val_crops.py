"""Experiment E helper: full-frame validation stacks for every v7-predicted action (no point estimate used).

  scripts/docker_run.sh python scripts/exp_e_fullframe_val_crops.py
"""
import pickle

from ra.crops import build

preds = pickle.load(open("/data/crops5/validation_pred_preds.pkl", "rb"))  # v7 pose-ensemble actions
build("/data/extracted", "/data/cache", "validation", "pred", "/data/crops5f", preds=preds, nf=5, full=320)
