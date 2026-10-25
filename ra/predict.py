"""Inference: run (an ensemble of) PoseTAL checkpoints on a split and write the Kaggle CSV.

  scripts/docker_run.sh python -m ra.predict --ckpt runs/base/model.pt --split challenge --csv submissions/base.csv
  scripts/docker_run.sh python -m ra.predict --ckpt runs/base/model.pt --split validation   # local mAP
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from .data import build_cache, sample_x
from .metric import map_global
from .model import MAX_T, PoseTAL, flip_x


@torch.no_grad()
def run_model(models, samples, bs=256, tta=True):
    """Returns per-sample (heatmap prob [T], offsets [T,2], points [T,2,2]) averaged over models (+ flip TTA)."""
    for m in models:
        m.eval()
    outs = []
    for b in range(0, len(samples), bs):
        chunk = samples[b : b + bs]
        variants = [False, True] if tta else [False]
        acc = None
        for flip in variants:  # loop order kept so old ensembles reproduce their CSVs bit-for-bit
            for m in models:
                x = np.zeros((len(chunk), MAX_T, m.in_dim), np.float32)
                pad = np.ones((len(chunk), MAX_T), bool)
                for i, s in enumerate(chunk):
                    T = min(len(s["t"]), MAX_T)
                    xi = sample_x(s, m.extra, m.vid)
                    if flip:
                        xi, _ = flip_x(xi, np.zeros((1, 2, 2), np.float32))
                    x[i, :T], pad[i, :T] = xi[:T], False
                h, o, p = m(torch.from_numpy(x).cuda(), torch.from_numpy(pad).cuda())
                h, o, p = torch.sigmoid(h).float().cpu().numpy(), o.float().cpu().numpy(), p.float().cpu().numpy()
                if flip:
                    p[..., 0] = 1 - p[..., 0]
                acc = [h, o, p] if acc is None else [acc[0] + h, acc[1] + o, acc[2] + p]
        n = len(models) * len(variants)
        for i, s in enumerate(chunk):
            T = min(len(s["t"]), MAX_T)
            outs.append((acc[0][i, :T] / n, acc[1][i, :T] / n, acc[2][i, :T] / n))
    return outs


def decode(outs, samples, topk=5, min_score=0.01, sec=1.0):
    """Heatmap peaks -> actions. `sec` scales the score of every peak after the strongest one."""
    preds = {}
    for (h, o, p), s in zip(outs, samples):
        t = s["t"][: len(h)]
        hp = np.pad(h, 1, constant_values=-1)
        peak = (h >= hp[:-2]) & (h >= hp[2:])
        idx = [i for i in np.argsort(-h) if peak[i] and h[i] >= min_score][:topk]
        acts = []
        for r, i in enumerate(idx):
            st = float(np.clip(t[i] + o[i, 0], 0, 1))
            en = float(np.clip(t[i] + o[i, 1], 0, 1))
            if en <= st + 1e-3:
                st, en = max(0.0, t[i] - 0.02), min(1.0, t[i] + 0.02)
            acts.append({"score": float(h[i]) * (sec if r else 1.0), "start": st, "end": en,
                         "pts": np.clip(p[i], 0, 1).tolist()})
        preds[s["id"]] = acts
    return preds


def to_kaggle(acts):
    return {"actions": [{
        "score": round(a["score"], 6), "start": round(a["start"], 6), "end": round(a["end"], 6),
        "spatial": {"action_cam": {
            "rank0": {"x": round(a["pts"][0][0], 6), "y": round(a["pts"][0][1], 6)},
            "rank1": {"x": round(a["pts"][1][0], 6), "y": round(a["pts"][1][1], 6)},
        }},
    } for a in acts]}


def write_csv(preds, ids, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "prediction"])
        for i in ids:
            w.writerow([i, json.dumps(to_kaggle(preds.get(i, [])))])


def load_models(paths):
    models = []
    for pth in paths:
        ck = torch.load(pth, map_location="cuda")
        m = PoseTAL(d=ck["args"]["d"], layers=ck["args"]["layers"], pointer=bool(ck["args"].get("pointer", 0)),
                    extra=bool(ck["args"].get("extra", 0)), vid=bool(ck["args"].get("vid", 0)), enc=ck["args"].get("enc", "tf")).cuda()
        m.load_state_dict(ck["state"])
        models.append(m)
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/extracted")
    ap.add_argument("--cache", default="/data/cache")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--split", default="challenge")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--sample-sub", default="/data/extracted/sample_submission.csv")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--sec", type=float, default=1.0, help="score factor for secondary peaks")
    ap.add_argument("--wh", type=float, nargs=2, default=[600, 600])
    a = ap.parse_args()
    samples = build_cache(a.root, a.split, a.cache)
    preds = decode(run_model(load_models(a.ckpt), samples), samples, topk=a.topk, sec=a.sec)
    if samples and samples[0]["labels"] is not None:
        print(json.dumps({"split": a.split, "map_global": round(map_global(preds, samples, a.wh), 4)}))
    if a.csv:
        ids = [r["id"] for r in csv.DictReader(open(a.sample_sub, newline=""))] if os.path.exists(a.sample_sub) else sorted(preds)
        missing = set(ids) - set(preds)
        empty = sum(1 for i in ids if not preds.get(i))
        print(f"writing {len(ids)} rows to {a.csv} ({len(missing)} ids not loaded, {empty} with no detected action)")
        write_csv(preds, ids, a.csv)


if __name__ == "__main__":
    main()
