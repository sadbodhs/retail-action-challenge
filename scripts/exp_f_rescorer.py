"""Experiment F: learned rescorer, 5-fold cross-validated on the validation split (v8 pipeline outputs).

Per detection: features from the pose model and the refiners; target = fraction of the metric's
(tIoU x distance) threshold pairs at which the detection is a true positive under greedy matching
in v8-score order. A small MLP predicts it; its output becomes the detection score.
  scripts/docker_run.sh python scripts/exp_f_rescorer.py
"""
import json
import pickle

import numpy as np
import torch
import torch.nn as nn

from ra.crops import WH
from ra.data import JOINTS, build_cache
from ra.metric import D_THRS, T_THRS, _pair_mats, m_per_px, map_global
from ra.refine import apply_points, ensemble_points, rescore

S, B, Z, E = "weights/v6/refiner5_cnxs.pt", "weights/v7/refiner5_cnxb.pt", "weights/v8/refinerz_cnxs.pt", "weights/v8/refinerf_cnxs.pt"
va = build_cache("/data/extracted", "validation", "/data/cache")
by_id = {s["id"]: s for s in va}
preds = pickle.load(open("/data/crops5/validation_pred_preds.pkl", "rb"))
meta = np.load("/data/crops5/validation_pred_meta.npz")
M = np.load("/data/crops5/validation_pred_members.npy")
N = len(M)
f = M.reshape(N, -1, 2)
spread_q = np.linalg.norm((f - f.mean(1, keepdims=True)) * WH, axis=-1).mean(1) * meta["mpp"]
st1 = ensemble_points([S, B], "/data/crops5", "validation", "pred")
zp = ensemble_points([Z], "/data/crops5z", "validation", "pred")
pe = ensemble_points([E], "/data/crops5f", "validation", "pred")
p1 = 0.75 * zp + 0.25 * st1
fin = 0.7 * p1 + 0.3 * pe
pose_pts = np.array([preds[str(meta["sid"][q])][int(meta["act"][q])]["pts"][int(meta["view"][q])] for q in range(N)])
m_ = lambda a, b: np.linalg.norm((a - b) * WH, axis=-1) * meta["mpp"]
d_full, d_zoom, d_ref = m_(p1, pe), m_(zp, st1), m_(st1, pose_pts)

# v8 detections and scores (the exact v8 formula)
v8 = apply_points(preds, fin, meta)
rescore(v8, spread_q + 0.5 * d_full, meta, 0.15)
for acts in v8.values():
    for a in acts[1:]:
        a["score"] *= 0.5
print("v8 baseline:", round(map_global(v8, va, (WH, WH)), 4))

# per-detection features
HANDS = [JOINTS.index("right_hand"), JOINTS.index("left_hand")]
qidx = {}
for q in range(N):
    qidx.setdefault((str(meta["sid"][q]), int(meta["act"][q])), []).append(q)
rows, keys = [], []
for sid, acts in preds.items():
    s = by_id[sid]
    T = len(s["t"])
    top = max([a["score"] for a in acts], default=1e-6)
    for i, a in enumerate(acts):
        qs = qidx[(sid, i)]
        fr = np.arange(int(a["start"] * T), max(int(np.ceil(a["end"] * T)), int(a["start"] * T) + 1)).clip(0, T - 1)
        hand = np.mean([s["kp"][v][fr][:, HANDS, 2].max(1).mean() > 0 for v in range(2)])
        rows.append([a["score"], a["score"] / top, i, len(acts), a["end"] - a["start"],
                     np.mean(spread_q[qs]), np.max(spread_q[qs]), np.mean(d_full[qs]), np.mean(d_zoom[qs]),
                     np.mean(d_ref[qs]), hand, np.log(np.mean(meta["mpp"][qs]))])
        keys.append((sid, i))
X = np.array(rows, np.float32)
FEATS = ["pose_score", "score_ratio", "rank", "n_peaks", "duration", "spread_mean", "spread_max", "d_full",
         "d_zoom", "d_ref", "hand_visible", "log_mpp"]


def tp_fraction(P):
    """Fraction of (t, d) threshold pairs at which each detection is a TP (greedy, metric order)."""
    y = {}
    for s in va:
        gts, ps = s["labels"] or [], P.get(s["id"], [])
        order = sorted(range(len(ps)), key=lambda i: -ps[i]["score"])
        if not ps:
            continue
        if not gts:
            for i in order:
                y[(s["id"], i)] = 0.0
            continue
        mpp = [m_per_px(s["kp"][v], np.array([WH, WH])) for v in range(2)]
        tiou, dist = _pair_mats([ps[i] for i in order], gts, mpp, np.array([WH, WH]))
        ok = (tiou[..., None, None] >= T_THRS[None, None, :, None]) & (dist[..., None, None] <= D_THRS[None, None, None, :])
        used = np.zeros((len(gts), len(T_THRS), len(D_THRS)), bool)
        for r, i in enumerate(order):
            cand = ok[r] & ~used
            sc = np.where(cand, tiou[r][:, None, None], -1)
            hit = sc.max(0) >= 0
            ti, di = np.nonzero(hit)
            used[sc.argmax(0)[ti, di], ti, di] = True
            y[(s["id"], i)] = float(hit.mean())
    return y


yd = tp_fraction(v8)
y = np.array([yd[k] for k in keys], np.float32)
print("detections:", len(y), " mean target %.3f" % y.mean())


def fit_predict(Xtr, ytr, Xte, seed=0, epochs=300):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    xt, xe = torch.tensor((Xtr - mu) / sd), torch.tensor((Xte - mu) / sd)
    net = nn.Sequential(nn.Linear(X.shape[1], 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
    yt = torch.tensor(ytr)[:, None]
    for _ in range(epochs):
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(net(xt), yt).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(xe)).numpy()[:, 0]


rng = np.random.default_rng(0)
sids = np.array(sorted({k[0] for k in keys}))
fold_of = dict(zip(sids, rng.permutation(len(sids)) % 5))
fold = np.array([fold_of[k[0]] for k in keys])
pred_y = np.zeros(len(y), np.float32)
for k in range(5):
    tr, te = fold != k, fold == k
    pred_y[te] = np.mean([fit_predict(X[tr], y[tr], X[te], seed) for seed in range(3)], 0)


def with_scores(sc):
    P = {k: [dict(a) for a in v] for k, v in v8.items()}
    for (sid, i), v in zip(keys, sc):
        P[sid][i]["score"] = float(v)
    return round(map_global(P, va, (WH, WH)), 4)


v8s = np.array([v8[sid][i]["score"] for sid, i in keys])
res = {"v8": with_scores(v8s), "learned (CV)": with_scores(pred_y)}
for w in (0.3, 0.5, 0.7):
    res[f"v8^{1 - w:.1f} * learned^{w}"] = with_scores(v8s ** (1 - w) * np.clip(pred_y, 1e-6, 1) ** w)
res["oracle: true target as score"] = with_scores(y)
print(json.dumps(res, indent=1))
corr = {n: round(float(np.corrcoef(X[:, j], y)[0, 1]), 3) for j, n in enumerate(FEATS)}
print("feature correlation with target:", json.dumps(corr))
