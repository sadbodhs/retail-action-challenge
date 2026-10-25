"""Local re-implementation of the challenge's mAP Global (our reading of the Data/Evaluation tabs).

A prediction matches a GT action when tIoU >= t_thr AND spatial distance (metres) <= d_thr.
Distance = mean over the two views of the pixel distance times that view's metres-per-pixel
factor, estimated from the subject's 2D bone lengths vs. average 3D bone lengths.
AP is computed per (t_thr, d_thr) from precision/recall at 101 confidence thresholds, then averaged.
The official scorer is not public, so absolute numbers can differ slightly; use it for ranking models.
"""
import numpy as np

from .data import JOINTS

BONE_LENGTH_MEANS = {
    ("neck", "left_shoulder"): 0.18136, ("neck", "right_shoulder"): 0.18081,
    ("left_shoulder", "left_elbow"): 0.27096, ("left_elbow", "left_wrist"): 0.21228,
    ("right_shoulder", "right_elbow"): 0.27210, ("right_elbow", "right_wrist"): 0.21316,
    ("left_hip", "left_knee"): 0.39204, ("left_knee", "left_ankle"): 0.39530,
    ("right_hip", "right_knee"): 0.39266, ("right_knee", "right_ankle"): 0.39322,
    ("left_shoulder", "right_shoulder"): 0.35484, ("left_hip", "right_hip"): 0.17150,
    ("left_shoulder", "left_hip"): 0.51375, ("right_shoulder", "right_hip"): 0.51226,
}
M_PX_FACTOR_AVG = 3.07 / 1000  # metres per pixel
T_THRS = np.linspace(0.05, 0.5, 10)
D_THRS = np.linspace(1.0, 0.05, 20)
C_THRS = np.linspace(1.0, 0.0, 101)
_JI = {n: i for i, n in enumerate(JOINTS)}


def m_per_px(kp, wh):
    """kp: [T, J, 3] normalised keypoints of one view -> metres per pixel."""
    ratios = []
    for (a, b), m in BONE_LENGTH_MEANS.items():
        if a not in _JI or b not in _JI:
            continue
        pa, pb = kp[:, _JI[a]], kp[:, _JI[b]]
        ok = (pa[:, 2] > 0) & (pb[:, 2] > 0)
        if not ok.any():
            continue
        px = np.linalg.norm((pa[ok, :2] - pb[ok, :2]) * wh, axis=1)
        px = px[px > 2]
        ratios.extend(m / px)
    return float(np.median(ratios)) if ratios else M_PX_FACTOR_AVG


def _pair_mats(preds, gts, mpp, wh):
    """tIoU and metric distance between every pred and GT of one video."""
    ps = np.array([[p["start"], p["end"]] for p in preds])
    gs = np.array([[g["start"], g["end"]] for g in gts])
    inter = np.clip(np.minimum(ps[:, None, 1], gs[None, :, 1]) - np.maximum(ps[:, None, 0], gs[None, :, 0]), 0, None)
    union = (ps[:, 1] - ps[:, 0])[:, None] + (gs[:, 1] - gs[:, 0])[None, :] - inter
    tiou = inter / np.maximum(union, 1e-9)
    pp = np.array([p["pts"] for p in preds], np.float64)  # [P, 2 views, 2]
    gp = np.array([g["pts"] for g in gts], np.float64)    # [G, 2, 2]
    d = np.linalg.norm((pp[:, None] - gp[None]) * wh, axis=-1) * np.asarray(mpp)  # [P, G, 2]
    dist = np.nanmean(d, axis=-1)
    return tiou, np.nan_to_num(dist, nan=1e9)


def map_global(pred_by_id, samples, wh=(1, 1)):
    """pred_by_id: {id: [{"score","start","end","pts":[[x,y],[x,y]]}]}; samples: cached dicts with labels."""
    wh = np.asarray(wh, np.float64)
    nT, nD = len(T_THRS), len(D_THRS)
    scores, tps = [], []  # per prediction: score, and TP flag per (t, d) combo
    n_gt = 0
    for s in samples:
        gts = s["labels"] or []
        preds = sorted(pred_by_id.get(s["id"], []), key=lambda p: -p["score"])
        n_gt += len(gts)
        if not preds:
            continue
        tp = np.zeros((len(preds), nT, nD), bool)
        if gts:
            mpp = [m_per_px(s["kp"][v], wh) for v in range(2)]
            tiou, dist = _pair_mats(preds, gts, mpp, wh)
            ok = (tiou[..., None, None] >= T_THRS[None, None, :, None]) & (dist[..., None, None] <= D_THRS[None, None, None, :])
            used = np.zeros((len(gts), nT, nD), bool)
            for i in range(len(preds)):
                cand = ok[i] & ~used                       # [G, nT, nD]
                score_g = np.where(cand, tiou[i][:, None, None], -1)
                best = score_g.argmax(0)                  # [nT, nD]
                hit = score_g.max(0) >= 0
                tp[i] = hit
                ti, di = np.nonzero(hit)
                used[best[ti, di], ti, di] = True
        scores.append(np.array([p["score"] for p in preds]))
        tps.append(tp)
    if n_gt == 0:
        return 0.0
    scores = np.concatenate(scores) if scores else np.zeros(0)
    tps = np.concatenate(tps) if tps else np.zeros((0, nT, nD), bool)
    order = np.argsort(-scores, kind="stable")
    scores, tps = scores[order], tps[order]
    cum_tp = np.cumsum(tps, 0)
    # predictions kept at each confidence threshold
    n_keep = np.searchsorted(-scores, -C_THRS, side="right")
    prec = np.ones((len(C_THRS), nT, nD))
    rec = np.zeros((len(C_THRS), nT, nD))
    nz = n_keep > 0
    tp_at = cum_tp[np.maximum(n_keep - 1, 0)]
    prec[nz] = tp_at[nz] / n_keep[nz, None, None]
    rec[nz] = tp_at[nz] / n_gt
    # precision envelope (non-increasing as recall grows), then area under PR
    prec = np.maximum.accumulate(prec[::-1], axis=0)[::-1]
    drec = np.diff(np.concatenate([np.zeros((1, nT, nD)), rec]), axis=0)
    ap = (drec * prec).sum(0)
    return float(ap.mean())
