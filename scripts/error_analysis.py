"""Where does v7 still lose points? Spatial and temporal error breakdown on the validation split.

Run after the v7 validation crops exist (/data/crops5, built from the v7 pose ensemble):
  scripts/docker_run.sh python scripts/error_analysis.py
"""
import json
import pickle

import numpy as np

from ra.crops import CROP, WH
from ra.data import JOINTS, build_cache
from ra.metric import m_per_px
from ra.refine import apply_points, ensemble_points

REF = ["weights/v6/refiner5_cnxs.pt", "weights/v7/refiner5_cnxb.pt"]
va = build_cache("/data/extracted", "validation", "/data/cache")
pose = pickle.load(open("/data/crops5/validation_pred_preds.pkl", "rb"))
meta = np.load("/data/crops5/validation_pred_meta.npz")
ref = apply_points(pose, ensemble_points(REF, "/data/crops5", "validation", "pred"), meta)
for acts in ref.values():
    for a in acts[1:]:
        a["score"] *= 0.5
HANDS = {"right": JOINTS.index("right_hand"), "left": JOINTS.index("left_hand")}


def tiou(p, g):
    i = max(0.0, min(p["end"], g["end"]) - max(p["start"], g["start"]))
    return i / max((p["end"] - p["start"]) + (g["end"] - g["start"]) - i, 1e-9)


def nearest_hand(kp, frames, xy):
    """Which hand (right/left) is closest to xy over the given frames; None if neither visible."""
    best, side = np.inf, None
    for name, j in HANDS.items():
        k = kp[frames, j]
        ok = k[:, 2] > 0
        if ok.any():
            d = np.linalg.norm(k[ok, :2] - xy, axis=1).min()
            if d < best:
                best, side = d, name
    return side


sp, tm = [], []
fp_scores_neg, top_scores_pos, n_gt_by_count, hit_by_count = [], [], {}, {}
for s in va:
    gts, ps = s["labels"] or [], ref[s["id"]]
    T = len(s["t"])
    if not gts:
        fp_scores_neg.append(max([p["score"] for p in ps], default=0.0))
        continue
    top_scores_pos.append(max([p["score"] for p in ps], default=0.0))
    k = min(len(gts), 3)
    n_gt_by_count[k] = n_gt_by_count.get(k, 0) + len(gts)
    for g in gts:
        best = max(ps, key=lambda p: tiou(p, g), default=None)
        t = tiou(best, g) if best else 0.0
        tm.append({"tiou": t, "dur": g["end"] - g["start"], "n": len(gts)})
        if t >= 0.3:
            hit_by_count[k] = hit_by_count.get(k, 0) + 1
        if best is None or t < 0.05:
            continue
        pb = pose[s["id"]][ps.index(best)]
        frames = np.arange(int(g["start"] * T), max(int(np.ceil(g["end"] * T)), int(g["start"] * T) + 1)).clip(0, T - 1)
        for v in range(2):
            gt_xy = np.array(g["pts"][v])
            if not np.isfinite(gt_xy[0]):
                continue
            mpp = m_per_px(s["kp"][v], np.array([WH, WH]))
            e_ref = np.linalg.norm((np.array(best["pts"][v]) - gt_xy) * WH) * mpp
            off_px = np.abs((np.array(pb["pts"][v]) - gt_xy) * WH)
            kp = s["kp"][v]
            g_side = nearest_hand(kp, frames, gt_xy)
            r_side = nearest_hand(kp, frames, np.array(best["pts"][v]))
            sp.append({"e": e_ref, "outside": bool((off_px > (224 // 2 - 8)).any()), "view": v,
                       "wrong_hand": g_side is not None and r_side is not None and g_side != r_side,
                       "no_hand": g_side is None, "tiou": t})

e = np.array([x["e"] for x in sp])
big = e > 0.15
out = {
    "spatial": {
        "n_points": len(sp), "median_m": round(float(np.median(e)), 3), "p90_m": round(float(np.percentile(e, 90)), 3),
        "share_over_0.15m": round(float(big.mean()), 3),
        "of_big_misses": {
            "gt_outside_crop": round(float(np.mean([x["outside"] for x, b in zip(sp, big) if b])), 3),
            "wrong_hand": round(float(np.mean([x["wrong_hand"] for x, b in zip(sp, big) if b])), 3),
            "no_hand_keypoint": round(float(np.mean([x["no_hand"] for x, b in zip(sp, big) if b])), 3),
            "poor_timing_tiou_lt_0.3": round(float(np.mean([x["tiou"] < 0.3 for x, b in zip(sp, big) if b])), 3),
        },
        "median_m_by_view": [round(float(np.median([x["e"] for x in sp if x["view"] == v])), 3) for v in range(2)],
    },
    "temporal": {
        "n_gt": len(tm),
        "tiou_median": round(float(np.median([x["tiou"] for x in tm])), 3),
        "share_tiou_lt_0.3": round(float(np.mean([x["tiou"] < 0.3 for x in tm])), 3),
        "share_missed_tiou_lt_0.05": round(float(np.mean([x["tiou"] < 0.05 for x in tm])), 3),
        "recall_tiou_0.3_by_actions_in_clip": {str(k): round(hit_by_count.get(k, 0) / n, 3) for k, n in sorted(n_gt_by_count.items())},
        "short_actions_le_0.1_tiou_median": round(float(np.median([x["tiou"] for x in tm if x["dur"] <= 0.1] or [0])), 3),
    },
    "no_action_clips": {
        "n": len(fp_scores_neg),
        "median_top_score": round(float(np.median(fp_scores_neg)), 3),
        "share_top_score_gt_0.5": round(float(np.mean(np.array(fp_scores_neg) > 0.5)), 3),
        "median_top_score_in_action_clips": round(float(np.median(top_scores_pos)), 3),
    },
}
print(json.dumps(out, indent=1))
