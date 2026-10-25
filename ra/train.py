"""Train PoseTAL on cached splits; evaluate mAP on a held-out split each few epochs.

  scripts/docker_run.sh python -m ra.train --train train --val validation --out runs/base
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import build_cache, sample_x
from .metric import map_global, m_per_px
from .model import MAX_T, PoseTAL, flip_x
from .predict import decode, run_model


def make_targets(s):
    """Per-frame targets: heatmap [T], offsets [T,2], points [T,2,2], reg mask [T]."""
    t = s["t"]
    T = len(t)
    hm = np.zeros(T, np.float32)
    off = np.zeros((T, 2), np.float32)
    pts = np.full((T, 2, 2), np.nan, np.float32)
    best = np.full(T, np.inf)
    for a in s["labels"] or []:
        c = 0.5 * (a["start"] + a["end"])
        sig = max((a["end"] - a["start"]) / 4, 0.01)
        g = np.exp(-0.5 * ((t - c) / sig) ** 2)
        g[np.abs(t - c).argmin()] = 1.0
        hm = np.maximum(hm, g)
        d = np.abs(t - c)
        own = (d < best) & (g > 0.05)  # frames assigned to this action for regression
        best = np.where(own, d, best)
        off[own] = np.stack([a["start"] - t[own], a["end"] - t[own]], 1)
        pts[own] = np.array(a["pts"], np.float32)
    return hm, off, pts, np.isfinite(best)


def collate(samples, in_dim, extra, vid=False, aug=False):
    B = len(samples)
    x = np.zeros((B, MAX_T, in_dim), np.float32)
    mpp = np.ones((B, 2), np.float32)
    pad = np.ones((B, MAX_T), bool)
    hm = np.zeros((B, MAX_T), np.float32)
    off = np.zeros((B, MAX_T, 2), np.float32)
    pts = np.full((B, MAX_T, 2, 2), np.nan, np.float32)
    rm = np.zeros((B, MAX_T), bool)
    for i, s in enumerate(samples):
        T = min(len(s["t"]), MAX_T)
        h, o, p, m = make_targets(s)
        xi = sample_x(s, extra, vid)
        mpp[i] = s["mpu"]
        if aug and np.random.rand() < 0.5:
            xi, p = flip_x(xi, p)
        x[i, :T], pad[i, :T], hm[i, :T], off[i, :T], pts[i, :T], rm[i, :T] = xi[:T], False, h[:T], o[:T], p[:T], m[:T]
    tt = lambda a: torch.from_numpy(a).cuda(non_blocking=True)
    return tt(x), tt(pad), tt(hm), tt(off), tt(pts), tt(rm), tt(mpp)


def loss_fn(out, hm, off, pts, rm, pad, mpp=None):
    lh, lo, lp = out
    valid = ~pad
    # CenterNet-style focal loss on the heatmap
    p = torch.sigmoid(lh).clamp(1e-4, 1 - 1e-4)
    pos = (hm >= 0.999) & valid
    neg = (~pos) & valid
    l_pos = -((1 - p) ** 2 * torch.log(p))[pos].sum()
    l_neg = -((1 - hm) ** 4 * p ** 2 * torch.log(1 - p))[neg].sum()
    l_hm = (l_pos + l_neg) / pos.sum().clamp(min=1)
    l_off = F.smooth_l1_loss(lo[rm] * 10, off[rm] * 10, beta=0.1) if rm.any() else lh.sum() * 0
    if mpp is None:
        pm = rm[..., None, None] & torch.isfinite(pts)
        l_pts = F.l1_loss(lp[pm], pts[pm]) * 20 if pm.any() else lh.sum() * 0
    else:  # euclidean error in metres, the unit the metric thresholds on
        pm = rm[..., None] & torch.isfinite(pts).all(-1)                  # [B, T, 2]
        d = torch.sqrt(((lp - torch.nan_to_num(pts)) ** 2).sum(-1) + 1e-8) * mpp[:, None, :]
        l_pts = d[pm].mean() * 10 if pm.any() else lh.sum() * 0
    return l_hm + l_off + l_pts, (l_hm.item(), l_off.item(), l_pts.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/extracted")
    ap.add_argument("--cache", default="/data/cache")
    ap.add_argument("--train", nargs="+", default=["train"])
    ap.add_argument("--val", default="validation")
    ap.add_argument("--out", default="runs/base")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pointer", type=int, default=1)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--extra", type=int, default=0, help="use timing / dense sampling-score features")
    ap.add_argument("--enc", default="tf", choices=["tf", "gru", "tcn", "gcn"], help="temporal encoder")
    ap.add_argument("--vid", type=int, default=0, help="use frozen MViTv2 features (cache/vid_<split>.pkl)")
    ap.add_argument("--metres", type=int, default=0, help="touch-point loss in metres instead of normalised L1")
    ap.add_argument("--wh", type=float, nargs=2, default=[600, 600])
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    tr = [s for sp in a.train for s in build_cache(a.root, sp, a.cache) if s["labels"] is not None]
    va = build_cache(a.root, a.val, a.cache) if a.val else []
    print(f"train {len(tr)}  val {len(va)}")

    for s in tr:  # metres per normalised image unit, per view (videos are square, a.wh[0] px wide)
        s["mpu"] = [m_per_px(s["kp"][v], np.array(a.wh)) * a.wh[0] for v in range(2)]
    model = PoseTAL(d=a.d, layers=a.layers, drop=a.drop, pointer=bool(a.pointer), extra=bool(a.extra), vid=bool(a.vid), enc=a.enc).cuda()
    allx = np.concatenate([sample_x(s, model.extra, model.vid) for s in tr])
    model.mu.copy_(torch.from_numpy(allx.mean(0)))
    model.sd.copy_(torch.from_numpy(allx.std(0) + 1e-3))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    log = []
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(len(tr))
        t0, agg = time.time(), np.zeros(3)
        for b in range(len(tr) // a.bs):
            batch = collate([tr[i] for i in perm[b * a.bs : (b + 1) * a.bs]], model.in_dim, model.extra, model.vid, aug=True)
            x, pad, hm, off, pts, rm, mpp = batch
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, pad)
            loss, parts = loss_fn([o.float() for o in out], hm, off, pts, rm, pad, mpp if a.metres else None)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            agg += parts
        agg /= max(1, len(tr) // a.bs)
        msg = {"ep": ep, "hm": round(agg[0], 4), "off": round(agg[1], 4), "pts": round(agg[2], 4), "s": round(time.time() - t0, 1)}
        if va and va[0]["labels"] is not None and ((ep + 1) % a.eval_every == 0 or ep == a.epochs - 1):
            preds = decode(run_model([model], va), va)
            msg["val_map"] = round(map_global(preds, va, a.wh), 4)
        print(json.dumps(msg), flush=True)
        log.append(msg)
    torch.save({"state": model.state_dict(), "args": vars(a)}, os.path.join(a.out, "model.pt"))
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"))


if __name__ == "__main__":
    main()
