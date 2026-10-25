"""Crop refiner: move the pose model's touch point onto what the hand touches, from pixels.

Input: nf frames of a full-resolution crop centred on the current point estimate (nf is read from the crop
cache: 3 at 25/50/75 % or 5 at 10/30/50/70/90 % of the action; every shipped refiner uses 5), stacked to
3*nf channels. The backbone (--arch: resnet50, convnext_t/s/b, swin_t, effv2_s, dinov2_b or timm:<name>,
e.g. timm:convnextv2_base.fcmae_ft_in22k_in1k) has its first conv inflated to 3*nf channels. FPN-lite to
stride 8 (stride 4 with --s4 1), heatmap + soft-argmax -> sub-pixel point. Loss: Euclidean error in metres.
Crop refiners take 224 px sub-crops of 256 px cached crops; zoom refiners (--zoom 2) take 112 px of a 144 px
cache and enlarge it to 224; full-frame models (--in-size 288 --s4 1) take 288 px of whole frames resized to 320.

  scripts/docker_run.sh python -m ra.refine train --train train test --epochs 12 --arch convnext_s --lr 2e-4 \
      --crops /data/crops5 --out runs/ref5_cnxs
  scripts/docker_run.sh python -m ra.refine eval --ckpt runs/ref5_cnxs/refiner.pt --crops /data/crops5 --split validation
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .crops import CROP, WH

IN = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def _inflate(c0, nf=3):
    """Copy of a 3-channel input conv taking 3*nf channels (nf stacked frames), weights split evenly."""
    conv = nn.Conv2d(3 * nf, c0.out_channels, c0.kernel_size, c0.stride, c0.padding, bias=c0.bias is not None)
    conv.weight.data = c0.weight.data.repeat(1, nf, 1, 1) / nf
    if c0.bias is not None:
        conv.bias.data = c0.bias.data
    return conv


class Refiner(nn.Module):
    def __init__(self, pretrained=True, arch="resnet50", nf=3, s4=False, in_size=IN):
        super().__init__()
        self.in_size = in_size
        self.s4 = s4  # add the stride-4 level: heatmap at IN/4 instead of IN/8
        if arch == "resnet50":
            r = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            r.conv1 = _inflate(r.conv1, nf)
            self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)
            self.l2, self.l3, self.l4 = r.layer2, r.layer3, r.layer4
            chans = (512, 1024, 2048)
        elif arch.startswith("convnext_"):
            ctor = {"convnext_t": torchvision.models.convnext_tiny, "convnext_s": torchvision.models.convnext_small,
                    "convnext_b": torchvision.models.convnext_base}[arch]
            f = ctor(weights="IMAGENET1K_V1" if pretrained else None).features
            f[0][0] = _inflate(f[0][0], nf)
            self.stem = nn.Sequential(f[0], f[1])
            self.l2, self.l3, self.l4 = nn.Sequential(f[2], f[3]), nn.Sequential(f[4], f[5]), nn.Sequential(f[6], f[7])
            chans = tuple(f[i][0].block[0].in_channels for i in (3, 5, 7))  # stage widths at strides 8/16/32
        elif arch == "swin_t":  # hierarchical ViT; stages work channels-last
            f = torchvision.models.swin_t(weights="IMAGENET1K_V1" if pretrained else None).features
            f[0][0] = _inflate(f[0][0], nf)
            self.stem, self.l2, self.l3, self.l4 = nn.Sequential(f[0], f[1]), nn.Sequential(f[2], f[3]), nn.Sequential(f[4], f[5]), nn.Sequential(f[6], f[7])
            chans = (192, 384, 768)
        elif arch == "effv2_s":
            f = torchvision.models.efficientnet_v2_s(weights="IMAGENET1K_V1" if pretrained else None).features
            f[0][0] = _inflate(f[0][0], nf)
            self.stem, self.l2, self.l3, self.l4 = nn.Sequential(f[0], f[1], f[2]), f[3], nn.Sequential(f[4], f[5]), f[6]
            chans = (64, 160, 256)
        elif arch.startswith("timm:"):  # any timm CNN with a 4-level feature pyramid (strides 4/8/16/32)
            import timm
            self.tm = timm.create_model(arch[5:], pretrained=pretrained, features_only=True, in_chans=3 * nf, out_indices=(0, 1, 2, 3))
            c1, *chans = self.tm.feature_info.channels()
            chans = tuple(chans)
        elif arch == "dinov2_b":  # self-supervised ViT-B/14, single scale (stride 14)
            from transformers import Dinov2Config, Dinov2Model
            self.vit = Dinov2Model.from_pretrained("facebook/dinov2-base") if pretrained else Dinov2Model(Dinov2Config())
            pe = self.vit.embeddings.patch_embeddings
            pe.projection = _inflate(pe.projection, nf)
            pe.num_channels = self.vit.config.num_channels = 3 * nf
            chans = (self.vit.config.hidden_size,)
        else:
            raise ValueError(arch)
        self.arch = arch
        self.lat = nn.ModuleList([nn.Conv2d(c, 256, 1) for c in chans])
        if s4:
            c1 = c1 if arch.startswith("timm:") else {"resnet50": 256, "swin_t": 96, "effv2_s": 48}.get(arch)
            if arch.startswith("convnext_"):
                c1 = f[1][0].block[0].in_channels
            self.lat4 = nn.Conv2d(c1, 256, 1)
        self.head = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.GELU(), nn.Conv2d(256, 1, 1))
        self.register_buffer("mean", MEAN.repeat(1, nf, 1, 1))
        self.register_buffer("std", STD.repeat(1, nf, 1, 1))

    def forward(self, x):
        """x: [B, 3*nf, IN, IN] uint8-range floats -> points [B, 2] in input-crop pixels."""
        x = (x - self.mean) / self.std
        if self.arch == "dinov2_b":
            tok = self.vit(pixel_values=x).last_hidden_state[:, 1:]
            g = int(tok.shape[1] ** 0.5)
            f = tok.transpose(1, 2).reshape(x.shape[0], -1, g, g)
            p = F.interpolate(self.lat[0](f), size=(self.in_size // 8, self.in_size // 8), mode="bilinear", align_corners=False)
        else:
            if self.arch.startswith("timm:"):
                c1, c2, c3, c4 = self.tm(x)
            else:
                c1 = self.stem(x)
                c2 = self.l2(c1)
                c3 = self.l3(c2)
                c4 = self.l4(c3)
            if self.arch == "swin_t":  # NHWC -> NCHW for the FPN
                c1, c2, c3, c4 = (c.permute(0, 3, 1, 2) for c in (c1, c2, c3, c4))
            p = self.lat[2](c4)
            p = F.interpolate(p, size=c3.shape[-2:], mode="nearest") + self.lat[1](c3)
            p = F.interpolate(p, size=c2.shape[-2:], mode="nearest") + self.lat[0](c2)
            if self.s4:
                p = F.interpolate(p, size=c1.shape[-2:], mode="nearest") + self.lat4(c1)
        hm = self.head(p).float()                                   # [B, 1, h, w], stride 8 (4 with s4)
        B, _, h, w = hm.shape
        prob = torch.softmax(hm.view(B, -1), -1).view(B, h, w)
        ys = (torch.arange(h, device=x.device) + 0.5) * (self.in_size / h)
        xs = (torch.arange(w, device=x.device) + 0.5) * (self.in_size / w)
        pt = torch.stack([(prob.sum(1) * xs).sum(-1), (prob.sum(2) * ys).sum(-1)], -1)
        if getattr(self, "return_spread", False):  # heatmap spread (input px): the model's own uncertainty
            vx = (prob.sum(1) * (xs - pt[:, :1]) ** 2).sum(-1)
            vy = (prob.sum(2) * (ys - pt[:, 1:]) ** 2).sum(-1)
            return pt, torch.sqrt(vx + vy)
        return pt


class CropSet(torch.utils.data.Dataset):
    def __init__(self, crop_dir, split, mode, train, zoom=1, in_size=IN):
        self.path = os.path.join(crop_dir, f"{split}_{mode}.npy")
        self.X = None
        self.C = int(np.load(self.path, mmap_mode="r").shape[2])  # cached crop size
        self.zoom, self.in_size, self.sub = zoom, in_size, in_size // zoom  # sub-crop taken, then enlarged to in_size
        m = np.load(os.path.join(crop_dir, f"{split}_{mode}_meta.npz"))
        self.meta = {k: m[k] for k in m.files}
        self.scale = float(self.meta.get("scale", 1.0))             # cached px per frame px (full-frame mode < 1)
        self.idx = np.nonzero(self.meta["ok"])[0] if train else np.arange(len(self.meta["ok"]))
        self.train = train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.X is None:  # open lazily in each worker
            self.X = np.load(self.path, mmap_mode="r")
        k = self.idx[i]
        x = np.asarray(self.X[k])                                   # [nf, C, C, 3]
        m = self.C - self.sub
        ox, oy = (np.random.randint(0, m + 1, 2) if self.train else (m // 2, m // 2))
        x = x[:, oy : oy + self.sub, ox : ox + self.sub]
        org = self.meta["origin"][k] + np.array([ox, oy]) / self.scale  # frame px of input-crop (0, 0)
        tgt = (self.meta["target"][k] * WH - org) * self.scale * self.zoom  # target in input px (nan if unknown)
        flip = self.train and np.random.rand() < 0.5
        if flip:
            x = x[:, :, ::-1]
            tgt = np.array([self.in_size - tgt[0], tgt[1]])
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).reshape(-1, self.sub, self.sub).float()
        if self.zoom > 1:
            x = F.interpolate(x[None], size=(self.in_size, self.in_size), mode="bilinear", align_corners=False)[0]
        if self.train:
            x = x * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20)
        return x, torch.tensor(tgt, dtype=torch.float32), torch.tensor(org, dtype=torch.float32), \
            torch.tensor(self.meta["mpp"][k]), k


def train(a):
    sets = [CropSet(a.crops, s, "gt", True, zoom=a.zoom, in_size=a.in_size) for s in a.train]
    ds = torch.utils.data.ConcatDataset(sets)
    dl = torch.utils.data.DataLoader(ds, a.bs, shuffle=True, num_workers=10, drop_last=True, pin_memory=True, persistent_workers=True,
                                     worker_init_fn=lambda w: np.random.seed(torch.initial_seed() % 2**32))
    a.nf = int(np.load(sets[0].path, mmap_mode="r").shape[1])  # frames per crop stack, from the data
    model = Refiner(arch=a.arch, nf=a.nf, s4=bool(a.s4), in_size=a.in_size).cuda().to(memory_format=torch.channels_last)
    model.zoom = a.zoom  # predict_points reads it (per-epoch validation)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl), pct_start=0.05)
    os.makedirs(a.out, exist_ok=True)
    log = []
    for ep in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, tgt, _, mpp, _ in dl:
            x, tgt, mpp = x.cuda(non_blocking=True).to(memory_format=torch.channels_last), tgt.cuda(), mpp.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(x)
            loss = (torch.sqrt(((p - tgt) ** 2).sum(-1) + 1e-6) * mpp).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
            n += len(x)
        msg = {"ep": ep, "train_m": round(tot / n, 4), "s": round(time.time() - t0)}
        if a.val:
            msg.update(eval_gt(model, a.crops, a.val))
        print(json.dumps(msg), flush=True)
        log.append(msg)
        torch.save({"state": model.state_dict(), "args": vars(a)}, os.path.join(a.out, "refiner.pt"))
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"))


@torch.no_grad()
def predict_points(model, crop_dir, split, mode, tta=True):
    """Refined full-frame normalised points [N, 2] for every query of a crop set."""
    zoom = getattr(model, "zoom", 1)
    ds = CropSet(crop_dir, split, mode, False, zoom=zoom, in_size=model.in_size)
    dl = torch.utils.data.DataLoader(ds, int(os.environ.get("RA_PRED_BS", 128)), num_workers=8)  # smaller to share a busy GPU
    model.eval()
    out = np.zeros((len(ds), 2), np.float32)
    for x, _, org, _, k in dl:
        x = x.cuda().to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x).float()
            if tta:
                pf = model(torch.flip(x, dims=[-1])).float()
                p = 0.5 * (p + torch.stack([model.in_size - pf[:, 0], pf[:, 1]], -1))
        out[k.numpy()] = ((p.cpu() / zoom / ds.scale + org) / WH).numpy()
    return out


def eval_gt(model, crop_dir, split):
    """Median / p90 metre error on GT-centred (noised) crops, vs. the noised centre itself."""
    pts = predict_points(model, crop_dir, split, "gt")
    m = np.load(os.path.join(crop_dir, f"{split}_gt_meta.npz"))
    e_ref = np.linalg.norm((pts - m["target"]) * WH, axis=1) * m["mpp"]
    e_in = np.linalg.norm((m["centre"] - m["target"]) * WH, axis=1) * m["mpp"]
    return {"val_med_m": round(float(np.median(e_ref)), 4), "val_p90_m": round(float(np.percentile(e_ref, 90)), 4),
            "in_med_m": round(float(np.median(e_in)), 4)}


def load_refiner(path):
    ck = torch.load(path, map_location="cuda")
    model = Refiner(pretrained=False, arch=ck["args"].get("arch", "resnet50"), nf=ck["args"].get("nf", 3),
                    s4=bool(ck["args"].get("s4", 0)), in_size=ck["args"].get("in_size", IN)).cuda().to(memory_format=torch.channels_last)
    model.zoom = ck["args"].get("zoom", 1)
    state = ck["state"]
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})  # shipped weights are fp16
    return model


SHIFTS = [(16, 16), (0, 0), (32, 0), (0, 32), (32, 32)]  # crop offsets inside the cached CROP (first = centre)


@torch.no_grad()
def shift_spread(paths, crop_dir, split, mode):
    """Localisation uncertainty per query: mean distance (metres) of the refiners' shifted/flipped
    estimates from their mean. Disagreement marks a doubtful point (val: rescoring +0.012)."""
    X = np.load(os.path.join(crop_dir, f"{split}_{mode}.npy"), mmap_mode="r")
    meta = np.load(os.path.join(crop_dir, f"{split}_{mode}_meta.npz"))
    N, m = len(X), CROP - IN
    est = []
    for path in paths:
        model = load_refiner(path).eval()
        for ox, oy in SHIFTS:
            if ox > m or oy > m:
                continue
            for flip in (False, True):
                out = np.zeros((N, 2), np.float32)
                for b in range(0, N, 96):
                    x = torch.from_numpy(np.ascontiguousarray(np.asarray(X[b : b + 96])[:, :, oy : oy + IN, ox : ox + IN]))
                    x = x.permute(0, 1, 4, 2, 3).reshape(len(x), -1, IN, IN).float().cuda().to(memory_format=torch.channels_last)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        p = model(torch.flip(x, dims=[-1]) if flip else x).float()
                    if flip:
                        p = torch.stack([IN - p[:, 0], p[:, 1]], -1)
                    out[b : b + len(x)] = p.cpu().numpy() + meta["origin"][b : b + len(x)] + [ox, oy]
                est.append(out)
    est = np.stack(est, 1)
    return np.linalg.norm(est - est.mean(1, keepdims=True), axis=-1).mean(1) * meta["mpp"]


def rescore(preds, spread, meta, tau):
    """Scale each action's score by exp(-mean spread over its views / tau), in place."""
    per = {}
    for q in range(len(spread)):
        per.setdefault((str(meta["sid"][q]), int(meta["act"][q])), []).append(spread[q])
    for (sid, a), v in per.items():
        preds[sid][a]["score"] *= float(np.exp(-np.mean(v) / tau))


def ensemble_points(paths, crop_dir, split, mode):
    """Refined points averaged over several refiner checkpoints."""
    return np.mean([predict_points(load_refiner(p), crop_dir, split, mode) for p in paths], 0)


def apply_points(preds, pts, meta, alpha=1.0):
    """Copy of preds with each queried (action, view) point moved alpha of the way to the refined point."""
    new = {k: [dict(p, pts=[list(xy) for xy in p["pts"]]) for p in v] for k, v in preds.items()}
    for q in range(len(pts)):
        p = new[str(meta["sid"][q])][int(meta["act"][q])]
        v = int(meta["view"][q])
        p["pts"][v] = np.clip(alpha * pts[q] + (1 - alpha) * np.array(p["pts"][v]), 0, 1).tolist()
    return new


def eval_map(a):
    """mAP on a labelled split when the pose model's predicted points are replaced by refined ones."""
    from .data import build_cache
    from .metric import map_global
    samples = build_cache(a.root, a.split, a.cache)
    preds = pickle.load(open(os.path.join(a.crops, f"{a.split}_pred_preds.pkl"), "rb"))
    pts = ensemble_points(a.ckpt, a.crops, a.split, "pred")
    meta = np.load(os.path.join(a.crops, f"{a.split}_pred_meta.npz"))
    print(json.dumps({"pose_only": round(map_global(preds, samples, (WH, WH)), 4)}))
    for alpha in (0.75, 1.0):
        new = apply_points(preds, pts, meta, alpha)
        print(json.dumps({"alpha": alpha, "map_global": round(map_global(new, samples, (WH, WH)), 4)}))
        for sec in (0.5,):
            for acts in new.values():
                for j, p in enumerate(acts[1:]):
                    p["score"] *= sec
            print(json.dumps({"alpha": alpha, "sec": sec, "map_global": round(map_global(new, samples, (WH, WH)), 4)}))


def refine_twice(a):
    """Second pass: re-crop around the first pass's points and refine again (val experiment)."""
    from .crops import build
    from .data import build_cache
    from .metric import map_global
    samples = build_cache(a.root, a.split, a.cache)
    model = load_refiner(a.ckpt[0])
    preds = pickle.load(open(os.path.join(a.crops, f"{a.split}_pred_preds.pkl"), "rb"))
    meta = np.load(os.path.join(a.crops, f"{a.split}_pred_meta.npz"))
    p1 = apply_points(preds, predict_points(model, a.crops, a.split, "pred"), meta)
    build(a.root, a.cache, a.split, "pred2", a.crops, preds=p1)
    meta2 = np.load(os.path.join(a.crops, f"{a.split}_pred2_meta.npz"))
    pts2 = predict_points(model, a.crops, a.split, "pred2")
    for alpha in (0.5, 1.0):
        print(json.dumps({"pass": 2, "alpha": alpha, "map_global": round(map_global(apply_points(p1, pts2, meta2, alpha), samples, (WH, WH)), 4)}))
    print(json.dumps({"pass": 1, "map_global": round(map_global(p1, samples, (WH, WH)), 4)}))


def submit(a):
    """Full inference: pose ensemble -> crops around its points -> refiners -> [zoom stage] ->
    [full-frame fusion] -> [confidence rescoring] -> Kaggle CSV. With no optional stage set this is v3-v7."""
    import csv

    from .crops import build
    from .predict import write_csv
    nf = {torch.load(p, map_location="cpu")["args"].get("nf", 3) for p in a.ckpt}
    assert len(nf) == 1, "all refiners in one submit must use the same frame count"
    nf = nf.pop()
    build(a.root, a.cache, a.split, "pred", a.crops, a.pose, nf=nf)
    preds = pickle.load(open(os.path.join(a.crops, f"{a.split}_pred_preds.pkl"), "rb"))
    meta = np.load(os.path.join(a.crops, f"{a.split}_pred_meta.npz"))
    pts = ensemble_points(a.ckpt, a.crops, a.split, "pred")
    if not (a.zoom_ckpt or a.full_ckpt):  # v3-v7 path, kept byte-for-byte
        new = apply_points(preds, pts, meta, a.alpha)
        if a.rescore_tau > 0:
            rescore(new, shift_spread(a.ckpt, a.crops, a.split, "pred"), meta, a.rescore_tau)
    else:
        spread = shift_spread(a.ckpt, a.crops, a.split, "pred") if a.rescore_tau > 0 else None
        if a.zoom_ckpt:  # second stage: zoomed crops around the refined points (val: +0.002)
            nfz = torch.load(a.zoom_ckpt[0], map_location="cpu")["args"].get("nf", 3)
            build(a.root, a.cache, a.split, "zoom", a.crops, preds=apply_points(preds, pts, meta), nf=nfz, crop=a.zoom_crop)
            pts = a.zoom_alpha * ensemble_points(a.zoom_ckpt, a.crops, a.split, "zoom") + (1 - a.zoom_alpha) * pts
        if a.full_ckpt:  # independent full-frame model: blend its point, add its disagreement to the spread (val: +0.002)
            nff = torch.load(a.full_ckpt[0], map_location="cpu")["args"].get("nf", 3)
            build(a.root, a.cache, a.split, "full", a.crops, preds=preds, nf=nff, full=a.full_size)
            pe = ensemble_points(a.full_ckpt, a.crops, a.split, "full")
            if spread is not None:
                spread = spread + a.full_lambda * np.linalg.norm((pts - pe) * WH, axis=-1) * meta["mpp"]
            pts = (1 - a.full_weight) * pts + a.full_weight * pe
        new = apply_points(preds, pts, meta, a.alpha)
        if spread is not None:
            rescore(new, spread, meta, a.rescore_tau)
    for acts in new.values():  # down-weight secondary peaks (val: +0.004)
        for p in acts[1:]:
            p["score"] *= a.sec
    from .data import build_cache
    samples = build_cache(a.root, a.split, a.cache)
    if samples and samples[0]["labels"] is not None:  # labelled split: report the local metric
        from .metric import map_global
        print(json.dumps({"split": a.split, "map_global": round(map_global(new, samples, (WH, WH)), 4)}))
        if a.split != "challenge":  # the challenge split always gets its CSV, labelled copy or not
            return
    ids = [r["id"] for r in csv.DictReader(open(a.sample_sub, newline=""))]
    empty = sum(1 for i in ids if not new.get(i))
    print(f"writing {len(ids)} rows to {a.csv} ({len(set(ids) - set(new))} ids not loaded, {empty} with no detected action)")
    write_csv(new, ids, a.csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "twice", "submit"])
    ap.add_argument("--root", default="/data/extracted")
    ap.add_argument("--cache", default="/data/cache")
    ap.add_argument("--crops", default="/data/crops")
    ap.add_argument("--train", nargs="+", default=["train"])
    ap.add_argument("--val", default="validation")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--ckpt", nargs="+")
    ap.add_argument("--arch", default="resnet50", help="resnet50 | convnext_t/s/b | swin_t | effv2_s | dinov2_b | timm:<model name>")
    ap.add_argument("--out", default="runs/ref_r50")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pose", nargs="*", help="pose-model checkpoints (submit)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--sec", type=float, default=0.5)
    ap.add_argument("--s4", type=int, default=0, help="train: stride-4 heatmap level")
    ap.add_argument("--in-size", type=int, default=IN, help="train: network input size (full-frame model: 288)")
    ap.add_argument("--zoom", type=int, default=1, help="train: enlarge a 1/zoom sub-crop to the input size")
    ap.add_argument("--zoom-ckpt", nargs="*", default=[], help="submit: second-stage zoom refiners")
    ap.add_argument("--zoom-crop", type=int, default=144)
    ap.add_argument("--zoom-alpha", type=float, default=0.75, help="weight of the zoom point vs the stage-1 point")
    ap.add_argument("--full-ckpt", nargs="*", default=[], help="submit: independent full-frame models")
    ap.add_argument("--full-size", type=int, default=320)
    ap.add_argument("--full-weight", type=float, default=0.3, help="weight of the full-frame point")
    ap.add_argument("--full-lambda", type=float, default=0.5, help="full-frame disagreement added to the rescoring spread")
    ap.add_argument("--rescore-tau", type=float, default=0.0, help="localisation-confidence rescoring temperature in metres (0 = off; v8, v9, v10: 0.15)")
    ap.add_argument("--csv", default="submissions/submission.csv")
    ap.add_argument("--sample-sub", default="/data/extracted/sample_submission.csv")
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    {"train": train, "eval": eval_map, "twice": refine_twice, "submit": submit}[a.cmd](a)
