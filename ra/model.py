"""Pose-sequence encoders (transformer, BiGRU, TCN, skeleton GCN) with 1D CenterNet-style heads.

Per frame it predicts: action-centre heatmap logit, signed offsets to the action's start and
end (normalised time), and the interaction point (x, y) in both views.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import EXTRA_DIM, FEAT_DIM, J, VID_DIM, VIEW_DIM

MAX_T = 64
OFF_SCALE = 10.0  # offsets are predicted in units of 0.1 normalised time

# indices to mirror left/right joints when flipping x
_LR = {"right": "left", "left": "right"}


def _flip_perm():
    from .data import JOINTS
    perm = []
    for n in JOINTS:
        side = n.split("_")[0]
        perm.append(JOINTS.index(n.replace(side, _LR[side], 1)) if side in _LR else JOINTS.index(n))
    return np.array(perm)


FLIP_PERM = _flip_perm()


POINTER_JOINTS = ["right_hand", "left_hand", "right_wrist", "left_wrist", "right_elbow", "left_elbow"]


class TCNBlock(nn.Module):
    """Pre-norm residual dilated temporal conv block; padded frames are zeroed before each conv."""

    def __init__(self, d, dil, drop):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, 3, padding=dil, dilation=dil)
        self.proj = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, h, pad):
        z = self.norm(h).masked_fill(pad[..., None], 0).transpose(1, 2)
        z = self.proj(self.drop(F.gelu(self.conv(z)))).transpose(1, 2)
        return h + z


BONES = [("neck", "top_of_head"), ("neck", "right_shoulder"), ("right_shoulder", "right_elbow"),
         ("right_elbow", "right_wrist"), ("right_wrist", "right_hand"), ("neck", "left_shoulder"),
         ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("left_wrist", "left_hand"),
         ("neck", "middle_of_waist"), ("middle_of_waist", "right_hip"), ("right_hip", "right_knee"),
         ("right_knee", "right_ankle"), ("right_ankle", "right_foot"), ("middle_of_waist", "left_hip"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("left_ankle", "left_foot")]


def skeleton_adjacency():
    """Normalised adjacency over 2 views x J joints: bones within a view, self loops, same joint across views."""
    from .data import JOINTS
    N = 2 * J
    A = np.eye(N, dtype=np.float32)
    for v in range(2):
        for a, b in BONES:
            i, j = v * J + JOINTS.index(a), v * J + JOINTS.index(b)
            A[i, j] = A[j, i] = 1
    for j in range(J):
        A[j, J + j] = A[J + j, j] = 1
    d = A.sum(1) ** -0.5
    return d[:, None] * A * d[None, :]


class GCNBlock(nn.Module):
    """Spatial graph conv over joints (fixed skeleton + learned adjacency), then a dilated depthwise temporal conv."""

    def __init__(self, c, dil, drop, A):
        super().__init__()
        self.register_buffer("A", torch.from_numpy(A), persistent=False)
        self.B = nn.Parameter(torch.zeros_like(self.A))  # learned extra edges (CTR-GCN-style refinement)
        self.n1, self.n2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.spatial = nn.Linear(c, c)
        self.tconv = nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil, groups=c)
        self.tproj = nn.Conv1d(c, c, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x, pad):
        """x: [B, T, N, C]; pad: [B, T]."""
        z = torch.einsum("nm,btmc->btnc", self.A + self.B, self.n1(x))
        x = x + self.drop(F.gelu(self.spatial(z)))
        B, T, N, C = x.shape
        z = self.n2(x).masked_fill(pad[:, :, None, None], 0).permute(0, 2, 3, 1).reshape(B * N, C, T)
        z = self.tproj(self.drop(F.gelu(self.tconv(z)))).view(B, N, C, T).permute(0, 3, 1, 2)
        return x + z


class SkeletonGCN(nn.Module):
    """ST-GCN-style encoder over both views' skeletons -> per-frame feature [B, T, d]."""

    def __init__(self, d, c=96, blocks=6, drop=0.1):
        super().__init__()
        A = skeleton_adjacency()
        self.inp = nn.Linear(6, c)                     # x, y, score, present, dx, dy
        self.joint = nn.Parameter(torch.zeros(1, 1, 2 * J, c))
        nn.init.trunc_normal_(self.joint, std=0.02)
        self.blocks = nn.ModuleList([GCNBlock(c, 2 ** (i % 4), drop, A) for i in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(2 * c), nn.Linear(2 * c, d))

    def forward(self, x, pad):
        B, T = x.shape[:2]
        kp = torch.cat([x[..., v * VIEW_DIM : v * VIEW_DIM + J * 4].view(B, T, J, 4) for v in range(2)], 2)  # [B,T,2J,4]
        present = kp[..., 3:4]
        dxy = torch.zeros_like(kp[..., :2])
        dxy[:, 1:] = (kp[:, 1:, :, :2] - kp[:, :-1, :, :2]) * present[:, 1:] * present[:, :-1]
        h = self.inp(torch.cat([kp, dxy * 10], -1)) + self.joint
        for blk in self.blocks:
            h = blk(h, pad)
        return self.out(torch.cat([h.mean(2), h.amax(2)], -1))


class PoseTAL(nn.Module):
    def __init__(self, d=256, layers=6, heads=8, drop=0.1, pointer=False, extra=False, vid=False, enc="tf"):
        super().__init__()
        self.extra, self.vid = extra, vid
        self.in_dim = FEAT_DIM + (EXTRA_DIM if extra else 0) + (VID_DIM if vid else 0)
        self.inp = nn.Sequential(nn.Linear(2 * self.in_dim, d), nn.GELU(), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.zeros(1, MAX_T, d))
        self.enc_type = enc
        if enc == "tf":
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers)
        elif enc == "gru":
            self.rnn = nn.GRU(d, d // 2, layers, batch_first=True, bidirectional=True, dropout=drop)
        elif enc == "tcn":
            self.tcn = nn.ModuleList([TCNBlock(d, 2 ** (i % 4), drop) for i in range(layers)])
        elif enc == "gcn":  # skeleton graph network; `layers` GCN blocks, then 2 temporal blocks to fuse with frame features
            self.gcn = SkeletonGCN(d, blocks=layers, drop=drop)
            self.tcn = nn.ModuleList([TCNBlock(d, 2 ** i, drop) for i in range(2)])
        else:
            raise ValueError(enc)
        self.norm = nn.LayerNorm(d)
        self.hm = nn.Linear(d, 1)
        self.off = nn.Linear(d, 2)
        self.pts = nn.Linear(d, 4)
        self.pointer = pointer
        if pointer:
            from .data import JOINTS
            K = len(POINTER_JOINTS)
            self.register_buffer("pj", torch.tensor([JOINTS.index(n) for n in POINTER_JOINTS]), persistent=False)
            self.q = nn.Linear(d, 2 * d)          # per-view query from the predicting frame
            self.k = nn.Linear(d, 2 * d)          # per-view key for every frame
            self.jl = nn.Linear(d, 2 * K)         # per-view, per-joint logit bias for every frame
            self.free = nn.Linear(d, 2)           # logit of the "free regression" candidate
            self.res = nn.Linear(d, 4)            # small residual offset on the pooled point
            nn.init.zeros_(self.res.weight)
            nn.init.zeros_(self.res.bias)
        nn.init.constant_(self.hm.bias, -3.0)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.register_buffer("mu", torch.zeros(self.in_dim))
        self.register_buffer("sd", torch.ones(self.in_dim))

    def _pointer_pts(self, x, h, pad):
        """Attention over (frame, arm joint) candidates per view -> point [B, T, 2, 2]."""
        B, T, d = h.shape
        K = len(POINTER_JOINTS)
        direct = torch.sigmoid(self.pts(h)).view(B, T, 2, 2)
        q = self.q(h).view(B, T, 2, d)
        k = self.k(h).view(B, T, 2, d)
        att = torch.einsum("bivd,bjvd->bvij", q, k) / d ** 0.5            # [B, 2, Ti, Tj]
        jl = self.jl(h).view(B, T, 2, K).permute(0, 2, 1, 3)               # [B, 2, Tj, K]
        kp = torch.stack([x[..., v * VIEW_DIM : v * VIEW_DIM + J * 4].view(B, T, J, 4)[:, :, self.pj] for v in range(2)], 1)  # [B,2,T,K,4]
        valid = (kp[..., 3] > 0) & ~pad[:, None, :, None]                   # [B, 2, Tj, K]
        logits = att[..., None] + jl[:, :, None]                            # [B, 2, Ti, Tj, K]
        logits = logits.masked_fill(~valid[:, :, None], float("-inf")).flatten(3)  # [B, 2, Ti, Tj*K]
        free = self.free(h).permute(0, 2, 1)[..., None]                     # [B, 2, Ti, 1]
        w = torch.softmax(torch.cat([logits, free], -1).float(), -1)
        cand = kp[..., :2].flatten(2, 3)                                   # [B, 2, Tj*K, 2]
        pooled = torch.einsum("bvin,bvnc->bvic", w[..., :-1], cand.float())
        pooled = pooled + w[..., -1:] * direct.permute(0, 2, 1, 3).float()
        pooled = pooled.permute(0, 2, 1, 3) + 0.1 * torch.tanh(self.res(h).view(B, T, 2, 2).float())
        return pooled.clamp(0, 1)

    def _encode(self, h, pad):
        if self.enc_type == "tf":
            return self.enc(h, src_key_padding_mask=pad)
        if self.enc_type in ("tcn", "gcn"):
            for blk in self.tcn:
                h = blk(h, pad)
            return h
        lengths = (~pad).sum(1).clamp(min=1).cpu()  # BiGRU: pack so the backward pass skips padding
        with torch.autocast("cuda", enabled=False):
            packed = nn.utils.rnn.pack_padded_sequence(h.float(), lengths, batch_first=True, enforce_sorted=False)
            out, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=h.shape[1])
        return out.to(h.dtype)

    def forward(self, x, pad):
        """x: [B, T, in_dim] raw features (see data.sample_x); pad: [B, T] True where padded."""
        xn = (x - self.mu) / self.sd
        dx = torch.zeros_like(xn)
        dx[:, 1:] = xn[:, 1:] - xn[:, :-1]
        h = self.inp(torch.cat([xn, dx], -1)) + self.pos[:, : x.shape[1]]
        if self.enc_type == "gcn":
            h = h + self.gcn(x, pad)
        h = self.norm(self._encode(h, pad))
        pts = self._pointer_pts(x, h, pad) if self.pointer else torch.sigmoid(self.pts(h)).view(*h.shape[:2], 2, 2)
        return self.hm(h).squeeze(-1), self.off(h) * (1 / OFF_SCALE), pts


def flip_x(x, pts):
    """Mirror both views horizontally (x -> 1 - x, swap left/right joints). Works on numpy arrays."""
    x = x.copy()
    for v in range(2):
        o = v * VIEW_DIM
        kp = x[:, o : o + J * 4].reshape(-1, J, 4)
        kp = kp[:, FLIP_PERM]
        kp[..., 0] = np.where(kp[..., 3] > 0, 1 - kp[..., 0], 0)
        x[:, o : o + J * 4] = kp.reshape(-1, J * 4)
        f = o + J * 4
        x[:, f] = np.where(x[:, f + 2] > 0, 1 - x[:, f], 0)
    pts = pts.copy()
    pts[..., 0] = 1 - pts[..., 0]
    return x, pts
