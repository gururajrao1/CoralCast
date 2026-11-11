#!/usr/bin/env python3
"""Item 2: Learned SST downscaling (Residual Corrective Neural Network).

Maps coarse daily SST to reef-pixel resolution. A conv net produces an
initial high-res estimate from upsampled coarse SST + static channels
(depth proxy, reef mask); K refinement steps then predict
dynamically-scaled residuals. The loss up-weights extreme heat anomalies so
marine-heatwave fine structure survives. Falls back to bilinear upsampling
when torch is unavailable.

Usage:
    python models/sst_downscale_rcnn.py --demo
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # graceful fallback path
    HAS_TORCH = False


def bilinear_upsample(x: np.ndarray, factor: int) -> np.ndarray:
    """Bilinear upsampling fallback (also the baseline to beat)."""
    from scipy.ndimage import zoom
    return zoom(x, factor, order=1).astype(np.float32)


if HAS_TORCH:
    class RCNN(nn.Module):
        """U-Net-lite initial estimate + K residual refinement steps."""

        def __init__(self, in_ch: int = 3, hidden: int = 16, steps: int = 3):
            super().__init__()
            self.steps = steps
            self.init = nn.Sequential(
                nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(),
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
                nn.Conv2d(hidden, 1, 3, padding=1))
            self.refine = nn.Sequential(
                nn.Conv2d(2, hidden, 3, padding=1), nn.ReLU(),
                nn.Conv2d(hidden, 1, 3, padding=1))
            self.gate = nn.Sequential(
                nn.Conv2d(2, hidden, 3, padding=1), nn.ReLU(),
                nn.Conv2d(hidden, 1, 3, padding=1), nn.Sigmoid())

        def forward(self, x):
            est = x[:, :1] + self.init(x)  # residual around bilinear input
            for _ in range(self.steps):
                feat = torch.cat([est, x[:, :1]], dim=1)
                est = est + self.gate(feat) * self.refine(feat)
            return est

    def extremes_loss(pred, target, q: float = 0.9, w: float = 2.0):
        """MSE with extra weight `w` on top-(1-q) heat anomalies (R3: ablatable)."""
        thresh = torch.quantile(target.detach(), q)
        wgt = torch.where(target >= thresh, w, 1.0)
        return ((pred - target) ** 2 * wgt).mean()


class RCNNDownscaler:
    """Coarse (Hc, Wc) SST -> fine (Hc*f, Wc*f) SST, factor f."""

    def __init__(self, factor: int = 4, hidden: int = 16, steps: int = 3,
                 epochs: int = 120, lr: float = 1e-3, seed: int = 0,
                 val_fraction: float = 0.2, weight_extreme: float = 2.0,
                 extreme_quantile: float = 0.9, patience: int = 15):
        self.factor = factor
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.hidden = hidden
        self.steps = steps
        self.net = None  # built in fit(), after seeding (deterministic)
        # R3 ablations: hold out patch fraction for early stopping; the
        # extremes up-weight and its quantile are constructor args now.
        self.val_fraction = val_fraction
        self.weight_extreme = weight_extreme
        self.extreme_quantile = extreme_quantile
        self.patience = patience
        self.val_rmse_: float | None = None

    @staticmethod
    def pack(coarse: np.ndarray, static: np.ndarray,
             factor: int) -> np.ndarray:
        up = bilinear_upsample(coarse, factor)
        return np.concatenate([up[None], np.asarray(static)], axis=0)

    @staticmethod
    def patches(coarse: np.ndarray, fine: np.ndarray,
                static: np.ndarray, size: int = 32,
                stride: int = 16, factor: int = 4) -> list:
        """R3: overlapping (coarse-patch, fine-patch, static-patch) triples.

        Trains on patches — not one image — so the net must generalize
        across the scene instead of memorizing it.
        """
        h, w = np.asarray(fine).shape
        out = []
        for i in range(0, max(h - size + 1, 1), stride):
            for j in range(0, max(w - size + 1, 1), stride):
                fp = np.asarray(fine)[i:i + size, j:j + size]
                ci0, ci1 = i // factor, (i + size) // factor
                cj0, cj1 = j // factor, (j + size) // factor
                cp = np.asarray(coarse)[ci0:ci1, cj0:cj1]
                sp = np.asarray(static)[:, i:i + size, j:j + size]
                if (fp.shape[0] == size and fp.shape[1] == size
                        and cp.size and sp.shape[1] == size
                        and sp.shape[2] == size):
                    out.append((cp, fp, sp))
        return out or [(coarse, fine, static)]

    def fit(self, coarse: np.ndarray, fine: np.ndarray,
            static: np.ndarray) -> "RCNNDownscaler":
        if not HAS_TORCH:
            return self  # fallback predicts bilinear; nothing to fit
        import copy
        torch.manual_seed(self.seed)
        if self.net is None:
            self.net = RCNN(1 + np.asarray(static).shape[0],
                            self.hidden, self.steps)
        rng = np.random.default_rng(self.seed)
        items = self.patches(coarse, fine, static, factor=self.factor)
        idx = rng.permutation(len(items))
        n_val = max(1, int(len(items) * self.val_fraction)) \
            if len(items) > 1 else 0
        val_idx, tr_idx = set(idx[:n_val].tolist()), idx[n_val:]

        def to_tensors(sub):
            xs = torch.stack([torch.tensor(
                self.pack(c, s, self.factor),
                dtype=torch.float32) for c, _, s in sub])
            ys = torch.stack([torch.tensor(
                np.asarray(f, dtype=np.float32)[None]) for _, f, _ in sub])
            return xs, ys

        tr = [items[i] for i in tr_idx] or items
        va = [items[i] for i in sorted(val_idx)]
        xtr, ytr = to_tensors(tr)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        best, best_state, bad = float("inf"), None, 0
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = extremes_loss(self.net(xtr), ytr, q=self.extreme_quantile,
                                 w=self.weight_extreme)
            loss.backward()
            opt.step()
            if va:
                self.net.eval()
                with torch.no_grad():
                    xv, yv = to_tensors(va)
                    v = float(((self.net(xv) - yv) ** 2).mean().sqrt())
                self.net.train()
                if v < best - 1e-6:
                    best, best_state, bad = v, copy.deepcopy(
                        self.net.state_dict()), 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.val_rmse_ = best
        return self

    def predict(self, coarse: np.ndarray, static: np.ndarray) -> np.ndarray:
        if not HAS_TORCH:
            return bilinear_upsample(coarse, self.factor)
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(self.pack(coarse, static, self.factor)[None],
                             dtype=torch.float32)
            return self.net(x).numpy()[0, 0]


def demo(seed: int = 0) -> dict:
    """Coarsen a synthetic SST field with static-correlated fine structure.

    R3: the fine field carries a 4 px oscillation gated by a static reef
    channel that 4x averaging destroys, so bilinear interpolation cannot
    recover it but a net that uses the static channels can. A smooth-only
    toy would be won by bilinear by construction and prove nothing.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:64, 0:64]
    reef = ((xx % 8) < 4).astype(np.float32)
    fine = (27.0 + 1.5 * np.sin(2 * np.pi * xx / 64)
            + 2.2 * np.exp(-((xx - 44) ** 2 + (yy - 20) ** 2) / 60)
            + 0.6 * np.sin(2 * np.pi * xx / 4) * reef
            + rng.standard_normal((64, 64)) * 0.02).astype(np.float32)
    coarse = fine.reshape(16, 4, 16, 4).mean(axis=(1, 3))
    depth = (xx / 64).astype(np.float32)
    static = np.stack([depth, reef])
    model = RCNNDownscaler(epochs=400, patience=30).fit(coarse, fine, static)
    pred = model.predict(coarse, static)
    base = bilinear_upsample(coarse, 4)
    rmse = lambda a: float(np.sqrt(np.mean((a - fine) ** 2)))
    out = {"rmse_bilinear": rmse(base), "rmse_rcnn": rmse(pred),
           "rmse_val": model.val_rmse_}
    print(out)
    assert out["rmse_rcnn"] < out["rmse_bilinear"], "RCNN must beat bilinear"
    return out


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    demo()
