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

    def extremes_loss(pred, target, q: float = 0.9):
        """MSE with mild extra weight on top-(1-q) heat anomalies."""
        thresh = torch.quantile(target.detach(), q)
        w = torch.where(target >= thresh, 2.0, 1.0)
        return ((pred - target) ** 2 * w).mean()


class RCNNDownscaler:
    """Coarse (Hc, Wc) SST -> fine (Hc*f, Wc*f) SST, factor f."""

    def __init__(self, factor: int = 4, hidden: int = 16, steps: int = 3,
                 epochs: int = 120, lr: float = 1e-3, seed: int = 0):
        self.factor = factor
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.hidden = hidden
        self.steps = steps
        self.net = None  # built in fit(), after seeding (deterministic)

    @staticmethod
    def pack(coarse: np.ndarray, static: np.ndarray,
             factor: int) -> np.ndarray:
        up = bilinear_upsample(coarse, factor)
        return np.concatenate([up[None], np.asarray(static)], axis=0)

    def fit(self, coarse: np.ndarray, fine: np.ndarray,
            static: np.ndarray) -> "RCNNDownscaler":
        if not HAS_TORCH:
            return self  # fallback predicts bilinear; nothing to fit
        torch.manual_seed(self.seed)
        if self.net is None:
            self.net = RCNN(3, self.hidden, self.steps)
        x = torch.tensor(self.pack(coarse, static, self.factor)[None])
        y = torch.tensor(np.asarray(fine, dtype=np.float32)[None, None])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = extremes_loss(self.net(x), y)
            loss.backward()
            opt.step()
        return self

    def predict(self, coarse: np.ndarray, static: np.ndarray) -> np.ndarray:
        if not HAS_TORCH:
            return bilinear_upsample(coarse, self.factor)
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(self.pack(coarse, static, self.factor)[None])
            return self.net(x).numpy()[0, 0]


def demo(seed: int = 0) -> dict:
    """Coarsen a synthetic SST field with a fine heatwave anomaly; downscale."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:64, 0:64]
    fine = (27.0 + 1.5 * np.sin(2 * np.pi * xx / 64)
            + 2.2 * np.exp(-((xx - 44) ** 2 + (yy - 20) ** 2) / 60)
            + rng.standard_normal((64, 64)) * 0.05).astype(np.float32)
    coarse = fine.reshape(16, 4, 16, 4).mean(axis=(1, 3))
    depth = (xx / 64).astype(np.float32)
    mask = np.ones_like(depth)
    model = RCNNDownscaler().fit(coarse, fine, np.stack([depth, mask]))
    pred = model.predict(coarse, np.stack([depth, mask]))
    base = bilinear_upsample(coarse, 4)
    rmse = lambda a: float(np.sqrt(np.mean((a - fine) ** 2)))
    out = {"rmse_bilinear": rmse(base), "rmse_rcnn": rmse(pred)}
    print(out)
    assert out["rmse_rcnn"] < out["rmse_bilinear"], "RCNN must beat bilinear"
    return out


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    demo()
