#!/usr/bin/env python3
"""Item 3: SITS temporal encoder for bleaching-risk classification.

Classifies a Sentinel-2 image time series (n, time, bands) into the 3
bleaching-risk classes. Backbone resolution order:
  1. mamba_ssm selective SSM (if installed: pip install mamba-ssm)
  2. torch GRU / Transformer encoder (default; verified on CPU)
  3. sklearn HistGradientBoosting on the flattened series (no torch)
NaN timesteps (cloud gaps) are masked, not interpolated.

Usage:
    python models/sits_mamba.py --demo
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from mamba_ssm import Mamba  # optional; heavy dep, not required
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


if HAS_TORCH:
    class TemporalNet(nn.Module):
        def __init__(self, n_feat: int, hidden: int = 32,
                     backbone: str = "gru", n_class: int = 3):
            super().__init__()
            if backbone == "mamba" and HAS_MAMBA:
                self.enc = Mamba(d_model=n_feat, d_state=16,
                                 d_conv=4, expand=2)
                h = n_feat
            elif backbone == "transformer":
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden, nhead=4, dim_feedforward=64,
                    batch_first=True)
                self.proj = nn.Linear(n_feat, hidden)
                self.enc = nn.TransformerEncoder(layer, num_layers=2)
                h = hidden
            else:
                if backbone == "mamba":
                    warnings.warn("mamba_ssm missing; using GRU backbone")
                self.enc = nn.GRU(n_feat, hidden, batch_first=True)
                h = hidden
                self.proj = None
            self.backbone = backbone
            self.h = h
            self.head = nn.Linear(h, n_class)
            # SITSMamba-style dual branch: reconstruct masked timesteps
            self.decoder = nn.Linear(h, n_feat)

        def encode(self, x, mask):
            if self.backbone == "transformer":
                x = self.proj(x)
            h = self.enc(x)
            h = h[0] if isinstance(h, tuple) else h
            return h * mask.unsqueeze(-1)

        def forward(self, x, mask):
            h = self.encode(x, mask)
            pooled = h.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            return self.head(pooled), self.decoder(h)


class SITSClassifier:
    """(n, time, bands) -> 3-class risk; handles NaN (cloud) timesteps."""

    def __init__(self, backbone: str = "gru", hidden: int = 32,
                 epochs: int = 60, lr: float = 3e-3, seed: int = 0,
                 aux_weight: float = 0.1, patience: int = 30,
                 val_fraction: float = 0.2):
        self.backbone = backbone
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.hidden = hidden
        self.aux_weight = aux_weight
        # R7: early stopping on a held-out split instead of fixed epochs.
        self.patience = patience
        self.val_fraction = val_fraction
        self.net = None
        self.fallback = None
        self.recon_mse_: float | None = None
        self.val_acc_: float | None = None

    @staticmethod
    def _mask(X: np.ndarray) -> np.ndarray:
        return (~np.isnan(X).any(axis=2)).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SITSClassifier":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        mask = self._mask(X)
        Xf = np.nan_to_num(X, nan=0.0)
        if not HAS_TORCH:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.fallback = HistGradientBoostingClassifier(
                max_iter=200, random_state=self.seed)
            self.fallback.fit(Xf.reshape(len(Xf), -1), y)
            return self
        import copy
        torch.manual_seed(self.seed)
        self.net = TemporalNet(X.shape[2], self.hidden, self.backbone)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        mse_fn = nn.MSELoss()
        Xt = torch.tensor(Xf)
        mt = torch.tensor(mask)
        yt = torch.tensor(y)
        wrng = torch.Generator().manual_seed(self.seed)
        # R7: internal train/val split (seeded) drives early stopping.
        n = len(Xf)
        perm = np.random.default_rng(self.seed).permutation(n)
        n_val = max(10, int(n * self.val_fraction))
        va, tr = perm[:n_val], perm[n_val:]
        Xt_tr, mt_tr, yt_tr = Xt[tr], mt[tr], yt[tr]
        Xt_va, mt_va, yt_va = Xt[va], mt[va], yt[va]
        self.net.train()
        if self.aux_weight > 0:
            # Stage A (SITSMamba-style): masked-reconstruction pretraining.
            # Corrupt 40% of valid steps, reconstruct originals, no labels.
            for _ in range(80):
                opt.zero_grad()
                drop = ((torch.rand(mt_tr.shape, generator=wrng) < 0.40)
                        & (mt_tr > 0))
                if not drop.any():
                    continue
                Xc = Xt_tr * (~drop).unsqueeze(-1).float()
                _, recon = self.net(Xc, mt_tr)
                loss = mse_fn(recon[drop], Xt_tr[drop])
                loss.backward()
                opt.step()
            # R7: recon metric on FRESH masked positions only (seeded), not
            # on all valid steps — the old denominator included visible
            # steps and flattered the "beats the mean" claim.
            with torch.no_grad():
                drop = ((torch.rand(mt_tr.shape,
                                    generator=torch.Generator()
                                    .manual_seed(self.seed + 999)) < 0.40)
                        & (mt_tr > 0))
                if drop.any():
                    Xc = Xt_tr * (~drop).unsqueeze(-1).float()
                    _, recon = self.net(Xc, mt_tr)
                    self.recon_mse_ = float(mse_fn(recon[drop],
                                                   Xt_tr[drop]))
        # Stage B: supervised classification with val early stopping.
        best, best_state, bad = float("inf"), None, 0
        for _ in range(self.epochs):
            opt.zero_grad()
            logits, _ = self.net(Xt_tr, mt_tr)
            loss = loss_fn(logits, yt_tr)
            loss.backward()
            opt.step()
            self.net.eval()
            with torch.no_grad():
                vl, _ = self.net(Xt_va, mt_va)
                v = float(loss_fn(vl, yt_va))
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
        self.net.eval()
        with torch.no_grad():
            vl, _ = self.net(Xt_va, mt_va)
            self.val_acc_ = float((vl.argmax(1) == yt_va).float().mean())
        self.net.train()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Xf = np.nan_to_num(X, nan=0.0)
        if self.fallback is not None:
            return self.fallback.predict_proba(Xf.reshape(len(Xf), -1))
        self.net.eval()
        with torch.no_grad():
            out, _ = self.net(torch.tensor(Xf),
                              torch.tensor(self._mask(X)))
            return torch.softmax(out, dim=1).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def demo(seed: int = 0) -> dict:
    """Synthetic SITS: bleaching signal = late-season NIR drop + DHW ramp."""
    rng = np.random.default_rng(seed)
    n, t, f = 300, 12, 4
    X = rng.standard_normal((n, t, f)).astype(np.float32) * 0.1 + 0.3
    # smooth shared seasonal structure per band (as in real SITS)
    tt = np.arange(t, dtype=np.float32)
    for j in range(f):
        X[:, :, j] += 0.15 * np.sin(2 * np.pi * tt / t + j * np.pi / 2)
    y = rng.integers(0, 3, n)
    for i in range(n):
        X[i, 6:, 3] -= 0.05 * y[i]          # class-graded NIR drop
        X[i, :, 0] += 0.01 * y[i] * np.arange(t)
    X[rng.random((n, t)) < 0.1] = np.nan     # cloud gaps
    tr = slice(0, 200)
    clf = SITSClassifier(epochs=60, seed=seed).fit(X[tr], y[tr])
    acc = float((clf.predict(X[200:]) == y[200:]).mean())
    out = {"accuracy": acc, "torch": HAS_TORCH, "mamba": HAS_MAMBA,
           "recon_mse": clf.recon_mse_, "val_acc": clf.val_acc_}
    print(out)
    assert acc > 0.5, "temporal encoder must beat chance (0.33)"
    if HAS_TORCH:
        valid = X[tr][~np.isnan(X[tr])]
        assert clf.recon_mse_ < float(valid.var()), \
            "dual branch must reconstruct better than the mean"
    return out


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    demo()
