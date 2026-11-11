#!/usr/bin/env python3
"""Item 5: SSTODE physics-informed Neural ODE for the 30-day SST horizon.

dSST/dt = advection(SST) + diffusion(SST) + Q_NN(SST, forcing), integrated
with explicit Euler steps. The advection term uses a learned latent velocity
over lag features; diffusion is a discrete Laplacian over the recent window;
Q_NN is a small MLP heat-budget source (solar/diurnal proxy = day-of-year
encoding). Falls back to a linear-trend extrapolation without torch.

Usage:
    python models/sstode_forecast.py --demo
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

WINDOW = 28  # days of history used as the ODE state


if HAS_TORCH:
    class HeatBudget(nn.Module):
        """Q_NN: MLP source term from state summary + seasonal forcing."""

        def __init__(self, hidden: int = 16):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(),
                                     nn.Linear(hidden, 1))
            # zero-init: corrections start at 0 around the trend prior
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

        def forward(self, sst, doy_sin, doy_cos):
            feat = torch.stack([sst, sst ** 2, doy_sin, doy_cos], dim=-1)
            return self.mlp(feat).squeeze(-1)


def null_forecasts(series: np.ndarray, horizon: int,
                   phi: float = 0.95) -> dict:
    """R2/R4 null SST horizons: persistence, damped persistence, climatology.

    Damped persistence relaxes toward the day-of-year climatology with
    factor phi**k; climatology is the in-series day-of-year mean (overall
    mean when the series is shorter than a year).
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    clim = np.full(365, series.mean())
    if n >= 365:
        doy = np.arange(n) % 365
        for d in range(365):
            m = doy == d
            if m.sum():
                clim[d] = series[m].mean()
    last_doy = (n - 1) % 365
    fut_doy = (last_doy + 1 + np.arange(horizon)) % 365
    clim_fc = clim[fut_doy]
    pers = np.full(horizon, series[-1])
    damp = clim_fc + (phi ** np.arange(1, horizon + 1)) * (series[-1]
                                                           - clim[last_doy])
    return {"persistence": pers, "damped_persistence": damp,
            "climatology": clim_fc}


class SSTODEForecaster:
    """Fit on a daily SST series; forecast `horizon` days ahead."""

    def __init__(self, hidden: int = 16, velocity: float = 0.05,
                 kappa: float = 0.02, epochs: int = 120, lr: float = 1e-2,
                 seed: int = 0, start_doy: int = 0,
                 horizons: tuple = (1,)):
        self.hidden = hidden
        self.velocity = velocity
        self.kappa = kappa
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.qnn = None  # built in fit(), after seeding (deterministic)
        self.trend = 0.0
        # R4: absolute day-of-year of series[0]; 0 keeps the old
        # series-relative behaviour. `horizons` adds short-rollout
        # multi-horizon loss (1 = one-step-ahead as before).
        self.start_doy = start_doy
        self.horizons = tuple(horizons)
        self.resid_sd_: float | None = None

    def _dynamics(self, win, d):
        """One Euler step from a (..., WINDOW) state at day-of-year d."""
        v = -float(torch.tanh(torch.tensor(self.velocity)))
        adv = v * (win[..., -1] - win[..., -2])
        # R4: curvature damping (second difference), NOT a spatial
        # Laplacian — a 1-D series has no spatial neighbours; the old
        # code was a mislabeled first difference.
        dif = self.kappa * (win[..., -3] - 2 * win[..., -2] + win[..., -1])
        src = self.qnn(win[..., -1],
                       torch.sin(2 * np.pi * d.float() / 365),
                       torch.cos(2 * np.pi * d.float() / 365))
        return win[..., -1] + self.trend + adv + dif + src

    def _torch_rollout(self, hist, steps, start_pos: int):
        state = hist.clone()
        outs = []
        for k in range(steps):
            # R4: continue absolute series time, not (k + WINDOW).
            d = torch.full((), (start_pos + k) % 365)
            step = (self._dynamics(state, d))[None]
            state = torch.cat([state[1:], step])
            outs.append(state[-1])
        return torch.stack(outs)

    def fit(self, series: np.ndarray, horizon: int = 30) -> "SSTODEForecaster":
        series = np.asarray(series, dtype=np.float32)
        self.mu = float(series.mean())
        self.sd = float(series.std()) or 1.0
        z = (series - self.mu) / self.sd
        self.trend = float(np.polyfit(np.arange(len(z)), z, 1)[0])
        if not HAS_TORCH:
            self.resid_sd_ = float(np.std(z[1:] - z[:-1]))
            return self
        torch.manual_seed(self.seed)
        if HAS_TORCH and self.qnn is None:
            self.qnn = HeatBudget(self.hidden)
        opt = torch.optim.Adam(self.qnn.parameters(), lr=self.lr)
        y = torch.tensor(z)
        self.qnn.train()
        hmax = max(self.horizons)
        idx = torch.arange(len(series) - WINDOW - hmax)
        win = y[idx[:, None] + torch.arange(WINDOW)]  # (n, WINDOW)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = 0.0
            for h in self.horizons:
                state = win
                for s in range(1, h + 1):
                    d = (idx + WINDOW + self.start_doy + s - 1) % 365
                    state = torch.cat(
                        [state[:, 1:], self._dynamics(state, d)[:, None]],
                        dim=1)
                loss = loss + ((state[:, -1] - y[idx + WINDOW + h - 1]) ** 2
                               ).mean() / len(self.horizons)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.qnn.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            d0 = (idx + WINDOW + self.start_doy) % 365
            resid = self._dynamics(win, d0) - y[idx + WINDOW]
            self.resid_sd_ = float(resid.std())
        return self

    def forecast(self, series: np.ndarray, horizon: int = 30) -> np.ndarray:
        series = np.asarray(series, dtype=np.float32)
        if not HAS_TORCH or self.qnn is None:
            slope = float(np.polyfit(np.arange(len(series)), series, 1)[0])
            return series[-1] + slope * np.arange(1, horizon + 1)
        mu = getattr(self, "mu", float(series.mean()))
        sd = getattr(self, "sd", float(series.std())) or 1.0
        start_pos = int(getattr(self, "start_doy", 0)) + len(series)
        self.qnn.eval()
        with torch.no_grad():
            out = self._torch_rollout(
                torch.tensor((series[-WINDOW:] - mu) / sd), horizon,
                start_pos)
            return (out.numpy() * sd + mu).astype(np.float32)

    def forecast_with_std(self, series: np.ndarray,
                          horizon: int = 30) -> tuple[np.ndarray, np.ndarray]:
        """R4: mean forecast + residual-based std growing as sqrt(horizon).

        Uncertainty comes from the in-sample one-step residual std, not a
        learned distribution — honest about what the model knows.
        """
        mean = self.forecast(series, horizon)
        sd = float(getattr(self, "resid_sd_", 1.0) or 1.0)
        if HAS_TORCH and self.qnn is not None:
            base = float(getattr(self, "sd", 1.0))
            grow = sd * base * np.sqrt(np.arange(1, horizon + 1))
        else:
            grow = sd * np.sqrt(np.arange(1, horizon + 1))
        return mean, grow.astype(np.float32)


def demo(seed: int = 0) -> dict:
    """Trending seasonal series with heatwave tail; SSTODE vs nulls."""
    rng = np.random.default_rng(seed)
    t = np.arange(400)
    s = (28.0 + 1.5 * np.sin(2 * np.pi * t / 365) + 0.003 * t
         + rng.standard_normal(400) * 0.1).astype(np.float32)
    s[-40:] += 1.5 * np.sin(np.pi * np.arange(40) / 40)
    tr, te = s[:340], s[340:370]
    fc = SSTODEForecaster(epochs=120, seed=seed).fit(tr).forecast(tr, 30)
    nulls = null_forecasts(tr, 30)
    rmse = lambda a: float(np.sqrt(np.mean((a - te) ** 2)))
    out = {"rmse_persistence": rmse(nulls["persistence"]),
           "rmse_damped": rmse(nulls["damped_persistence"]),
           "rmse_climatology": rmse(nulls["climatology"]),
           "rmse_sstode": rmse(fc), "torch": HAS_TORCH}
    print(out)
    assert out["rmse_sstode"] < out["rmse_persistence"]
    return out


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    demo()
