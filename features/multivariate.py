#!/usr/bin/env python3
"""Item 6: Multivariate feature set + SHAP-gated selection.

Implements the vetted predictor list: accumulated + instantaneous heat
stress, SST anomaly/variance/frequency/rate-of-change, seasonal phase,
ENSO-like 12-month anomaly, optional met/ocean columns (air temp, winds,
solar, currents) passed through when present, plus geography/depth and
optical aggregates. prune_features() keeps top_k via SHAP when installed,
else LightGBM gain.

Usage:
    python features/multivariate.py --demo
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

MET_COLS = ["air_temp", "wind_speed", "solar_rad", "current_speed",
            "iod_index", "nino34"]


def build_multivariate_features(sst: pd.DataFrame, mmm: float,
                               lat: float = -16.0, depth_m: float = 5.0,
                               optical: dict | None = None) -> pd.DataFrame:
    """One row per day of the SST series; NaN-padded for windowed stats."""
    s = sst.sort_values("date").reset_index(drop=True)
    v = pd.to_numeric(s["sst_c"], errors="coerce").to_numpy(dtype=float)
    anom = v - mmm
    hot = np.clip(anom - 1.0, 0, None)          # HotSpot vs MMM+1
    # R5: windows require >=50% coverage (NaN until then) instead of
    # min_periods=1 zeros that biased early-series rows downward.
    feats: dict[str, np.ndarray] = {
        "sst": v,
        "ssta": anom,
        "hotspot": np.clip(anom, 0, None),
        "hotspot_inst": hot,
        "ssta_4w_mean": pd.Series(anom).rolling(28, min_periods=14).mean()
        .to_numpy(),
        "ssta_12w_var": pd.Series(anom).rolling(84, min_periods=42).var()
        .to_numpy(),
        "freq_above_mmm1_12w": pd.Series((v >= mmm + 1.0).astype(float))
        .rolling(84, min_periods=42).mean().to_numpy(),
        "sst_rate_4w": pd.Series(v).diff(28).to_numpy() / 28.0,
        "enso12": pd.Series(anom).rolling(365, min_periods=183).mean()
        .to_numpy(),
        "doy_sin": np.sin(2 * np.pi * np.arange(len(v)) / 365.25),
        "doy_cos": np.cos(2 * np.pi * np.arange(len(v)) / 365.25),
        "lat": np.full(len(v), lat),
        "abs_lat": np.full(len(v), abs(lat)),
        "depth_m": np.full(len(v), depth_m),
    }
    for c in MET_COLS:
        if c in s.columns:
            feats[c] = pd.to_numeric(s[c], errors="coerce") \
                .fillna(0).to_numpy(dtype=float)
    if optical:
        for k, arr in optical.items():
            a = np.asarray(arr, dtype=float)
            feats[f"opt_{k}_mean"] = np.full(len(v), float(np.nanmean(a)))
            feats[f"opt_{k}_p10"] = np.full(len(v), float(np.nanpercentile(a, 10)))
    return pd.DataFrame(feats, index=s["date"])


def prune_features(X: pd.DataFrame, y: np.ndarray, top_k: int = 12,
                   seed: int = 0) -> list[str]:
    """Rank columns by SHAP (if installed) else LightGBM gain; top_k names.

    R5: rows whose windowed features are still NaN (short series start)
    are dropped instead of zero-filled, so imputation cannot invent
    calm early windows.
    """
    import lightgbm as lgb
    keep_rows = X.notna().all(axis=1).to_numpy()
    if keep_rows.sum() < 10:  # degenerate: fall back to zero-fill
        X = X.fillna(0)
    else:
        X = X.loc[keep_rows].reset_index(drop=True)
        y = np.asarray(y)[keep_rows]
    try:
        import shap  # optional
        model = lgb.LGBMClassifier(n_estimators=100, verbose=-1,
                                   random_state=seed).fit(X, y)
        sv = shap.TreeExplainer(model).shap_values(X)
        imp = np.abs(np.asarray(sv)).mean(axis=(0, 2)) \
            if np.asarray(sv).ndim == 3 else np.abs(sv).mean(axis=0)
        rank = np.argsort(imp)[::-1]
    except ImportError:
        model = lgb.LGBMClassifier(n_estimators=100, verbose=-1,
                                   random_state=seed).fit(X, y)
        gain = model.booster_.feature_importance(importance_type="gain")
        rank = np.argsort(gain)[::-1]
    return list(X.columns[rank[:top_k]])


def demo(seed: int = 0) -> dict:
    """Synthetic series + optical; pruned set must contain the signal."""
    rng = np.random.default_rng(seed)
    n = 500
    t = np.arange(n)
    sst = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n),
        "sst_c": 28.5 + 1.5 * np.sin(2 * np.pi * t / 365)
        + rng.standard_normal(n) * 0.2,
        "wind_speed": rng.standard_normal(n) + 5,
    })
    X = build_multivariate_features(
        sst, 28.5, optical={"ndvi": rng.standard_normal((20, 20))})
    y = (X["freq_above_mmm1_12w"] > 0.05).astype(int).to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        y[::7] = 1
    kept = prune_features(X, y, top_k=8, seed=seed)
    out = {"n_features": int(X.shape[1]), "kept": kept}
    print(out)
    assert "freq_above_mmm1_12w" in kept
    return out


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    demo()
