#!/usr/bin/env python3
"""Adaptive-threshold DHW + trend/stationarity features for year-to-year transfer.

Motivation: fixed-baseline heat metrics (``SSTA_DHW`` with NOAA tiers
``dhw_baseline_predict`` at DHW>=4/8, ``eval/gcbd_benchmark.py:43``) are
nonstationary under warming, and ``forward_temporal_scores``
(``eval/gcbd_benchmark.py:154``, TimeSeriesSplit past-trains/future-tests)
sits at ~0.49 while the stratified split reaches 0.70. This module
re-expresses heat relative to a *train-years-only* reference so the same
 LightGBM head sees stationary inputs when tested on future years.

Causality contract (never pooled): every reference constant is fit on
train-fold rows only via :func:`fit_adaptive_thresholds`; test rows are
transformed with :func:`apply_adaptive_thresholds` using those frozen
constants. No test year, test median, or full-frame statistic is ever
used. :func:`forward_temporal_delta` enforces this structurally by
fitting thresholds inside each TimeSeriesSplit fold on the train side.

GCBD has no per-site SST series (single-row CoRTAD covariates per
survey), so adaptation is year-level: a train-fit drift slope plus
train-fit DHW/SSTA quantiles and z-score constants.
"""

from __future__ import annotations

DHW_COL = "SSTA_DHW"
TSA_DHW_COL = "TSA_DHW"
SSTA_MAX_COL = "SSTA_Maximum"
SSTA_COL = "SSTA"
YEAR_COL = "Date_Year"

ADAPTIVE_COLS = [
    "adapt_dhw_excess_q50",
    "adapt_dhw_excess_q75",
    "adapt_dhw_ratio_q75",
    "adapt_dhw_ge_q75",
    "adapt_dhw_detrended",
    "adapt_year_offset",
    "adapt_year_x_dhw",
    "adapt_tsa_excess_q50",
    "adapt_ssta_z",
    "adapt_heat_max",
]


def _num(df, col: str):
    """Numeric series for col, NaN-filled with 0.0 (missing col -> zeros)."""
    import pandas as pd

    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)


def fit_adaptive_thresholds(train_df) -> dict:
    """Fit reference constants on TRAIN-years rows only.

    Returns a JSON-serializable dict with: train year range, per-year
    median-DHW drift slope (least squares of annual median SSTA_DHW vs
    year), quantiles of SSTA_DHW / TSA_DHW / SSTA_Maximum, and mean/std
    of SSTA for z-scoring. Callers must never pass test rows here.
    """
    import numpy as np

    dhw = _num(train_df, DHW_COL).to_numpy()
    tsa = _num(train_df, TSA_DHW_COL).to_numpy()
    smax = _num(train_df, SSTA_MAX_COL).to_numpy()
    ssta = _num(train_df, SSTA_COL).to_numpy()
    year = _num(train_df, YEAR_COL).to_numpy()

    def _q(a, q):
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return 0.0
        return float(np.quantile(a, q))

    # Drift slope: annual median DHW vs year (train years only).
    slope = 0.0
    ref_year = float(np.median(year)) if year.size else 0.0
    if year.size >= 4 and len(set(np.round(year, 1).tolist())) >= 2:
        yrs = np.round(year, 0)
        uy = np.unique(yrs)
        if uy.size >= 2:
            med = np.array([np.median(dhw[yrs == y]) for y in uy])
            A = np.vstack([uy, np.ones_like(uy)]).T
            slope = float(np.linalg.lstsq(A, med, rcond=None)[0][0])

    mu, sd = float(np.mean(ssta)), float(np.std(ssta))
    return {
        "n_train": int(len(train_df)),
        "train_year_min": float(np.min(year)) if year.size else 0.0,
        "train_year_max": float(np.max(year)) if year.size else 0.0,
        "ref_year": float(ref_year),
        "dhw_drift_per_year": float(slope),
        "dhw_q50": _q(dhw, 0.50),
        "dhw_q75": _q(dhw, 0.75),
        "tsa_q50": _q(tsa, 0.50),
        "smax_q50": _q(smax, 0.50),
        "ssta_mean": mu,
        "ssta_std": sd if sd > 1e-9 else 1.0,
    }


def apply_adaptive_thresholds(df, thresholds: dict):
    """Add the 10 ADAPTIVE_COLS to df using frozen train-fit constants."""
    import pandas as pd

    dhw = _num(df, DHW_COL)
    tsa = _num(df, TSA_DHW_COL)
    ssta = _num(df, SSTA_COL)
    year = _num(df, YEAR_COL)
    q50 = float(thresholds["dhw_q50"])
    q75 = float(thresholds["dhw_q75"])
    denom = q75 if abs(q75) > 1e-9 else 1.0
    slope = float(thresholds["dhw_drift_per_year"])
    ref = float(thresholds["ref_year"])
    tmax = float(thresholds["train_year_max"])
    tsa_q50 = float(thresholds["tsa_q50"])
    smu, ssd = float(thresholds["ssta_mean"]), float(thresholds["ssta_std"])

    out = pd.DataFrame(index=df.index)
    out["adapt_dhw_excess_q50"] = dhw - q50
    out["adapt_dhw_excess_q75"] = dhw - q75
    out["adapt_dhw_ratio_q75"] = dhw / denom
    out["adapt_dhw_ge_q75"] = (dhw >= q75).astype(float)
    out["adapt_dhw_detrended"] = dhw - slope * (year - ref)
    out["adapt_year_offset"] = year - tmax
    out["adapt_year_x_dhw"] = (year - tmax) * dhw
    out["adapt_tsa_excess_q50"] = tsa - tsa_q50
    out["adapt_ssta_z"] = (ssta - smu) / (abs(ssd) if abs(ssd) > 1e-9 else 1.0)
    out["adapt_heat_max"] = (
        (dhw - q50) / (abs(q75 - q50) if abs(q75 - q50) > 1e-9 else 1.0)
    ).clip(lower=0.0)
    return out[ADAPTIVE_COLS]


def add_adaptive_features_train_test(train_df, test_df):
    """Fit thresholds on train_df, transform both; returns (thr, A_tr, A_te)."""
    thr = fit_adaptive_thresholds(train_df)
    return thr, apply_adaptive_thresholds(train_df, thr), \
        apply_adaptive_thresholds(test_df, thr)


def _base_frame(df):
    """Same numeric feature frame as load_gcbd_frame (no leakage)."""
    import pandas as pd
    from eval.gcbd_benchmark import DROP_COLS

    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    return df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def forward_temporal_delta(df, seed: int = 0, n_estimators: int = 50) -> dict:
    """Forward-temporal CV: base arm vs base+adaptive arm, same splits.

    Rows are ordered by ``Date``; TimeSeriesSplit(3) past-trains /
    future-tests each arm identically. Adaptive thresholds are fit on
    each fold's train rows only. Folds whose train or test side is
    single-class are skipped (same rule as ``forward_temporal_scores``).
    """
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import TimeSeriesSplit

    from eval.gcbd_benchmark import map_labels

    work = df.reset_index(drop=True)
    order = np.argsort(pd.to_datetime(work["Date"], errors="coerce").to_numpy())
    work = work.iloc[order].reset_index(drop=True)
    y = map_labels(work["Percent_Bleaching"]).to_numpy()
    Xb = _base_frame(work)

    def _clf():
        return lgb.LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=n_estimators,
            learning_rate=0.05, num_leaves=15, min_child_samples=10,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=5.0,
            class_weight="balanced", verbose=-1, random_state=seed)

    base_accs, adapt_accs = [], []
    for tr, te in TimeSeriesSplit(n_splits=3).split(Xb):
        if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
            continue
        cb = _clf()
        cb.fit(Xb.to_numpy()[tr], y[tr])
        base_accs.append(float(accuracy_score(
            y[te], cb.predict(Xb.to_numpy()[te]))))
        thr = fit_adaptive_thresholds(work.iloc[tr])
        A_tr = apply_adaptive_thresholds(work.iloc[tr], thr).to_numpy()
        A_te = apply_adaptive_thresholds(work.iloc[te], thr).to_numpy()
        # Adaptive arm: base cols + per-fold adaptive cols, where the
        # adaptive block on each side comes from the train-fit transform
        # (no test pooling).
        Xtr = np.hstack([Xb.to_numpy()[tr], A_tr])
        Xte = np.hstack([Xb.to_numpy()[te], A_te])
        ca = _clf()
        ca.fit(Xtr, y[tr])
        adapt_accs.append(float(accuracy_score(y[te], ca.predict(Xte))))
    base_m = float(np.mean(base_accs)) if base_accs else float("nan")
    adapt_m = float(np.mean(adapt_accs)) if adapt_accs else float("nan")
    return {"seed": int(seed), "n_estimators": int(n_estimators),
            "base_per_fold": [round(a, 4) for a in base_accs],
            "adapt_per_fold": [round(a, 4) for a in adapt_accs],
            "base_mean": round(base_m, 4) if base_accs else None,
            "adapt_mean": round(adapt_m, 4) if adapt_accs else None,
            "delta_mean": round(adapt_m - base_m, 4)
            if base_accs and adapt_accs else None,
            "n_folds": int(len(base_accs))}
