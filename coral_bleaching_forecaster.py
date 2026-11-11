#!/usr/bin/env python3
"""Project 6.3: Multispectral Coral Bleaching Predictor.

End-to-end framework that ingests Sentinel-2 multispectral imagery
(Bands B2/blue, B3/green, B4/red, B8/NIR) plus a daily SST time series,
engineers coral-health indices and NOAA Degree Heating Weeks (DHW),
trains a LightGBM 3-class bleaching-risk classifier, and exports a
georeferenced 30-day forecast risk map (GeoTIFF) + evaluation metrics (JSON).

Usage (synthetic demo, no inputs needed — fully runnable out of the box):
    python coral_bleaching_forecaster.py

Usage with real data:
    python coral_bleaching_forecaster.py --blue B2.tif --green B3.tif \\
        --red B4.tif --nir B8.tif --sst_csv sst_noaa_daily.csv \\
        --mmm 28.5 --horizon_days 30

Outputs (written to the current directory):
    coral_bleaching_risk_forecast.tif
    bleaching_evaluation_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RISK_LABELS = {0: "No Stress", 1: "Bleaching Watch/Warning",
               2: "Severe Bleaching Expected"}
N_CLASSES = 3
DHW_WINDOW_DAYS = 84          # 12-week historical window
FORECAST_HORIZON_DAYS = 30
DEFAULT_MMM = 28.5            # degC Maximum Monthly Mean baseline (NOAA)
S2_SCALE = 10000.0            # Sentinel-2 L2A reflectance scale factor
SYNTH_SHAPE = (200, 200)      # demo raster rows, cols
SYNTH_RES_M = 10.0            # Sentinel-2 10 m visible/NIR resolution
# Synthetic reef site: northern Great Barrier Reef (UTM 55S)
SYNTH_CRS = "EPSG:32755"
SYNTH_ORIGIN_X, SYNTH_ORIGIN_Y = 400000.0, 8350000.0
RNG_SEED = 63


# ---------------------------------------------------------------------------
# 0. Synthetic demo data (used when no real SAFE/GeoTIFF inputs are given)
# ---------------------------------------------------------------------------
def _smooth_noise(rng: np.random.Generator, shape, scale: float = 8.0) -> np.ndarray:
    """Low-frequency spatial field in [0, 1] via smoothed white noise."""
    from scipy.ndimage import gaussian_filter
    field = gaussian_filter(rng.standard_normal(shape), sigma=scale)
    field -= field.min()
    field /= field.max() + 1e-12
    return field


def generate_synthetic_bands(shape=SYNTH_SHAPE, seed: int = RNG_SEED) -> dict[str, np.ndarray]:
    """Create plausible Sentinel-2-like surface reflectances (0..1 floats).

    Scene layout: land strip on the west edge, shallow reef flat, deeper
    lagoon to the east, with patchy coral/algae texture in the reef zone.
    Keys: blue (B2), green (B3), red (B4), nir (B8).
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    eastward = xx / cols                                   # 0 west -> 1 east
    depth = np.clip(1.5 + 9.0 * eastward
                    + 2.0 * _smooth_noise(rng, shape), 0, 15)   # metres
    land = xx < 12                                         # western land strip
    reef_zone = (~land) & (depth < 5.0)
    patch = _smooth_noise(rng, shape, scale=4.0)           # coral/algae patchiness

    # Base water-leaving reflectance spectra (healthy coral / sand / algae mix)
    blue = 0.10 + 0.06 * np.exp(-depth / 6.0) + 0.02 * patch
    green = 0.09 + 0.07 * np.exp(-depth / 5.0) + 0.03 * patch
    red = 0.05 + 0.05 * np.exp(-depth / 3.0) + 0.02 * patch
    nir = 0.020 + 0.05 * np.exp(-depth / 1.5) * (0.4 + 0.6 * patch)
    for band in (blue, green, red, nir):
        band += rng.standard_normal(shape) * 0.004
    # Land: bright soil/vegetation spectrum
    blue[land] = 0.08 + rng.standard_normal(land.sum()) * 0.01
    green[land] = 0.12 + rng.standard_normal(land.sum()) * 0.01
    red[land] = 0.14 + rng.standard_normal(land.sum()) * 0.01
    nir[land] = 0.35 + rng.standard_normal(land.sum()) * 0.02
    out = {"blue": blue, "green": green, "red": red, "nir": nir}
    return {k: np.clip(v, 0.001, 1.0).astype(np.float32) for k, v in out.items()}


def generate_synthetic_sst(days: int = 730, mmm: float = DEFAULT_MMM,
                           seed: int = RNG_SEED) -> "pd.DataFrame":
    """Two years of daily SST: seasonal cycle + warming trend + noise + event.

    A marine heatwave is embedded in the final ~60 days so the demo forecast
    contains all three risk classes.
    """
    import pandas as pd
    rng = np.random.default_rng(seed + 1)
    t = np.arange(days)
    seasonal = 1.6 * np.sin(2 * np.pi * (t - 60) / 365.25)
    trend = 0.002 * t                                             # demo warming drift
    sst = (mmm - 1.2) + seasonal + trend + rng.standard_normal(days) * 0.15
    sst[-60:] += 2.8 * np.sin(np.pi * np.arange(60) / 60) ** 0.5  # heatwave
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    return pd.DataFrame({"date": dates, "sst_c": np.round(sst, 3)})


# ---------------------------------------------------------------------------
# 1. Satellite Raster Processing Subsystem + coral health indices
# ---------------------------------------------------------------------------
def load_band(path: str | None, synthetic: np.ndarray | None) -> np.ndarray:
    """Load one band via Rasterio, or fall back to the synthetic array."""
    if path is not None:
        import rasterio
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float64)
    assert synthetic is not None
    return np.asarray(synthetic, dtype=np.float64)


def to_reflectance(dn: np.ndarray) -> np.ndarray:
    """Normalise raw DN to 0..1 reflectance (handles L2A uint16 scaling)."""
    dn = np.asarray(dn, dtype=np.float64)
    if dn.max() > 1.5:                       # scaled integers -> divide
        dn = dn / S2_SCALE
    return np.clip(dn, 0.0, 1.0)


def atmospheric_correction(band: np.ndarray) -> np.ndarray:
    """Dark-Object Subtraction (DOS): subtract the 1st-percentile path radiance."""
    path_radiance = np.percentile(band[np.isfinite(band)], 1.0)
    return np.clip(band - path_radiance, 1e-4, 1.0)


def water_column_adjustment(band: np.ndarray, depth_proxy: np.ndarray,
                            k: float = 0.25) -> np.ndarray:
    """First-order water-column correction: compensate exponential attenuation.

    depth_proxy in [0, 1] (0 shallow, 1 deep); R_corrected = R * exp(k * depth).
    """
    return band * np.exp(k * np.clip(depth_proxy, 0.0, 1.0))


def compute_indices(blue: np.ndarray, green: np.ndarray,
                    red: np.ndarray, nir: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NDVI_submerged, Submerged Color Index, and a water mask (True = water)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi_sub = (nir - red) / (nir + red + 1e-12)
        sci = green / (blue + 1e-12)
    ndvi_sub = np.clip(np.nan_to_num(ndvi_sub, nan=0.0), -1.0, 1.0)
    sci = np.clip(np.nan_to_num(sci, nan=1.0), 0.0, 5.0)
    water = ~((nir > 0.12) & (ndvi_sub > 0.2))   # vegetated land / clouds out
    return ndvi_sub, sci, water


def tls_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Perpendicular (total-least-squares) slope + ordinary R^2."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc, yc = x - x.mean(), y - y.mean()
    cov = np.cov(np.stack([xc, yc]))
    vals, vecs = np.linalg.eigh(cov)
    slope = float(vecs[1, 1] / max(vecs[0, 1], 1e-12))
    ss = float((yc ** 2).sum())
    r2 = float(1 - ((yc - slope * xc) ** 2).sum() / max(ss, 1e-12))
    return slope, max(min(r2, 1.0), 0.0)


def depth_invariant_indices(blue: np.ndarray, green: np.ndarray,
                           red: np.ndarray, nir: np.ndarray,
                           water: np.ndarray, seed: int = RNG_SEED
                           ) -> tuple[dict[str, np.ndarray], dict]:
    """Lyzenga DII per Sen2Coral: DII_ij = ln(Ri) - (ki/kj)*ln(Rj).

    Attenuation ratio ki/kj fit by perpendicular regression over same-bottom
    (bright sand: high blue, low NIR) pixels spanning depths. Returns DII
    maps for (B2,B3), (B2,B4), (B3,B4) plus per-pair fit params.
    """
    rng = np.random.default_rng(seed + 5)
    valid = (water & np.isfinite(blue) & np.isfinite(green)
             & np.isfinite(red) & (blue > 0) & (green > 0) & (red > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red + 1e-12)
    sand = valid & (blue >= np.percentile(blue[valid], 75)) & (ndvi < 0.1)
    sidx = np.where(sand.ravel())[0]
    if len(sidx) < 50:                       # no clean sand ROI: use water
        sidx = np.where(valid.ravel())[0]
    if len(sidx) > 3000:
        sidx = rng.choice(sidx, 3000, replace=False)
    ln = {k: np.log(np.clip(v, 1e-6, None))
          for k, v in (("b", blue), ("g", green), ("r", red))}
    out, params = {}, {}
    for name, (a, b) in {"dii_23": ("b", "g"), "dii_24": ("b", "r"),
                         "dii_34": ("g", "r")}.items():
        slope, r2 = tls_slope(ln[b].ravel()[sidx], ln[a].ravel()[sidx])
        with np.errstate(invalid="ignore"):
            dii = np.nan_to_num(ln[a] - slope * ln[b], nan=0.0)
        out[name] = dii.astype(np.float32)
        params[name] = {"ki_kj": round(slope, 4), "sand_r2": round(r2, 4),
                        "n_sand": int(len(sidx))}
    return out, params


def process_satellite(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Full raster subsystem: reflectance -> DOS -> water-column -> indices."""
    refl = {k: to_reflectance(v) for k, v in bands.items()}
    corrected = {k: atmospheric_correction(v) for k, v in refl.items()}
    # Depth proxy from blue attenuation (clear water: blue fades with depth).
    b = corrected["blue"]
    depth_proxy = 1.0 - (b - b.min()) / (b.max() - b.min() + 1e-12)
    adjusted = {k: water_column_adjustment(v, depth_proxy) for k, v in corrected.items()}
    ndvi_sub, sci, water = compute_indices(
        adjusted["blue"], adjusted["green"], adjusted["red"], adjusted["nir"])
    diis, dii_params = depth_invariant_indices(
        adjusted["blue"], adjusted["green"], adjusted["red"],
        adjusted["nir"], water)
    return {"ndvi_sub": ndvi_sub, "sci": sci, "water": water,
            "depth_proxy": depth_proxy, **diis, "dii_params": dii_params}


# ---------------------------------------------------------------------------
# 2. Oceanographic Thermal Feature Engineering (NOAA DHW)
# ---------------------------------------------------------------------------
def compute_dhw_series(sst: np.ndarray, mmm: float,
                       window_days: int = DHW_WINDOW_DAYS) -> np.ndarray:
    """Degree Heating Weeks for each day from the trailing 12-week window.

    DHW = sum_{past 12 weeks} (SST - MMM) * I(SST >= MMM + 1.0 C),
    with daily values divided by 7 to express the total in C-weeks.
    """
    sst = np.asarray(sst, dtype=np.float64)
    hotspot = np.where(sst >= mmm + 1.0, sst - mmm, 0.0)
    dhw = np.zeros_like(sst)
    cumsum = np.concatenate([[0.0], np.cumsum(hotspot)])
    for i in range(len(sst)):
        lo = max(0, i - window_days + 1)
        dhw[i] = (cumsum[i + 1] - cumsum[lo]) / 7.0
    return dhw


def thermal_features(sst: np.ndarray, mmm: float,
                     window_days: int = DHW_WINDOW_DAYS,
                     dates=None) -> dict[str, float]:
    """Scalar thermal metrics for the latest window + re-tuning grid + MHW."""
    w = np.asarray(sst[-window_days:], dtype=np.float64)
    hotspot = np.maximum(0.0, w - (mmm + 1.0))
    dhw_now = float(compute_dhw_series(sst, mmm, window_days)[-1])
    out = {
        "dhw": dhw_now,
        "sst_mean": float(w.mean()),
        "sst_max": float(w.max()),
        "sst_std": float(w.std()),
        "hotspot_max": float((w - mmm).max()),
        "weeks_above_mmm1": float(np.sum(w >= mmm + 1.0) / 7.0),
        "hotspot_sum": float(hotspot.sum()),
        "warming_rate": float(np.polyfit(np.arange(len(w)), w, 1)[0] * 7.0),
        "dhw_t05_w42": compute_dhw_threshold(sst, mmm, 0.5, 42),
        "dhw_t05_w84": compute_dhw_threshold(sst, mmm, 0.5, 84),
        "dhw_t10_w42": compute_dhw_threshold(sst, mmm, 1.0, 42),
        "dhw_t10_w84": compute_dhw_threshold(sst, mmm, 1.0, 84),
    }
    out.update(mhw_features(np.asarray(sst, dtype=np.float64), dates,
                            window_days))
    out.update(transplant_features(np.asarray(sst, dtype=np.float64), mmm,
                                   window_days))
    return out


def load_sst(csv_path: str | None, mmm: float) -> "pd.DataFrame":
    """Load NOAA-style daily SST CSV (columns: date, sst_c) or synthesize it."""
    import pandas as pd
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = next(c for c in df.columns if "date" in c or "time" in c)
        sst_col = next(c for c in df.columns if "sst" in c or "temp" in c)
        df = df.rename(columns={date_col: "date", sst_col: "sst_c"})
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    df = generate_synthetic_sst(mmm=mmm)
    df.to_csv("sst_noaa_daily.csv", index=False)
    print("No --sst_csv given: wrote synthetic SST series to sst_noaa_daily.csv")
    return df


def forecast_sst(sst: np.ndarray, horizon_days: int = FORECAST_HORIZON_DAYS,
                 seed: int = RNG_SEED) -> np.ndarray:
    """Project SST `horizon_days` ahead: linear-trend + seasonal persistence."""
    rng = np.random.default_rng(seed + 2)
    sst = np.asarray(sst, dtype=np.float64)
    slope = float(np.polyfit(np.arange(len(sst)), sst, 1)[0])
    yearly = sst[max(0, len(sst) - 365):]
    seasonal = yearly.mean() - sst.mean() if len(yearly) else 0.0
    proj = sst[-1] + slope * np.arange(1, horizon_days + 1) + 0.3 * seasonal
    proj += rng.standard_normal(horizon_days) * sst[-30:].std()
    return np.concatenate([sst, proj])


# ---------------------------------------------------------------------------
# 3. Spatiotemporal Risk Model (LightGBM, 3 bleaching-risk classes)
# ---------------------------------------------------------------------------
PIXEL_FEATURES = ["ndvi_sub", "sci", "depth_proxy", "dii_23", "dii_24",
                  "dii_34", "light_heat_dose"]
PIXEL_RAW = [k for k in PIXEL_FEATURES if k != "light_heat_dose"]
THERMAL_FEATURES = ["dhw", "sst_mean",
                    "sst_max", "sst_std", "hotspot_max",
                    "weeks_above_mmm1",
                    "hotspot_sum", "warming_rate",
                    "dhw_t05_w42", "dhw_t05_w84", "dhw_t10_w42",
                    "dhw_t10_w84",
                    "mhw_events", "mhw_max_cat", "mhw_max_days",
                    "mhw_cum_int",
                    "gdd", "cwsi_mean", "cwsi_max", "days_above_30",
                    "abs_excess_sum", "slope7_rate", "hot_streak_now",
                    "hot_streak_max", "var30", "var90", "std_surplus",
                    "lag4_mean", "lag4_mean_sq", "lag8_mean"]
FEATURE_NAMES = PIXEL_FEATURES + THERMAL_FEATURES
N_PIXEL = len(PIXEL_FEATURES)
DHW_COL = N_PIXEL  # dhw is the first thermal column after pixel features

# Monotonicity priors: heat/severity features increase risk (+1), high SST
# variance is protective per Thompson & van Woesik / Sully (-1), optical and
# lag terms left unconstrained (0) since their response is nonlinear.
_INCREASING = {"dhw", "hotspot_max", "hotspot_sum", "weeks_above_mmm1",
               "sst_max", "dhw_t05_w42", "dhw_t05_w84", "dhw_t10_w42",
               "dhw_t10_w84", "mhw_events", "mhw_max_cat", "mhw_max_days",
               "mhw_cum_int", "gdd", "cwsi_mean", "cwsi_max",
               "days_above_30", "abs_excess_sum", "slope7_rate",
               "hot_streak_now", "hot_streak_max", "std_surplus",
               "light_heat_dose"}
_DECREASING = {"sst_std", "var30", "var90"}


def monotone_constraints() -> list[int]:
    """Per-feature LightGBM monotone constraints aligned to FEATURE_NAMES."""
    return [1 if f in _INCREASING else (-1 if f in _DECREASING else 0)
            for f in FEATURE_NAMES]


def select_features(X: np.ndarray, y: np.ndarray, seed: int = RNG_SEED,
                    n_estimators: int = 100) -> list[int]:
    """R5: keep columns with nonzero variance AND nonzero LightGBM gain.

    A quick probe fit finds dead transplants (e.g. days_above_30 on cool
    scenes); the caller slices every downstream block with the returned
    indices. Never drops the DHW column.
    """
    import lightgbm as lgb
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    keep = [i for i in range(X.shape[1])
            if float(np.var(X[:, i])) > 0.0]
    probe = lgb.LGBMClassifier(objective="multiclass", num_class=N_CLASSES,
                               n_estimators=n_estimators, learning_rate=0.05,
                               num_leaves=31, min_child_samples=20,
                               verbose=-1, random_state=seed)
    probe.fit(X[:, keep], y)
    gain = probe.booster_.feature_importance(importance_type="gain")
    keep = [i for i, g in zip(keep, gain) if float(g) > 0.0]
    if DHW_COL not in keep:
        keep = sorted(keep + [DHW_COL])
    return keep


def compute_dhw_threshold(sst: np.ndarray, mmm: float, offset: float,
                          window_days: int) -> float:
    """DHW with HotSpot threshold MMM+offset over a custom window (C-weeks)."""
    sst = np.asarray(sst, dtype=np.float64)[-window_days:]
    hotspot = np.where(sst >= mmm + offset, sst - mmm, 0.0)
    return float(hotspot.sum() / 7.0)


# R2: redefined DHW (Coral Reefs 2024): HotSpot cutoff MMM+0.4 C over an
# 11-week (77 d) window, bleaching threshold 3 C-weeks. Reported +~70%
# global skill vs the standard 1.0 C / 12-wk / >=4 definition, so it is
# the harder, fairer baseline every model must beat alongside persistence.
REDEFINED_OFFSET = 0.4
REDEFINED_WINDOW_DAYS = 77
REDEFINED_THRESHOLD = 3.0


def redefined_dhw(sst: np.ndarray, mmm: float) -> float:
    """Redefined DHW (0.4 C cutoff, 77-day window) in C-weeks."""
    return compute_dhw_threshold(np.asarray(sst, dtype=np.float64), mmm,
                                 REDEFINED_OFFSET, REDEFINED_WINDOW_DAYS)


def redefined_dhw_predict(dhw: np.ndarray) -> np.ndarray:
    """Redefined tiers: >=2x threshold severe (2), >=threshold watch (1)."""
    dhw = np.asarray(dhw, dtype=float)
    return np.where(dhw >= 2 * REDEFINED_THRESHOLD, 2,
                    np.where(dhw >= REDEFINED_THRESHOLD, 1, 0))


def transplant_features(sst: np.ndarray, mmm: float,
                        window_days: int = DHW_WINDOW_DAYS
                        ) -> dict[str, float]:
    """Cross-domain transplants as 30-day bleaching covariates.

    GDD-style coral degree-heating-days; CWSI-style (anom-lower)/(upper-lower)
    normalization; absolute-temperature days-above-30C; 7-day heating-rate
    slope + consecutive hot-day streaks; SST-variance protection windows;
    SPEI-style standardized surplus; DLNM-style lagged quadratic terms.
    """
    w = np.asarray(sst[-window_days:], dtype=float)
    anom = w - mmm
    hot = anom >= 1.0
    gdd = float(np.maximum(0.0, anom).sum() / 7.0)
    cwsi = np.clip(anom / 2.0, 0.0, 1.0)
    d30 = float(np.sum(w >= 30.0))
    abs_sum = float(np.maximum(0.0, w - 30.0).sum())
    slope7 = float(np.polyfit(np.arange(min(7, len(w))), w[-7:], 1)[0] * 7.0)
    streak, best, cur = 0, 0, 0
    for h in hot:
        cur = cur + 1 if h else 0
        best = max(best, cur)
    streak = cur
    var30 = float(np.var(w[-30:])) if len(w) >= 30 else float(np.var(w))
    var90 = float(np.var(w[-90:])) if len(w) >= 90 else float(np.var(w))
    w90 = np.asarray(sst[-90:], dtype=float) if len(sst) >= 90 else w
    std_surplus = float((w90.mean() - mmm) / (w90.std() + 1e-9))
    lag4 = np.asarray(sst[max(0, len(sst) - 56):len(sst) - 28], dtype=float)
    lag8 = np.asarray(sst[max(0, len(sst) - 84):len(sst) - 56], dtype=float)
    lag4m = float(lag4.mean()) if len(lag4) else float(w.mean())
    lag8m = float(lag8.mean()) if len(lag8) else float(w.mean())
    return {"gdd": gdd,
            "cwsi_mean": float(cwsi.mean()),
            "cwsi_max": float(cwsi.max()),
            "days_above_30": d30,
            "abs_excess_sum": abs_sum,
            "slope7_rate": slope7,
            "hot_streak_now": float(streak),
            "hot_streak_max": float(best),
            "var30": var30,
            "var90": var90,
            "std_surplus": std_surplus,
            "lag4_mean": lag4m,
            "lag4_mean_sq": lag4m ** 2,
            "lag8_mean": lag8m}


def mhw_features(sst: np.ndarray, dates=None,
                 window_days: int = DHW_WINDOW_DAYS) -> dict[str, float]:
    """Hobday-style marine-heatwave features over the trailing window.

    Event = SST above the day-of-year 90th percentile for >=5 consecutive
    days (<=2-day gaps merged); categories I-IV by exceedance above the
    climatological mean in multiples of (P90 - mean). Climatology is
    estimated from the series itself (day-of-year grouping, smoothed).
    """
    import pandas as pd
    sst = np.asarray(sst, dtype=np.float64)
    n = len(sst)
    if dates is None:
        doy = np.arange(n) % 365
    else:
        doy = pd.to_datetime(dates).dayofyear.to_numpy() - 1
    clim_mean = np.zeros(n)
    clim_p90 = np.zeros(n)
    for d in range(365):
        m = doy == d
        if m.sum() == 0:
            continue
        clim_mean[m] = sst[m].mean()
        clim_p90[m] = (np.percentile(sst[m], 90) if m.sum() >= 3
                       else sst[m].max())
    clim_mean = pd.Series(clim_mean).rolling(31, min_periods=1,
                                             center=True).mean().to_numpy()
    clim_p90 = pd.Series(clim_p90).rolling(31, min_periods=1,
                                          center=True).mean().to_numpy()
    lo = max(0, n - window_days)
    exceed = sst[lo:] > clim_p90[lo:]
    # merge <=2-day gaps, keep runs >=5 days
    runs, start, prev = [], None, -10
    for i, e in enumerate(exceed):
        if e and (start is None or i - prev > 3):
            if start is not None and prev - start + 1 >= 5:
                runs.append((start, prev))
            start = i
        if e:
            prev = i
    if start is not None and prev - start + 1 >= 5:
        runs.append((start, prev))
    cats, days, cum = [], [], 0.0
    for a, b in runs:
        denom = max(clim_p90[lo + a:lo + b + 1].mean()
                    - clim_mean[lo + a:lo + b + 1].mean(), 1e-6)
        mult = ((sst[lo + a:lo + b + 1]
                 - clim_mean[lo + a:lo + b + 1]).max() / denom)
        cats.append(min(int(mult), 4))
        days.append(b - a + 1)
        cum += float(np.maximum(0.0, sst[lo + a:lo + b + 1]
                                - clim_mean[lo + a:lo + b + 1]).sum() / 7.0)
    return {"mhw_events": float(len(runs)),
            "mhw_max_cat": float(max(cats) if cats else 0),
            "mhw_max_days": float(max(days) if days else 0),
            "mhw_cum_int": float(cum)}


def pixel_block(features: dict[str, np.ndarray], idx: np.ndarray,
                therm: dict[str, float], dhw_map: np.ndarray) -> np.ndarray:
    """Assemble one feature matrix block for flat pixel indices.

    Columns follow FEATURE_NAMES: raw pixel features, light×heat dose,
    thermal scalars tiled, with the DHW column replaced by per-pixel DHW.
    Shared by snapshot training, site-supervised training, and forecast.
    """
    idx = np.asarray(idx)
    n = len(idx)
    scalar = np.array([therm[f] for f in THERMAL_FEATURES], dtype=np.float64)
    block = np.column_stack(
        [features[k].ravel()[idx] for k in PIXEL_RAW]
        + [np.zeros(n, dtype=np.float32), np.tile(scalar, (n, 1))])
    block[:, DHW_COL] = np.asarray(dhw_map).ravel()[idx]
    dose = PIXEL_FEATURES.index("light_heat_dose")
    sci = PIXEL_FEATURES.index("sci")
    block[:, dose] = block[:, sci] * np.clip(block[:, DHW_COL] / 4.0, 0, 2.0)
    return block


def rule_labels(dhw_px: np.ndarray, ndvi_px: np.ndarray) -> np.ndarray:
    """NOAA-tier rule labels with coral-cover modulation (shared by heads)."""
    dhw_px = np.asarray(dhw_px, dtype=float)
    ndvi_px = np.asarray(ndvi_px, dtype=float)
    y = np.where(dhw_px < 2.0, 0, np.where(dhw_px > 4.0, 2, 1)).astype(int)
    upgrade = (y == 1) & (ndvi_px < np.median(ndvi_px)) & (dhw_px > 3.0)
    y[upgrade] = 2
    downgrade = (y == 1) & (ndvi_px > np.median(ndvi_px)) & (dhw_px < 2.5)
    y[downgrade] = 0
    return y


def load_stack(stack_dir: str | None) -> list | None:
    """Load a real multi-temporal stack: {BAND}_{YYYYMMDD}.tif files.

    Returns [(date_str, {blue,green,red,nir})] sorted by date, or None when
    the directory holds fewer than 2 complete dates (caller falls back to
    the synthetic temporal stack).
    """
    import re
    from pathlib import Path as _P
    if not stack_dir:
        return None
    files = list(_P(stack_dir).glob("*.tif")) + list(_P(stack_dir).glob("*.TIF"))
    by_date: dict[str, dict] = {}
    for f in files:
        m = re.match(r"(?i)(B2|B3|B4|B8|blue|green|red|nir)[-_]?(\d{8})",
                     f.stem)
        if not m:
            continue
        band = {"b2": "blue", "b3": "green", "b4": "red", "b8": "nir"}.get(
            m.group(1).lower(), m.group(1).lower())
        by_date.setdefault(m.group(2), {})[band] = str(f)
    stack = []
    for date in sorted(by_date):
        if set(by_date[date]) == {"blue", "green", "red", "nir"}:
            stack.append((date, {k: load_band(p, None)
                                 for k, p in by_date[date].items()}))
    return stack if len(stack) >= 2 else None


def _stack_positions(stack_dates: list, sst_dates, n: int) -> list:
    """R5: map each stack YYYYMMDD date to its SST-series cut position.

    Falls back to the full series (old behaviour) when dates are absent
    or unparseable, so old callers are unaffected.
    """
    if sst_dates is None:
        return [n - 1] * len(stack_dates)
    try:
        ref = np.asarray(sst_dates).astype("datetime64[D]")
        out = []
        for d in stack_dates:
            q = np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:]}")
            pos = int(np.searchsorted(ref, q, side="right")) - 1
            out.append(min(max(pos, DHW_WINDOW_DAYS), n - 1))
        return out
    except Exception:
        return [n - 1] * len(stack_dates)


def build_temporal_dataset(features: dict[str, np.ndarray],
                           sst: np.ndarray, mmm: float,
                           n_times: int = 6, n_pixels: int = 4000,
                           seed: int = RNG_SEED,
                           stack: list | None = None,
                           sst_dates=None
                           ) -> tuple[np.ndarray, np.ndarray, list]:
    """Per-pixel sequences (n, t, [ndvi, sci, depth, dhw]) + final labels.

    Synthetic path: cumulative-DHW-driven bleaching perturbation of the base
    scene over `n_times` snapshot dates. Real path: each stack acquisition is
    processed through the satellite subsystem; DHW comes from the SST series
    at the matching date (fallback: evenly spaced).
    """
    rng = np.random.default_rng(seed + 21)
    water = features["water"]
    idx_all = np.where(water.ravel())[0]
    samp = rng.choice(idx_all, size=min(n_pixels, len(idx_all)),
                        replace=False)
    if stack is None:
        ends = np.linspace(DHW_WINDOW_DAYS, len(sst) - 1,
                           n_times).astype(int)
        dhw_series = compute_dhw_series(np.asarray(sst, dtype=float), mmm)
        seqs, dhw_px_last, ndvi_last = [], None, None
        base_ndvi = features["ndvi_sub"].ravel()[samp]
        base_sci = features["sci"].ravel()[samp]
        depth = features["depth_proxy"].ravel()[samp]
        for end in ends:
            dhw_map = np.clip(dhw_series[end] * (
                0.85 + 0.30 * features["depth_proxy"]), 0, None)
            dhw_px = dhw_map.ravel()[samp]
            stress = np.clip(dhw_px / 8.0, 0, 1)
            ndvi_t = base_ndvi - 0.25 * stress * (0.5 + 0.5 * rng.random(len(samp)))
            ndvi_t += rng.standard_normal(len(samp)) * 0.02
            sci_t = base_sci + 0.10 * stress + rng.standard_normal(len(samp)) * 0.01
            seqs.append(np.column_stack([ndvi_t, sci_t, depth, dhw_px]))
        X_seq = np.stack(seqs, axis=1)
        y = rule_labels(X_seq[:, -1, 3], X_seq[:, -1, 0])
        dates = [f"t{k}" for k in range(n_times)]
    else:
        seqs = []
        cuts = _stack_positions([d for d, _ in stack], sst_dates, len(sst))
        for (date, bands), cut in zip(stack, cuts):
            f = process_satellite(bands)
            dhw_here = thermal_features(
                np.asarray(sst[:cut + 1], dtype=float), mmm)["dhw"]
            seqs.append(np.column_stack([
                f["ndvi_sub"].ravel()[samp], f["sci"].ravel()[samp],
                f["depth_proxy"].ravel()[samp],
                np.full(len(samp), dhw_here)]))
            dates = [d for d, _ in stack]
        X_seq = np.stack(seqs, axis=1)
        y = rule_labels(X_seq[:, -1, 3], X_seq[:, -1, 0])
    flip = rng.random(len(y)) < 0.05
    y[flip] = rng.integers(0, N_CLASSES, size=int(flip.sum()))
    return X_seq.astype(np.float32), y, dates


def quadrant_grid(shape) -> np.ndarray:
    """Quadrant label (NE/NW/SE/SW) per pixel; the R8 deployment basin axis.

    Same N/S-by-row, E/W-by-column rule historically inline in
    build_training_matrix, factored out so train-time and export-time basin
    assignment cannot drift apart.
    """
    h, w = int(shape[0]), int(shape[1])
    rr, cc = np.mgrid[0:h, 0:w]
    return np.char.add(np.where(rr < h // 2, "N", "S"),
                       np.where(cc < w // 2, "W", "E"))


def build_training_matrix(features: dict[str, np.ndarray],
                          sst: np.ndarray, mmm: float,
                          n_snapshots: int = 10,
                          seed: int = RNG_SEED
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatiotemporal pixel feature matrix X, stress labels y, water index.

    Rows are (water pixel x historical snapshot) pairs: for each of
    `n_snapshots` end-dates spread over the SST record, per-pixel DHW is the
    snapshot DHW modulated by a fixed spatial pattern (inshore/offshore
    gradient + patchiness), so training spans calm and heatwave regimes.
    Labels follow NOAA practice (DHW < 2: no stress; 2-4: watch/warning;
    > 4: severe) modulated by coral-cover signal (low NDVI_submerged on reef
    pixels upgrades borderline cases), with 5% label noise for robustness.
    """
    rng = np.random.default_rng(seed + 3)
    water = features["water"]
    idx_all = np.where(water.ravel())[0]
    pattern = (0.85 + 0.30 * features["depth_proxy"]
               + rng.standard_normal(features["depth_proxy"].shape) * 0.05)
    # DHW-stratified snapshot dates: uniform coverage of the observed DHW
    # range so transitional (watch/warning) regimes are represented.
    dhw_series = compute_dhw_series(np.asarray(sst, dtype=np.float64), mmm)
    valid = np.arange(DHW_WINDOW_DAYS, len(sst))
    targets = np.linspace(dhw_series[valid].min(), dhw_series[valid].max(),
                          n_snapshots)
    ends, used = [], set()
    for t in targets:
        end = int(valid[np.argmin(np.abs(dhw_series[valid] - t))])
        if end not in used:
            used.add(end)
            ends.append(end)
    X_blocks, y_blocks, samp_idx = [], [], []
    for end in ends:
        idx = rng.choice(idx_all, size=min(8000, len(idx_all)), replace=False)
        samp_idx.append(idx)
        n = len(idx)
        therm = thermal_features(sst[:end + 1], mmm)
        dhw_map = np.clip(therm["dhw"] * pattern, 0, None)
        block = pixel_block(features, idx, therm, dhw_map)
        X_blocks.append(block)
        dhw_px = X_blocks[-1][:, DHW_COL]
        ndvi_px = X_blocks[-1][:, 0]
        y_blocks.append(rule_labels(dhw_px, ndvi_px))
    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    flip = rng.random(len(y)) < 0.05
    y[flip] = rng.integers(0, N_CLASSES, size=int(flip.sum()))
    final_end = len(sst) - 1
    height, width = features["water"].shape
    row_idx = np.concatenate(samp_idx)
    row_regions = np.where((row_idx % width) < width // 2, "west", "east")
    row_quad = quadrant_grid(features["water"].shape).ravel()[row_idx]
    row_leads_wk = np.concatenate([
        np.full(len(b), (final_end - end) / 7.0) for b, end in
        zip(X_blocks, ends)])
    return X, y, idx, row_leads_wk, row_regions, row_quad


def build_site_supervised_matrix(features: dict[str, np.ndarray],
                                 site_labels: "pd.DataFrame",
                                 therm: dict[str, float],
                                 dhw_map: np.ndarray,
                                 transform, crs: str,
                                 radius_px: int = 5,
                                 seed: int = RNG_SEED
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel training rows from in-situ site labels (real-label path).

    Each survey (lat, lon, label) contributes a water-only square window of
    radius `radius_px` around its projected pixel; every sampled pixel
    inherits the site label. Returns (X, y, site_index). Empty arrays when
    no site falls on water. NOTE: labels may predate the imagery — treat
    scores as a consistency check unless dates match the scene.
    """
    import rasterio
    from rasterio.warp import transform as warp_xy
    rng = np.random.default_rng(seed + 31)
    H, W = features["water"].shape
    water_flat = features["water"].ravel()
    xs, ys = warp_xy("EPSG:4326", crs,
                     np.asarray(site_labels["lon"], dtype=float),
                     np.asarray(site_labels["lat"], dtype=float))
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    Xb, yb, sb = [], [], []
    for s, (r, c, lab) in enumerate(zip(rows, cols,
                                       site_labels["label"].to_numpy())):
        rr = np.arange(max(0, r - radius_px), min(H, r + radius_px + 1))
        cc = np.arange(max(0, c - radius_px), min(W, c + radius_px + 1))
        if len(rr) == 0 or len(cc) == 0:
            continue
        rr_g, cc_g = np.meshgrid(rr, cc, indexing="ij")
        pix = (rr_g * W + cc_g).ravel()
        pix = pix[water_flat[pix]]
        if len(pix) == 0:
            continue
        if len(pix) > 400:
            pix = rng.choice(pix, 400, replace=False)
        Xb.append(pixel_block(features, pix, therm, dhw_map))
        yb.append(np.full(len(pix), int(lab)))
        sb.append(np.full(len(pix), s))
    if not Xb:
        z = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
        return z, np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return (np.vstack(Xb).astype(np.float64),
            np.concatenate(yb).astype(int),
            np.concatenate(sb).astype(int))


def train_risk_model(X: np.ndarray, y: np.ndarray, seed: int = RNG_SEED,
                     extra: tuple = (), cols: list[int] | None = None):
    """Stratified split + LightGBM; `extra` arrays are split in sync.

    `cols` is an R5 prune mask: constraints are sliced to match, and the
    fitted model expects caller blocks sliced with the same mask.
    """
    from sklearn.model_selection import train_test_split
    parts = train_test_split(X, y, *extra, test_size=0.25,
                             random_state=seed, stratify=y)
    Xtr, Xte, ytr, yte = parts[0], parts[1], parts[2], parts[3]
    rest = parts[4:]
    extras = [(rest[2 * i], rest[2 * i + 1]) for i in range(len(extra))]
    import lightgbm as lgb
    mono = monotone_constraints()
    if cols is not None:
        mono = [mono[i] for i in cols]
    model = lgb.LGBMClassifier(objective="multiclass", num_class=N_CLASSES,
                               n_estimators=400, learning_rate=0.05,
                               num_leaves=63, min_child_samples=40,
                               subsample=0.9, colsample_bytree=0.9,
                               reg_lambda=5.0, class_weight="balanced",
                               monotone_constraints=mono,
                               random_state=seed, verbose=-1)
    model.fit(Xtr, ytr)
    return model, (Xtr, Xte, ytr, yte), extras


def train_per_basin_models(X: np.ndarray, y: np.ndarray, basins,
                           seed: int = RNG_SEED,
                           cols: list[int] | None = None, min_n: int = 50,
                           n_estimators: int = 400) -> dict:
    """R8 deployment: one constrained LightGBM head per basin.

    Same hyperparams and monotone constraints as train_risk_model (`cols`
    slices the constraints exactly like there; the caller slices X).
    Basins with n < min_n or a single class get model None and fall back
    to the global head in predict_per_basin.
    """
    import lightgbm as lgb
    Xa = np.asarray(X)
    ya = np.asarray(y)
    ba = np.asarray(basins)
    mono = monotone_constraints()
    if cols is not None:
        mono = [mono[i] for i in cols]
    out = {}
    for b in sorted(set(ba.tolist())):
        m = ba == b
        n = int(m.sum())
        if n < min_n or len(set(ya[m].tolist())) < 2:
            out[str(b)] = {"model": None, "n_train": n,
                           "note": f"n<{min_n} or single-class: "
                                   "global fallback"}
            continue
        clf = lgb.LGBMClassifier(objective="multiclass", num_class=N_CLASSES,
                                 n_estimators=n_estimators, learning_rate=0.05,
                                 num_leaves=63, min_child_samples=40,
                                 subsample=0.9, colsample_bytree=0.9,
                                 reg_lambda=5.0, class_weight="balanced",
                                 monotone_constraints=mono,
                                 random_state=seed, verbose=-1)
        clf.fit(Xa[m], ya[m])
        out[str(b)] = {"model": clf, "n_train": n}
    return out


def predict_per_basin(models: dict, fallback, X: np.ndarray,
                      basins) -> np.ndarray:
    """Route each row to its basin head; skipped/unknown basins use fallback."""
    Xa = np.asarray(X)
    ba = np.asarray(basins)
    proba = fallback.predict_proba(Xa)
    for b, entry in models.items():
        clf = entry.get("model")
        if clf is None:
            continue
        m = ba == b
        if m.sum():
            proba[m] = clf.predict_proba(Xa[m])
    return proba


# ---------------------------------------------------------------------------
# 4. Evaluation + 30-day forecast GeoTIFF export
# ---------------------------------------------------------------------------
def evaluate_model(model, Xte: np.ndarray, yte: np.ndarray,
                   names: list[str] | None = None) -> dict:
    """ROC-AUC (OvR), per-class Precision-Recall, DHW importance, confusion.

    `names` overrides FEATURE_NAMES when the model trained on an R5-pruned
    column subset.
    """
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 confusion_matrix, classification_report)
    proba = model.predict_proba(Xte)
    pred = proba.argmax(axis=1)
    metrics: dict = {
        "roc_auc_ovr_macro": float(roc_auc_score(yte, proba, multi_class="ovr")),
        "per_class": {},
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
        "classification_report": classification_report(
            yte, pred, target_names=[RISK_LABELS[i] for i in range(N_CLASSES)],
            output_dict=True, zero_division=0),
        "accuracy": float((pred == yte).mean()),
        "n_test": int(len(yte)),
    }
    for c in range(N_CLASSES):
        bin_true = (yte == c).astype(int)
        metrics["per_class"][str(c)] = {
            "label": RISK_LABELS[c],
            "average_precision": float(average_precision_score(bin_true, proba[:, c])),
            "roc_auc_ovr": float(roc_auc_score(bin_true, proba[:, c])),
            "support": int(bin_true.sum()),
        }
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    names = list(names) if names is not None else FEATURE_NAMES
    metrics["feature_importance_gain"] = {
        f: float(gain[i]) for i, f in enumerate(names)}
    metrics["feature_importance_split"] = {
        f: int(split[i]) for i, f in enumerate(names)}
    dhw_gain = metrics["feature_importance_gain"]["dhw"]
    total_gain = sum(metrics["feature_importance_gain"].values()) or 1.0
    metrics["dhw_gain_share"] = float(dhw_gain / total_gain)
    return metrics


def export_forecast_geotiff(model, features: dict[str, np.ndarray],
                            therm_fc: dict[str, float],
                            dhw_map_fc: np.ndarray,
                            profile: dict, out_path: str,
                            temperature: float = 1.0,
                            cols: list[int] | None = None,
                            basin_models: dict | None = None
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Predict per-pixel 30-day risk; write class + P(severe) GeoTIFF.

    `cols` slices blocks with an R5 prune mask when the model trained
    on a column subset. `basin_models` (R8) routes each pixel to its
    quadrant head via quadrant_grid, falling back to `model`.
    """
    import rasterio
    water = features["water"]
    shape = water.shape
    _npix = shape[0] * shape[1]
    Xf = pixel_block(features, np.arange(_npix), therm_fc, dhw_map_fc)
    if cols is not None:
        Xf = Xf[:, cols]
    if basin_models is not None:
        proba = predict_per_basin(basin_models, model, Xf,
                                  quadrant_grid(shape).ravel())
    else:
        proba = model.predict_proba(Xf)
    if temperature != 1.0:                   # item 7b: calibrated map
        from eval.severity_leadtime import apply_temperature
        proba = apply_temperature(proba, temperature)
    risk = proba.argmax(axis=1).reshape(shape).astype(np.uint8)
    p_severe = (proba[:, 2] * 100).reshape(shape).astype(np.uint8)
    risk[~water] = 255          # nodata over land
    p_severe[~water] = 255
    profile = dict(profile)
    profile.update(count=2, dtype="uint8", nodata=255)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(risk, 1)
        dst.write(p_severe, 2)
        dst.update_tags(1, CLASS="risk_class_0_no_stress_1_watch_2_severe")
        dst.update_tags(2, CLASS="P_severe_percent_0_100")
        dst.update_tags(ns="bleaching",
                        horizon_days=FORECAST_HORIZON_DAYS,
                        dhw_forecast_C_weeks=round(therm_fc["dhw"], 3))
    return risk, p_severe


# ---------------------------------------------------------------------------
# 5. Orchestration
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multispectral coral bleaching predictor")
    p.add_argument("--blue", default=None, help="Sentinel-2 B2 GeoTIFF path")
    p.add_argument("--green", default=None, help="Sentinel-2 B3 GeoTIFF path")
    p.add_argument("--red", default=None, help="Sentinel-2 B4 GeoTIFF path")
    p.add_argument("--nir", default=None, help="Sentinel-2 B8 GeoTIFF path")
    p.add_argument("--sst_csv", default=None, help="NOAA daily SST CSV path")
    p.add_argument("--mmm", type=float, default=DEFAULT_MMM,
                   help="Maximum Monthly Mean baseline (degC)")
    p.add_argument("--horizon_days", type=int, default=FORECAST_HORIZON_DAYS)
    p.add_argument("--seed", type=int, default=RNG_SEED)
    p.add_argument("--sst_model", default="persistence",
                   choices=["persistence", "sstode"],
                   help="30-day SST projection method (item 5)")
    p.add_argument("--multivariate", action="store_true",
                   help="report SHAP/gain-pruned multivariate features (item 6)")
    p.add_argument("--eval_report", action="store_true",
                   help="write severity/lead-time eval report (item 7)")
    p.add_argument("--labels_csv", default=None,
                   help="external label CSV (item 1 join output) for DHW "
                        "baseline comparison")
    p.add_argument("--temporal", default="off", choices=["off", "sits"],
                   help="train SITS temporal head on multi-temporal stack "
                        "and compare vs LightGBM (item 3)")
    p.add_argument("--stack_dir", default=None,
                   help="real stack dir with {BAND}_{YYYYMMDD}.tif files; "
                        "falls back to synthetic stack when absent")
    p.add_argument("--calibrate", action="store_true",
                   help="temperature-scale probabilities; report ECE "
                        "before/after (item 7)")
    p.add_argument("--loro", action="store_true",
                   help="leave-one-region-out (quadrant) CV with the same "
                        "constrained head; report mean/std skill")
    p.add_argument("--site_labels_csv", default=None,
                   help="in-situ site labels (lat,lon,label) for the "
                        "real-label pixel training path")
    p.add_argument("--site_radius_px", type=int, default=5,
                   help="water-pixel window radius sampled per site")
    p.add_argument("--prune_dead", action="store_true",
                   help="R5: drop zero-variance/zero-gain features via a "
                        "probe fit before training (recorded in metrics)")
    p.add_argument("--downscale", action="store_true",
                   help="R3: sharpen the forecast DHW map with the RCNN "
                        "downscaler (depth + water static channels, "
                        "patch/val split)")
    p.add_argument("--per_basin", action="store_true",
                   help="R8: train one head per quadrant basin (+ global "
                        "fallback) and deploy the routed set for the map; "
                        "report deployed vs global skill on the same split")
    return p.parse_args()


def default_profile(shape) -> dict:
    from affine import Affine
    return {"driver": "GTiff", "height": shape[0], "width": shape[1],
            "count": 2, "dtype": "uint8", "crs": SYNTH_CRS,
            "transform": Affine(SYNTH_RES_M, 0.0, SYNTH_ORIGIN_X,
                                0.0, -SYNTH_RES_M, SYNTH_ORIGIN_Y)}


def main() -> None:
    args = parse_args()
    real_inputs = all([args.blue, args.green, args.red, args.nir])

    if real_inputs:
        import rasterio
        bands = {k: load_band(getattr(args, k), None)
                 for k in ("blue", "green", "red", "nir")}
        with rasterio.open(args.blue) as src:
            profile = dict(src.profile)
        print(f"Loaded real Sentinel-2 bands, shape={bands['blue'].shape}")
    else:
        synth = generate_synthetic_bands(seed=args.seed)
        bands = {k: load_band(None, v) for k, v in synth.items()}
        profile = default_profile(bands["blue"].shape)
        print(f"Using synthetic Sentinel-2 scene, shape={bands['blue'].shape}")

    # --- 1. Satellite indices ------------------------------------------------
    features = process_satellite(bands)
    n_water = int(features["water"].sum())
    print(f"NDVI_submerged range: [{features['ndvi_sub'].min():.3f}, "
          f"{features['ndvi_sub'].max():.3f}] | water pixels: {n_water}")

    # --- 2. Thermal features --------------------------------------------------
    sst_df = load_sst(args.sst_csv, args.mmm)
    sst = sst_df["sst_c"].to_numpy(dtype=np.float64)
    therm_now = thermal_features(sst, args.mmm,
                                 dates=sst_df["date"].to_numpy())
    print(f"Current DHW: {therm_now['dhw']:.2f} C-weeks "
          f"(MMM={args.mmm} C, {len(sst)} daily obs)")

    # --- 3. Train -------------------------------------------------------------
    X, y, _, row_leads_wk, row_regions, row_quad = build_training_matrix(
        features, sst, args.mmm, n_snapshots=12, seed=args.seed)
    dhw_col = X[:, DHW_COL].copy()
    prune_cols = None
    if args.prune_dead:  # R5: probe-prune dead features before training
        prune_cols = select_features(X, y, seed=args.seed)
        X = X[:, prune_cols]
        print(f"R5 pruned features: kept "
              f"{[FEATURE_NAMES[i] for i in prune_cols]}")
    uniq, counts = np.unique(y, return_counts=True)
    print(f"Training pixels: {len(y)} | class distribution: "
          f"{dict(zip(uniq.tolist(), counts.tolist()))}")
    model, (Xtr, Xte, ytr, yte), [(leads_tr, leads_te),
                                  (regs_tr, regs_te),
                                  (quad_tr, quad_te),
                                  (_dhw_tr, dhw_te)] = train_risk_model(
        X, y, args.seed, extra=(row_leads_wk, row_regions, row_quad,
                                dhw_col),
        cols=prune_cols)

    # --- 4. Evaluate ----------------------------------------------------------
    metrics = evaluate_model(
        model, Xte, yte,
        names=[FEATURE_NAMES[i] for i in prune_cols]
        if prune_cols is not None else None)
    if prune_cols is not None:
        metrics["pruned_features"] = [FEATURE_NAMES[i]
                                      for i in prune_cols]
    metrics["thermal_now_C_weeks"] = therm_now
    metrics["mmm_baseline_C"] = args.mmm
    from eval.run_registry import new_run as _new_run
    metrics["run"] = _new_run(seed=args.seed)
    # R2: harder baselines scored on the same test pixels.
    _dhw_te = np.asarray(dhw_te, dtype=float)
    _std_rule = np.where(_dhw_te >= 8.0, 2, np.where(_dhw_te >= 4.0, 1, 0))
    _red_rule = redefined_dhw_predict(_dhw_te)
    _maj = int(np.bincount(np.asarray(yte)).argmax())
    metrics["baselines"] = {
        "dhw_rule_accuracy": round(float((_std_rule == yte).mean()), 4),
        "redefined_dhw_accuracy": round(float((_red_rule == yte).mean()), 4),
        "majority_accuracy": round(float((np.full_like(yte, _maj)
                                          == yte).mean()), 4),
        "redefined_def": {"offset_C": REDEFINED_OFFSET,
                          "window_days": REDEFINED_WINDOW_DAYS,
                          "threshold_C_weeks": REDEFINED_THRESHOLD},
    }
    print(f"Baselines on test pixels: DHW rule "
          f"{metrics['baselines']['dhw_rule_accuracy']} | redefined-DHW "
          f"{metrics['baselines']['redefined_dhw_accuracy']} | majority "
          f"{metrics['baselines']['majority_accuracy']} | model "
          f"{metrics['accuracy']:.3f}")
    print(f"ROC-AUC (OvR macro): {metrics['roc_auc_ovr_macro']:.3f} | "
          f"accuracy: {metrics['accuracy']:.3f} | "
          f"DHW gain share: {metrics['dhw_gain_share']:.3f}")

    basin_models, proba_dep = None, None
    if args.per_basin:  # R8: one deployed head per quadrant basin.
        # Heads train on the global TRAIN split only, so routing the held-out
        # test rows scores the deployed system with no leakage, on the same
        # test set as the global model above.
        from sklearn.metrics import roc_auc_score as _roc
        basin_models = train_per_basin_models(Xtr, ytr, quad_tr,
                                              args.seed, cols=prune_cols)
        proba_dep = predict_per_basin(basin_models, model, Xte, quad_te)
        pred_dep = proba_dep.argmax(axis=1)
        _yte = np.asarray(yte)
        _pb = {"basins": {},
               "deployed_accuracy": round(float((pred_dep == _yte).mean()),
                                          4),
               "deployed_roc_auc_ovr": round(
                   float(_roc(_yte, proba_dep, multi_class="ovr")), 4),
               "global_accuracy": metrics["accuracy"]}
        for _b, _entry in basin_models.items():
            _m = np.asarray(quad_te) == _b
            _pb["basins"][_b] = {
                "n_train": _entry["n_train"],
                "n_test": int(_m.sum()),
                "accuracy": round(float((pred_dep[_m] == _yte[_m]).mean()),
                                  4) if _m.sum() else None}
            if "note" in _entry:
                _pb["basins"][_b]["note"] = _entry["note"]
        _pb["delta_vs_global"] = round(_pb["deployed_accuracy"]
                                      - _pb["global_accuracy"], 4)
        metrics["per_basin"] = _pb
        print(f"Per-basin deployed (R8): acc={_pb['deployed_accuracy']:.3f} "
              f"vs global {metrics['accuracy']:.3f} "
              f"(delta {_pb['delta_vs_global']:+.4f}) | " +
              ", ".join(f"{_b}: {_pb['basins'][_b]['accuracy']}"
                        for _b in sorted(_pb["basins"])))

    if args.multivariate:  # item 6: report pruned multivariate features
        import pandas as pd
        from features.multivariate import prune_features
        kept = prune_features(pd.DataFrame(X, columns=FEATURE_NAMES), y,
                              top_k=8, seed=args.seed)
        metrics["multivariate_pruned_features"] = kept
        print(f"Pruned features (item 6): {kept}")

    if args.labels_csv:  # item 1: score DHW baseline on external labels
        import pandas as pd
        from data.join_reefcheck_cortad import (dhw_baseline_predict,
                                                score_by_region)
        ext = pd.read_csv(args.labels_csv)
        ext["__pred"] = dhw_baseline_predict(ext["dhw"].to_numpy())
        metrics["external_labels_baseline"] = score_by_region(ext)
        print(f"External-label DHW baseline (item 1): "
              f"{metrics['external_labels_baseline']['overall_accuracy']:.3f}")

    if args.temporal == "sits":  # item 3: temporal head vs LightGBM
        from sklearn.model_selection import train_test_split as _tts
        from models.sits_mamba import SITSClassifier
        stack = load_stack(args.stack_dir)
        Xs, ys, sdates = build_temporal_dataset(
            features, sst, args.mmm, stack=stack, seed=args.seed,
            sst_dates=sst_df["date"].to_numpy())
        tr_i, te_i = _tts(np.arange(len(ys)), test_size=0.3,
                          random_state=args.seed, stratify=ys)
        tclf = SITSClassifier(epochs=30, seed=args.seed).fit(Xs[tr_i], ys[tr_i])
        tacc = float((tclf.predict(Xs[te_i]) == ys[te_i]).mean())
        metrics["temporal_sits"] = {
            "accuracy": tacc,
            "lightgbm_test_accuracy": metrics["accuracy"],
            "n_times": int(Xs.shape[1]),
            "source": "real_stack" if stack else "synthetic_stack",
            "dates": sdates,
        }
        print(f"SITS temporal head (item 3, {metrics['temporal_sits']['source']}): "
              f"acc={tacc:.3f} vs LightGBM {metrics['accuracy']:.3f}")

    if args.site_labels_csv:  # real-label pixel path: site buffers
        import pandas as pd
        _sites = pd.read_csv(args.site_labels_csv)
        _dhw_now = np.clip(therm_now["dhw"]
                           * (0.85 + 0.30 * features["depth_proxy"]), 0, None)
        Xs, ys, sids = build_site_supervised_matrix(
            features, _sites, therm_now, _dhw_now, profile["transform"],
            profile["crs"], radius_px=args.site_radius_px, seed=args.seed)
        if prune_cols is not None:
            Xs = Xs[:, prune_cols]
        n_sites = int(len(set(sids.tolist()))) if len(sids) else 0
        metrics["site_supervised"] = {"n_sites_used": n_sites,
                                      "n_pixels": int(len(ys)),
                                      "note": "labels may predate imagery; "
                                              "consistency check unless "
                                              "dates match the scene"}
        if len(ys) >= 30 and len(set(ys.tolist())) >= 2:
            from sklearn.model_selection import train_test_split as _ss
            tri, tei = _ss(np.arange(len(ys)), test_size=0.3,
                           random_state=args.seed, stratify=ys)
            import lightgbm as _lgb
            _mono = monotone_constraints()
            if prune_cols is not None:
                _mono = [_mono[i] for i in prune_cols]
            _clf = _lgb.LGBMClassifier(
                objective="multiclass", num_class=N_CLASSES,
                n_estimators=200, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=5.0, class_weight="balanced",
                monotone_constraints=_mono,
                random_state=args.seed, verbose=-1).fit(Xs[tri], ys[tri])
            _pred = _clf.predict(Xs[tei])
            _agree = float((_clf.predict(Xs[tei])
                            == model.predict(Xs[tei])).mean())
            metrics["site_supervised"].update({
                "accuracy": round(float((_pred == ys[tei]).mean()), 4),
                "holdout_n": int(len(tei)),
                "rule_head_agreement": round(_agree, 4)})
            print(f"Site-supervised head: n={len(ys)} px / {n_sites} sites, "
                  f"acc={metrics['site_supervised']['accuracy']}, "
                  f"rule-head agreement={_agree:.3f}")
        else:
            print(f"Site-supervised head: {n_sites} sites, {len(ys)} px "
                  f"(insufficient for a split)")

    temperature = 1.0
    if args.calibrate:  # item 7b: temperature scaling on a cal split
        from eval.severity_leadtime import (apply_temperature,
                                            brier_skill_score,
                                            expected_calibration_error,
                                            fit_temperature,
                                            fit_temperature_cv)
        # R8: calibrate the deployed (routed) system when per-basin is on.
        proba_te = proba_dep if proba_dep is not None else \
            model.predict_proba(Xte)
        half = len(yte) // 2
        cal = fit_temperature(yte[:half], proba_te[:half])
        cal_cv = fit_temperature_cv(yte, proba_te, seed=args.seed)
        temperature = cal["temperature"]
        ece_before = expected_calibration_error(
            yte[half:], proba_te[half:, 2])["ece"]
        ece_after = expected_calibration_error(
            yte[half:], apply_temperature(proba_te[half:],
                                          temperature)[:, 2])["ece"]
        metrics["calibration"] = {**cal, "ece_before": ece_before,
                                  "ece_after": ece_after,
                                  "cross_fit": cal_cv,
                                  "brier_skill": brier_skill_score(
                                      yte, proba_te)}
        print(f"Calibration (item 7): T={temperature}, "
              f"ECE {ece_before} -> {ece_after}")

    if args.loro:  # honest spatial generalization via quadrant holdouts
        from eval.severity_leadtime import leave_one_region_out
        metrics["loro_cv"] = leave_one_region_out(X, y, row_quad,
                                                  seed=args.seed)
        print(f"LORO-CV quadrants: "
              f"{metrics['loro_cv']['mean_accuracy']} "
              f"(+/-{metrics['loro_cv']['std_accuracy']}) acc, "
              f"{metrics['loro_cv']['mean_roc_auc']} ROC-AUC")

    # --- 5. 30-day forecast ----------------------------------------------------
    if args.sst_model == "sstode":  # item 5: physics-informed SST horizon
        from models.sstode_forecast import SSTODEForecaster
        fc = SSTODEForecaster(seed=args.seed).fit(sst, args.horizon_days)
        sst_fc = np.concatenate([sst, fc.forecast(sst, args.horizon_days)])
        print("SST horizon: SSTODE Neural ODE (item 5)")
    else:
        sst_fc = forecast_sst(sst, args.horizon_days, args.seed)
    import pandas as _pd
    _fc_dates = _pd.DatetimeIndex(
        list(sst_df["date"].to_numpy())
        + list(_pd.date_range(sst_df["date"].iloc[-1]
                              + _pd.Timedelta(days=1),
                              periods=args.horizon_days, freq="D"))).to_numpy()
    therm_fc = thermal_features(sst_fc, args.mmm, dates=_fc_dates)
    dhw_map_fc = np.clip(
        therm_fc["dhw"] * (0.85 + 0.30 * features["depth_proxy"]), 0, None)
    if args.downscale:  # R3: reef-scale DHW via patch-trained downscaler
        from models.sst_downscale_rcnn import RCNNDownscaler
        _h, _w = dhw_map_fc.shape
        _ch, _cw = _h // 4, _w // 4
        _coarse = dhw_map_fc[:_ch * 4, :_cw * 4].reshape(
            _ch, 4, _cw, 4).mean(axis=(1, 3))
        _static = np.stack([features["depth_proxy"][:_ch * 4, :_cw * 4],
                            features["water"][:_ch * 4, :_cw * 4]
                            .astype(np.float32)])
        _ds = RCNNDownscaler(seed=args.seed).fit(
            _coarse, dhw_map_fc[:_ch * 4, :_cw * 4], _static)
        dhw_map_fc = np.clip(_ds.predict(_coarse, _static), 0, None)
        metrics["downscale"] = {"val_rmse": _ds.val_rmse_,
                                "static": ["depth_proxy", "water"]}
        print(f"RCNN downscale (R3): val RMSE={_ds.val_rmse_}")
    risk, p_severe = export_forecast_geotiff(
        model, features, therm_fc, dhw_map_fc, profile,
        "coral_bleaching_risk_forecast.tif", temperature=temperature,
        cols=prune_cols, basin_models=basin_models)
    water_risk = risk[features["water"]]
    for c in range(N_CLASSES):
        frac = float((water_risk == c).mean())
        print(f"  forecast class {c} ({RISK_LABELS[c]}): {frac * 100:.1f}% of reef")
    print(f"Forecast DHW (+{args.horizon_days}d): {therm_fc['dhw']:.2f} C-weeks")

    metrics["thermal_forecast_C_weeks"] = therm_fc
    metrics["forecast_horizon_days"] = args.horizon_days
    metrics["forecast_class_fractions"] = {
        str(c): float((water_risk == c).mean()) for c in range(N_CLASSES)}
    if args.eval_report:  # item 7: severity/lead-time product evaluation
        from eval.severity_leadtime import (outlook_style_summary,
                                            plot_reliability)
        proba_te = proba_dep if proba_dep is not None else \
            model.predict_proba(Xte)
        if temperature != 1.0:
            from eval.severity_leadtime import apply_temperature as _at
            proba_te = _at(proba_te, temperature)
        lead_wk = np.clip(np.round(leads_te).astype(int), 1, None)
        sev = outlook_style_summary(yte, proba_te.argmax(axis=1),
                                    proba_te[:, 2], lead_wk, regs_te)
        metrics["severity_leadtime"] = sev
        plot_reliability(sev, "eval/reliability.png")
        print(f"Severity/lead-time report (item 7): "
              f"ECE={sev['calibration']['ece']}, "
              f"{len(sev['by_lead'])} lead x region rows")
    with open("bleaching_evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    from eval.run_registry import record as _record
    print("Wrote coral_bleaching_risk_forecast.tif + bleaching_evaluation_metrics.json "
          f"+ {_record(metrics, 'forecaster')}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        shown = np.where(features["water"], risk, np.nan)
        ax[0].imshow(shown, cmap=ListedColormap(["#2ca02c", "#ffbf00", "#d62728"]),
                     vmin=0, vmax=2)
        ax[0].set_title(f"30-day bleaching risk (DHW {therm_fc['dhw']:.1f} C-wks)")
        ax[0].axis("off")
        ax[1].imshow(np.where(features["water"], p_severe, np.nan),
                     cmap="Reds", vmin=0, vmax=100)
        ax[1].set_title("P(severe) %")
        ax[1].axis("off")
        fig.tight_layout()
        fig.savefig("forecast_quicklook.png", dpi=110)
        print("Wrote forecast_quicklook.png")
    except Exception as e:  # quicklook is optional eye-candy, never fatal
        print(f"Skipped quicklook PNG ({e})")


if __name__ == "__main__":
    main()


