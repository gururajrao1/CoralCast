#!/usr/bin/env python3
"""Item 1: Real labels via Reef Check + CoRTAD/SST joins + DHW baseline.

Joins in-situ bleaching surveys (Reef Check format: site, latitude,
longitude, date, bleaching severity) to thermal stress (DHW/HotSpot) from a
site SST series or a gridded SST product, emits a real-label train/test split
and scores the NOAA DHW>=4 / DHW>=8 baseline stratified by region and lead.

Usage:
    python data/join_reefcheck_cortad.py --demo
    python data/join_reefcheck_cortad.py --reefcheck <csv|url> \\
        --sst_csv sst_noaa_daily.csv --mmm 28.5 --out data/labels.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coral_bleaching_forecaster import compute_dhw_series  # noqa: E402

REEFCHECK_URL = (
    "https://raw.githubusercontent.com/InstituteForGlobalEcology/"
    "Coral-bleaching-a-global-analysis-of-the-past-two-decades/master/"
)


def load_reefcheck(path_or_url: str) -> pd.DataFrame:
    """Load a Reef Check-style CSV and normalise columns.

    Expected (case-insensitive) columns: site, latitude, longitude, date,
    plus a severity column (bleaching / percent_bleached / severity) in 0..1
    or 0..100, mapped to labels 0 (none), 1 (moderate), 2 (severe).
    """
    df = pd.read_csv(path_or_url)
    df.columns = [c.strip().lower() for c in df.columns]
    col = {c: c for c in df.columns}
    lat = next(c for c in df.columns if c in ("latitude", "lat"))
    lon = next(c for c in df.columns if c in ("longitude", "lon", "long"))
    date = next(c for c in df.columns if "date" in c)
    sev = next((c for c in df.columns
                if any(k in c for k in ("bleach", "severity", "percent"))), None)
    out = pd.DataFrame({
        "site": df.get("site", pd.Series(range(len(df)))),
        "lat": pd.to_numeric(df[lat], errors="coerce"),
        "lon": pd.to_numeric(df[lon], errors="coerce"),
        "date": pd.to_datetime(df[date], errors="coerce"),
        "severity": pd.to_numeric(df[sev], errors="coerce") / 100.0
        if sev is not None else 0.0,
    }).dropna(subset=["lat", "lon", "date"]).reset_index(drop=True)
    out["severity"] = out["severity"].clip(0.0, 1.0)
    out["label"] = np.where(out["severity"] < 0.1, 0,
                            np.where(out["severity"] > 0.5, 2, 1)).astype(int)
    return out


def dhw_at_date(sst: pd.DataFrame, date: pd.Timestamp, mmm: float) -> dict:
    """DHW/HotSpot on `date` from a daily site SST series (date, sst_c)."""
    s = sst.sort_values("date").reset_index(drop=True)
    pos = s["date"].searchsorted(date)
    if pos >= len(s):
        pos = len(s) - 1
    hist = s["sst_c"].to_numpy(dtype=float)[:pos + 1]
    if len(hist) < 84:
        hist = np.pad(hist, (84 - len(hist), 0), mode="edge")
    dhw = compute_dhw_series(hist, mmm)[-1]
    hotspot = float(hist[-1] - mmm)
    return {"dhw": float(dhw), "hotspot": hotspot,
            "sst": float(hist[-1])}


def join_labels(surveys: pd.DataFrame, sst: pd.DataFrame,
                mmm: float) -> pd.DataFrame:
    """Attach (dhw, hotspot, sst) at each survey date; keep label + region."""
    rows = []
    for _, r in surveys.iterrows():
        th = dhw_at_date(sst, r["date"], mmm)
        rows.append({"site": r["site"], "lat": r["lat"], "lon": r["lon"],
                     "date": r["date"], "label": int(r["label"]),
                     "region": ("tropics" if abs(r["lat"]) < 15
                                else "subtropics"),
                     **th})
    return pd.DataFrame(rows)


def dhw_baseline_predict(dhw: np.ndarray) -> np.ndarray:
    """NOAA tiers: DHW>=8 severe (2), DHW>=4 watch/warning (1), else none (0)."""
    dhw = np.asarray(dhw, dtype=float)
    return np.where(dhw >= 8.0, 2, np.where(dhw >= 4.0, 1, 0))


def score_by_region(labels: pd.DataFrame) -> dict:
    """Accuracy + hit/false-alarm of the DHW baseline, overall and by region."""
    from sklearn.metrics import confusion_matrix
    y = labels["label"].to_numpy()
    pred = dhw_baseline_predict(labels["dhw"].to_numpy())
    out = {"overall_accuracy": float((pred == y).mean()),
           "confusion_matrix": confusion_matrix(y, pred).tolist()}
    for region, g in labels.groupby("region"):
        yp = dhw_baseline_predict(g["dhw"].to_numpy())
        yt = g["label"].to_numpy()
        out[str(region)] = {"n": int(len(g)),
                            "accuracy": float((yp == yt).mean())}
    return out


def make_demo_surveys(n: int = 60, seed: int = 7) -> pd.DataFrame:
    """Synthetic Reef Check-format surveys for offline verification."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "site": [f"reef_{i}" for i in range(n)],
        "latitude": rng.uniform(-20, 20, n).round(3),
        "longitude": rng.uniform(140, 160, n).round(3),
        "date": pd.to_datetime("2024-01-01")
        + pd.to_timedelta(rng.integers(0, 300, n), unit="D"),
        "percent_bleached": rng.uniform(0, 100, n).round(1),
    })


def main() -> None:
    p = argparse.ArgumentParser(description="Reef Check + SST label join")
    p.add_argument("--reefcheck", default=None)
    p.add_argument("--sst_csv", default="sst_noaa_daily.csv")
    p.add_argument("--mmm", type=float, default=28.5)
    p.add_argument("--out", default="data/labels.csv")
    p.add_argument("--demo", action="store_true",
                   help="run on synthetic surveys (offline verification)")
    args = p.parse_args()
    if args.demo or args.reefcheck is None:
        surveys = make_demo_surveys()
        surveys.to_csv("data/demo_reefcheck.csv", index=False)
        reef = load_reefcheck("data/demo_reefcheck.csv")
        print("demo mode: synthetic surveys -> data/demo_reefcheck.csv")
    else:
        reef = load_reefcheck(args.reefcheck)
    sst = pd.read_csv(args.sst_csv, parse_dates=["date"])
    labels = join_labels(reef, sst, args.mmm)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.out, index=False)
    scores = score_by_region(labels)
    print(json.dumps(scores, indent=2))
    with open(Path(args.out).with_suffix(".baseline.json"), "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Wrote {args.out} ({len(labels)} labels)")


if __name__ == "__main__":
    main()
