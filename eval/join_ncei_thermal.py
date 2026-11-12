#!/usr/bin/env python3
"""Thermal join for the NCEI bleaching archive (offline donors only).

Matches each NCEI 0228498 row to CoRTAD thermal covariates from the two
already-vendored donor frames (GCBD BCO-DMO + ReefCheck_CoRTAD) on
rounded coordinates + survey year. Emits a benchmark-ready CSV with the
GCBD column schema so ``eval/gcbd_benchmark.py`` consumes it directly.

Label mapping: numeric PERCENT_BLEACHED (or the MIN/MAX midpoint when the
main field is missing) plus range/category strings ('1%-10%' -> midpoint
5.5, '>60' -> 60, '<1'/'None_(<1)' -> 0) feed the standard
Percent_Bleaching 0/1/2 tiers. Realm is a single coarse scheme from
CORAL_REGIONS for every row (Caribbean/GoM -> Tropical Atlantic,
GBR -> Central Indo-Pacific, West-Indian -> Western Indo-Pacific,
Other Reef stays its own group) — NOT the 8-realm scheme.

Candidate rule: among donor rows on the same key, take the closest survey
month (GCBD donor wins ties); donor medians would smear event DHW across
the year. Missing NCEI month/day default to mid-year (Jul-01) for the
constructed Date. Unmatched rows are dropped (reported, not imputed).

Usage:
    python eval/join_ncei_thermal.py --out data/ncei_thermal_joined.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

NCEI_DEFAULT = "data/ncei_bleaching_0228498.csv"
GCBD_DEFAULT = "data/gcbd_bleaching_environmental.csv"
RC_DEFAULT = "data/ReefCheck_CoRTAD.csv"

RC_RENAME = {
    "Latitude.Degrees": "Latitude_Degrees",
    "Longitude.Degrees": "Longitude_Degrees",
    "Year": "Date_Year",
    "Average_bleaching": "Percent_Bleaching",
    "Depth": "Depth_m",
    "Reef.ID": "Site_ID",
    "Reef.Name": "Site_Name",
    "Country": "Country_Name",
}
RC_DROP = {"Ocean", "State.Province.Island", "City.Town", "Organism.Code",
           "S1", "S2", "S3", "S4", "Errors.", "What.errors.", "Region",
           "Diversity", "rate_of_SST_change"}

REALM_MAP = {
    "Caribbean/GoM": "Tropical Atlantic",
    "GBR": "Central Indo-Pacific",
    "West-Indian": "Western Indo-Pacific",
}

# Bleaching percents are never negative; hyphens are range separators
# ('1%-10%'), so only unsigned numbers are extracted.
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def parse_percent(v) -> tuple[float | None, str]:
    """NCEI bleaching entry -> (percent, provenance)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None, "missing"
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None, "missing"
    if s.startswith("<") or s.startswith("None"):
        return 0.0, "range_lt1"
    try:
        return float(s), "percent_numeric"
    except ValueError:
        pass
    nums = [float(t) for t in NUM_RE.findall(s)]
    if not nums:
        return None, "unparseable"
    if s.startswith(">"):
        return float(nums[0]), "range_gt"
    return float(sum(nums) / len(nums)), "range_mid"


def map_label(pct: float) -> int:
    """Same tiers as eval/gcbd_benchmark.map_labels."""
    if pct < 1.0:
        return 0
    if pct < 50.0:
        return 1
    return 2


def load_donors(gcbd_path: str, rc_path: str) -> pd.DataFrame:
    """Donor thermal pool with join keys (lat2, lon2, year, month)."""
    g = pd.read_csv(gcbd_path, low_memory=False)
    g["donor"] = "gcbd"
    g["key_month"] = pd.to_numeric(g["Date_Month"], errors="coerce")
    r = pd.read_csv(rc_path, low_memory=False)
    r = r.rename(columns=RC_RENAME).drop(columns=list(RC_DROP & set(r.columns)))
    r["donor"] = "reefcheck_cortad"
    r["Date"] = pd.to_datetime(r["Date"], format="mixed", errors="coerce")
    r["key_month"] = r["Date"].dt.month
    r["Date_Year"] = pd.to_numeric(r["Date_Year"], errors="coerce")
    keep = [c for c in g.columns if c in r.columns or c == "donor"]
    pool = pd.concat([g[keep].assign(_src=0),
                      r[[c for c in keep if c in r.columns]].assign(_src=1)],
                     ignore_index=True, sort=False)
    pool["lat2"] = pd.to_numeric(pool["Latitude_Degrees"],
                                 errors="coerce").round(2)
    pool["lon2"] = pd.to_numeric(pool["Longitude_Degrees"],
                                 errors="coerce").round(2)
    pool["yr"] = pd.to_numeric(pool["Date_Year"], errors="coerce")
    return pool.dropna(subset=["lat2", "lon2", "yr"])


def main() -> None:
    p = argparse.ArgumentParser(description="NCEI offline thermal join")
    p.add_argument("--ncei", default=NCEI_DEFAULT)
    p.add_argument("--gcbd", default=GCBD_DEFAULT)
    p.add_argument("--reefcheck", default=RC_DEFAULT)
    p.add_argument("--out", default="data/ncei_thermal_joined.csv")
    args = p.parse_args()

    ncei = pd.read_csv(args.ncei, low_memory=False, dtype=str)
    pool = load_donors(args.gcbd, args.reefcheck)

    ncei["lat2"] = pd.to_numeric(ncei["LATITUDE"], errors="coerce").round(2)
    ncei["lon2"] = pd.to_numeric(ncei["LONGITUDE"], errors="coerce").round(2)
    ncei["yr"] = pd.to_numeric(ncei["YEAR"], errors="coerce")
    ncei["mo"] = pd.to_numeric(ncei["MONTH"], errors="coerce")
    ncei["dy"] = pd.to_numeric(ncei["DAY"], errors="coerce")

    # Labels: main field, else MIN/MAX midpoint.
    pct, prov = [], []
    for _, row in ncei.iterrows():
        v, s = parse_percent(row["PERCENT_BLEACHED"])
        if v is None:
            lo, _ = parse_percent(row["MIN_PERCENT_BLEACHED"])
            hi, _ = parse_percent(row["MAX_PERCENT_BLEACHED"])
            if lo is not None and hi is not None:
                v, s = (lo + hi) / 2.0, "minmax_mid"
        pct.append(v)
        prov.append(s)
    ncei["_pct"] = pct
    ncei["label_source"] = prov
    ncei["Percent_Bleaching"] = ncei["_pct"]

    # Benchmark-ready identity columns.
    ncei["Date_Year"] = ncei["yr"]
    ncei["Date_Month"] = ncei["mo"].fillna(7).astype(int)
    ncei["Date_Day"] = ncei["dy"].fillna(1).astype(int)
    ncei["Date"] = pd.to_datetime(
        dict(year=ncei["Date_Year"].astype(int),
             month=ncei["Date_Month"], day=ncei["Date_Day"]),
        errors="coerce")
    ncei["Latitude_Degrees"] = pd.to_numeric(ncei["LATITUDE"],
                                             errors="coerce")
    ncei["Longitude_Degrees"] = pd.to_numeric(ncei["LONGITUDE"],
                                              errors="coerce")
    ncei["Site_Name"] = ncei["SITE_NAME"].fillna(ncei["LOCATION"])
    ncei["Realm_Name"] = ncei["CORAL_REGIONS"].map(REALM_MAP).fillna(
        ncei["CORAL_REGIONS"])

    key = ["lat2", "lon2", "yr"]
    merged = ncei.merge(pool, on=key, how="left", suffixes=("", "_donor"))
    merged["month_dist"] = (
        merged["key_month"] - merged["mo"]).abs().fillna(99.0)
    merged = merged.sort_values(
        ["RECORD_ID", "_src", "month_dist"]).drop_duplicates("RECORD_ID")
    matched = merged[merged["SSTA_DHW"].notna()].copy()
    matched["thermal_source"] = matched["donor"]
    # The benchmark maps NaN bleaching to tier 0, so unlabeled rows must
    # go here, not into the frame.
    n_unlabeled = int(matched["Percent_Bleaching"].isna().sum())
    matched = matched[matched["Percent_Bleaching"].notna()].copy()

    out_cols = ([c for c in pool.columns
                 if c not in ("donor", "key_month", "lat2", "lon2", "yr",
                              "_src")]
                + ["Percent_Bleaching", "Realm_Name", "Site_Name", "Date",
                   "Date_Year", "Date_Month", "Date_Day",
                   "Latitude_Degrees", "Longitude_Degrees",
                   "thermal_source", "label_source", "RECORD_ID",
                   "COUNTRY", "CORAL_REGIONS"])
    # Shared names (Date, Percent_Bleaching, coords) resolve to the NCEI
    # side; donor copies carry the _donor suffix. Dedupe keeps the first.
    out_cols = list(dict.fromkeys(out_cols))
    out = matched[out_cols].reset_index(drop=True)
    out.to_csv(args.out, index=False)
    rep = {
        "n_ncei": int(len(ncei)),
        "n_joined": int(len(out)),
        "n_dropped_unlabeled": n_unlabeled,
        "coverage": round(float(len(out)) / max(len(ncei), 1), 4),
        "label_source": out["label_source"].value_counts().to_dict(),
        "thermal_source": out["thermal_source"].value_counts().to_dict(),
        "realm": out["Realm_Name"].value_counts().to_dict(),
        "year_min": float(out["Date_Year"].min()),
        "year_max": float(out["Date_Year"].max()),
        "label_tiers": pd.Series(
            [map_label(float(v)) for v in out["Percent_Bleaching"]]
        ).value_counts().to_dict(),
    }
    print(json.dumps(rep, indent=2, default=str))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
