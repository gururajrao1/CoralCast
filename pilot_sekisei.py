#!/usr/bin/env python3
"""Sekisei Lagoon real-data pilot: labels + Sentinel-2 stack + MUR SST.

1. Builds data/pilot_labels.csv from ReefCheck_CoRTAD.csv (Sekisei bbox
   sites plus all bleached Japan rows for class balance).
2. Downloads 10 m B2/B3/B4/B8 windows for 2022-08-25 + 2022-09-29
   (tile 51RWH) into data/pilot_stack/ as {BAND}_{YYYYMMDD}.tif.
3. Fetches daily MUR SST (2020-01 to 2022-10) at the lagoon centre into
   data/pilot_sst.csv and prints the MMM to pass as --mmm.

Then run:
    python coral_bleaching_forecaster.py --blue data/pilot_stack/B2_20220929.tif ... --sst_csv data/pilot_sst.csv --mmm <MMM> --stack_dir data/pilot_stack --temporal sits --calibrate --eval_report --labels_csv data/pilot_labels.csv
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

OUT_STACK = Path("data/pilot_stack")
LON0, LAT0 = 123.71, 24.31  # Sekisei Lagoon centre
WIN = (123.65, 24.30, 123.80, 24.42)  # lon_min, lat_min, lon_max, lat_max

SCENES = {
    "20220825": ("S2A_51RWH_20220825_0_L2A", "2022/8"),
    "20220929": ("S2B_51RWH_20220929_0_L2A", "2022/9"),
}
BANDS = {"B02": "B2", "B03": "B3", "B04": "B4", "B08": "B8"}


def build_labels() -> pd.DataFrame:
    d = pd.read_csv("data/ReefCheck_CoRTAD.csv", low_memory=False)
    j = d[d["Country"].str.contains("Japan", na=False)].copy()
    sek = j[(j["Longitude.Degrees"].between(123.5, 124.5))
            & (j["Latitude.Degrees"].between(23.5, 25.0))]
    pos = j[pd.to_numeric(j["Average_bleaching"],
                          errors="coerce").fillna(0) > 0]
    use = pd.concat([sek, pos]).drop_duplicates().reset_index(drop=True)
    sev = pd.to_numeric(use["Average_bleaching"],
                        errors="coerce").fillna(0).clip(0, 1)
    out = pd.DataFrame({
        "site": use["Reef.Name"].fillna("unknown"),
        "lat": use["Latitude.Degrees"],
        "lon": use["Longitude.Degrees"],
        "date": pd.to_datetime(use["Date"], format="%d-%b-%y",
                               errors="coerce"),
        "label": np.where(sev < 0.1, 0, np.where(sev < 0.5, 1, 2)),
        "region": np.where(use["Latitude.Degrees"].abs() < 15, "tropics",
                           "subtropics"),
        "dhw": pd.to_numeric(use["SSTA_DHW"], errors="coerce").fillna(0),
        "hotspot": np.maximum(
            0.0, pd.to_numeric(use["Temperature_Mean"], errors="coerce")
            - pd.to_numeric(use["ClimSST"], errors="coerce")).fillna(0),
        "sst": pd.to_numeric(use["Temperature_Mean"], errors="coerce"),
    }).dropna(subset=["lat", "lon", "date", "sst"]).reset_index(drop=True)
    out.to_csv("data/pilot_labels.csv", index=False)
    print(f"pilot labels: {len(out)} rows, "
          f"classes={out['label'].value_counts().to_dict()}")
    return out


def download_stack() -> None:
    OUT_STACK.mkdir(parents=True, exist_ok=True)
    for datestr, (scene, ym) in SCENES.items():
        for cog, short in BANDS.items():
            href = (f"https://sentinel-cogs.s3.us-west-2.amazonaws.com/"
                    f"sentinel-s2-l2a-cogs/51/R/WH/{ym}/{scene}/{cog}.tif")
            out = OUT_STACK / f"{short}_{datestr}.tif"
            if out.exists():
                print(f"exists: {out}")
                continue
            with rasterio.open(f"/vsicurl/{href}") as src:
                b = transform_bounds("EPSG:4326", src.crs, *WIN)
                w = from_bounds(*b, src.transform)
                w = w.round_offsets().round_lengths()
                w = w.intersection(
                    from_bounds(*src.bounds, src.transform))
                arr = src.read(1, window=w)
                prof = dict(src.profile, height=w.height, width=w.width,
                            transform=rasterio.windows.transform(w,
                                                                 src.transform),
                            compress="deflate", tiled=True)
            with rasterio.open(out, "w", **prof) as dst:
                dst.write(arr, 1)
            print(f"wrote {out} {arr.shape}")


def fetch_sst() -> float:
    import time
    import urllib.request
    frames = []
    for start, end in [("2020-01-01", "2020-12-31"),
                       ("2021-01-01", "2021-12-31"),
                       ("2022-01-01", "2022-10-31")]:
        q = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.csv"
             f"?analysed_sst%5B({start}T09%3A00%3A00Z)%3A({end}T09%3A00%3A00Z)%5D"
             "%5B(24.31)%3A(24.31)%5D%5B(123.71)%3A(123.71)%5D")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(q, timeout=180) as r:
                    raw = r.read().decode()
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"SST fetch failed for {start}..{end}")
        frames.append(pd.read_csv(io.StringIO(raw), skiprows=[1]))
        time.sleep(2)
    df = pd.concat(frames, ignore_index=True)
    df.columns = ["date", "lat", "lon", "sst_c"]
    df["date"] = pd.to_datetime(df["date"])
    s = df.groupby("date")["sst_c"].mean().reset_index()
    s.to_csv("data/pilot_sst.csv", index=False)
    s["month"] = s["date"].dt.month
    mmm = float(s.groupby("month")["sst_c"].mean().max())
    print(f"pilot SST: {len(s)} days, MMM={mmm:.2f} C")
    return mmm


if __name__ == "__main__":
    build_labels()
    download_stack()
    mmm = fetch_sst()
    print(f"--mmm {mmm:.2f}")
