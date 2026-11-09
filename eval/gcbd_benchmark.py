#!/usr/bin/env python3
"""Real-world benchmark: GCBD bleaching outcomes vs DHW-only baseline.

Trains the LightGBM 3-class head on CoRTAD thermal covariates with
REAL bleaching labels (Percent_Bleaching 0-100 -> 0/1/2), evaluates with
a stratified split, leave-one-realm-out CV, and calibration, and compares
against the NOAA DHW>=4 / DHW>=8 baseline on the same records.

Usage:
    python eval/gcbd_benchmark.py [--csv data/gcbd_bleaching_environmental.csv]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

LABEL_COL = "label"
DHW_COL = "SSTA_DHW"
REGION_COL = "Realm_Name"
DROP_COLS = {"Site_ID", "Sample_ID", "Data_Source", "Reef_ID", "Ocean_Name",
             "Country_Name", "State_Island_Province_Name", "City_Town_Name",
             "Site_Name", "Realm_Name", "Ecoregion_Name", "Substrate_Name",
             "Bleaching_Level", "Percent_Bleaching", "Date", "Date_Day",
             "Date_Month", "Date_Year", "Site_Comments", "Sample_Comments",
             "Bleaching_Comments"}


def map_labels(pct: pd.Series) -> pd.Series:
    """Percent bleached -> 0 none (<1), 1 moderate (<50), 2 severe."""
    v = pd.to_numeric(pct, errors="coerce").fillna(0.0)
    return pd.Series(np.where(v < 1.0, 0, np.where(v < 50.0, 1, 2)),
                     index=pct.index, dtype=int)


def dhw_baseline_predict(dhw: np.ndarray) -> np.ndarray:
    """NOAA tiers: DHW>=8 severe, >=4 watch/warning, else none."""
    dhw = np.asarray(dhw, dtype=float)
    return np.where(dhw >= 8.0, 2, np.where(dhw >= 4.0, 1, 0))


GEO_COLS = {"Latitude_Degrees", "Longitude_Degrees"}


def load_gcbd(path: str, subsample: int | None = None,
              seed: int = 0, drop_geo: bool = False
              ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Feature frame, labels, region series from the GCBD CSV.

    drop_geo removes raw coordinates so skill must come from thermal
    features (matching deployment, where the pixel model never sees
    lat/lon).
    """
    df = pd.read_csv(path, low_memory=False)
    if subsample is not None:
        df = df.sample(min(subsample, len(df)),
                       random_state=seed).reset_index(drop=True)
    y = map_labels(df["Percent_Bleaching"])
    regions = df[REGION_COL].fillna("unknown").astype(str)
    feat_cols = [c for c in df.columns if c not in DROP_COLS
                 and (not drop_geo or c not in GEO_COLS)]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, y, regions


def run_benchmark(csv_path: str, subsample: int | None = None,
                  n_estimators: int = 400, seed: int = 0,
                  drop_geo: bool = False) -> dict:
    """Full benchmark: stratified test + LORO by realm + calibration."""
    import lightgbm as lgb
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 roc_auc_score)
    from sklearn.model_selection import train_test_split
    from eval.severity_leadtime import (expected_calibration_error,
                                        fit_temperature,
                                        leave_one_region_out)
    X, y, regions = load_gcbd(csv_path, subsample, seed, drop_geo)
    rep_drop_geo = bool(drop_geo)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=3,
                             n_estimators=n_estimators, learning_rate=0.05,
                             num_leaves=63, min_child_samples=40,
                             subsample=0.9, colsample_bytree=0.9,
                             reg_lambda=5.0, class_weight="balanced",
                             verbose=-1, random_state=seed)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    pred = proba.argmax(axis=1)
    per_class = {}
    for c in range(3):
        b = (yte.to_numpy() == c).astype(int)
        per_class[str(c)] = {
            "average_precision": float(average_precision_score(b, proba[:, c])),
            "support": int(b.sum())}
    base = dhw_baseline_predict(Xte[DHW_COL].to_numpy())
    base_acc = float(accuracy_score(yte, base))
    cal = fit_temperature(yte.to_numpy(), proba)
    ece = expected_calibration_error(yte.to_numpy(), proba[:, 2])["ece"]
    gain = clf.booster_.feature_importance(importance_type="gain")
    top = sorted(zip(X.columns, gain), key=lambda kv: -kv[1])[:8]
    return {
        "n": int(len(X)), "n_features": int(X.shape[1]),
        "drop_geo": rep_drop_geo,
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "roc_auc_ovr": round(float(roc_auc_score(yte, proba,
                                                 multi_class="ovr")), 4),
        "per_class": per_class,
        "dhw_baseline_accuracy": round(base_acc, 4),
        "loro_by_realm": leave_one_region_out(
            X.to_numpy(), y.to_numpy(), regions.to_numpy(),
            seed=seed, n_estimators=n_estimators),
        "ece": ece, "temperature_fit": cal,
        "top_gain_features": [(k, round(float(g), 1)) for k, g in top],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="GCBD real-world benchmark")
    p.add_argument("--csv", default="data/gcbd_bleaching_environmental.csv")
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--drop-geo", action="store_true",
                   help="drop raw lat/lon coordinates (thermal-only skill)")
    p.add_argument("--out", default="eval/gcbd_benchmark.json")
    args = p.parse_args()
    rep = run_benchmark(args.csv, args.subsample, drop_geo=args.drop_geo)
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("loro_by_realm", "temperature_fit")},
                     indent=2))
    print("LORO mean acc:", rep["loro_by_realm"]["mean_accuracy"],
          "| DHW baseline:", rep["dhw_baseline_accuracy"],
          "| model:", rep["accuracy"])
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
