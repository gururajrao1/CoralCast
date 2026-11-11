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

# R1: honest-validation helpers (Mestry 2026: random-split skill collapses
# under site-grouped / spatial-block / forward-temporal splits, and
# ReefCheck carries ~34.5% exact-duplicate records).
SITE_COLS = ("Site_ID", "Reef_ID", "Site_Name")
LATLON_COLS = ("Latitude_Degrees", "Longitude_Degrees")


def dedupe_events(df: "pd.DataFrame") -> "pd.DataFrame":
    """Drop exact-duplicate survey records (same site+date+label+thermal).

    Returns a reset copy; the dropped count is reported by run_benchmark.
    """
    import pandas as pd
    key = [c for c in ("Site_ID", "Reef_ID", "Site_Name", "Date",
                       "Percent_Bleaching", "SSTA_DHW", "TSA_DHW")
           if c in df.columns]
    if not key:
        return df.reset_index(drop=True)
    return df.drop_duplicates(subset=key).reset_index(drop=True)


def _site_groups(df: "pd.DataFrame") -> "np.ndarray":
    """Best-available site identifier for grouped splits."""
    import numpy as np
    for c in SITE_COLS:
        if c in df.columns:
            g = df[c].fillna("unknown").astype(str).to_numpy()
            if len(set(g)) > 1:
                return g
    return np.array(["all"] * len(df))


def grouped_split_scores(X, y, groups, seed: int = 0,
                         n_estimators: int = 400) -> dict:
    """Hold out whole sites (GroupShuffleSplit 75/25): no site in both."""
    import lightgbm as lgb
    import numpy as np
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(gss.split(X, y, groups))
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=3,
                             n_estimators=n_estimators, learning_rate=0.05,
                             num_leaves=63, min_child_samples=40,
                             subsample=0.9, colsample_bytree=0.9,
                             reg_lambda=5.0, class_weight="balanced",
                             verbose=-1, random_state=seed)
    clf.fit(np.asarray(X)[tr], np.asarray(y)[tr])
    pred = clf.predict(np.asarray(X)[te])
    return {"accuracy": round(float(accuracy_score(
        np.asarray(y)[te], pred)), 4),
        "n_test": int(len(te)),
        "n_groups_test": int(len(set(np.asarray(groups)[te])))}


def spatial_block_scores(X, y, lat, lon, seed: int = 0, block_deg: float = 1.0,
                         n_estimators: int = 400,
                         max_blocks: int = 25) -> dict:
    """Block CV over rounded lat/lon cells (default 1-deg; try 5-deg).

    Whole spatial blocks are held out, breaking the autocorrelation that
    makes random splits optimistic (blockCV / Roberts-Valavi practice).
    At most `max_blocks` held-out blocks (seeded sample) are evaluated —
    a 1-deg global grid has ~400 occupied cells, far too many for a full
    leave-one-block-out; the sample count and total are both reported.
    """
    import lightgbm as lgb
    import numpy as np
    from sklearn.metrics import accuracy_score
    lat = pd.to_numeric(pd.Series(np.asarray(lat).ravel()),
                          errors="coerce").fillna(0.0).to_numpy()
    lon = pd.to_numeric(pd.Series(np.asarray(lon).ravel()),
                          errors="coerce").fillna(0.0).to_numpy()
    blocks = (np.floor(lat / block_deg).astype(int).astype(str) + "x"
              + np.floor(lon / block_deg).astype(int).astype(str))
    uniq = sorted(set(blocks.tolist()))
    if len(uniq) > max_blocks:
        rng = np.random.default_rng(seed)
        uniq = sorted(rng.choice(uniq, size=max_blocks,
                                 replace=False).tolist())
    accs = []
    for held in uniq:
        tr, te = blocks != held, blocks == held
        if len(set(np.asarray(y)[tr].tolist())) < 2:
            continue
        clf = lgb.LGBMClassifier(objective="multiclass", num_class=3,
                                 n_estimators=n_estimators,
                                 learning_rate=0.05, num_leaves=63,
                                 min_child_samples=40, subsample=0.9,
                                 colsample_bytree=0.9, reg_lambda=5.0,
                                 class_weight="balanced", verbose=-1,
                                 random_state=seed)
        clf.fit(np.asarray(X)[tr], np.asarray(y)[tr])
        accs.append(float(accuracy_score(np.asarray(y)[te],
                                        clf.predict(np.asarray(X)[te]))))
    return {"mean_accuracy": round(float(np.mean(accs)), 4) if accs else None,
            "std_accuracy": round(float(np.std(accs)), 4) if accs else None,
            "n_blocks_eval": int(len(uniq)),
            "n_blocks_total": int(len(set(blocks.tolist()))),
            "n_blocks": int(len(set(blocks.tolist()))),
            "block_deg": float(block_deg)}


def forward_temporal_scores(X, y, dates, seed: int = 0,
                            n_estimators: int = 400) -> dict:
    """Train on the past, test on the future (TimeSeriesSplit, 3 folds)."""
    import lightgbm as lgb
    import numpy as np
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import TimeSeriesSplit
    order = np.argsort(pd.to_datetime(dates, errors="coerce").to_numpy())
    Xs, ys = np.asarray(X)[order], np.asarray(y)[order]
    accs = []
    for tr, te in TimeSeriesSplit(n_splits=3).split(Xs):
        if len(set(ys[tr].tolist())) < 2 or len(set(ys[te].tolist())) < 2:
            continue
        clf = lgb.LGBMClassifier(objective="multiclass", num_class=3,
                                 n_estimators=n_estimators,
                                 learning_rate=0.05, num_leaves=63,
                                 min_child_samples=40, subsample=0.9,
                                 colsample_bytree=0.9, reg_lambda=5.0,
                                 class_weight="balanced", verbose=-1,
                                 random_state=seed)
        clf.fit(Xs[tr], ys[tr])
        accs.append(float(accuracy_score(ys[te], clf.predict(Xs[te]))))
    return {"mean_accuracy": round(float(np.mean(accs)), 4) if accs else None,
            "n_folds": int(len(accs))}


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


def load_gcbd_frame(df: "pd.DataFrame", drop_geo: bool = False
                    ) -> tuple["pd.DataFrame", "pd.Series", "pd.Series"]:
    """Feature frame/labels/regions from an already-loaded GCBD frame."""
    y = map_labels(df["Percent_Bleaching"])
    regions = df[REGION_COL].fillna("unknown").astype(str)
    feat_cols = [c for c in df.columns if c not in DROP_COLS
                 and (not drop_geo or c not in GEO_COLS)]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, y, regions


def _fit_eval_split(X, y, n_estimators: int = 400,
                    seed: int = 0) -> dict:
    """One stratified 75/25 LightGBM fit; returns clf/proba/pred/scores."""
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.25, random_state=seed,
                              stratify=np.asarray(y))
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=3,
                             n_estimators=n_estimators, learning_rate=0.05,
                             num_leaves=63, min_child_samples=40,
                             subsample=0.9, colsample_bytree=0.9,
                             reg_lambda=5.0, class_weight="balanced",
                             verbose=-1, random_state=seed)
    clf.fit(np.asarray(X)[tr], np.asarray(y)[tr])
    proba = clf.predict_proba(np.asarray(X)[te])
    pred = proba.argmax(axis=1)
    yte = np.asarray(y)[te]
    return {"clf": clf, "proba": proba, "pred": pred,
            "yte": pd.Series(yte, index=te), "te_idx": te,
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "roc_auc_ovr": round(float(roc_auc_score(yte, proba,
                                                    multi_class="ovr")), 4)}


def majority_baseline_accuracy(y_true: np.ndarray) -> float:
    """R2 null: always predict the majority class (climatology rate)."""
    import numpy as np
    y_true = np.asarray(y_true)
    maj = np.bincount(y_true).argmax()
    return float((y_true == maj).mean())


def run_per_basin(X, y, regions, seed: int = 0, n_estimators: int = 400,
                  min_n: int = 50) -> dict:
    """R8: one model per realm (basin); within-basin 75/25 split.

    Realms with n < min_n are reported with accuracy None (too small to
    split honestly) instead of a noisy number.
    """
    import lightgbm as lgb
    import numpy as np
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    X, y, regions = np.asarray(X), np.asarray(y), np.asarray(regions)
    out = {}
    for basin in sorted(set(regions.tolist())):
        m = regions == basin
        if int(m.sum()) < min_n or len(set(y[m].tolist())) < 2:
            out[str(basin)] = {"n": int(m.sum()), "accuracy": None,
                               "note": f"n<{min_n} or single-class: skipped"}
            continue
        tr, te = train_test_split(np.where(m)[0], test_size=0.25,
                                  random_state=seed,
                                  stratify=y[m])
        clf = lgb.LGBMClassifier(objective="multiclass", num_class=3,
                                 n_estimators=n_estimators,
                                 learning_rate=0.05, num_leaves=63,
                                 min_child_samples=40, subsample=0.9,
                                 colsample_bytree=0.9, reg_lambda=5.0,
                                 class_weight="balanced", verbose=-1,
                                 random_state=seed)
        clf.fit(X[tr], y[tr])
        out[str(basin)] = {"n": int(m.sum()),
                           "accuracy": round(float(accuracy_score(
                               y[te], clf.predict(X[te]))), 4)}
    return out


def skill_by_severity(y_true: np.ndarray, y_pred: np.ndarray,
                      dhw: np.ndarray) -> dict:
    """R8 severity half of skill-vs-lead x severity: score on DHW subsets."""
    import numpy as np
    from sklearn.metrics import accuracy_score
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    dhw = np.asarray(dhw, dtype=float)
    out = {"overall_accuracy": round(float(accuracy_score(y_true,
                                                         y_pred)), 4)}
    for name, m in {"dhw_ge_4": dhw >= 4.0, "dhw_ge_8": dhw >= 8.0}.items():
        out[name] = {"n": int(m.sum()),
                     "accuracy": round(float(accuracy_score(y_true[m],
                                                            y_pred[m])), 4)
                     if m.sum() else None}
    return out


def add_covariate(df: "pd.DataFrame", csv: str | None,
                  on: tuple = ("Site_ID", "Date")) -> "pd.DataFrame":
    """R8: merge an external lead-time covariate (e.g. CFSv2/Outlook weekly
    bleaching probability) by site+date for the skill-vs-lead analysis.

    Returns the frame unchanged when csv is None. New columns become
    model features automatically (anything not in DROP_COLS).
    """
    if csv is None:
        return df
    import pandas as pd
    cov = pd.read_csv(csv, low_memory=False)
    keys = [k for k in on if k in df.columns and k in cov.columns]
    if not keys:
        raise ValueError(f"no shared join keys {on} in {csv}")
    return df.merge(cov, on=keys, how="left", suffixes=("", "_cov"))


def feature_parity_report(gcbd_cols: list) -> dict:
    """R5: which of the 37 deployment features exist in the GCBD frame.

    GCBD carries raw CoRTAD columns, not the forecaster's engineered names,
    so this documents the offline/online feature gap instead of assuming
    the benchmark validates the deployment set.
    """
    from coral_bleaching_forecaster import THERMAL_FEATURES
    have = [f for f in THERMAL_FEATURES if f in gcbd_cols]
    return {"n_deployment_thermal": len(THERMAL_FEATURES),
            "n_present_in_gcbd": len(have), "present": have,
            "note": "benchmark uses raw CoRTAD columns; overlap by exact "
                    "name is expected to be ~0"}


def run_benchmark(csv_path: str, subsample: int | None = None,
                  n_estimators: int = 400, seed: int = 0,
                  drop_geo: bool = False,
                  seeds: tuple = (0,), block_deg: float = 1.0,
                  dedupe: bool = True,
                  covariate_csv: str | None = None,
                  max_blocks: int = 25) -> dict:
    """Full benchmark: stratified test + LORO by realm + calibration.

    R1 honest validation: optional dedupe of repeat survey rows, seed
    sweep over `seeds` (mean/std beside the point estimate), site-grouped
    holdout, spatial-block CV, and forward-temporal CV. R6: cross-fit
    temperature + Brier/BSS (see severity_leadtime helpers).
    """
    import lightgbm as lgb
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 roc_auc_score)
    from sklearn.model_selection import train_test_split
    from eval.run_registry import new_run
    from eval.severity_leadtime import (brier_skill_score,
                                        expected_calibration_error,
                                        fit_temperature_cv,
                                        leave_one_region_out)
    raw = pd.read_csv(csv_path, low_memory=False)
    if subsample is not None:
        raw = raw.sample(min(subsample, len(raw)),
                         random_state=seed).reset_index(drop=True)
    n_raw = int(len(raw))
    if dedupe:
        raw = dedupe_events(raw)
    raw = add_covariate(raw, covariate_csv)
    X, y, regions = load_gcbd_frame(raw, drop_geo)
    rep_drop_geo = bool(drop_geo)
    # seed sweep (R1): identical stratified pipeline per seed; the primary
    # `seed` run supplies proba/pred for the detailed tables below.
    first = _fit_eval_split(X, y, n_estimators=n_estimators, seed=seed)
    sweep_acc, sweep_roc = [first["accuracy"]], [first["roc_auc_ovr"]]
    for sd in seeds:
        if sd == seed:
            continue
        r = _fit_eval_split(X, y, n_estimators=n_estimators, seed=sd)
        sweep_acc.append(r["accuracy"])
        sweep_roc.append(r["roc_auc_ovr"])
    clf, proba, pred = first["clf"], first["proba"], first["pred"]
    yte = first["yte"]
    per_class = {}
    for c in range(3):
        b = (yte.to_numpy() == c).astype(int)
        per_class[str(c)] = {
            "average_precision": float(average_precision_score(b, proba[:, c])),
            "support": int(b.sum())}
    base = dhw_baseline_predict(X.iloc[first["te_idx"]][DHW_COL].to_numpy())
    base_acc = float(accuracy_score(yte, base))
    maj_acc = majority_baseline_accuracy(yte.to_numpy())
    cal = fit_temperature_cv(yte.to_numpy(), proba)
    ece = expected_calibration_error(yte.to_numpy(), proba[:, 2])["ece"]
    bss = brier_skill_score(yte.to_numpy(), proba)
    gain = clf.booster_.feature_importance(importance_type="gain")
    top = sorted(zip(X.columns, gain), key=lambda kv: -kv[1])[:8]
    Xn, yn, reg = (X.to_numpy(), y.to_numpy(), regions.to_numpy())
    lat = raw["Latitude_Degrees"] if "Latitude_Degrees" in raw.columns else None
    lon = raw["Longitude_Degrees"] if "Longitude_Degrees" in raw.columns else None
    dates = raw["Date"] if "Date" in raw.columns else None
    return {
        "n": int(len(X)), "n_features": int(X.shape[1]),
        "drop_geo": rep_drop_geo, "covariate_csv": covariate_csv,
        "dedupe": {"n_raw": int(n_raw), "n_deduped": int(len(raw)),
                   "n_dropped": int(n_raw - len(raw))},
        "run": new_run(seed=seed),
        "accuracy": first["accuracy"],
        "roc_auc_ovr": first["roc_auc_ovr"],
        "seed_sweep": {"seeds": [int(s) for s in seeds],
                       "mean_accuracy": round(float(np.mean(sweep_acc)), 4),
                       "std_accuracy": round(float(np.std(sweep_acc)), 4),
                       "mean_roc_auc": round(float(np.mean(sweep_roc)), 4)},
        "per_class": per_class,
        "dhw_baseline_accuracy": round(base_acc, 4),
        "majority_baseline_accuracy": round(maj_acc, 4),
        "grouped_site_holdout": grouped_split_scores(
            Xn, yn, _site_groups(raw), seed=seed,
            n_estimators=n_estimators),
        "spatial_block_cv": spatial_block_scores(
            Xn, yn, lat, lon, seed=seed, block_deg=block_deg,
            n_estimators=n_estimators, max_blocks=max_blocks),
        "forward_temporal_cv": forward_temporal_scores(
            Xn, yn, dates, seed=seed, n_estimators=n_estimators),
        "per_basin": run_per_basin(Xn, yn, reg, seed=seed,
                                   n_estimators=n_estimators),
        "skill_by_severity": skill_by_severity(
            yte.to_numpy(), pred, X.iloc[first["te_idx"]][DHW_COL].to_numpy()),
        "feature_parity": feature_parity_report(list(X.columns)),
        "loro_by_realm": leave_one_region_out(
            Xn, yn, reg, seed=seed, n_estimators=n_estimators),
        "ece": ece, "temperature_fit": cal, "brier_skill": bss,
        "top_gain_features": [(k, round(float(g), 1)) for k, g in top],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="GCBD real-world benchmark")
    p.add_argument("--csv", default="data/gcbd_bleaching_environmental.csv")
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--drop-geo", action="store_true",
                   help="drop raw lat/lon coordinates (thermal-only skill)")
    p.add_argument("--out", default="eval/gcbd_benchmark.json")
    p.add_argument("--seeds", default="0",
                   help="comma-separated seed sweep, e.g. '0,1,2,3,4'")
    p.add_argument("--block-deg", type=float, default=1.0,
                   help="spatial-block size in degrees (try 5.0)")
    p.add_argument("--max-blocks", type=int, default=25,
                   help="held-out blocks sampled for block CV")
    p.add_argument("--no-dedupe", action="store_true",
                   help="skip duplicate-survey-row removal")
    p.add_argument("--covariate", default=None,
                   help="R8: external lead-time covariate CSV merged by "
                        "Site_ID+Date (e.g. CFSv2 outlook probabilities)")
    args = p.parse_args()
    from eval.run_registry import record
    rep = run_benchmark(args.csv, args.subsample, drop_geo=args.drop_geo,
                        seeds=tuple(int(s) for s in args.seeds.split(",")),
                        block_deg=args.block_deg,
                        dedupe=not args.no_dedupe,
                        covariate_csv=args.covariate,
                        max_blocks=args.max_blocks)
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("loro_by_realm", "temperature_fit",
                                   "per_basin")},
                     indent=2))
    print("LORO mean acc:", rep["loro_by_realm"]["mean_accuracy"],
          "| grouped-site:", rep["grouped_site_holdout"]["accuracy"],
          "| block:", rep["spatial_block_cv"]["mean_accuracy"],
          "| temporal:", rep["forward_temporal_cv"]["mean_accuracy"],
          "| DHW baseline:", rep["dhw_baseline_accuracy"],
          "| model:", rep["accuracy"])
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"Wrote {args.out} + {record(rep, 'gcbd_benchmark')}")


if __name__ == "__main__":
    main()
