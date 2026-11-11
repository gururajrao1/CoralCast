#!/usr/bin/env python3
"""Item 7: Severity / lead-time evaluation harness.

Reframes scoring as the operational product: calibration (ECE + reliability
table, optional PNG), hit-rate / false-alarm-ratio by lead week and region,
and BAA-tier mapping (Warning / Alert 1 / Alert 2) so a 30-day
none/moderate/severe forecast can be compared against the 4-month outlook
benchmark style.

Usage:
    python eval/severity_leadtime.py --demo
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

BAA_TIERS = {0: "No Stress", 1: "Warning", 2: "Alert Level 1",
             3: "Alert Level 2"}


def baa_tier(dhw: np.ndarray, hotspot: np.ndarray) -> np.ndarray:
    """NOAA tiers from DHW + same-day HotSpot (Warning needs HotSpot>=1)."""
    dhw = np.asarray(dhw, dtype=float)
    hotspot = np.asarray(hotspot, dtype=float)
    tier = np.where(dhw >= 8.0, 3, np.where(dhw >= 4.0, 2, 0))
    warn = (tier == 0) & (hotspot >= 1.0)
    tier[warn] = 1
    return tier


def fit_temperature(y_true: np.ndarray, proba: np.ndarray,
                    lo: float = 0.05, hi: float = 10.0,
                    iters: int = 60) -> dict:
    """Single temperature T scaling multiclass probabilities (NLL, golden search).

    Returns T and NLL before/after, fit on the given (held-out) split.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-9, 1.0)
    logits = np.log(proba)
    idx = (np.arange(len(y_true)), y_true)

    def nll(t: float) -> float:
        e = np.exp(logits / t)
        return float(-np.log((e / e.sum(axis=1, keepdims=True))[idx]).mean())
    gr = (5 ** 0.5 - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        if nll(c) < nll(d):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    t = (a + b) / 2
    return {"temperature": round(float(t), 4),
            "nll_before": round(nll(1.0), 4),
            "nll_after": round(nll(t), 4)}


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale probabilities by a fitted temperature."""
    proba = np.clip(np.asarray(proba, dtype=float), 1e-9, 1.0)
    logits = np.log(proba) / max(float(temperature), 1e-6)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature_cv(y_true: np.ndarray, proba: np.ndarray,
                       n_splits: int = 3, **kw) -> dict:
    """R6: cross-fit temperature scaling (fit T on K-1 folds, eval NLL out).

    Unlike fit_temperature (in-sample), the reported nll_after is the mean
    out-of-fold NLL, and `temperature` is the mean fold-T applied to new
    data. Deterministic for a fixed seed kwarg.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    seed = int(kw.pop("seed", 0))
    nlls, ts = [], []
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=seed).split(proba, y_true):
        fold = fit_temperature(y_true[tr], proba[tr], **kw)
        ts.append(fold["temperature"])
        e = np.exp(np.log(np.clip(proba[te], 1e-9, 1.0))
                   / max(float(fold["temperature"]), 1e-6))
        e = e / e.sum(axis=1, keepdims=True)
        nlls.append(float(-np.log(e[np.arange(len(te)),
                                    y_true[te]]).mean()))
    base = float(-np.log(np.clip(proba[np.arange(len(y_true)), y_true],
                                 1e-9, 1.0)).mean())
    return {"temperature": round(float(np.mean(ts)), 4),
            "temperature_std": round(float(np.std(ts)), 4),
            "nll_before": round(base, 4),
            "nll_after": round(float(np.mean(nlls)), 4),
            "n_splits": n_splits, "cross_fit": True}


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Multiclass Brier score (mean squared prob error) + per-class."""
    import numpy as np
    y_true = np.asarray(y_true, dtype=int)
    proba = np.clip(np.asarray(proba, dtype=float), 0, 1)
    n_class = proba.shape[1]
    onehot = np.eye(n_class)[y_true]
    per = {str(c): round(float(np.mean((proba[:, c]
                                        - (y_true == c)) ** 2)), 4)
           for c in range(n_class)}
    return {"brier": round(float(np.mean((proba - onehot) ** 2)), 4),
            "per_class": per}


def brier_skill_score(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """BSS vs a climatology (class-frequency) reference forecast.

    Positive = better than predicting base rates; 1 = perfect.
    """
    import numpy as np
    y_true = np.asarray(y_true, dtype=int)
    proba = np.clip(np.asarray(proba, dtype=float), 0, 1)
    freq = np.bincount(y_true, minlength=proba.shape[1]) / len(y_true)
    ref = np.tile(freq, (len(y_true), 1))
    bs = brier_score(y_true, proba)["brier"]
    bs_ref = brier_score(y_true, ref)["brier"]
    return {"brier": bs, "brier_ref_climatology": bs_ref,
            "bss_vs_climatology": round(float(1 - bs / max(bs_ref, 1e-12)),
                                        4)}


def expected_calibration_error(y_true: np.ndarray, p_severe: np.ndarray,
                               n_bins: int = 10) -> dict:
    """ECE + reliability table for P(severe)."""
    y = np.asarray(y_true) == 2
    p = np.clip(np.asarray(p_severe, dtype=float), 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p <= hi)
        if m.sum() == 0:
            continue
        acc, conf = float(y[m].mean()), float(p[m].mean())
        ece += abs(acc - conf) * m.sum() / len(p)
        table.append({"bin": f"({lo:.1f},{hi:.1f}]", "n": int(m.sum()),
                      "mean_p": round(conf, 3), "frac_severe": round(acc, 3)})
    return {"ece": round(float(ece), 4), "reliability": table}


def hit_falsealarm_by_lead(y_true: np.ndarray, y_pred: np.ndarray,
                           leads_wk: np.ndarray,
                           regions: np.ndarray | None = None) -> pd.DataFrame:
    """Hit-rate (recall of severe) and false-alarm-ratio by lead week."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    leads_wk = np.asarray(leads_wk)
    regions = (np.asarray(regions) if regions is not None
               else np.array(["all"] * len(y_true)))
    rows = []
    for (lead, region), idx in pd.DataFrame(
            {"l": leads_wk, "r": regions}).groupby(["l", "r"]).groups.items():
        i = np.asarray(list(idx))
        tp = int(((y_pred[i] == 2) & (y_true[i] == 2)).sum())
        fn = int(((y_pred[i] != 2) & (y_true[i] == 2)).sum())
        fp = int(((y_pred[i] == 2) & (y_true[i] != 2)).sum())
        rows.append({"lead_week": int(lead), "region": str(region),
                     "n": int(len(i)),
                     "hit_rate": round(tp / max(tp + fn, 1), 3),
                     "false_alarm_ratio": round(fp / max(tp + fp, 1), 3)})
    return pd.DataFrame(rows).sort_values(["lead_week", "region"])


def outlook_style_summary(y_true, y_pred, proba_severe, leads_wk,
                          regions=None) -> dict:
    """One JSON-ready report: ECE, by-lead table, BAA-tier confusion."""
    from sklearn.metrics import confusion_matrix
    return {
        "calibration": expected_calibration_error(y_true, proba_severe),
        "by_lead": hit_falsealarm_by_lead(
            y_true, y_pred, leads_wk, regions).to_dict("records"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def leave_one_region_out(X, y, regions, seed: int = 0,
                         n_estimators: int = 400) -> dict:
    """Leave-one-region-out CV: train on all-but-one spatial region.

    Returns per-held-out-region accuracy + macro ROC-AUC and the mean/std,
    the honest spatial-generalization counterpart to the stratified split.
    Uses the same LightGBM head (with monotone constraints when the
    FEATURE_NAMES match is available).
    """
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, roc_auc_score
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    regions = np.asarray(regions)
    try:
        from coral_bleaching_forecaster import (FEATURE_NAMES, N_CLASSES,
                                                monotone_constraints)
        mono = (monotone_constraints() if len(FEATURE_NAMES) == X.shape[1]
                else None)
        n_class = N_CLASSES
    except ImportError:
        mono, n_class = None, int(y.max() + 1)
    rows = []
    for held in sorted(set(regions.tolist())):
        tr = regions != held
        if len(set(y[tr].tolist())) < 2 or len(set(y[~tr].tolist())) < 2:
            continue
        kw = {} if mono is None else {"monotone_constraints": mono}
        clf = lgb.LGBMClassifier(objective="multiclass", num_class=n_class,
                                 n_estimators=n_estimators,
                                 learning_rate=0.05, num_leaves=63,
                                 min_child_samples=40, subsample=0.9,
                                 colsample_bytree=0.9, reg_lambda=5.0,
                                 class_weight="balanced", verbose=-1,
                                 random_state=seed, **kw)
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[~tr])
        pred = proba.argmax(axis=1)
        row = {"held_out": str(held), "n_test": int((~tr).sum()),
               "accuracy": round(float(accuracy_score(y[~tr], pred)), 4),
               "roc_auc_ovr": None}
        if set(y[~tr].tolist()) == set(range(n_class)):
            row["roc_auc_ovr"] = round(float(roc_auc_score(
                y[~tr], proba, multi_class="ovr")), 4)
        rows.append(row)
    accs = [r["accuracy"] for r in rows]
    aucs = [r["roc_auc_ovr"] for r in rows if r["roc_auc_ovr"] is not None]
    return {"per_region": rows,
            "mean_accuracy": round(float(np.mean(accs)), 4) if accs else None,
            "std_accuracy": round(float(np.std(accs)), 4) if accs else None,
            "mean_roc_auc": round(float(np.mean(aucs)), 4) if aucs else None,
            "n_roc_folds": len(aucs)}


def plot_reliability(report: dict, path: str = "eval/reliability.png") -> str:
    """Reliability diagram PNG (Agg backend; optional eye-candy)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tab = report["calibration"]["reliability"]
    xs = [r["mean_p"] for r in tab]
    ys = [r["frac_severe"] for r in tab]
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.plot(xs, ys, "o-", label=f"model (ECE={report['calibration']['ece']})")
    ax.set_xlabel("mean P(severe)")
    ax.set_ylabel("fraction severe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def demo(seed: int = 0) -> dict:
    """Synthetic 4-week-lead predictions; checks table shape + ECE range."""
    rng = np.random.default_rng(seed)
    n = 2000
    y = rng.integers(0, 3, n, endpoint=False)
    y = np.where(rng.random(n) < 0.4, 2, y)          # enrich severe
    p = np.clip((y == 2) * 0.7 + rng.standard_normal(n) * 0.2, 0.01, 0.99)
    pred = np.where(p > 0.66, 2, np.where(p > 0.33, 1, 0))
    leads = rng.integers(1, 5, n)
    regions = np.where(rng.random(n) < 0.5, "west", "east")
    rep = outlook_style_summary(y, pred, p, leads, regions)
    print({k: (v if k != "by_lead" else f"{len(v)} rows") for k, v in rep.items()})
    assert 0.0 <= rep["calibration"]["ece"] <= 1.0
    assert len(rep["by_lead"]) == 8  # 4 leads x 2 regions
    return rep


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    demo()
