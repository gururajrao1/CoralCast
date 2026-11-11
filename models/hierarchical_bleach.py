#!/usr/bin/env python3
"""Hierarchical (partial-pooling) bleaching classifier with cross-basin borrowing.

Method: empirical-Bayes shrunk per-realm blend (methods-spike rank #1,
``eval/gcbd_benchmark.py:246`` :func:`run_per_basin` has no pooling while
``coral_bleaching_forecaster.py:795`` :func:`train_per_basin_models` falls
back to the global head for small basins).

For each realm *b* with *n_b* training rows::

    P_test = w_b * P_basin + (1 - w_b) * P_global,  w_b = n_b / (n_b + kappa)

``kappa -> inf`` recovers the global head, ``kappa -> 0`` recovers
independent per-basin heads. Realms with ``n < min_n`` or a single class get
no head and use the global probabilities (same fallback convention as
``train_per_basin_models``); rows from realms unseen at fit time also use
the global head, so strict leave-one-realm-out accuracy is kappa-invariant
by construction (reported as such in the probe).

sklearn-compatible: ``fit(X, y, groups)`` / ``predict`` / ``predict_proba``
plus ``get_params``/``set_params`` via :class:`BaseEstimator`. ``groups`` is
the realm label vector (``eval/gcbd_benchmark.py:27`` ``REGION_COL``).
Any feature frame works, including one extended with
``eval/adaptive_features.py`` ``ADAPTIVE_COLS`` — pass the concatenated
matrix as ``X``; this module adds no features itself.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


def _lgbm(n_estimators: int, seed: int, monotone=None):
    """Benchmark-head LightGBM (eval/gcbd_benchmark.py:92, :167, :268)."""
    import lightgbm as lgb

    kw = {}
    if monotone is not None:
        kw["monotone_constraints"] = monotone
    return lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=n_estimators, learning_rate=0.05,
        num_leaves=63, min_child_samples=40,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=5.0, class_weight="balanced",
        verbose=-1, random_state=seed, **kw)


class HierarchicalBleachClassifier(BaseEstimator, ClassifierMixin):
    """Shrunk per-realm LightGBM blend with a global head.

    Parameters
    ----------
    kappa : float
        Shrinkage strength (effective prior sample size). Per-realm
        weight ``w_b = n_b / (n_b + kappa)``.
    n_estimators : int
        Trees per head (benchmark default 400; probes use <= 50).
    min_n : int
        Minimum rows for a realm to get its own head (benchmark
        convention: ``eval/gcbd_benchmark.py:246`` ``min_n=50``).
    seed : int
        Random state for every head.
    """

    _estimator_type = "classifier"

    def __init__(self, kappa: float = 100.0, n_estimators: int = 400,
                 min_n: int = 50, seed: int = 0):
        self.kappa = kappa
        self.n_estimators = n_estimators
        self.min_n = min_n
        self.seed = seed

    def fit(self, X, y, groups=None):
        """Fit the global head plus one head per eligible realm."""
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=int)
        if ya.ndim != 1 or not np.issubdtype(ya.dtype, np.integer):
            raise ValueError("y must be a 1d integer label vector")
        if groups is None:
            ga = np.array(["global"] * len(ya))
        else:
            ga = np.asarray(groups).ravel().astype(str)
        self.classes_ = np.unique(ya)
        self.n_features_in_ = Xa.shape[1]
        self.global_model_ = _lgbm(self.n_estimators, self.seed)
        self.global_model_.fit(Xa, ya)
        self.basin_models_ = {}
        self.basin_n_ = {}
        for b in sorted(set(ga.tolist())):
            m = ga == b
            n = int(m.sum())
            self.basin_n_[str(b)] = n
            if n < self.min_n or len(set(ya[m].tolist())) < 2:
                self.basin_models_[str(b)] = None
                continue
            clf = _lgbm(self.n_estimators, self.seed)
            clf.fit(Xa[m], ya[m])
            self.basin_models_[str(b)] = clf
        # Realm weight w_b = n_b / (n_b + kappa); ineligible realms get 0.
        self.basin_weight_ = {
            b: (n / (n + float(self.kappa))
                if self.basin_models_[b] is not None else 0.0)
            for b, n in self.basin_n_.items()
        }
        return self

    def predict_proba(self, X, groups=None):
        """Shrunk blend; unknown/ineligible realms use the global head."""
        check_is_fitted(self, ["global_model_", "basin_models_"])
        Xa = np.asarray(X, dtype=float)
        p_global = np.asarray(self.global_model_.predict_proba(Xa))
        if groups is None:
            return p_global
        ga = np.asarray(groups).ravel().astype(str)
        out = p_global.copy()
        for b, clf in self.basin_models_.items():
            if clf is None:
                continue
            m = ga == b
            if not m.sum():
                continue
            w = self.basin_weight_[b]
            out[m] = w * np.asarray(clf.predict_proba(Xa[m])) \
                + (1.0 - w) * p_global[m]
        return out

    def predict(self, X, groups=None):
        """Argmax of the shrunk-blend probabilities."""
        return np.asarray(self.predict_proba(X, groups=groups)).argmax(axis=1)


def tune_kappa(X, y, groups, site_groups, kappas=(10.0, 50.0, 100.0, 500.0),
               seed: int = 0, n_estimators: int = 50,
               min_n: int = 15) -> dict:
    """Pick kappa by grouped-site holdout (same sites never in both sides).

    Strict realm-LORO is kappa-invariant under the unseen-realm global
    fallback, so it cannot tune kappa; grouped-site holdout keeps realm
    labels seen while holding out whole sites, which is kappa-sensitive.
    """
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupShuffleSplit

    Xa, ya = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    ga = np.asarray(groups).ravel()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(gss.split(Xa, ya, np.asarray(site_groups)))
    out = {}
    for k in kappas:
        clf = HierarchicalBleachClassifier(kappa=float(k),
                                           n_estimators=n_estimators,
                                           min_n=min_n, seed=seed)
        clf.fit(Xa[tr], ya[tr], groups=ga[tr])
        out[float(k)] = round(float(accuracy_score(
            ya[te], clf.predict(Xa[te], groups=ga[te]))), 4)
    best = max(out, key=out.get)
    return {"per_kappa": out, "best_kappa": best,
            "best_accuracy": out[best]}
