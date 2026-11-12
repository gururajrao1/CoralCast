"""Verification suite for research items 1-7 (run: python -m unittest)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestAllSeven(unittest.TestCase):
    def test_1_join(self):
        from data.join_reefcheck_cortad import (
            dhw_baseline_predict, join_labels, load_reefcheck,
            make_demo_surveys, score_by_region)
        import pandas as pd
        import tempfile
        # load_reefcheck expects a path; write demo CSV first
        with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False) as tmp:
            make_demo_surveys(n=30).to_csv(tmp.name, index=False)
            reef = load_reefcheck(tmp.name)
        self.assertIn("label", reef.columns)
        sst = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=400),
            "sst_c": 28.5})
        labels = join_labels(reef, sst, 28.5)
        self.assertTrue((labels["dhw"] >= 0).all())
        self.assertIn("overall_accuracy", score_by_region(labels))

    def test_2_downscale(self):
        from models.sst_downscale_rcnn import demo
        out = demo()
        self.assertLess(out["rmse_rcnn"], out["rmse_bilinear"])

    def test_3_sits(self):
        from models.sits_mamba import demo
        self.assertGreater(demo()["accuracy"], 0.5)

    def test_4_satmae(self):
        from models.pretrain_satmae import demo
        out = demo()
        self.assertLess(out["mse"], out["data_variance"])

    def test_5_sstode(self):
        from models.sstode_forecast import demo
        out = demo()
        self.assertLess(out["rmse_sstode"], out["rmse_persistence"])

    def test_6_multivariate(self):
        from features.multivariate import demo
        out = demo()
        self.assertIn("freq_above_mmm1_12w", out["kept"])

    def test_7_eval(self):
        from eval.severity_leadtime import baa_tier, demo
        import numpy as np
        self.assertEqual(list(baa_tier(
            np.array([0.0, 5.0, 9.0]), np.array([1.5, 0.0, 0.0])).tolist()),
            [1, 2, 3])
        rep = demo()
        self.assertEqual(len(rep["by_lead"]), 8)


class TestCodePhase(unittest.TestCase):
    def test_temperature_scaling(self):
        import numpy as np
        from eval.severity_leadtime import (apply_temperature,
                                            expected_calibration_error,
                                            fit_temperature)
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 400)
        p = np.full((400, 3), 1 / 3) + rng.standard_normal((400, 3)) * 0.05
        cal = fit_temperature(y[:200], p[:200])
        self.assertLess(cal["nll_after"], cal["nll_before"] + 1e-9)
        after = apply_temperature(p[200:], cal["temperature"])
        self.assertAlmostEqual(float(after.sum(axis=1).mean()), 1.0, places=6)
        ece = expected_calibration_error(y[200:], after[:, 2])["ece"]
        self.assertGreaterEqual(ece, 0.0)

    def test_heat_grid_and_mhw(self):
        import numpy as np
        from coral_bleaching_forecaster import (
            compute_dhw_threshold, mhw_features, thermal_features,
            generate_synthetic_sst, FEATURE_NAMES)
        df = generate_synthetic_sst()
        sst = df["sst_c"].to_numpy()
        t = thermal_features(sst, 28.5, dates=df["date"].to_numpy())
        for k in ["dhw_t05_w42", "dhw_t05_w84", "dhw_t10_w42", "dhw_t10_w84",
                  "mhw_events", "mhw_max_cat", "mhw_max_days", "mhw_cum_int"]:
            self.assertIn(k, t)
            self.assertGreaterEqual(t[k], 0.0)
        self.assertEqual(compute_dhw_threshold(np.full(100, 20.0), 28.5,
                                              1.0, 84), 0.0)
        self.assertEqual(len(FEATURE_NAMES), 37)
        m = mhw_features(np.full(200, 28.0))
        self.assertEqual(m["mhw_events"], 0.0)

    def test_dii(self):
        import numpy as np
        from coral_bleaching_forecaster import (
            depth_invariant_indices, generate_synthetic_bands,
            process_satellite)
        feats = process_satellite(generate_synthetic_bands())
        for k in ["dii_23", "dii_24", "dii_34"]:
            self.assertIn(k, feats)
            self.assertTrue(bool(np.isfinite(feats[k]).all()))
        for k, p in feats["dii_params"].items():
            self.assertGreaterEqual(p["sand_r2"], 0.0)
            self.assertLessEqual(p["sand_r2"], 1.0)

    def test_transplants_and_constraints(self):
        import numpy as np
        from coral_bleaching_forecaster import (
            FEATURE_NAMES, monotone_constraints, transplant_features)
        sst = np.full(200, 28.0)
        sst[-12:] = 30.0
        f = transplant_features(sst, 28.5)
        self.assertGreater(f["gdd"], 0.0)
        self.assertEqual(f["hot_streak_now"], 12.0)
        self.assertEqual(f["hot_streak_max"], 12.0)
        self.assertTrue(0.0 <= f["cwsi_mean"] <= 1.0)
        self.assertGreaterEqual(f["var30"], 0.0)
        self.assertEqual(f["days_above_30"], 12.0)
        mc = monotone_constraints()
        self.assertEqual(len(mc), len(FEATURE_NAMES))
        self.assertTrue(set(mc) <= {-1, 0, 1})
        self.assertEqual(mc[FEATURE_NAMES.index("dhw")], 1)
        self.assertEqual(mc[FEATURE_NAMES.index("var30")], -1)
        self.assertEqual(mc[FEATURE_NAMES.index("ndvi_sub")], 0)

    def test_loro(self):
        import numpy as np
        from eval.severity_leadtime import leave_one_region_out
        rng = np.random.default_rng(2)
        X = rng.standard_normal((240, 6))
        y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 1).astype(int)
        y = np.clip(y, 0, 2)
        regs = np.array(["A", "B", "C", "D"] * 60)
        rep = leave_one_region_out(X, y, regs, seed=2, n_estimators=20)
        self.assertEqual(len(rep["per_region"]), 4)
        self.assertGreater(rep["mean_accuracy"], 0.5)

    def test_gcbd_mapping_and_baseline(self):
        import numpy as np
        import pandas as pd
        from eval.gcbd_benchmark import (dhw_baseline_predict, load_gcbd,
                                         map_labels, run_benchmark)
        self.assertEqual(list(map_labels(pd.Series([0.0, 0.5, 30.0, 80.0]))),
                         [0, 0, 1, 2])
        self.assertEqual(list(dhw_baseline_predict(
            np.array([0.0, 5.0, 9.0]))), [0, 1, 2])
        X, y, regs = load_gcbd("data/gcbd_bleaching_environmental.csv",
                               subsample=120, drop_geo=True)
        self.assertNotIn("Latitude_Degrees", list(X.columns))
        self.assertTrue(set(np.unique(y)) <= {0, 1, 2})
        rep = run_benchmark("data/gcbd_bleaching_environmental.csv",
                            subsample=300, n_estimators=20, drop_geo=True)
        self.assertIn("accuracy", rep)
        self.assertIn("dhw_baseline_accuracy", rep)
        self.assertIn("loro_by_realm", rep)
        self.assertTrue(rep["drop_geo"])

    def test_site_supervised_path(self):
        import numpy as np
        import pandas as pd
        import rasterio
        from rasterio.warp import transform as warp_xy
        from coral_bleaching_forecaster import (
            build_site_supervised_matrix, build_training_matrix,
            default_profile, generate_synthetic_bands,
            generate_synthetic_sst, process_satellite, rule_labels,
            thermal_features)
        feats = process_satellite(generate_synthetic_bands())
        sst = generate_synthetic_sst()["sst_c"].to_numpy()
        therm = thermal_features(sst, 28.5)
        prof = default_profile(feats["ndvi_sub"].shape)
        water_idx = np.where(feats["water"].ravel())[0]
        rng = np.random.default_rng(11)
        pick = rng.choice(water_idx, size=8, replace=False)
        rows, cols = pick // 200, pick % 200
        xs, ys = rasterio.transform.xy(prof["transform"], rows, cols)
        lon, lat = warp_xy(prof["crs"], "EPSG:4326", xs, ys)
        dhw_map = np.clip(therm["dhw"] * (0.85 + 0.30
                                          * feats["depth_proxy"]), 0, None)
        ndvi_px = feats["ndvi_sub"].ravel()[pick]
        y = rule_labels(dhw_map.ravel()[pick], ndvi_px)
        y[::2] = 0
        y[1::2] = 2  # force two-class separability for the split
        sites = pd.DataFrame({"lat": lat, "lon": lon, "label": y})
        Xs, ys2, sids = build_site_supervised_matrix(
            feats, sites, therm, dhw_map, prof["transform"], prof["crs"],
            radius_px=2, seed=11)
        self.assertGreater(len(ys2), 0)
        self.assertEqual(Xs.shape[1], 37)
        self.assertEqual(len(set(sids.tolist())), 8)

    def test_temporal_dataset_and_rules(self):
        import numpy as np
        from coral_bleaching_forecaster import (build_temporal_dataset,
                                               load_stack, rule_labels)
        self.assertEqual(list(rule_labels(
            np.array([1.0, 3.0, 9.0]), np.array([0.5, -0.5, 0.0]))), [0, 1, 2])
        self.assertIsNone(load_stack(None))
        from coral_bleaching_forecaster import (generate_synthetic_bands,
                                               generate_synthetic_sst,
                                               process_satellite)
        feats = process_satellite(generate_synthetic_bands())
        sst = generate_synthetic_sst()["sst_c"].to_numpy()
        Xs, ys, dates = build_temporal_dataset(feats, sst, 28.5, n_times=4,
                                              n_pixels=300)
        self.assertEqual(Xs.shape, (300, 4, 4))
        self.assertEqual(len(dates), 4)
        self.assertTrue(set(np.unique(ys)) <= {0, 1, 2})


class TestRoadmap(unittest.TestCase):
    """R10: pin the honest-validation / baseline / calibration gates."""

    def test_dedupe_events(self):
        import pandas as pd
        from eval.gcbd_benchmark import dedupe_events
        df = pd.DataFrame({
            "Site_ID": ["a", "a", "b"], "Date": ["2020-01-01"] * 3,
            "Percent_Bleaching": [10.0, 10.0, 50.0],
            "SSTA_DHW": [1.0, 1.0, 5.0], "TSA_DHW": [0.5, 0.5, 2.0]})
        out = dedupe_events(df)
        self.assertEqual(len(out), 2)

    def test_redefined_dhw(self):
        import numpy as np
        from coral_bleaching_forecaster import (
            compute_dhw_series, redefined_dhw, redefined_dhw_predict)
        # +0.6 over MMM: invisible to the standard >=1.0 cutoff,
        # caught by the redefined >=0.4 cutoff (the point of R2).
        sst = np.full(200, 29.1)
        self.assertEqual(float(compute_dhw_series(sst, 28.5)[-1]), 0.0)
        self.assertGreater(redefined_dhw(sst, 28.5), 0.0)
        self.assertEqual(list(redefined_dhw_predict(
            np.array([0.0, 3.5, 7.0]))), [0, 1, 2])

    def test_null_forecasts(self):
        import numpy as np
        from models.sstode_forecast import null_forecasts
        s = 28.0 + np.sin(2 * np.pi * np.arange(400) / 365)
        n = null_forecasts(s, 30)
        self.assertTrue((n["persistence"] == s[-1]).all())
        self.assertEqual(len(n["damped_persistence"]), 30)
        self.assertEqual(len(n["climatology"]), 30)

    def test_majority_baseline(self):
        import numpy as np
        from eval.gcbd_benchmark import majority_baseline_accuracy
        self.assertAlmostEqual(majority_baseline_accuracy(
            np.array([2, 2, 2, 0, 1])), 0.6)

    def test_crossfit_calibration(self):
        import numpy as np
        from eval.severity_leadtime import (brier_skill_score,
                                            fit_temperature_cv)
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 300)
        p = np.full((300, 3), 1 / 3) + rng.standard_normal((300, 3)) * 0.05
        p = np.clip(p, 1e-9, 1.0)
        p /= p.sum(axis=1, keepdims=True)
        c1 = fit_temperature_cv(y, p, seed=0)
        c2 = fit_temperature_cv(y, p, seed=0)
        self.assertTrue(c1["cross_fit"])
        self.assertEqual(c1["temperature"], c2["temperature"])
        self.assertGreaterEqual(c1["temperature"], 0.05)
        self.assertLessEqual(c1["temperature"], 10.0)
        onehot = np.eye(3)[y]
        self.assertAlmostEqual(brier_skill_score(y, onehot)
                               ["bss_vs_climatology"], 1.0, places=3)

    def test_severity_and_basins(self):
        import numpy as np
        from eval.gcbd_benchmark import run_per_basin, skill_by_severity
        y = np.array([0, 1, 2, 2, 0, 1])
        pred = np.array([0, 1, 2, 1, 0, 0])
        dhw = np.array([0.0, 5.0, 9.0, 9.0, 1.0, 2.0])
        s = skill_by_severity(y, pred, dhw)
        self.assertEqual(s["dhw_ge_8"]["n"], 2)
        self.assertEqual(s["dhw_ge_8"]["accuracy"], 0.5)
        regs = np.array(["A"] * 200 + ["B"] * 10)
        X = np.random.default_rng(0).standard_normal((210, 4))
        yy = (X[:, 0] > 0).astype(int)
        b = run_per_basin(X, yy, regs, n_estimators=10, min_n=50)
        self.assertIsNotNone(b["A"]["accuracy"])
        self.assertIsNone(b["B"]["accuracy"])  # n=10 < min_n: skipped

    def test_honest_splits_synthetic(self):
        import numpy as np
        from eval.gcbd_benchmark import (forward_temporal_scores,
                                         grouped_split_scores,
                                         spatial_block_scores)
        rng = np.random.default_rng(1)
        X = rng.standard_normal((120, 4))
        y = (X[:, 0] > 0).astype(int)
        groups = np.array([f"site{i % 12}" for i in range(120)])
        lat = rng.uniform(-20, 20, 120)
        lon = rng.uniform(140, 160, 120)
        g = grouped_split_scores(X, y, groups, seed=0, n_estimators=10)
        self.assertGreaterEqual(g["accuracy"], 0.0)
        self.assertGreater(g["n_groups_test"], 0)
        b = spatial_block_scores(X, y, lat, lon, seed=0, n_estimators=10)
        self.assertGreater(b["n_blocks"], 1)
        import pandas as pd
        dates = pd.date_range("2020-01-01", periods=120)
        t = forward_temporal_scores(X, y, dates, seed=0, n_estimators=10)
        self.assertGreaterEqual(t["n_folds"], 1)

    def test_select_features(self):
        import numpy as np
        from coral_bleaching_forecaster import (DHW_COL, FEATURE_NAMES,
                                                select_features)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, len(FEATURE_NAMES)))
        X[:, 1] = 1.0  # zero-variance dead column
        y = (X[:, DHW_COL] > 0).astype(int)
        keep = select_features(X, y, seed=0, n_estimators=10)
        self.assertNotIn(1, keep)
        self.assertIn(DHW_COL, keep)
        # pruned constraint vector must match pruned width (site-head crash)
        import lightgbm as lgb
        from coral_bleaching_forecaster import monotone_constraints
        mono = [monotone_constraints()[i] for i in keep]
        clf = lgb.LGBMClassifier(n_estimators=5, verbose=-1,
                                 monotone_constraints=mono)
        clf.fit(X[:, keep], y)  # must not raise
        self.assertEqual(len(mono), X[:, keep].shape[1])

    def test_stack_positions(self):
        import numpy as np
        from coral_bleaching_forecaster import _stack_positions
        dates = np.array(["2022-01-01", "2022-06-01"],
                         dtype="datetime64[D]")
        sst_dates = np.arange(np.datetime64("2022-01-01"),
                              np.datetime64("2023-01-01"),
                              dtype="datetime64[D]")
        pos = _stack_positions(["20220101", "20220601"], sst_dates, 365)
        self.assertEqual(pos[0], 84)  # clamped to DHW window floor
        self.assertGreater(pos[1], pos[0])
        self.assertEqual(_stack_positions(["20220101"], None, 365), [364])

    def test_change_map_and_balance(self):
        import numpy as np
        import pandas as pd
        from pilot_sekisei import balance_report, bleaching_change_map
        base = np.full((10, 10), 0.10)
        event = base.copy()
        event[:5] = 0.15
        cm = bleaching_change_map(base, event, thresh=0.03)
        self.assertAlmostEqual(cm["fraction_flagged"], 0.5)
        lab = pd.DataFrame({"label": [0] * 96 + [2] * 4})
        br = balance_report(lab)
        self.assertAlmostEqual(br["minority_fraction"], 0.04)

    def test_run_registry(self):
        import tempfile
        from eval.run_registry import new_run, record
        run = new_run(argv=["--demo"], seed=0)
        self.assertIn("run_id", run)
        self.assertEqual(run["seed"], 0)
        with tempfile.NamedTemporaryFile(
                suffix=".jsonl", delete=False) as tmp:
            path = record({"run": run, "accuracy": 0.9}, "probe",
                          registry=tmp.name)
        with open(path) as f:
            line = f.read().strip()
        self.assertIn("probe", line)

    def test_add_covariate(self):
        import pandas as pd
        from eval.gcbd_benchmark import add_covariate
        df = pd.DataFrame({"Site_ID": ["a", "b"], "Date": ["2020-01-01"] * 2,
                           "x": [1.0, 2.0]})
        self.assertIs(add_covariate(df, None), df)
        import tempfile
        with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, mode="w") as tmp:
            tmp.write("Site_ID,Date,outlook_p\n"
                      "a,2020-01-01,0.7\nb,2020-01-01,0.2\n")
            path = tmp.name
        out = add_covariate(df, path)
        self.assertIn("outlook_p", out.columns)
        self.assertAlmostEqual(out["outlook_p"].iloc[0], 0.7)

    def test_downscale_patches(self):
        import numpy as np
        from models.sst_downscale_rcnn import RCNNDownscaler
        rng = np.random.default_rng(0)
        fine = rng.standard_normal((64, 64)).astype(np.float32)
        coarse = fine.reshape(16, 4, 16, 4).mean(axis=(1, 3))
        static = np.stack([np.ones((64, 64), dtype=np.float32)] * 2)
        items = RCNNDownscaler.patches(coarse, fine, static, factor=4)
        self.assertGreater(len(items), 1)

    def test_quadrant_grid(self):
        import numpy as np
        from coral_bleaching_forecaster import quadrant_grid
        g = quadrant_grid((4, 6))
        self.assertEqual(g.shape, (4, 6))
        self.assertEqual(g[0, 0], "NW")
        self.assertEqual(g[0, 5], "NE")
        self.assertEqual(g[3, 0], "SW")
        self.assertEqual(g[3, 5], "SE")
        self.assertEqual(set(np.unique(g)), {"NW", "NE", "SW", "SE"})

    def test_hierarchical_blend_math(self):
        import lightgbm as lgb
        import numpy as np
        from models.hierarchical_bleach import HierarchicalBleachClassifier
        rng = np.random.default_rng(11)
        X = rng.standard_normal((140, 5))
        y = ((X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int))
        groups = np.array(["A"] * 80 + ["B"] * 60)
        kappa = 50.0
        clf = HierarchicalBleachClassifier(kappa=kappa, n_estimators=10,
                                          min_n=10, seed=0).fit(X, y, groups)
        # Manual shrunk blend with identically configured heads must match.
        def _head(m):
            c = lgb.LGBMClassifier(
                objective="multiclass", num_class=3, n_estimators=10,
                learning_rate=0.05, num_leaves=63, min_child_samples=40,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=5.0,
                class_weight="balanced", verbose=-1, random_state=0)
            return c.fit(X[m], y[m])
        g = _head(np.ones(len(y), bool))
        pa = _head(groups == "A")
        w_a = 80 / (80 + kappa)
        expect_a = (w_a * pa.predict_proba(X[groups == "A"])
                    + (1 - w_a) * g.predict_proba(X[groups == "A"]))
        np.testing.assert_allclose(
            clf.predict_proba(X, groups)[groups == "A"], expect_a,
            rtol=1e-6)
        # No groups -> pure global head.
        np.testing.assert_allclose(clf.predict_proba(X),
                                   g.predict_proba(X), rtol=1e-6)

    def test_hierarchical_fallback(self):
        import numpy as np
        from models.hierarchical_bleach import HierarchicalBleachClassifier
        rng = np.random.default_rng(12)
        X = rng.standard_normal((120, 5))
        y = ((X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int))
        # tiny (n=8 < min_n) + single-class realms fall back to global.
        groups = np.array(["BIG"] * 100 + ["tiny"] * 8 + ["flat"] * 12)
        y = np.where(groups == "flat", 1, y)
        clf = HierarchicalBleachClassifier(kappa=100.0, n_estimators=10,
                                          min_n=50, seed=0).fit(X, y, groups)
        self.assertIsNone(clf.basin_models_["tiny"])
        self.assertIsNone(clf.basin_models_["flat"])
        self.assertIsNotNone(clf.basin_models_["BIG"])
        self.assertEqual(clf.basin_weight_["tiny"], 0.0)
        proba = clf.predict_proba(X, groups)
        self.assertEqual(proba.shape, (120, 3))
        # Unknown realm at predict time also falls back without raising.
        q = clf.predict_proba(X[:5], np.array(["XX"] * 5))
        np.testing.assert_allclose(
            q, clf.global_model_.predict_proba(X[:5]), rtol=1e-6)

    def test_tune_kappa_smoke(self):
        import numpy as np
        from models.hierarchical_bleach import tune_kappa
        rng = np.random.default_rng(13)
        X = rng.standard_normal((150, 5))
        y = ((X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int))
        groups = np.array(["A"] * 90 + ["B"] * 60)
        sites = np.array([f"s{i % 30}" for i in range(150)])
        out = tune_kappa(X, y, groups, sites, kappas=(10.0, 500.0),
                         seed=0, n_estimators=10, min_n=10)
        self.assertIn(out["best_kappa"], (10.0, 500.0))
        self.assertEqual(set(out["per_kappa"]), {10.0, 500.0})

    def test_adaptive_causal_contract(self):
        import numpy as np
        import pandas as pd
        from eval.adaptive_features import (ADAPTIVE_COLS,
                                            add_adaptive_features_train_test)
        tr = pd.DataFrame({
            "SSTA_DHW": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "TSA_DHW": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "SSTA": [0.1, 0.2, 0.1, 0.3, 0.2, 0.4],
            "SSTA_Maximum": [0.5, 0.6, 0.5, 0.7, 0.6, 0.8],
            "Date_Year": [2000, 2000, 2001, 2001, 2002, 2002]})
        te = pd.DataFrame({
            "SSTA_DHW": [7.0, 8.0],
            "TSA_DHW": [3.5, 4.0],
            "SSTA": [0.5, 0.6],
            "SSTA_Maximum": [0.9, 1.0],
            "Date_Year": [2005, 2006]})  # novel future years
        thr, A_tr, A_te = add_adaptive_features_train_test(tr, te)
        self.assertEqual(list(A_te.columns), ADAPTIVE_COLS)
        self.assertFalse(np.isnan(A_te.to_numpy()).any())
        # Train-only reference: excess equals manual dhw - q75.
        np.testing.assert_allclose(
            A_te["adapt_dhw_excess_q75"].to_numpy(),
            te["SSTA_DHW"].to_numpy() - thr["dhw_q75"], rtol=1e-9)
        # Deterministic: same input -> same output.
        _, _, A_te2 = add_adaptive_features_train_test(tr, te)
        pd.testing.assert_frame_equal(A_te, A_te2)
        # Test rows never leak into the reference.
        self.assertLessEqual(thr["train_year_max"], 2002.0)

    def test_adaptive_temporal_smoke(self):
        import pandas as pd
        from eval.adaptive_features import forward_temporal_delta
        from eval.gcbd_benchmark import dedupe_events
        df = pd.read_csv("data/gcbd_bleaching_environmental.csv",
                         low_memory=False)
        df = dedupe_events(df).reset_index(drop=True)
        out = forward_temporal_delta(df, seed=0, n_estimators=10)
        self.assertGreaterEqual(out["n_folds"], 1)
        self.assertEqual(len(out["base_per_fold"]), out["n_folds"])
        self.assertIsInstance(out["delta_mean"], float)

    def test_hierarchical_benchmark_wiring(self):
        from eval.gcbd_benchmark import run_benchmark
        kw = dict(subsample=300, n_estimators=10, drop_geo=True,
                  hier_kappa=500.0, max_blocks=4)
        rep = run_benchmark("data/gcbd_bleaching_environmental.csv",
                            seed=0, **kw)
        h = rep["hierarchical"]
        self.assertEqual(h["kappa"], 500.0)
        # Delta is exactly hier minus global on the same split.
        self.assertAlmostEqual(
            h["grouped_site"]["delta"],
            h["grouped_site"]["hier_accuracy"]
            - h["grouped_site"]["global_accuracy"], places=9)
        ft = h["forward_temporal"]
        if ft["delta_mean"] is None:
            self.assertEqual(ft["n_folds"], 0)
        else:
            self.assertAlmostEqual(
                ft["delta_mean"],
                ft["hier_mean"] - ft["global_mean"], places=9)
        # Opt-out: no flag means no section, and the wiring is
        # deterministic on the same split.
        rep0 = run_benchmark("data/gcbd_bleaching_environmental.csv",
                             subsample=300, n_estimators=10, drop_geo=True)
        self.assertIsNone(rep0["hierarchical"])
        rep2 = run_benchmark("data/gcbd_bleaching_environmental.csv",
                             seed=0, **kw)
        self.assertEqual(rep2["hierarchical"], h)

    def test_ncei_label_parse(self):
        from eval.join_ncei_thermal import map_label, parse_percent
        self.assertEqual(parse_percent("12.5"), (12.5, "percent_numeric"))
        self.assertEqual(parse_percent("1%-10%"), (5.5, "range_mid"))
        self.assertEqual(parse_percent(">60"), (60.0, "range_gt"))
        self.assertEqual(parse_percent("<1"), (0.0, "range_lt1"))
        self.assertEqual(parse_percent("None_(<1)"), (0.0, "range_lt1"))
        self.assertEqual(parse_percent("N/A"), (None, "unparseable"))
        self.assertEqual(parse_percent(None), (None, "missing"))
        self.assertEqual(parse_percent(float("nan")), (None, "missing"))
        # Tiers match the benchmark: <1 none, <50 moderate, else severe.
        self.assertEqual([map_label(v) for v in (0.5, 5.5, 60.0)], [0, 1, 2])
        self.assertEqual(map_label(1.0), 1)

    def test_per_basin_train_and_route(self):
        import numpy as np
        from coral_bleaching_forecaster import (predict_per_basin,
                                                train_per_basin_models,
                                                train_risk_model)
        rng = np.random.default_rng(2)
        X = rng.standard_normal((240, 6))
        y = ((X[:, 0] > 0.5).astype(int)
             + (X[:, 1] > 0.5).astype(int))
        self.assertEqual(set(np.unique(y)), {0, 1, 2})
        # NE has n=20 < min_n=50: skipped to the global fallback.
        # Narrow test matrix: cols must match the constraint count, the
        # same convention as train_risk_model (real X is 37 wide).
        basins = np.array(["NW"] * 120 + ["SE"] * 100 + ["NE"] * 20)
        cols6 = [0, 1, 2, 3, 4, 5]
        models = train_per_basin_models(X, y, basins, seed=0,
                                        cols=cols6, n_estimators=10)
        self.assertIsNotNone(models["NW"]["model"])
        self.assertIsNotNone(models["SE"]["model"])
        self.assertIsNone(models["NE"]["model"])
        self.assertIn("note", models["NE"])
        # Single-class basin is skipped, not crashed on.
        tiny = train_per_basin_models(
            X[:60], np.zeros(60, dtype=int),
            np.array(["SW"] * 60), seed=0, cols=cols6,
            n_estimators=10)
        self.assertIsNone(tiny["SW"]["model"])
        # Prune-mask convention: sliced X + sliced constraints must fit.
        keep = [0, 1, 2]
        masked = train_per_basin_models(X[:, keep], y, basins, seed=0,
                                        cols=keep, n_estimators=10)
        self.assertIsNotNone(masked["NW"]["model"])
        # Routing: trained-basin rows use their head, skipped and unknown
        # basin rows exactly match the global fallback.
        fallback, _, _ = train_risk_model(X, y, 0, cols=cols6)
        proba = predict_per_basin(models, fallback, X, basins)
        self.assertEqual(proba.shape, (240, 3))
        fb = fallback.predict_proba(X)
        np.testing.assert_allclose(proba[basins == "NE"],
                                   fb[basins == "NE"])
        unknown = predict_per_basin(models, fallback, X[:5],
                                    np.array(["XX"] * 5))
        np.testing.assert_allclose(unknown, fb[:5])


if __name__ == "__main__":
    unittest.main()
