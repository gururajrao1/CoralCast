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


if __name__ == "__main__":
    unittest.main()
