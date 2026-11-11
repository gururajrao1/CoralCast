# CoralCast — 30-Day Coral Bleaching Risk Forecaster

> Status: research prototype, re-measured Nov 11 after the R1–R10 build.
> Real-data headline: **0.70 accuracy (±0.01 over 3 seeds) vs 0.33 DHW /
> 0.47 majority baselines (ROC-AUC 0.85)** on GCBD n=4,113 deduped rows,
> thermal-only. Deployment rule: **train per basin with a minimum-n
> cutoff** — skill does not transfer across ocean basins (realm-LORO
> ~0.50, temporal CV ~0.50). Do not quote the stratified 0.70 without an
> honest-split number beside it.

## 1. What this is

CoralCast predicts **coral bleaching risk 30 days ahead** at reef scale by
fusing two independent signals: what the reef looks like from space right
now, and how much heat stress the water has accumulated.

- **Nowcast vs forecast.** NOAA Coral Reef Watch DHW is a diagnostic
  (heat accumulated *so far*). CoralCast adds a true forecast horizon:
  SST is forecast forward, DHW is recomputed ahead, and a severity /
  lead-time harness reports skill per lead week, against the operational
  targets of ~6 weeks (best published XGBoost early-warning, Univ. Miami
  2025) and 4 months (NOAA CRW bleaching Outlook).
- **Problem statement.** Given multispectral imagery + a daily SST series
  (+ optional survey labels), output per-pixel risk
  {0 No Stress, 1 Watch/Warning, 2 Severe Bleaching Expected} 30 days out
  as a georeferenced map with calibrated probabilities.
- **History.** Built Nov 8 as "Project 6.3 Multispectral Coral Bleaching
  Predictor" (~41 files, 13 probe scripts), verified Nov 9 (15/15 tests,
  synthetic + GCBD benchmarks), pushed to
  `https://github.com/gururajrao1/CoralCast` (10 commits, 40 files), docs
  pass `f0298cc`. Later sessions moved to other projects, so coral numbers
  below are the frozen Nov 9 state unless re-run.

## 2. How we approach it

```
Sentinel-2 B2/B3/B4/B8 ──> DOS correction + water-column adj ──> indices
   (submerged NDVI, blue/green SCI, Lyzenga DII_23/24/34 via sand-ROI regression)
SST series ──> DHW(12wk, MMM) + Hobday MHW + transplants(GDD/CWSI/days>30C/...) ──> 37 features
37 features ──> LightGBM (monotonic priors: risk rises w/ heat, falls w/ variance) ──> risk map
Optional heads: SITS temporal encoder │ SSTODE SST forecast │ RCNN downscaler │ SatMAE pretrain
Labels: ReefCheck+CoRTAD join │ site-supervised pixels │ GCBD benchmark │ Sekisei pilot
Eval: temperature scaling │ realm-LORO │ severity-by-lead (hit/FAR) │ reliability plots
```

- **Feature set (37).** `PIXEL_FEATURES` (7, optical) + `THERMAL_FEATURES`
  (30, SST-derived) = `FEATURE_NAMES`, assembled by the shared
  `pixel_block` used identically in training, site path, and forecast
  export. Monotonicity priors encode domain knowledge (heat ↑ risk,
  thermal variance ↓ risk at fixed mean).
- **Synthetic path.** `generate_synthetic_bands` / `generate_synthetic_sst`
  let the full pipeline run with no downloads. Labels come from
  `rule_labels()` (DHW + NDVI rules + 5% injected noise) — so synthetic
  skill is **partly circular**; treat 0.965 as a pipeline smoke test, not
  a skill claim.
- **Real paths.** Tabular: ReefCheck surveys joined to CoRTAD weekly SST
  (`data/join_reefcheck_cortad.py`); GCBD benchmark (`eval/`); spatial:
  Sekisei pilot S2 stack + MUR SST + ReefCheck labels (`pilot_sekisei.py`
  fetch-only; the run itself is the forecaster CLI).
- **Outputs.** `coral_bleaching_risk_forecast.tif` (georeferenced risk
  map), `bleaching_evaluation_metrics.json` (ROC-AUC, PR, DHW importance,
  calibration, LORO), `forecast_quicklook.png`, `eval/reliability.png`.
  **Warning: outputs are overwritten every run (G10)** — copy them aside
  with run-id/seed/git-hash before comparing runs. Every run now stamps
  run-id/seed/git-hash/argv into its metrics JSON and appends one JSONL line
  to `runs/registry.jsonl` (`eval/run_registry.py`; forecaster, benchmark,
  and pilot all record) — consult the registry instead of diffing bare
  files.

## 3. Data sources

| File in repo | Upstream source | Notes / seam |
|---|---|---|
| `data/ReefCheck_raw.csv` | Sully et al. 2019 repo ([raw CSV](https://raw.githubusercontent.com/InstituteForGlobalEcology/Coral-bleaching-a-global-analysis-of-the-past-two-decades/master/Reef%20Check%20Data%20Raw.csv)), paper [doi:10.1038/s41467-019-09238-2](https://doi.org/10.1038/s41467-019-09238-2) | Survey-level bleaching % |
| `data/ReefCheck_CoRTAD.csv` | Same repo ([joined CSV](https://raw.githubusercontent.com/InstituteForGlobalEcology/Coral-bleaching-a-global-analysis-of-the-past-two-decades/master/Reef_Check_with_cortad_variables_with_annual_rate_of_SST_change.csv)); CoRTAD = [NOAA NCEI](http://www.ncei.noaa.gov/products/coral-reef-temperature-anomaly-database), [accession](https://accession.nodc.noaa.gov/NCEI-CoRTADv6) | Weekly 4 km AVHRR 1982–2022. **Frozen end-2022**: append CRW 5km / OISST / OSTIA for inference |
| `data/gcbd_bleaching_environmental.csv` | GCBD BCO-DMO 773466 v2 ([dataset](https://www.bco-dmo.org/dataset/773466)), paper [doi:10.1038/s41597-022-01121-y](https://doi.org/10.1038/s41597-022-01121-y); full 33k-record NCEI Accession 0228498 ([catalog](https://catalog.data.gov/dataset/global-coral-bleaching-database-ncei-accession-0228498)) | Vendored copy n=4,219 (1980–2016, zero S2-era bleached rows). Full-NCEI pull JS/302-gated (G2) |
| `data/pilot_stack/*.tif` | [Sentinel-2 L2A COGs](https://registry.opendata.aws/sentinel-2-l2a-cogs/), searched via [Earth Search STAC](https://earth-search.aws.element84.com/v1/search) | Tile 51RWH, 2022-08-25 + 2022-09-29, Sekisei box lon 123.65–123.80 / lat 24.30–24.42 |
| `data/pilot_sst.csv` | JPL MUR SST v4.1 via [CoastWatch ERDDAP](https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.html) | 1,033 d at 24.31N 123.71E, MMM 29.98 C. Chunk yearly; multi-year single queries drop |
| `data/pilot_labels.csv` | Derived: ReefCheck_CoRTAD Japan/Sekisei bbox + all bleached Japan rows | 69 rows; surveys are 2003–2004 vs 2022 imagery (mismatch, G7) |
| `sst_noaa_daily.csv` | **Not external — synthetic fallback** written by `load_sst` when no `--sst_csv` is given | Provenance hazard: name looks observational; content is generated with a hard-coded heatwave tail |
| `data/labels.csv` (60 rows), `data/demo_sites.csv` | Synthetic demo fixtures | Toy baselines only (0.233 on n=60) |

Fetch notes (hard-won, keep): Earth Search STAC needs POST + full RFC3339
datetimes; MUR SST via ERDDAP griddap — chunk yearly, single-point queries
are reliable, multi-year single queries drop; GeoNetwork POST search works
(`catalogue.eatlas.org.au/geonetwork/srv/api/search/records/_search`);
CoralWatch UID `dr17828` returns 0 (data lives in the BioCollect project
store `9c55416c-f56a-4917-a65d-daadcb8f51f7`); eAtlas nextcloud sharealias
`/download` links 404.

Operational references (not vendored): NOAA CRW 5km v3.1 DHW
([product](https://coralreefwatch.noaa.gov/product/5km/tutorial/crw04a_intro_products.php),
HotSpot=SST−MMM, DHW=Σ HotSpot≥1 C over 12 wk; ≥4 possible, ≥8 severe)
and the 4-month CFSv2 Outlook ([Eakin et al. 2018](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2018.00057/full));
OSTIA gap-free SST; Allen Coral Atlas [methods](https://allencoralatlas.com/methods/)
(triggered baseline-vs-event brightening *detection*, not a forecast).

## 4. How to run

```bash
uv venv
uv pip install --python .venv/Scripts/python.exe rasterio lightgbm scikit-learn geopandas matplotlib xarray numpy torch --index-url https://download.pytorch.org/whl/cpu
python coral_bleaching_forecaster.py
```

Full pipeline (all flags):

```bash
python data/join_reefcheck_cortad.py --demo
python coral_bleaching_forecaster.py --sst_model sstode --multivariate --eval_report --labels_csv data/labels.csv --temporal sits --calibrate --loro --site_labels_csv data/demo_sites.csv --prune_dead --downscale
python eval/gcbd_benchmark.py --drop-geo --seeds 0,1,2 --out eval/gcbd_benchmark_nogeo.json
python eval/gcbd_benchmark.py --seeds 0,1,2
python -m pytest tests/test_all7.py -q
```

CLI flags: `--sst_model sstode`, `--multivariate`, `--eval_report`,
`--labels_csv`, `--temporal sits`, `--stack_dir`, `--calibrate`, `--loro`,
`--site_labels_csv`, `--site_radius_px`, `--prune_dead` (R5 probe-prune of
zero-variance/zero-gain features), `--downscale` (R3 patch-trained RCNN
sharpening of the forecast DHW map), `--per_basin` (R8 quadrant-basin
heads + global fallback, deployed-vs-global on the same test split),
plus `--blue/--green/--red/--nir`
`--sst_csv --mmm` for pilot runs. Benchmark flags: `--seeds` (sweep with
CIs), `--max-blocks`, `--block-deg`, `--no-dedupe`, `--covariate` (R8
external lead-time covariate CSV). Per-basin models apply a minimum-n rule
(realms with n < 50 reported as skipped) with no CLI override.
Every run is stamped into `runs/registry.jsonl` (see §2). Environment notes: `mamba_ssm` needs
Linux+CUDA (GRU fallback shipped, G3); ACOLITE unavailable here — DOS +
DII is the verified chain (G4); Prithvi path is a stub (G5); PowerShell
may show exit 1 on passing unittest runs (stderr artifact — trust `OK`).

## 5. Benchmarks

GCBD numbers below are re-measured Nov 11 (R1–R10 build, project venv,
3-seed sweep). Synthetic pipeline numbers are the Nov 9 state plus the new
R2/R6 rows and the R8 `--per_basin` run, both re-measured Nov 11. `--prune_dead` + `--downscale` variant
(re-measured Nov 11): 21/37 features kept, model parity 0.965/0.947, DHW
gain share 0.231 → 0.300, downscale val RMSE 0.076, site 0.4931 — pruning
removes dead weight without losing skill. Test suite: 35 tests
(15 pre-existing + 20 roadmap), 35/35 green Nov 11 on the project venv.

### 5.1 Main synthetic pipeline (n_test = 24,000)

| Metric | Value |
|---|---|
| Accuracy | 0.9655 |
| ROC-AUC (ovr macro) | 0.9467 |
| AP per class 0/1/2 (supports 4385/2465/17150) | 0.926 / 0.859 / 0.965 |
| Confusion | [[4070,42,273],[59,2096,310],[67,78,17005]] |
| Calibration T / NLL / ECE | 0.9876, 0.2071 → 0.2070, 0.0327 → 0.0313 (fit on the `--per_basin` deployed system) |
| Quadrant-LORO acc / ROC | 0.9652 ± 0.0011 / 0.9459 |
| SITS temporal head vs LightGBM | 0.965 vs 0.9655 (tie) |
| Site-supervised path (10 sites, 1210 px) | acc 0.4904, rule-head agreement 0.5372 |
| Top gain features (dhw_gain_share 0.231) | sst_mean, dhw, mhw_events, sst_max, ndvi_sub |
| Forecast map at DHW 5.60 | 100% class 2 (saturated demo regime — expected, not a bug) |
| Severity/lead-time ECE | 0.0294; hit_rate 0.978–1.0 at leads 15–30 d |
| R2 baselines, same test pixels (re-measured Nov 11) | DHW rule 0.6137, redefined-DHW **0.7496**, majority 0.7146, model 0.965 |
| Per-basin deployed vs global (same test split) | deployed 0.9625 / ROC 0.9479 vs global 0.9655 (delta −0.003); NE 0.9616 / NW 0.9649 / SE 0.9617 / SW 0.9618 |
| Calibration Brier (synthetic) | 0.0261, BSS **+0.82** vs climatology |

### 5.2 Real GCBD benchmark (re-measured Nov 11, R1–R10 build)

Seed sweep (3 seeds) + dedupe: 106 exact-duplicate rows dropped (4,219 →
4,113). With-geo honest splits: grouped-site **0.7223**, block **0.6895 ±
0.3506**, temporal **0.5071** — geo features lift block transfer (0.69 vs
0.58 thermal-only) while adding nothing to stratified accuracy, the classic
leakage-adjacent signature; Latitude_Degrees still dominates gain (17,639).

| Setup | Acc | ROC-AUC | AP 0/1/2 | DHW base | Majority base | LORO | ECE (T, cross-fit) | BSS |
|---|---|---|---|---|---|---|---|---|
| Thermal-only (39 feats, `--drop-geo`) | 0.6958 ± 0.0071 | 0.8499 (sweep 0.8538) | 0.900/0.715/0.741 | 0.3324 | 0.4674 | 0.4924 ± 0.1555 | 0.1234 (1.915 ± 0.076, NLL 0.717 → 0.620) | +0.241 vs climatology |
| With-geo (41 feats) | 0.6987 ± 0.0088 | 0.8537 (sweep 0.8570) | 0.890/0.723/0.744 | 0.3324 | 0.4674 | 0.5055 ± 0.1104 | 0.1206 (1.948 ± 0.073, NLL 0.720 → 0.616) | +0.249 vs climatology |

New honest splits (thermal-only): grouped-site holdout **0.7036**
(965 test rows, 675 sites — holds up); spatial-block CV **0.5833 ± 0.4011**
(25 sampled of 396 occupied 1° cells — very noisy, as expected);
forward-temporal CV **0.4919** (3 folds). Severity cut: DHW≥4 0.6964
(n=494), DHW≥8 0.7203 (n=261). Top gain: Temperature_Mean, TSA_Mean,
Depth_m. Feature parity: 0 of 30 deployment thermal names exist in GCBD
(raw CoRTAD columns instead — expected, documented).

Full-scale global-predictor probes (Nov 11, n=4113, 400 trees,
same-split both arms — not seed sweeps, report means over 2–3 seeds):

| Probe | Global | Challenger | Delta |
|---|---|---|---|
| Grouped-site hier-500 | 0.700 | 0.705 | +0.005, positive all 3 seeds |
| Forward-temporal hier-500 | 0.495 | 0.514 | +0.019, both seeds |
| Forward-temporal adaptive | 0.488 | 0.526 | +0.038, both seeds (fold spread dominates) |
| LORO CORAL | 0.494 | 0.416 | −0.078, rejected and removed |

Hierarchical shrink (`models/hierarchical_bleach.py`, kappa=500) is the
only approach that helps on both axes; adaptive heat features
(`eval/adaptive_features.py`, train-year-only fit) help temporal only.
CORAL helps only Tropical Atlantic (+0.04) while wrecking TNP/EIP —
covariance alignment is not a global story on 4k rows. Nothing here
moves strict LORO (structural); the shippable gains are grouped-site
and temporal.

### 5.3 Realm leave-one-out (thermal-only; ROC null where a fold lacks a class)

| Realm | n | LORO acc | LORO ROC | Per-basin acc |
|---|---|---|---|---|
| Central Indo-Pacific | 1214 | 0.4498 | 0.6579 | 0.7039 |
| Eastern Indo-Pacific | 102 | 0.5588 | null | 0.3077 |
| Temperate Australasia | 7 | 0.1429 | null | skipped (n<50) |
| Temperate Northern Atlantic | 5 | skipped (single-class) | null | skipped (n<50) |
| Temperate Northern Pacific | 188 | 0.6383 | 0.7099 | 0.6809 |
| Tropical Atlantic | 2253 | 0.4794 | 0.5941 | 0.6454 |
| Tropical Eastern Pacific | 51 | 0.6078 | null | 0.3077 |
| Western Indo-Pacific | 293 | 0.5700 | null | 0.7432 |

LORO mean acc 0.4924 ± 0.1555, mean ROC 0.654 (3/7 folds). TNA (n=5) was
always present (4,214 + 5 = 4,219) but silently skipped by LORO — now
reported as skipped instead. Per-basin models hold 0.65–0.74 where n is
adequate (CIP/TNP/TAtl/WIP) and collapse at n≈50–100 (EIP/TEP 0.31) —
hence train-per-basin *with a minimum-n rule*.

### 5.4 Per-module benchmark (the 7 extensions + main; demo gates)

| # | Module | Gate (demo asserts) | Nov 9 result | Limitation |
|---|---|---|---|---|
| 1 | Real labels join (`data/join_reefcheck_cortad.py`) | DHW baseline runs, regional scores | demo baseline 0.233 (n=60 toy) | severity/100 scale assumption; single global SST series |
| 2 | RCNN downscale (`models/sst_downscale_rcnn.py`) | rmse_rcnn < bilinear | pass | trains on ONE coarse/fine pair; synthetic depth/mask; unwired into CLI |
| 3 | SITS encoder (`models/sits_mamba.py`) | acc > 0.5, recon < var | acc ~0.54 (thin) | mamba_ssm uninstallable here; no val early-stop; HGB fallback flattens time |
| 4 | SatMAE (`models/pretrain_satmae.py`) | mse < var | pass | loss over ALL patches incl. visible; no encoder.encode for classifier; unseeded masking |
| 5 | SSTODE (`models/sstode_forecast.py`) | rmse < persistence | pass | diffusion term is mislabeled damping; train/infer seasonal-phase mismatch; no uncertainty |
| 6 | Multivariate (`features/multivariate.py`) | kept incl. freq_above_mmm1_12w | pass | fillna(0)/min_periods=1 biases early windows; optical → global mean/p10 only |
| 7 | Severity/lead-time (`eval/severity_leadtime.py`) | by_lead len 8 | pass (8-row demo; 22-row full) | ECE on P(severe) only; temp fit on test split; silent null-AUC folds |
| — | Main forecaster | 35/35 `tests/test_all7.py` | 35/35 OK Nov 11 (project venv, ~80 s) | roadmap tests pin redefined-DHW tiers, nulls, cross-fit determinism, BSS=1 on perfect, prune-mask width, block patches, registry, hier-blend math/fallback, adaptive causal contract |

### 5.5 Pilot Sekisei (all real, n = 69 labels)

Model and DHW baseline both 1.0 on the 6 in-window sites; DHW 0.681 on
all labels; model ROC-AUC 0.534 — no in-window heatwave, so nothing to
discriminate. 96% no-stress imbalance; 2003–2004 labels vs 2022 imagery.

### 5.6 How to read these numbers (honest-CV warning)

Random-split 0.71 overstates skill vs realm-LORO ~0.47. Independent work
(Mestry 2026, EarthArXiv 15638: pooled RF R² 0.688 random → 0.53 new
sites → 0.18 1-deg blocks → 0.02 10-deg blocks; ReefCheck ~34.5%
duplicates) shows why: always report site-grouped + spatial-block +
forward-temporal CV beside stratified, dedupe substrate rows to sampling
events, sweep seeds for CIs, and keep ranking-severe-events AUC (~0.73 at
5-deg blocks) as the transfer metric.

## 6. Research-backed improvements (ranked)

Each item: what to change, why (literature), expected gain, effort.

- **R1 Honest validation first (M).** Add site-grouped + spatial-block
  (blockCV / Roberts–Valavi) + forward-temporal CV; dedupe ReefCheck
  substrate rows to sampling events; seed sweeps with CIs. Why: Mestry
  2026 ([EarthArXiv 15638](https://eartharxiv.org/repository/view/15638),
  [data](https://doi.org/10.5281/zenodo.23511976)) shows random-split
  skill collapsing under honest splits. Gain: credible numbers; severe
  ranking keeps ~0.73 AUC at 5-deg blocks.
- **R2 Harder baselines (S).** Add redefined-DHW (0.4 C cutoff, 11-wk
  window, threshold 3 DHW; +~70% global skill,
  [Coral Reefs 2024](https://doi.org/10.1007/s00338-024-02512-w)) plus
  persistence / damped-persistence / climatology nulls (Brier-skill
  reference). Gain: stops beating a strawman; thermal-gap work
  ([Frontiers 2026](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2026.1835611/full))
  shows standard DHW AUC falls to ~0.57 in the hottest seas.
- **R3 Wire real downscaling (M–L).** Fix RCNN `fit()` single-pair
  overfit → patch/val split, real bathymetry + reef mask, ablate the
  extremes weight (2.0/q=0.9), wire into the forecaster path. Why:
  residual-corrective U-Net downscaling ([arXiv 2608.10022](https://arxiv.org/abs/2608.10022v1))
  and 250 m downscaled DHW ([PeerJ 21269](https://peerj.com/articles/21269))
  improve bleaching prediction where 5 km DHW is flat.
- **R4 Fix + ablate the SST forecast head (L).** Repair the diffusion
  term (currently −1st-difference mislabeled as Laplacian), unify the
  train/infer seasonal phase, add multi-horizon loss + NLL uncertainty;
  order baselines persistence → linear → U-Net → ConvLSTM →
  hybrid U-Net+ConvLSTM → SSTODE-style ODE
  ([SSTODE](https://arxiv.org/abs/2511.05629)). Target the 6-week Miami
  XGBoost horizon ([doi:10.1088/2515-7620/ae2570](https://doi.org/10.1088/2515-7620/ae2570),
  ~1-week tolerance) and 4-month Outlook horizon.
- **R5 Feature hygiene + parity (M).** Per-date SST windowing (today one
  global `therm["dhw"]` is reused for all dates), proper short-window
  guards (replace fillna(0)/min_periods=1), drop dead 0-gain transplant
  features, and validate the actual deployment 37 (benchmark uses raw
  CoRTAD columns, not the 37). Gain: removes bias, aligns offline/online.
- **R6 Calibration overhaul (S–M).** Per-class (or Dirichlet) temperature
  with cross-fit, reliability diagrams + Brier/BSS vs persistence
  (ACCESS-S1 practice,
  [Front. Mar. Sci. 2021](https://www.frontiersin.org/articles/10.3389/fmars.2021.687833/full));
  fix GCBD ECE ~0.10–0.11 overconfidence (T≈1.77) vs synthetic 0.03.
- **R7 SITS for gappy series (L).** ALISE-style alignment for irregular /
  cloud-gappy SITS ([arXiv 2407.08448](https://arxiv.org/pdf/2407.08448))
  + Mamba head ([SITSMamba](https://github.com/XiaoleiQinn/SITSMamba)),
  masked-only loss, seeded masking, val early-stop — always vs a
  from-scratch UNet (MAE pretraining can lose on dense tasks).
- **R8 Per-basin models + outlook covariate (M).** Train per basin, drop
  geo coordinates (leakage-adjacent: Latitude_Degrees gain 18395 adds
  zero accuracy), add CFSv2/Outlook or short-range forecast SST as a
  lead-time feature; plot skill-vs-lead × severity (DHW≥4 vs ≥8).
  Deployment wiring shipped Nov 11: `quadrant_grid` basin axis (single
  source of truth for train-time and export-time assignment),
  `train_per_basin_models` / `predict_per_basin` with the min-n /
  single-class global-fallback rule, `--per_basin` end-to-end (heads train
  on the train split only, routed skill scored on the same held-out test
  split — no leakage). Measured Nov 11 (same flags + `--per_basin`,
  run `f6b92864375b`): deployed 0.9625 / ROC 0.9479 vs global 0.9655
  (delta −0.003; NE 0.9616, NW 0.9649, SE 0.9617, SW 0.9618 over
  ~5.6–6.5k test rows per basin) — parity, expected on the smooth toy
  field. Deep-research follow-up (Nov 11, full-scale GCBD probes, n=4113,
  400 trees, same-split both arms): shrunk per-realm LightGBM blend
  (`models/hierarchical_bleach.py`, w=n/(n+kappa), kappa=500 wins all 3
  tune seeds) beats the global head on grouped-site (+0.005 mean, positive
  all 3 seeds) and forward-temporal (+0.019 mean, both seeds — the
  subsample fold-1 collapse did not persist); train-year-only adaptive
  heat features (`eval/adaptive_features.py`, +0.035/+0.042 temporal,
  both seeds, fold spread dominates); CORAL covariance alignment
  (`eval/coral_da.py`) HURTS overall (LORO 0.41 vs 0.49 — helps only
  Tropical Atlantic +0.04, wrecks TNP 0.64→0.10 and EIP) and was removed,
  not merged. Strict LORO stays ~0.49 by construction (unseen realm has
  no head); grouped-site is the shippable gain.
- **R9 Atlas-style pilot redo (M).** Warning-triggered biweekly
  baseline-vs-2wk-post median brightening with QA60/SCL masking (Allen
  Atlas design); fix the 2003–04 vs 2022 mismatch; handle 96% imbalance.
  Shipped Nov 11: retry-hardened COG fetch (backoff 5/10/20 s), label
  `balance_report` with inverse-frequency weights, pilot runs recorded in
  the registry. First credible real-site number lives here.
- **R10 Test/registry hardening (S).** Pin GCBD 0.71/0.86 + ECE/LORO +
  byte-identical determinism, rule-label tiers, `load_stack` parsing,
  export/CLI flags, stronger-than-trivial gates. Shipped Nov 11: 13 roadmap
  tests plus `eval/run_registry.py` stamping
  run-id/seed/git-hash/argv into every metrics JSON and `runs/registry.jsonl`
  (35/35 green after the global-predictor probes added 7 more).

Deferred with reason: GNN / CNN-LSTM / PINN bleaching heads — no credible
peer-reviewed skill numbers found (2022–2026 hits were non-peer-reviewed
pages without DOIs/data); revisit only after an XGBoost+blocked-CV
baseline exists, on dense time series. Foundation models (Prithvi-WxC,
SatMAE, Chronos/Lag-Llama): reliable SST only to ~6–12 h / ~7 d
([arXiv 2409.13598](http://arxiv.org/html/2409.13598v1),
[Bustos-Usta 2025](https://www.mdpi.com/2072-4292/17/3/517)) — future SST
feature providers, not bleaching predictors.

## 7. Open gaps (nothing omitted)

G1 Cross-basin transfer (biggest): LORO-by-realm 0.46/0.65 vs stratified
0.71/0.86 — thermal-gap / shifting baselines, not code. Shipped: per-basin
rule + LORO harness + shrunk per-realm LightGBM blend (partial pooling
*inside* LightGBM, kappa=500; grouped-site +0.005, temporal +0.019 at
full scale). CORAL alignment tested and rejected (LORO 0.41 vs 0.49).
Remaining: AIMS 2022 pull for data scale; strict LORO is structural, so
ship grouped-site gains and stop chasing LORO with blending.
Do NOT claim global skill.
G2 S2 heatwave pilot with contemporaneous labels (blocked, 6 failed
attempts): BCO-DMO GCBD ends 2016; NCEI full GCBD (accession 0228498,
33,244 records 1963–2021 — landing page verified reachable with plain
HTTPS+FTP links, bytes not yet pulled); figshare has no copy;
Japan ReefCheck ends 2015; CoralWatch
lives in the BioCollect project store (needs project-API work); Allen
Atlas layers unverified (likely GEE asset); AIMS dashboard is a JS shell.
Ranked next: AIMS LTMP 2022 via data.aims.gov.au, 2020 heatwave fallback,
Allen Atlas via GEE (needs auth), MOE endpoint hunt.
G3 mamba-ssm uninstallable here (no Windows wheels; source build fails).
GRU+Transformer verified; `mamba_ssm` stays optional with fallback.
G4 ACOLITE uninstallable here (not on PyPI; source install hung 30+ min).
DOS + DII is the verified chain.
G5 Prithvi-EO-2.0 path is a stub (`load_prithvi` raises unless terratorch
configured). Local SatMAE pretraining is real and tested.
G6 SITS demo margin thin (acc ~0.54 toy). Real fix = real SITS labels (G2).
G7 Pilot temporal mismatch (labels 2003–2004 vs scenes 2022) — consistency
checks, not validation.
G8 Class imbalance (pilot 96% no-stress; synthetic Watch 10%) — check
per-class recall, not accuracy.
G9 Research items not implemented: DLinear/PatchTST/xLSTM SST heads,
RT-consistency loss, delta-DII head, BAA-tier ordinal targets, INT8 delta
(unmeasured), hierarchical pooling, Himawari diurnal SST, PACE covariates,
NCRMP eval, DESIS/SkySat licensing, GCRMN 2025 / MOE endpoints, JGR/GRL
landmarks (unverifiable — do not cite).
G10 Outputs overwritten every run — mitigated Nov 11: run-id/seed/hash/argv
are stamped into each metrics JSON and logged to `runs/registry.jsonl`, but
the .tif/.png files themselves are still overwritten, so copy them aside
for head-to-head map comparisons.
G11 `pilot_sekisei.py` is fetch-only; pilot runs go through the forecaster
CLI. eAtlas sharealias `/download` links 404 — do not chase further.
G12 Environment: project-local `.venv` (Python 3.11, torch 2.15 CPU,
rasterio, lightgbm, sklearn, geopandas, xarray). Reinstall:
`uv pip install --python .\.venv\Scripts\python.exe <pkgs>`. No
`.gitignore` committed (policy) — `git add` named files only, never
`git add -A`. PowerShell exit-1-on-pass is a stderr artifact.
New audit gaps (Nov 11): N1 rule-label circularity (synthetic skill is a
smoke test); N2 single global DHW reused for all dates in
`build_temporal_dataset`; N3 SSTODE train/infer seasonal-phase mismatch;
N4 several transplant gains exactly 0.0 (dead features shipped); N5
temperature/ECE fit on the test split (optimistic); N6 `sst_noaa_daily.csv`
name implies observations, content is synthetic.

## 8. Resume guide

```bash
# fastest re-verification (~1 min here; 13–15 min with torch demos on full env)
python -m pytest tests/test_all7.py -q
# full synthetic run, all flags
python coral_bleaching_forecaster.py --temporal sits --calibrate --eval_report --loro --site_labels_csv data/demo_sites.csv
# real-data benchmark (thermal-only + with-geo comparison)
python eval/gcbd_benchmark.py --drop-geo --out eval/gcbd_benchmark_nogeo.json
python eval/gcbd_benchmark.py
```

Project log: Nov 8 built 41 files + 13 probes, 15/15 tests, metrics above,
committed `d924ea5` (no remote yet), 12 gaps logged. Nov 9 pushed to
CoralCast main (10 commits, 40 files, in sync), README data-sources
`f0298cc`, freed ~1.07 GB (`.venv`), layered commits around pilot-tif 408
timeouts. Nov 11 built the full R1–R10 roadmap (honest splits, harder
baselines, RCNN/SSTODE/SITS rework, feature hygiene, cross-fit calibration,
per-basin models, pilot hardening, 13 roadmap tests + run registry),
re-measured all numbers above, 28/28 tests green. Per-basin deployment
wiring (R8, `--per_basin`) followed the same day, 30/30 tests green.
Deep-research combo followed: hierarchical shrink (kappa=500, +0.005
grouped / +0.019 temporal) and adaptive heat features (+0.038 temporal)
both win at full scale; CORAL rejected (0.41 vs 0.49 LORO); 35/35 tests
green, CORAL module removed unmerged.
Next session: re-run the fastest line above, then either pull the NCEI
33k-record archive (landing page verified, bytes not yet pulled) or wire
the hier-500/adaptive winners into `run_benchmark` for tracked numbers.

## 9. References

- NOAA CRW 5km v3.1 DHW tutorial; CRW thermal history; R `dhw` package
  (v3.1 methods); NOAA CRW 4-month Outlook (CFSv2).
- Eakin et al. 2018, Front. Mar. Sci. (Outlook system).
- CoRTAD v6 (NCEI product page + accession NCEI-CoRTADv6, 1982–2022).
- Allen Coral Atlas methods + bleaching method PDF (v1).
- Univ. Miami XGBoost early-warning, Environ. Res. Commun. 2025,
  doi:10.1088/2515-7620/ae2570 (6-week horizon).
- Mestry 2026, EarthArXiv 15638 + Zenodo 10.5281/zenodo.23511976
  (honest-CV collapse; ReefCheck duplicates).
- Coral Reefs 2024, doi:10.1007/s00338-024-02512-w (redefined DHW, +70%).
- Frontiers Mar. Sci. 2026 thermal-gap paper (DHW AUC → 0.57 hottest seas).
- Bustos-Usta et al. 2025, Remote Sens. 17(3):517 (Chronos SST ~7 d).
- Prithvi-WxC, arXiv 2409.13598; SSTODE, arXiv 2511.05629.
- PeerJ 21269 (250 m downscaled DHW); residual U-Net downscaling,
  arXiv 2608.10022.
- SatMAE (NeurIPS 2022); SITSMamba; ALISE 2024; SatMamba survey.
- Roberts et al. 2017 / Valavi blockCV (spatial CV); ACCESS-S1
  Front. Mar. Sci. 2021 (reliability/Brier practice).
- Sully et al. 2019, Nat. Commun. (ReefCheck analysis); van Woesik &
  Burkepile 2022, Sci. Data (GCBD).
