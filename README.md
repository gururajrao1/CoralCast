# Multispectral Coral Bleaching Predictor

End-to-end 30-day coral bleaching-risk forecaster (Project 6.3) plus the
seven deep-research extensions.

## What this app does

CoralCast predicts **coral bleaching risk 30 days ahead** by fusing two
independent signals: what the reef looks like from space right now, and how
much heat stress the water has accumulated.

**1. It reads the reef from Sentinel-2 satellites.** Given multispectral
imagery (bands B2 blue, B3 green, B4 red, B8 near-infrared), it computes
coral-health indices per pixel: submerged NDVI, a blue/green color index,
and Lyzenga depth-invariant indices (DII_23/24/34 via per-scene sand-ROI
regression) that cancel out water-depth effects so shallow sand is not
mistaken for healthy coral. Atmospheric correction (dark-object
subtraction) and a water-column adjustment are applied first.

**2. It quantifies ocean heat the way NOAA does.** From a daily sea-surface
temperature (SST) series it builds the standard Degree Heating Week (DHW)
metric over 12-week windows against the local maximum-monthly-mean (MMM)
baseline, plus Hobday marine-heatwave events and categories, heating rates,
hot streaks, days above 30 C, growing-degree-days, and thermal-variance
features (variance is protective — steady heat bleaches less than spiky
heat at the same mean).

**3. It learns risk from pixels + heat, honestly calibrated.** A LightGBM
classifier with monotonicity priors (risk rises with heat, falls with
variance) predicts three levels per pixel — 0 No Stress, 1 Watch/Warning,
2 Severe Bleaching Expected. An optional SITS temporal head (GRU/Transformer
with masked-reconstruction pretraining) reads multi-date stacks; an SSTODE
neural-ODE module forecasts SST itself; learned RCNN downscaling sharpens
coarse SST; temperature scaling calibrates the probabilities so a "70%"
means 70%. Leave-one-region-out cross-validation and a severity/lead-time
harness report how skill degrades with distance and forecast horizon.

**4. It trains on real bleaching observations, not just physics.** A join
pipeline matches Reef Check survey records with CoRTAD SST and DHW
baselines; a site-supervised path samples water pixels around survey
coordinates to train the pixel head; the GCBD benchmark (4,219 real rows)
scores the model against a DHW-only baseline with coordinates ablated.

**Outputs.** Every run writes a georeferenced 30-day risk map
(`coral_bleaching_risk_forecast.tif`), an evaluation report
(`bleaching_evaluation_metrics.json`: ROC-AUC, precision-recall, DHW
importance, calibration, LORO), and quicklook plots
(`forecast_quicklook.png`, `eval/reliability.png`).

**Bottom line from measurement, not hype.** Thermal-only model: 0.71
accuracy vs 0.31 DHW baseline on real GCBD outcomes (ROC-AUC 0.86, matching
published skill). Synthetic pipeline: ROC-AUC 0.947 with calibration error
cut from 0.072 to 0.030. Known limit, stated plainly: skill does not
transfer across ocean basins (LORO 0.46–0.65), so the deployment rule is
train-per-basin — see G1/G2 in the gaps section.

## Quick start

```bash
uv venv
uv pip install --python .venv/Scripts/python.exe rasterio lightgbm scikit-learn geopandas matplotlib xarray numpy torch --index-url https://download.pytorch.org/whl/cpu
python coral_bleaching_forecaster.py
```

Outputs: `coral_bleaching_risk_forecast.tif`, `bleaching_evaluation_metrics.json`.

## The 7 research extensions

| # | Module | Run |
|---|--------|-----|
| 1 | Real labels: Reef Check + SST join + DHW baseline | `python data/join_reefcheck_cortad.py --demo` |
| 2 | Learned SST downscaling (RCNN) | `python models/sst_downscale_rcnn.py --demo` |
| 3 | SITS temporal encoder (Mamba/GRU) | `python models/sits_mamba.py --demo` |
| 4 | SatMAE pretraining (+Prithvi option) | `python models/pretrain_satmae.py --demo` |
| 5 | SSTODE physics-informed SST forecast | `python models/sstode_forecast.py --demo` |
| 6 | Multivariate features + SHAP pruning | `python features/multivariate.py --demo` |
| 7 | Severity/lead-time eval harness | `python eval/severity_leadtime.py --demo` |

Wired into the main CLI: `--sst_model sstode`, `--multivariate`,
`--eval_report`, `--labels_csv <join output>`, `--temporal sits`,
`--stack_dir <dir>`, `--calibrate`.

Also in the main path (no flags needed): DHW re-tuning grid (thresholds
MMM+0.5/+1.0 × 6/12-week windows), Hobday MHW event/category features,
Lyzenga depth-invariant indices (DII_23/24/34 via per-scene sand-ROI
regression), and cross-domain transplants (coral GDD, CWSI normalization,
days-above-30C, 7-day heating-rate + hot streaks, variance-protection
windows, SPEI-style surplus, DLNM lags, light×heat dose). SITS head does
masked-reconstruction pretraining + supervised fine-tuning
(SITSMamba-style two-stage). LightGBM carries monotonicity priors
(heat ↑, variance ↓). `--loro` runs leave-one-region-out quadrant CV.

## Real-world benchmark (GCBD, n=4,219, `python eval/gcbd_benchmark.py`)

Model vs DHW-only baseline on real bleaching outcomes: accuracy 0.71 vs
0.31, ROC-AUC 0.86, per-class AP 0.89/0.75/0.76 — matching published
absolute-temperature skill (0.884), with thermal-only features (no lat/lon).
Leave-one-realm-out CV collapses to 0.46–0.65: skill does not transfer
across basins. Within-basin training holds: Tropical Atlantic 0.68/0.84,
Central Indo-Pacific 0.70/0.86, both well above baseline. Deployment rule:
train per basin (or basin-calibrate), never one global model.

## Pilot status

Round 1 (Sekisei Lagoon, all real): 69 Reef Check labels, 2-date S2 stack,
1,033-day MUR SST — model/ DHW baseline both 1.0 on 6 in-window sites;
DHW baseline 0.681 on all labels; model ROC-AUC 0.534 (no in-window
heatwave, so nothing to discriminate). Round 2 scouting (blocked on
S2-era labels, 6 access attempts): BCO-DMO GCBD copy downloaded (4,219
rows, 1980–2016, zero S2-era bleached rows); NCEI full-GCBD download flow
JS/302-gated; figshare API search found no GCBD copy; Japan Reef Check
ends 2015; CoralWatch absent from ALA biocache (dr17828 + dataset_name
both 0 — data lives in BioCollect project store). Next, easiest first:
AIMS LTMP 2022 in-water data via data.aims.gov.au; 2020 heatwave year as
fallback; Allen Atlas layers via GEE asset; MOE Sites 1000 endpoint hunt.

Full pipeline with everything on:

```bash
python data/join_reefcheck_cortad.py --demo
python coral_bleaching_forecaster.py --sst_model sstode --multivariate --eval_report --labels_csv data/labels.csv --temporal sits --calibrate --loro --site_labels_csv data/demo_sites.csv
python eval/gcbd_benchmark.py --drop-geo --out eval/gcbd_benchmark_nogeo.json
python -m unittest tests.test_all7
```

## Project handoff — state, verification log, and every open gap

Last verified: 15/15 tests pass (`python -m unittest tests.test_all7`).
Synthetic headline numbers: ROC-AUC 0.947, accuracy 0.966, per-class AP
0.926/0.859/0.965, calibration ECE 0.072 -> 0.030 (T=0.80), quadrant-LORO
0.965 +/- 0.001. Real-data headline numbers (GCBD n=4,219, thermal-only):
accuracy 0.71 vs DHW baseline 0.31, ROC-AUC 0.86, per-class AP
0.89/0.75/0.76. Main script has 37 features
(`FEATURE_NAMES`; pixel list `PIXEL_FEATURES`, thermal list
`THERMAL_FEATURES`, shared assembler `pixel_block` used by training,
site path, and forecast export).

### Verified working (do not redo)

- End-to-end runs: synthetic demo, real Sekisei pilot (S2 + MUR SST +
  Reef Check labels), GCBD tabular benchmark, all CLI flags
  (`--sst_model sstode`, `--multivariate`, `--eval_report`,
  `--labels_csv`, `--temporal sits`, `--stack_dir`, `--calibrate`,
  `--loro`, `--site_labels_csv`, `--site_radius_px`).
- Determinism fixes kept: torch nets are built after `manual_seed`
  (init-before-seed caused run-order-dependent results); SSTODE trains on
  standardized series with one-step-ahead loss (rollout BPTT diverged);
  SITS uses two-stage pretrain-then-finetune (joint MT loss fought itself);
  NLL temperature fit uses an explicit softmax (operator-precedence bug
  fixed); `load_stack` accepts >= 2 dates; LORO ROC skips folds missing a
  class instead of crashing.
- Data provenance: `data/ReefCheck_CoRTAD.csv` + `data/ReefCheck_raw.csv`
  (GitHub, ~3 MB each), `data/gcbd_bleaching_environmental.csv` (BCO-DMO
  dataset 773466, 4,219 rows), `data/pilot_stack/` (8 real S2 L2A COG
  windows, tile 51RWH, 2022-08-25 + 2022-09-29, Sekisei box
  lon 123.65-123.80 / lat 24.30-24.42), `data/pilot_sst.csv` (1,033 d MUR
  SST at 24.31N 123.71E via coastwatch ERDDAP, MMM 29.98 C),
  `data/pilot_labels.csv` (69 rows, Sekisei bbox + all bleached Japan
  rows), `data/demo_sites.csv` (rule-consistent synthetic demo sites).
- Earth Search STAC works with POST + full RFC3339 datetimes
  (`https://earth-search.aws.element84.com/v1/search`); MUR SST via
  `https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.csv`
  (chunk yearly; single-point queries are reliable, multi-year single
  queries drop); GeoNetwork POST search works
  (`catalogue.eatlas.org.au/geonetwork/srv/api/search/records/_search`);
  CoralWatch data resource UID is `dr17828` (BIE), BioCollect project
  `9c55416c-f56a-4917-a65d-daadcb8f51f7`.

### Open gaps (nothing omitted)

- G1 Cross-basin transfer (biggest). LORO-by-realm 0.46 acc / 0.65 AUC
  vs stratified 0.71 / 0.86. Root cause per literature: thermal-gap /
  shifting baselines, not code. Mitigation shipped: per-basin deployment
  rule + LORO harness. Real fix needs: AIMS 2022 pull, or hierarchical
  partial pooling (not implemented — LightGBM can't do it; would need a
  mixed model), or domain adaptation. Do NOT claim global skill.
- G2 S2 heatwave pilot with contemporaneous labels (blocked, 6 failed
  attempts documented): BCO-DMO GCBD copy ends 2016 (zero S2-era bleached
  rows); NCEI full GCBD (35k records, accession 0228498) download flow is
  JS/302-gated (`.../archive/accession/download/228498` fails);
  figshare API search found no GCBD copy; Japan Reef Check ends 2015;
  CoralWatch returns 0 records under `dr17828` and `dataset_name`
  (data lives in the BioCollect project store — needs project-API work);
  Allen Atlas layers path unverified (Next.js app, likely GEE asset);
  MOE Sites 1000 endpoint unverified; AIMS dashboard is a JS shell with
  no reachable data API and the 2021/22 status PDF is sector-level prose
  with no reef tables. Ranked next: AIMS LTMP 2022 via data.aims.gov.au
  (needs navigation/request), 2020 heatwave fallback, Allen Atlas via
  GEE (needs auth), MOE endpoint hunt.
- G3 mamba-ssm uninstallable here: no Windows wheels, needs Linux +
  CUDA, source build fails (`bare_metal_version` NameError). GRU +
  Transformer backbones verified; `mamba_ssm` import stays optional with
  GRU fallback + warning. Revisit only on a Linux+CUDA machine.
- G4 ACOLITE uninstallable here: not on PyPI; GitHub source install
  hung 30+ min and was killed; venv verified healthy after. DOS +
  Lyzenga DII is the verified aquatic chain. Revisit with conda or a
  Linux box if a BOA comparison is wanted.
- G5 Prithvi-EO-2.0 path is a stub (`load_prithvi` raises informative
  error unless terratorch is configured). SatMAE local pretraining is
  real and tested.
- G6 SITS demo margin is thin (acc 0.54 on toy data; recon verified
  below variance). Real fix = real SITS labels (blocked by G2).
- G7 Pilot validation caveat: Sekisei survey labels are 2003-2004, S2
  scenes 2022 — temporal mismatch, so site checks are consistency
  checks, not validation. The 6 in-window sites were all observed 0.
- G8 Class imbalance: real pilot is 96% no-stress; synthetic Watch class
  is 10%. Any future heatwave run must check per-class recall, not just
  accuracy.
- G9 Research items NOT yet implemented (all sourced in session):
  DLinear/PatchTST/xLSTM SST heads, RT-consistency loss off the Lee
  model, delta-DII change-detection head, BAA-tier ordinal targets,
  INT8-quantization delta (unmeasured), hierarchical partial pooling,
  diurnal Himawari SST feature, PACE water-quality covariates, NCRMP
  US-reef eval, DESIS/SkySat licensing + quotas, GCRMN 2025 raw endpoint,
  MOE/COrMoC endpoint, JGR/GRL/TGRS bleaching landmark papers (could not
  verify landmark claims — do not cite any).
- G10 Metrics/outputs are overwritten every run (`coral_bleaching_risk
  _forecast.tif`, `bleaching_evaluation_metrics.json`,
  `forecast_quicklook.png`, `eval/reliability.png`). Copy them aside
  before comparing runs.
- G11 `pilot_sekisei.py` is fetch-only; the pilot run itself is the
  forecaster CLI with `--blue/--green/--red/--nir` + `--sst_csv` +
  `--mmm` + `--stack_dir` + `--labels_csv` (see Quickstart). eAtlas
  nextcloud sharealias downloads do NOT resolve to direct links
  (`/download` variants 404) — do not chase further without the alias
  API.
- G12 Environment: project-local `.venv` (Python 3.11) has torch 2.15
  CPU, rasterio, lightgbm, sklearn, geopandas, xarray, pdfplumber.
  Reinstall pattern that worked: `uv pip install --python
  .\.venv\Scripts\python.exe <pkgs>` (+ torch CPU index URL). No
  `.gitignore` is committed (policy) — always `git add` named files,
  never `git add -A`, to keep `.venv/` out. PowerShell shows exit 1 on
  unittest runs even when tests pass (stderr capture artifact) — trust
  the `OK` line, not the exit code.

### Resume commands

```bash
# fastest re-verification (13-15 min, mostly torch demos)
.\.venv\Scripts\python.exe -m unittest tests.test_all7
# full synthetic run, all flags
.\.venv\Scripts\python.exe coral_bleaching_forecaster.py --temporal sits --calibrate --eval_report --loro --site_labels_csv data/demo_sites.csv
# real-data benchmark (thermal-only + with-geo comparison)
.\.venv\Scripts\python.exe eval/gcbd_benchmark.py --drop-geo --out eval/gcbd_benchmark_nogeo.json
.\.venv\Scripts\python.exe eval/gcbd_benchmark.py
```

### Session log — 2026-11-09 (resume, verify, benchmark, push)

- Re-verified from scratch: `tests.test_all7` 15/15 OK (166 s).
- GCBD thermal-only (`--drop-geo`): model acc 0.71 vs DHW baseline
  0.3118, LORO mean acc 0.4757. Output byte-identical to committed
  `eval/gcbd_benchmark_nogeo.json` (deterministic).
- GCBD with-geo: model acc 0.709 vs baseline 0.3118, LORO 0.4645.
  Geo features add nothing — thermal-only is the deployable path.
- Full synthetic run (all flags): ROC-AUC 0.947, acc 0.965,
  SITS 0.965 vs LightGBM 0.965, calibration T=0.7628
  (ECE 0.0717 -> 0.0299), quadrant-LORO 0.9652 +/- 0.0011,
  site-supervised demo acc 0.4904 (n=1,210 px / 10 synthetic sites),
  severity/lead-time ECE 0.0294 over 24 lead x region rows.
  Synthetic scene sits under heat (DHW 5.60) so the forecast map is
  100% class 2 — expected for the demo, not a bug.
- Remote set to `https://github.com/gururajrao1/CoralCast` (origin).
- Upgrades deliberately deferred (see G9): DLinear/PatchTST/xLSTM SST
  heads, RT-consistency loss, delta-DII head, BAA-tier ordinal targets,
  INT8 quantization, hierarchical partial pooling, Himawari diurnal
  SST, PACE covariates, NCRMP eval, GEE/GCRMN/MOE data pulls. G1 (cross-
  basin transfer) and G2 (S2-era labels) remain the two load-bearing
  gaps; everything else needs data access or a Linux+CUDA box, not
  more code in this repo state.
