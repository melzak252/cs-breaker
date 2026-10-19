# Kedro 8-Layer Data Engineering Catalog — CS-Breaker

## 1. Hierarchy Overview

The data architecture adheres to the 8-layer progressive disclosure standard:
- **`01_raw`**: Immutable raw data exports, scraper extracts, and original archives.
- **`02_intermediate`**: Cleansed, normalized, and typed entity records.
- **`03_primary`**: Canonical domain tables with consolidated match identities.
- **`04_feature`**: Feature matrices, lag indicators, and multi-system ratings.
- **`05_model_input`**: Pinned, reproducible training/validation/test partitions.
- **`06_models`**: Frozen estimator checkpoints, model bundles, and rating states.
- **`07_model_output`**: Out-of-time probability predictions and paired comparison panels.
- **`08_reporting`**: Metric ledgers, drift tables, simulation outputs, and markdown reports.

---

## 2. Layer Specifications

### Layer 01: Raw Data Sources (`data/01_raw/`)

| Dataset Key | File Path | Expected Rows | SHA-256 Checksum | Description |
|---|---|---:|---|---|
| `kaggle_results_only_raw` | `data/01_raw/kaggle_results_only.csv` | 7,033 | `c5c243fb8be2618f8aebc024a8034cec74c6142f7c378b21a0b1330648efdf27` | Historical CS2 match results (2024–2025) from Kaggle export. |
| `hltv_matches_raw` | `data/01_raw/hltv_matches.csv` | 122,038 | `7ff59a5d942210ce3be2e94c2ae4529f3a5326b8373ba7be4929818cf6b9b606` | Full historical HLTV professional match results archive. |
| `strandedpond_matches_raw` | `data/01_raw/strandedpond_matches.csv` | 2,299 | `c2b0b2c9ea404ecc5f4f98256fd12a51f22e89d2c7dbbca144dec2368fc763e4` | StrandedPond map-level professional series records. |
| `opencs2_maps_raw` | `data/01_raw/opencs2_maps.csv` | 1,988 | — | OpenCS2 SteamID-level round and map event archive. |

### Layer 02: Intermediate Data (`data/02_intermediate/`)

| Dataset Key | File Path | Description |
|---|---|---|
| `match_lineups_processed` | `data/02_intermediate/match_lineups.csv` | Extracted and normalized starting five player rosters per match. |

### Layer 03: Primary Domain Tables (`data/03_primary/`)

| Dataset Key | File Path | Expected Rows | SHA-256 Checksum | Description |
|---|---|---:|---|---|
| `master_matches_consolidated` | `data/03_primary/master_matches_consolidated.csv` | 122,038 | `cbd32f49584dcf8528b5ea7b0169d5788cffb4602815b80f32ff8ad94ecbc0d8` | Deduplicated, joined master match table with canonical HLTV identifiers. |

### Layer 04: Feature Matrices (`data/04_feature/`)

| Dataset Key | File Path | Expected Rows | SHA-256 Checksum | Description |
|---|---|---:|---|---|
| `match_features_with_ratings` | `data/04_feature/match_features_with_ratings.csv` | 675 | `8f78a0efc68fcac4ba01c40e7165b98e700326280800578310bb7f935f5e3639` | 38 pre-match predictors including TrueSkill, Glicko-2, Elo, and 20-series form. |
| `player_rankings_master` | `data/04_feature/player_rankings_master.csv` | 621 | `c7d6f4971138eebc45b5db623787e5e615da63acb94786754062a1a87701724a` | Multi-system ratings per professional player identity. |

### Layer 05: Model Inputs (`data/05_model_input/`)

| Dataset Key | File Path | Expected Rows | SHA-256 Checksum | Description |
|---|---|---:|---|---|
| `canonical_benchmark_paired` | `data/05_model_input/canonical_benchmark_paired.csv` | 203 | `a6b3cd4a237bb940a7ffde8fa6b1ea118da5ae18e26749d3cbef6a2243c56cea` | Aligned test cohort with pre-match model probabilities and bookmaker quotes. |
| `development_features` | `data/05_model_input/development_features.csv` | — | — | Training and validation partitions for model candidate selection. |

### Layer 06: Model Checkpoints (`data/06_models/`)

| Dataset Key | File Path | SHA-256 Checksum | Description |
|---|---|---|---|
| `baseline_model` | `data/06_models/baseline_model.pkl` | `1733cc14e7aaeab6d29dad8b561fa4c0ae5f08b919e48ce5f2073bda62348357` | Retrospective team Elo model checkpoint ($K=32$). |
| `player_rating_state` | `data/06_models/player_rating_state.pkl` | `78e0fd7f8b79bbc8b1a64fe86936afb29647bebe27a1690dab58b09bef0c3215` | Multi-system player rating historical state bank. |
| `professional_selected_model` | `data/06_models/professional_selected_model.pkl` | `2006f25bf1a889588c75bf8e82cc6132bf32dfdfd596632ee1853a6ce10b8cde` | Ridge Bayesian dynamic challenger model checkpoint. |

### Layer 07: Model Outputs (`data/07_model_output/`)

| Dataset Key | File Path | Expected Rows | SHA-256 Checksum | Description |
|---|---|---:|---|---|
| `baseline_test_predictions` | `data/07_model_output/baseline_test_predictions.csv` | 164 | `3af807b37d6f04ac33fa44dd36c1dc98edfa39b8cbbceca62861b73131ef96da` | Out-of-time test cohort predictions from the baseline model. |
| `bookmaker_comparison_matches` | `data/07_model_output/bookmaker_comparison_matches.csv` | 203 | `a6b3cd4a237bb940a7ffde8fa6b1ea118da5ae18e26749d3cbef6a2243c56cea` | Matched test predictions joined with Pinnacle, Bet365, and Unibet quotes. |

### Layer 08: Reporting (`data/08_reporting/`)

| Dataset Key | File Path | Description |
|---|---|---|
| `baseline_model_metrics` | `data/08_reporting/baseline_model_metrics.json` | Comprehensive baseline evaluation metrics, calibration bins, and fold summaries. |
| `simulation_pipeline_report` | `data/08_reporting/simulation_pipeline/SIMULATION_REPORT.md` | Generated markdown report from the 5-stage simulation pipeline. |
