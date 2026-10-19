# Audited CS2 model experiment: observed results, 2026-10-09

## Bottom line

The fixed experiment selected `blend_player_archive_elo` by **VALIDATION log loss**, not TEST performance. It is exactly 0.5 × player Elo probability + 0.5 × no-decay archive Elo probability: no learned blend weights and no bookmaker inputs. On the same 164 previously observed TEST series, log loss fell from 0.644669 to 0.640090. This is a modest **exploratory observed gain**, not fresh confirmation, a universal best model, SOTA, or an established statistical improvement.

**Important regressions:** ten-bin ECE worsened from 0.087954 to 0.102596; the final chronological TEST block worsened in log loss and Brier; and the identical 33-match all-bookmaker cohort worsened from 0.641430 to 0.664955 log loss. All primary model-minus-book and overall before/after paired intervals include zero. There is **no demonstrated bookmaker edge, ROI, CLV, executable-price advantage, or actionable betting claim**.

Research rationale: [research synthesis](cs2-model-research-2026-10-09.md) and [source ledger](cs2-model-research-sources-2026-10-09.json). This results report uses local execution evidence, not external benchmark rankings.

## Cohort and provenance

The frozen [protocol](cs2-model-experiment-protocol-2026-10-09.json) was saved before evaluating new candidates; the previous TEST had already been observed. The master archive has 122,038 native HLTV IDs. Exact existing lineup eligibility yields 675 target series: 670 historical map-derived lineups and five match-page captures, with 621 observed ranked players. Nickname namespaces and native IDs remain separate; no inferred cross-source merge. Kaggle scrape-time roster/stat snapshots are excluded.

| Partition | Source-calendar dates | Series |
|---|---|---:|
| TRAIN | 2025-12-08–2026-06-07 | 396 |
| VALIDATION | 2026-06-09–2026-08-13 | 115 |
| TEST | 2026-08-14–2026-10-07 | 164 |

`after/summary.json` verifies all 675 IDs, dates, team orientation, labels, split membership and all 31 original predictor values unchanged. Seven archive fields extend the schema to 38 predictors. Archive history contains 23,568 valid complete BO3/BO5 results, 22,893 history-only rows, 1,768 literal team names and 992 source days (2024-01-09–2026-10-07). The minimum eligibility date is 2024-01-01: a modern-era proxy, not independently certified CS2-version or competition-level evidence. Literal names are not organization IDs or an alias resolver. Broader history never supplies new target labels. Audit: [`modeling_data_audit.json`](../data/audits/modeling_data_audit.json).

Every source day's target probabilities/features precede all updates from that day. Ratings replay earlier evaluation-day outcomes into later days; estimator/scaler/calibration fits use TRAIN only, with no TRAIN+VALIDATION refit. Archive Elo uses K=32, scale 400, initial 1500; the alternative decay channel has a fixed 180-day half-life. Neutral cold starts carry explicit support counts. Historical played lineups, source dates and postgame records do not prove prematch publication, original timezone, substitution coverage or obtainable information. This is a retrospective actual-lineup-conditional series experiment.

## Fixed candidate comparison

Minimum VALIDATION log loss among the 16 predeclared candidates selects the recipe; exact ties favor incumbent player Elo, then declared order. There are only 115 validation cases, so selection optimism is a material limitation. Original classifiers retain the original feature schema/hyperparameters. Scalar calibration uses TRAIN prequential raw logits only, a nonnegative zero-intercept slope bounded from zero to three. Compact ridge recipes and both half/half blends are specified in the protocol. TEST columns below are descriptive, not a second selection criterion.

| Candidate | VALIDATION LL | TEST LL | TEST Brier | TEST accuracy | TEST AUC |
|---|---:|---:|---:|---:|---:|
| `pure_elo` | 0.684419 | 0.644669 | 0.226488 | 60.98% | 0.676777 |
| `pure_glicko` | 0.902859 | 0.799350 | 0.272424 | 58.54% | 0.598192 |
| `pure_trueskill` | 0.821497 | 0.714018 | 0.248486 | 57.32% | 0.618413 |
| `pure_openskill` | 0.836523 | 0.727081 | 0.242098 | 60.98% | 0.657629 |
| `logistic_regression` | 0.746588 | 0.646871 | 0.227395 | 64.02% | 0.670343 |
| `gradient_boosting` | 0.839654 | 0.667481 | 0.227965 | 64.02% | 0.695466 |
| `pure_archive_elo` | 0.683847 | 0.646536 | 0.226739 | 64.63% | 0.667739 |
| `pure_archive_decay` | 0.701231 | 0.669945 | 0.238606 | 64.02% | 0.619179 |
| `calibrated_elo` | 0.710921 | 0.645623 | 0.225417 | 60.98% | 0.676777 |
| `calibrated_glicko` | 0.696819 | 0.676045 | 0.241197 | 58.54% | 0.598192 |
| `calibrated_trueskill` | 0.703720 | 0.666060 | 0.236248 | 57.32% | 0.618413 |
| `calibrated_openskill` | 0.711590 | 0.659663 | 0.230505 | 60.98% | 0.657629 |
| `ridge_rating_logits` | 0.701955 | 0.650399 | 0.226741 | 64.63% | 0.677083 |
| `ridge_strength_form` | 0.692460 | 0.652714 | 0.227289 | 64.02% | 0.673407 |
| `blend_player_archive_elo` | 0.673296 | 0.640090 | 0.224272 | 64.02% | 0.681219 |
| `blend_player_archive_decay` | 0.688632 | 0.653052 | 0.230465 | 62.20% | 0.667279 |

Controls are excluded from candidate selection; the literal TRAIN-frequency prior is orientation-asymmetric and diagnostic. A 0.5 tie predicts team 1, so accuracy is not simulated coin-flip accuracy.

| Control | VALIDATION LL | TEST LL | TEST Brier | TEST accuracy | TEST AUC |
|---|---:|---:|---:|---:|---:|
| `constant_0_5` | 0.693147 | 0.693147 | 0.250000 | 58.54% | 0.500000 |
| `class_prior` | 0.697058 | 0.678941 | 0.242928 | 58.54% | 0.500000 |

Full precision, calibration bins, support and all recipe parameters: [`after/metrics.json`](../artifacts/model_research_20261009/after/metrics.json).

## Selected-before versus selected-after

Paired after-minus-before differences (negative favors after): log loss **-0.004580 (-0.021467, +0.012746)**; Brier **-0.002216 (-0.010352, +0.006043)**. These percentile intervals resample whole source-calendar days (39 days, 5,000 draws, seed 42), retaining match-weighted means. Both include zero; observed improvement is not statistically established.

### Three expanding TRAIN-only diagnostics

These prefix refits are diagnostics, not additional selection or prospective TEST results. Full per-candidate fold metrics/parameters are in `after/metrics.json` → `development_walk_forward`; the incumbent and selected recipe are summarized here.

| Fold | Training prefix | Diagnostic dates | Diagnostic N | Player Elo LL | Blend LL | Status |
|---|---|---|---:|---:|---:|---|
| 1 | 2025-12-08–2026-03-21 | 2026-03-22–2026-04-10 | 80 | 0.686123 | 0.613212 | evaluated |
| 2 | 2025-12-08–2026-04-10 | 2026-04-11–2026-05-10 | 53 | 0.615685 | 0.541306 | evaluated |
| 3 | 2025-12-08–2026-05-10 | 2026-05-11–2026-06-07 | 75 | 0.613309 | 0.600508 | evaluated |

### Three descriptive chronological TEST blocks

No retuning followed these blocks. The final block is an explicit regression, not hidden by the overall mean.

| Block | Source dates | N | Before LL | After LL | Before Brier | After Brier |
|---|---|---:|---:|---:|---:|---:|
| 1 | 2026-08-14–2026-08-30 | 52 | 0.618658 | 0.602252 | 0.213484 | 0.205607 |
| 2 | 2026-08-31–2026-09-17 | 66 | 0.665437 | 0.651969 | 0.235907 | 0.229422 |
| 3 | 2026-09-18–2026-10-07 | 46 | 0.644277 | 0.665819 | 0.227674 | 0.237983 |

## Bookmaker comparison: same games, unchanged prices

Offline read-only SQLite comparison makes zero provider requests and does not mutate application mapping decisions or refit/reselect the model. Native HLTV-ID/result/orientation checks and strict latest-observation gates reject unusable latest quotes rather than falling back to older ones. Both price source clocks precede the earlier scheduled/actual provider start; current observations also require pre-start receipt. These are latest stored eligible observations, not certified closing prices.

[`markets/validation.json`](../artifacts/model_research_20261009/markets/validation.json) verifies before/after equality of original primary quote IDs, raw prices, clocks, orientation, labels and book probabilities: **203 quote pairs, 92 distinct games, 33 all-book-common games**. Full-cohort 164-series scores must never replace these smaller matched scores.

### Identical 33-match all-book intersection, proportional de-vig

| Forecast | Log loss | Brier | Accuracy | AUC |
|---|---:|---:|---:|---:|
| Previous player Elo | 0.641430 | 0.224437 | 66.67% | 0.681481 |
| Selected blend | 0.664955 | 0.235420 | 63.64% | 0.640741 |
| pinnacle | 0.625063 | 0.214726 | 63.64% | 0.744444 |
| bet365 | 0.629480 | 0.215742 | 66.67% | 0.740741 |
| unibet | 0.607781 | 0.208049 | 63.64% | 0.772222 |

The selected blend is worse than previous player Elo here despite its overall TEST gain. N=33 is too small for confident bookmaker ranking. Paired selected-model-minus-book intervals on these identical games:

| Book | LL difference (95% interval) | Brier difference (95% interval) |
|---|---|---|
| pinnacle | +0.039892 (-0.084467, +0.127467) | +0.020695 (-0.026077, +0.056997) |
| bet365 | +0.035475 (-0.077644, +0.122828) | +0.019679 (-0.021764, +0.053705) |
| unibet | +0.057174 (-0.068916, +0.143681) | +0.027371 (-0.020711, +0.063469) |

### Per-book matched cohorts, proportional pre-start

Each row pairs model and book on exactly the same IDs, but rows have different populations: do not rank books using these scores.

| Book | N | Selected-model LL | Book LL | Model − book LL (95% interval) | Model − book Brier (95% interval) |
|---|---:|---:|---:|---|---|
| pinnacle | 67 | 0.636018 | 0.618459 | +0.017558 (-0.052805, +0.081612) | +0.011288 (-0.014655, +0.036266) |
| bet365 | 74 | 0.624941 | 0.610523 | +0.014418 (-0.037696, +0.067997) | +0.007299 (-0.013948, +0.030177) |
| unibet | 62 | 0.683737 | 0.656401 | +0.027336 (-0.054237, +0.124253) | +0.015858 (-0.017918, +0.056702) |

All primary intervals cross zero. Positive point differences favor books, not the model. Source-day bootstrap uses 5,000 draws and seed 42. Artifacts retain raw pair evidence, matched IDs, calibration bins, exclusions and frozen selection metadata.

### De-vig and source-day sensitivities

| Mode | Quote pairs | Games with any book | All-book common | Pinnacle / Bet365 / Unibet N |
|---|---:|---:|---:|---|
| proportional_pre_start | 203 | 92 | 33 | 67/74/62 |
| power_pre_start | 203 | 92 | 33 | 67/74/62 |
| proportional_source_day | 13 | 13 | 0 | 1/0/12 |
| power_source_day | 13 | 13 | 0 | 1/0/12 |

Power de-vig solves q1^a + q2^a = 1 separately from each quote's prices; it is never fitted to TEST outcomes. Cohorts are identical between proportional and power at each cutoff. The sensitivity does not establish an edge. Source-day additionally gates source clocks and current receipts before nominal source-date midnight UTC: it filters latest stored observations, not reconstructed T−24/T−6/T−1h snapshots. The 13-game proxy has one Pinnacle game, no Bet365 games, 12 Unibet games and no common intersection. Singleton-day bootstrap intervals can degenerate; empty comparisons are null, not zero. It is not equal-information, original source-release-time or closing-price certification. No market-informed blend, ROI backtest or CLV measurement was performed.

## Reproduction and saved predictor

Run from the repository root with the existing downloaded inputs and frozen `before/` evidence. The evaluator checks cohort and declared recipe provenance and writes exact after evidence/manifest; it does not train a model, and opaque pickle bytes alone cannot prove training. Independent actual Kedro node execution and saved inference provide the runtime evidence.

```bash
uv pip install --python .venv -r requirements-modeling.txt
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KEDRO_DISABLE_TELEMETRY=1 .venv/bin/kedro run --pipelines modeling
.venv/bin/python scripts/evaluate_model_research.py

.venv/bin/python -m pipeline.cli compare-bookmakers --devig-method proportional --quote-cutoff pre_start --mapping-dir artifacts/model_research_20261009/markets/proportional_pre_start/mapping --audit-dir artifacts/model_research_20261009/markets/proportional_pre_start/audit --output-dir artifacts/model_research_20261009/markets/proportional_pre_start
.venv/bin/python -m pipeline.cli compare-bookmakers --devig-method power --quote-cutoff pre_start --mapping-dir artifacts/model_research_20261009/markets/power_pre_start/mapping --audit-dir artifacts/model_research_20261009/markets/power_pre_start/audit --output-dir artifacts/model_research_20261009/markets/power_pre_start
.venv/bin/python -m pipeline.cli compare-bookmakers --devig-method proportional --quote-cutoff source_day --mapping-dir artifacts/model_research_20261009/markets/proportional_source_day/mapping --audit-dir artifacts/model_research_20261009/markets/proportional_source_day/audit --output-dir artifacts/model_research_20261009/markets/proportional_source_day
.venv/bin/python -m pipeline.cli compare-bookmakers --devig-method power --quote-cutoff source_day --mapping-dir artifacts/model_research_20261009/markets/power_source_day/mapping --audit-dir artifacts/model_research_20261009/markets/power_source_day/audit --output-dir artifacts/model_research_20261009/markets/power_source_day

.venv/bin/python -m pipeline.cli predict-player --team1 Legacy --team2 M80 \
  --roster1 hltv:player:19686 hltv:player:15698 hltv:player:17215 hltv:player:19045 hltv:player:19869 \
  --roster2 hltv:player:16599 hltv:player:21665 hltv:player:22921 hltv:player:15370 hltv:player:20104 \
  --day 2026-10-09
```

The supplied Legacy/M80 keys are historical scenario evidence from HLTV ID 2398746, **not verified current starters or a live prediction**. The example day is after saved history through 2026-10-07. Actual saved CLI output gives Legacy 0.5738714545 and M80 0.4261285455; support reports minimum one prior player match, archive prior counts 205/193 and no measured-performance availability. There is no automatic roster acquisition. Require two explicit disjoint five-key rosters and a day strictly after both saved history engines.

Trusted local pickle bundle **version 4**, rating state **schema 2** are a clean cutover: rebuild obsolete artifacts with modeling, do not load old versions or add compatibility shims. Untrusted pickle files can execute code. [`inference_smoke.json`](../artifacts/model_research_20261009/inference_smoke.json) records complementary forward/reverse forecasts, unchanged query state/artifact bytes, rejection of historical-day and obsolete-version queries, and reproduction of all 164 saved forecasts with maximum absolute error 8.326672684688674e-17.

The audited Kedro modeling / `predict-player` path is distinct from generic `pipeline.cli train` / `run`, whose experimental selection uses TEST and is outside this change. Standalone organization/player-map rating studies and the application workspace/models are unchanged and use different cohorts.

## Evidence inventory and executed verification

Root: `artifacts/model_research_20261009/`.

- `before/`: preserved original features, metrics, test predictions, selected model, reproduction metadata and original bookmaker comparison.
- `after/summary.json`: exact before/after scores, paired intervals, chronological blocks, full 675-row validation and limits.
- `after/metrics.json`, `after/test_predictions.csv`, `after/model.pkl`, `after/state.pkl`, `after/features.csv` and `after/paired_model_predictions.csv`: saved experiment evidence; exact snapshot hashes are declared by the manifest.
- [`after/manifest.json`](../artifacts/model_research_20261009/after/manifest.json): SHA256 and byte counts for frozen protocol, inputs, artifacts and source code; original feature SHA256 `054d8dbb10ce61caa24978f5cd780e6c432dd60fb7ad09137710efbd1193f545`.
- `markets/{proportional_pre_start,power_pre_start,proportional_source_day,power_source_day}/bookmaker_comparison.json` and pair CSVs: complete market scores and quote provenance; `markets/validation.json`: cross-mode/cohort equality evidence.
- [`environment.json`](../artifacts/model_research_20261009/environment.json): actual Python 3.12.3 and installed numerical package versions; requirements declare the already exercised SciPy 1.18.1 dependency.
- `inference_smoke.json`: independently exercised persisted model/API/CLI inference evidence.
- [`verification.json`](../artifacts/model_research_20261009/verification.json): parent-executed checks found all 11 input hashes, 38 source hashes and seven archived output hashes valid; the six original candidates' VALIDATION/TEST metrics and all 164 per-candidate probabilities are exactly unchanged. The requirements file pins all exercised installed versions.
- `documentation_verification.json`: 22 bibliography sources matched to the ledger, 30 relative links and 15 local evidence records checked, all 18 candidate/control table rows matched to saved metrics, and the documented CLI example reproduced exactly. Citation linkage passed; it checks references and URL syntax, not factual truth or source availability.

Executed full implementation gate: **398 tests passed + 218 subtests passed**. Actual feature and training nodes completed; the evaluator and all four offline bookmaker commands completed; serialized and CLI inference were exercised. An evidence-runner reduced-schema certification defect and hard-link overwrite guard defect were reproduced in isolated copies and corrected before the final gate. Original project inputs were preserved. No LSP was configured; no lint/type-check claim. The numerical experiment remains exploratory; additional independent future data is required before claiming a reliable improvement or market advantage.
