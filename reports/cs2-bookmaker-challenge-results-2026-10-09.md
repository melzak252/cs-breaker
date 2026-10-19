# Broader CS2 model: frozen bookmaker confirmation

## Decision

**The bookmaker-competitiveness goal is not achieved.** The validation-selected sports-only model fails the predeclared confirmation rules. Persist it as an experimental predictor with `competitiveness_qualified=false`; do not promote it as a bookmaker-equivalent or profitable betting model.

This is a previously unscored **retrospective** comparison, not a prospective trial or wholly unseen raw-data study. The result archive already existed locally. Earlier prediction-ledger IDs and the entire earlier player cohort were excluded from scored confirmation cases. The selected model, code and artifacts were frozen before generating these TEST forecasts; no TEST-driven reselection or coefficient refit followed.

- [Research synthesis](cs2-model-research-2026-10-09.md)
- [Methods, data audit and cited sources](cs2-bookmaker-challenge-methods-2026-10-09.md)
- [Predeclared protocol](cs2-bookmaker-challenge-protocol-2026-10-09.json)
- [Confirmation and frozen acceptance decision](../artifacts/bookmaker_challenge_20261009/verified/confirmation.json)
- [Executed verification evidence](../artifacts/bookmaker_challenge_20261009/verified/verification.json)

## What was implemented and trained

The new workflow uses the broad historical series archive instead of restricting supervised learning to the 675 lineup-conditional examples. Its history contains 23,568 complete BO3/BO5 series and 4,420 completed map-coded observations. Maps update historical strength but are not invented series targets. Coefficients fit 16,164 series from 2024-07-01 through 2026-04-30; earlier results warm the ratings. Selection uses May–June 2026 only, without a TRAIN+VALIDATION refit.

The archive is not results-only: **670 eligible series have measured historical participant statistics, including 661 from 2026 and nine from 2025**. This is sparse relative to the result archive and does not establish contemporary starting rosters or historical announcement clocks. The acquisition also saved 27 original historical Valve snapshots with 7,828 ranking rows, using first-introduction Git provenance and a conservative availability lag.

The 128 predeclared candidates cover raw/calibrated series Elo, map-score Elo, daily Glicko2, ridge-logistic learners, shallow histogram boosting and fixed blends. Features include past-only ratings, uncertainty, form, opponent-adjusted residuals, rest/activity, measured performance and historical Valve standings. Bookmaker probabilities are never model inputs.

Selected: **`ridge_C1_ratings_form_hnone`** — ridge logistic regression with ratings and recent-form predictors, C=1, no sample-recency weighting, mirrored training and symmetric prediction. The selected recipe does not consume the measured-performance or Valve covariates. Their failure to win this selection is not proof that better roster or performance data would be useless.

### Development results, not confirmation

All rows below use the same 545 market-covered validation matches. Family winners were chosen within development data; their small differences are selection-optimistic, not independent superiority estimates.

| Candidate | Validation log loss |
|---|---:|
| Selected ratings/form ridge | 0.625585 |
| Best shallow boosting recipe | 0.628566 |
| Best predeclared blend | 0.628815 |
| Best raw rating: Glicko2, RD 200 | 0.630332 |
| Best calibrated rating: Glicko2, RD 200 | 0.630853 |
| Raw series Elo, K=32, no decay | 0.632885 |
| Available-book consensus | 0.623213 |

The selected model's full 1,487-match validation log loss is 0.619840. Three TRAIN-only expanding diagnostics are saved in `selection.json`; they did not use the reserved period.

The first development run in the parent artifact directory is preserved but superseded. Review corrections bound the actual consumed inputs and pre-fit source bytes, handle empty timed cohorts, and include individual-book checks in the two-day sensitivity. The reviewed rebuild produced **byte-identical `selection.json`, `selected_model.pkl` and `validation_predictions.csv`**, then created a new valid source freeze under `verified/` before confirmation.

## Primary reserved comparison

The reserved period is 2026-07-01–2026-10-07. Of 2,728 reserved native series, 570 have eligible native/provider forecast-origin evidence and **419 have validated one-hour bookmaker comparisons across 72 source days**. The remaining 2,158 lack an eligible origin; another 151 timed cases lack an admitted quote pair. These are coverage/identity/timing exclusions, not exclusions based on prediction losses.

Each model forecast is paired with the equal-weight average of available, validated Pinnacle/Bet365/Unibet probabilities for that same native match and orientation. Both price-side clocks precede the one-hour origin; the model's source-day opening does not follow it. Proportional de-vig is primary. The consensus is not an executable bookmaker price.

| Same 419 reserved matches | Model | Bookmaker consensus |
|---|---:|---:|
| Log loss, lower is better | 0.632088 | 0.600528 |
| Brier, lower is better | 0.221130 | 0.206919 |
| Accuracy | 62.05% | 67.78% |
| AUC | 0.682701 | 0.739706 |
| ECE, bin-dependent diagnostic | 0.046099 | 0.047170 |

Slightly lower binned ECE does not compensate for worse proper scores and discrimination. Accuracy or calibration-bin appearance cannot replace the frozen log-loss/Brier decision.

### Frozen uncertainty and acceptance checks

Differences are model minus consensus; positive values favor bookmakers. Bootstrap: 5,000 paired resamples, seed 42. Day clustering has 72 blocks; fixed seven-calendar-day clustering, anchored at the frozen TEST start, has 14 populated blocks.

| Criterion | Observed | Required maximum | Result |
|---|---:|---:|---|
| Log-loss point difference | +0.031560 | +0.005000 | Fail |
| Log-loss one-sided 95% upper, day blocks | +0.045779 | +0.010000 | Fail |
| Log-loss one-sided 95% upper, seven-day blocks | +0.050031 | +0.010000 | Fail |
| Brier point difference | +0.014211 | +0.002500 | Fail |
| Brier one-sided 95% upper, day blocks | +0.020778 | +0.005000 | Fail |
| Brier one-sided 95% upper, seven-day blocks | +0.022755 | +0.005000 | Fail |

The two-sided 95% log-loss intervals are **[+0.015649, +0.048596]** with day blocks and **[+0.012808, +0.053423]** with seven-day blocks. Both lie above zero under these resampling assumptions. Minimum sample/day requirements pass, but every proper-score and individual-book rule fails. This is not merely insufficient power to establish equivalence; the point estimates themselves are materially worse. The bootstrap is still conditional on the declared population, timing proxies and dependence assumptions.

### Individual bookmaker populations

Each row compares identical matches within that row. Different rows have different populations and must not be used to rank bookmakers.

| Bookmaker | Paired matches | Model LL | Book LL | Model minus book LL | 95% day-block interval |
|---|---:|---:|---:|---:|---|
| Pinnacle | 250 | 0.622296 | 0.587756 | +0.034540 | [+0.016517, +0.054856] |
| Bet365 | 155 | 0.632285 | 0.592446 | +0.039839 | [+0.010433, +0.066016] |
| Unibet | 346 | 0.635589 | 0.604587 | +0.031002 | [+0.011897, +0.050236] |

All exceed the frozen +0.01 point-difference ceiling. The 73-match all-three-book intersection also favors every book over the model; its exact metrics and paired intervals are in `comparisons/one_hour_proportional.json`. The primary export has 751 bookmaker rows, **not 751 independent matches**.

## Predeclared sensitivities

The following are robustness diagnostics, not opportunities to choose a favorable benchmark afterward. Populations differ across horizons.

| Quote origin | De-vig | Matches | Model LL | Consensus LL | Difference |
|---|---|---:|---:|---:|---:|
| One hour | Power | 419 | 0.632088 | 0.595654 | +0.036434 |
| Six hours | Proportional | 325 | 0.632850 | 0.601824 | +0.031026 |
| Six hours | Power | 325 | 0.632850 | 0.596612 | +0.036238 |
| Source-day opening | Proportional | 170 | 0.646209 | 0.613287 | +0.032922 |
| Source-day opening | Power | 170 | 0.646209 | 0.611063 | +0.035146 |

Every alternative has a positive log-loss gap. Source-day coverage is below the primary minimum sample requirement. A separate **two-day result-history lag**, with the same 419 primary quotes, produces log-loss difference +0.032799 and Brier difference +0.014614; it also fails all score/uncertainty/individual-book rules. Valve availability remains under its original conservative policy, not relabeled as result history.

Across all 2,728 reserved series, model log loss is 0.612459, Brier 0.212469, accuracy 65.43%, AUC 0.712183 and ECE 0.024023. **These full-population metrics cannot be compared directly with bookmaker metrics on 419 matches**, or with the previous player model's different 164-match cohort. No independent before/after improvement over that old model is established here.

## Runnable predictor and exercised verification

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KEDRO_DISABLE_TELEMETRY=1 \
  .venv/bin/python -m pipeline.cli predict-series \
  --team1 Vitality --team2 "Natus Vincere" --day 2026-10-09 --format bo3
```

This is a hypothetical literal-organization scenario, **not an observed upcoming fixture or verified lineup**. The actual saved-model smoke returned Vitality probability 0.654453469, minimum prior-series support 178, history through 2026-10-07, coefficients through 2026-04-30, and qualification false. Reversing the sides complements the probability within 4.5e-16. The actual BO5 path ran; two unseen teams produced 0.5 with explicit cold support. Querying a day within saved history was rejected with exit code 2. All inference artifact bytes remained unchanged.

The command defaults to the trusted local artifacts in `artifacts/bookmaker_challenge_20261009/verified/`. **Only load trusted pickles: unpickling executes code.** This experimental command does not replace the existing application models or the separate player-roster predictor. The generic `train`/`run` path is not this audited experiment.

Executed checks, recorded in `verified/verification.json`:

- Full project suite: **472 tests and 356 subtests passed** in 72.63 seconds.
- Actual public Valve acquisition, offline history loading, preparation, all 128 development fits, frozen selection, six bookmaker comparisons and two-day-history confirmation ran.
- Independently verified four frozen-input hashes, six preparation-artifact hashes, 16 source hashes, three selected-artifact hashes and all three confirmation artifact hashes. Comparison JSON/CSV hashes are saved separately in verification.
- Independently checked all 2,728 TEST native identities, orientations, labels and source dates; no overlap with the 8,318-ID earlier-evaluation exclusion list.
- Independently recomputed full-TEST and all six consensus comparisons from saved CSVs, agreeing on log loss, Brier and accuracy within 1e-12. Checked 3,136 exported bookmaker-row clock orderings across those comparisons; these are repeated-mode checks, not independent observations.
- Actual CLI attempts to reselect or reconfirm the already-frozen experiment exited 2 before fitting/scoring and left the artifacts unchanged.
- No lint/type-check, live execution, betting-profit or production UI verification claim.

### Reproduction without overwriting evidence

Use the pinned modeling environment, unchanged frozen inputs and saved Valve snapshot directory. The existing `verified/` run deliberately rejects overwrite/reselection. A fresh directory can reproduce the same experiment:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KEDRO_DISABLE_TELEMETRY=1 \
  .venv/bin/python scripts/evaluate_bookmaker_challenge.py prepare \
  --output-dir artifacts/bookmaker_challenge_20261009/reproduction
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KEDRO_DISABLE_TELEMETRY=1 \
  .venv/bin/python scripts/evaluate_bookmaker_challenge.py develop \
  --output-dir artifacts/bookmaker_challenge_20261009/reproduction
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KEDRO_DISABLE_TELEMETRY=1 \
  .venv/bin/python scripts/evaluate_bookmaker_challenge.py confirm \
  --output-dir artifacts/bookmaker_challenge_20261009/reproduction
```

A rerun is **reproduction of already observed TEST results**, not fresh confirmation. Do not modify candidates, thresholds or exclusions and reuse this TEST to claim an independent win. Exact native IDs, sources, environment versions, fitted recipe and hashes are in `frozen_cohorts.json`, `prepared_manifest.json`, `selection.json` and `selection_freeze.json`.

## Remaining evidence and development requirements

**Verified facts:** the archive ends 2026-10-07, with zero native rows after the reserved endpoint; the complete eligible modern-series population is already allocated to TRAIN, VALIDATION, this now-exposed TEST or prior-evaluated history. The new learner is better than the listed rating baselines on validation, but falls short of bookmakers on confirmation. More observations alone would narrow uncertainty, not remove the observed point-score deficit.

**Unproven hypotheses:** roster/stand-in information, map-pool/veto information, event conditions, organization identity changes and population drift could explain or reduce some of the gap. This experiment does not isolate their causal contribution. A more complex learner is not automatically a better model; boosting and added covariates already failed to win the development selection.

The next defensible campaign needs:

1. Verified as-of player/roster identities and announcement clocks, plus richer match-specific historical measurements. Never substitute scrape-time current squads for historical starters.
2. Map-specific strength and veto information that was actually available at the declared origin, rather than the maps observed after play. Evaluate any added feature in development-only rolling-origin folds.
3. A newly frozen candidate and **new independent later matches**, preferably forecasts and contemporaneous odds recorded before outcomes. Keep exact kickoff, feature-availability and quote clocks, native IDs, source bytes and the selected model hash.
4. The same paired proper-score discipline, uncertainty bounds and explicit failure reporting. Do not manufacture competitiveness by copying the benchmark into the model or selecting the best recipe on this exposed TEST.

No later native result cohort exists in the current archive to provide that fresh confirmation. The goal therefore remains open; this report records a completed experiment and an unsuccessful promotion decision, not a promise of eventual bookmaker superiority.
