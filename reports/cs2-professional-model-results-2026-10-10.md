# Research-driven Bayesian challenger: measured development results

Date: 2026-10-10. Selected model: `ridge_C1_ratings_form_bayesian_hnone`. Qualification: **false**.

## Decision

The new challenger improves development log loss slightly and is available through `predict-series`. It does **not** establish bookmaker competitiveness. Its estimated development advantage over the incumbent is small, its diagnostic uncertainty interval crosses zero, and its bookmaker point scores remain worse. No new forecasts or quality scores were generated for the exposed July–October TEST.

The default experimental CLI now uses the new model and schema-2 state. This is a development-candidate cutover, **not a promotion to a bookmaker-qualified model** and not a replacement of the existing application's other predictors.

## Research and implementation

- [Sourced professional-methods research](cs2-professional-model-research-2026-10-10.md): public vendor practices, dynamic skill literature, map/roster prerequisites and counterevidence.
- [Evidence ledger](cs2-professional-model-research-sources-2026-10-10.json): quoted source evidence and directly measured local data coverage.
- [Frozen experiment protocol](cs2-professional-model-protocol-2026-10-10.json): defined before new challenger forecasts or scores.
- `pipeline/features/bayesian_series.py`: 16 Gaussian latent-strength channels; uncertainty integration; random-walk/mean-reverting dynamics; exact conditional stopped-series score likelihood versus winner-only ablations; approximate Gaussian day-site posterior updates.
- `scripts/develop_bayesian_series.py`: frozen development, strict native-population join, original-baseline regression check, paired comparison and update-only publication.
- `pipeline/models/series_forecast.py` and `pipeline/cli.py`: schema-2 paired-state inference with history agreement and used-channel reporting. Legacy schema-1 inference states are rejected, not silently upgraded.

The terminal-score likelihood is exact **conditional on the exchangeable-map assumption**. Its Gaussian inference and fixed 64-point integration are approximations. The selected ridge outputs only a match probability; it is not a full coherent multi-market pricing model.

## Population and chronology

The raw master was filtered to dates at most 2026-06-30 **before** historical validation/replay. Development history contained **20,593 series and 4,125 map-coded observations**. Coefficient fitting used **16,164 series from 2024-07-01 through 2026-04-30**; earlier history warmed state only. Validation contained **1,487 series**, including the same **545** one-hour proportional-de-vig available-book consensus IDs used by the incumbent.

Forty-nine predeclared recipes were evaluated: one original incumbent, sixteen raw Bayesian forecasts, sixteen TRAIN-slope-calibrated forecasts, eight ridge combinations and eight fixed blends. Coefficients were not refitted on TRAIN+validation. Bookmaker prices were never predictors.

Publication subsequently advanced both history banks through **2026-10-07**, using completed outcomes via `update_day` only. The archive contains **27,988** accepted games; **3,270** trailing games were updates, not new TEST predictions. The selected coefficients remained unchanged.

## Same-population bookmaker comparison

All figures below use the same **545 validation matches across 56 source days**. Lower log loss/Brier is better. Accuracy and AUC are diagnostics; they do not override probability quality.

| Metric | Previous model | Research challenger | Available-book consensus |
|---|---:|---:|---:|
| Log loss | 0.625585 | 0.624766 | 0.623213 |
| Brier score | 0.217833 | 0.217521 | 0.216906 |
| Accuracy | 0.656881 | 0.656881 | 0.647706 |
| ROC AUC | 0.706369 | 0.706980 | 0.708869 |
| ECE (diagnostic) | 0.029311 | 0.038113 | 0.050511 |

Challenger minus incumbent log loss: **−0.000819064**. Diagnostic day-block 95% interval: **[−0.005548287, +0.003796444]**. Seven-day interval: **[−0.004785099, +0.003025745]**.

Challenger minus consensus log loss: **+0.001553317**. Diagnostic day-block interval: **[−0.013656340, +0.017396027]**. Seven-day interval: **[−0.004861963, +0.011049033]**.

These use 5,000 fixed-seed paired bootstrap replicates. They are **not selection-adjusted** and are not an independent confirmation after searching/reusing development data. An interval containing zero does not establish equivalence or competitiveness. Calibration-bin ECE actually worsened relative to the incumbent on this market subset, so the result is not an improvement on every diagnostic.

### Individual-book matched subsets

Each row has its own fixed subset; rows cannot be compared as if their populations were identical. The 26-match Bet365 subset is especially small.

| Book | Matches | Previous model log loss | Challenger log loss | Book log loss |
|---|---:|---:|---:|---:|
| pinnacle | 327 | 0.624272 | 0.623753 | 0.615610 |
| bet365 | 26 | 0.575660 | 0.580302 | 0.576754 |
| unibet | 467 | 0.620184 | 0.617727 | 0.616459 |

The challenger is worse than every individual bookmaker's point log loss here. It improves on the incumbent for Pinnacle and Unibet subsets, but worsens for Bet365. No bookmaker edge or ROI claim is supported.

### Full validation and temporal diagnostics

On all 1,487 validation series, incumbent/challenger log loss is **0.619840 / 0.617191**, and Brier is **0.215535 / 0.214429**. This broader population is not the bookmaker comparison population.

The selected and incumbent recipes were separately refitted at each earlier TRAIN-only cutoff:

| Coefficient cutoff | Evaluation window | Matches | Incumbent log loss | Challenger log loss |
|---|---|---:|---:|---:|
| 2025-06-30 | 2025-07-01–2025-08-29 | 1393 | 0.614997 | 0.610833 |
| 2025-12-31 | 2026-01-01–2026-03-01 | 1703 | 0.609767 | 0.603863 |
| 2026-02-28 | 2026-03-01–2026-04-29 | 2281 | 0.608198 | 0.604319 |

All three favor the challenger, but these are retrospective TRAIN-only diagnostics of a recipe selected later on validation—not fresh final tests. The latter two 60-day windows both include March 1; they are not independent disjoint replications.

### What the candidate ladder learned

The best raw score-aware Bayesian channel had validation log loss **0.629544**, versus **0.631598** for the best raw winner-only channel. Their best TRAIN-calibrated versions scored **0.628413** and **0.630429**. All four are worse than the incumbent. The useful gain came from combining Bayesian channels with existing ratings/form, not replacing the system with a single new rating. Every candidate score and native prediction is retained, not only the winner.

## Real inference smoke

The actual CLI loaded its default saved model/state and forecast a hypothetical Vitality versus Natus Vincere scenario on 2026-10-10:

- BO3 team1 probability: **0.650623565753**.
- Reversed teams: **0.349376434247**, complement within numerical tolerance.
- BO5 team1 probability: **0.694136702722**.
- Two unseen literal organizations: **0.5**.
- Query on the completed-history day 2026-10-07: rejected with exit code 2.
- Repeating `develop` or `publish` on the completed output: rejected with exit code 2 before rescoring/overwriting.
- Saved state bytes remained unchanged after queries and guard checks.

This is not a verified upcoming fixture, current starting roster or executable betting recommendation. Output retains `competitiveness_qualified=false`, coverage and source-time limitations. Only trusted local pickle artifacts may be loaded.

## Reproduction and artifacts

The completed output directory is deliberately immutable. To reproduce with the same input bytes and current implementation, choose a new empty directory:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KEDRO_DISABLE_TELEMETRY=1 \
  .venv/bin/python scripts/develop_bayesian_series.py develop \
  --output-dir artifacts/professional_model_20261010_reproduction

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KEDRO_DISABLE_TELEMETRY=1 \
  .venv/bin/python scripts/develop_bayesian_series.py publish \
  --output-dir artifacts/professional_model_20261010_reproduction
```

These are the same exercised phases with a different output directory, not a claimed second execution. Runtime/package versions and source/input hashes are recorded. The immutable original run is `artifacts/professional_model_20261010/`:

| Artifact | Purpose |
|---|---|
| `development_started_manifest.json` | Source/input/runtime freeze before replay/fitting. |
| `development_features.csv`, `bayesian_features.csv` | Past-only development feature rows. |
| `selection.json` | All recipes/scores and chronological diagnostics. |
| `development_comparison.json` | Identical-ID consensus and individual-book comparisons. |
| `candidate_validation_predictions.csv`, `validation_predictions.csv` | Exported predictions, labels and fixed market membership. |
| `selection_freeze.json` | Selected model, source/input/artifact hashes; no TEST selection. |
| `selected_model.pkl` | TRAIN-fitted coefficients. |
| `series_inference_state.pkl`, `publish_manifest.json` | Updated schema-2 future-scenario history and provenance. |
| `verification.json` | Exercised tests/commands, independent arithmetic, native-label audit and numerical limits. |

## Verification and remaining limits

- **492 tests and 384 subtests passed** in the complete pytest run; 18 focused Bayesian and 37 focused series tests also passed. Meaningful tests cover stopped-series probabilities, uncertainty integration, posterior reference integrals, same-day ordering, future-prefix invariance, input rollback, joins and persisted inference boundaries.
- Independent standard-library CSV arithmetic reproduced log loss, Brier and accuracy for all **49 candidates**, plus the primary selected/incumbent/consensus comparison.
- All **1,487 validation** native labels, dates, team orientations, formats and canonical URL prefixes were checked directly against the master. Prediction IDs have no overlap with the protected TEST IDs.
- **18 source hashes, 34 consumed-input hashes and 11 development-artifact hashes** were verified. Published state hashes and six protected previous model/selection/TEST artifacts also remained intact. Old source hashes describe the old code revision; they are not asserted to equal the intentionally updated inference code.
- The saved research's citation structure passed with 14 cited bibliography sources. Structural validation does not prove factual entailment; original-source scope/limitations were separately inspected.
- A 108-point comparison with adaptive integration found maximum sampled probability error **0.002273** for variances up to 32 and **0.022514** through variance 128. This is not a universal numerical bound; broad uncertainty deserves caution.
- No lint/type-check or live UI/application-model promotion claim. No new TEST leaderboard, paid-provider acquisition or database mutation was part of this experiment.

The next decisive prerequisites are verified as-of roster/map/veto information and a genuinely later reserved outcome/quote cohort. The current local archive ends on 2026-10-07, and the old TEST has already been exposed. **The overarching bookmaker-competitiveness goal remains unmet.**
