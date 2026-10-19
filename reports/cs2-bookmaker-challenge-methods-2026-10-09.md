# Broader CS2 modeling and bookmaker challenge: methods and data audit

This is the methods/data companion to the [frozen protocol](cs2-bookmaker-challenge-protocol-2026-10-09.json). It extends the [earlier literature synthesis](cs2-model-research-2026-10-09.md), rather than treating another paper's score as a CS2 performance ceiling. Final model quality must come from paired local predictions, not these data counts or passing software tests.

The completed [confirmation and inference report](cs2-bookmaker-challenge-results-2026-10-09.md) records **failed bookmaker qualification**. This methods document describes the frozen design, not a successful promotion claim.

## Decision

Use the broader complete-series archive for supervised learning, not only as auxiliary history for the 675 lineup-conditional cases. Add historical Valve standings recovered from their first Git introduction, prior measured team performance, uncertainty-aware strength channels and low-capacity symmetric learners. Keep bookmaker probabilities outside every learner. Reconstruct a fixed forecast-time benchmark from cached raw quote histories.

The evaluation reserve is **previously unscored, not prospective or wholly unseen raw data**. The archived results already existed locally; some informed earlier online states. Every native ID in any earlier prediction ledger, plus the entire old player cohort, is excluded from newly scored confirmation cases. Its earlier-day result may still update subsequent prequential history. No final-period quality is used to select a candidate.

## Actual source audit

The real bounded loader accepted **27,988 historical observations** from 122,038 unique native HLTV results: 23,568 complete BO3/BO5 series plus 4,420 completed map-coded observations. Five map scores were rejected as incomplete/unsupported modern round scores. Map observations update only map-strength history; they are not relabeled as additional series targets. There are no literal `bo1` formats in this native archive, and no observed repeated native IDs. Source-year >= 2024 is a modern-game proxy, not certified CS2 version evidence.

**670 series have usable verified historical participant measurements: 661 from 2026 and nine from 2025.** Their five-player totals pool player-round exposure, not unique team rounds; map support is counted once rather than five times. Current target lineups are not required or invented. Missing historical measurements remain unavailable. Malformed nonempty claimed-verified measurements stop intake rather than silently remove an otherwise eligible target from evaluation.

The collector acquired **27 original global Valve snapshots, 7,828 ranking rows**, from August 2024 through October 2026. Original Git blobs, SHA256/Git blob hashes, first-introduction commits and committer clocks are saved in `data/raw/valve_standings_20261009/`. Offline loading checks raw bytes, CSV equivalence and availability metadata. This is a bounded public-source acquisition; no paid odds-provider request or application database mutation occurred.

| Frozen partition | Source days | Complete series | One-hour market-covered series |
|---|---|---:|---:|
| TRAIN/history | 2024-01-09–2026-04-30 | 19,106 | 695 |
| VALIDATION | 2026-05-01–2026-06-30 | 1,487 | 545 |
| Previously unscored TEST reserve | 2026-07-01–2026-10-07 | 2,728 | 419 |
| Same TEST dates, prior-evaluated IDs used only as earlier history | 2026-07-03–2026-10-07 | 247 | 102 |

Coefficient fitting starts 2024-07-01: **16,164 fitted series**, with preceding accepted results used for rating warmup rather than additional coefficient fitting. Prepared development data contain 20,593 series and no reserved-TEST features or model forecasts. TRAIN contains 408 cases with both teams' prior measured performance and 10,065 with both teams in an eligible VRS snapshot; VALIDATION counts are 262 and 1,133. These counts are availability indicators, not certificates of exact roster or publication identity.

## Why these model families

Bradley–Terry/Elo-style strength differences give an interpretable paired-comparison starting point. Published esports rating benchmarks motivate testing uncertainty-aware ratings, but their historical population, targets and periods differ from this experiment; their scores are not local targets. [1][2]

The bounded 128-candidate ladder comprises:

- **40 raw/calibrated rating forecasts:** 12 series Elo channels (K 16/32/64 and 90/180/365-day/no decay), six map-score Elo channels (K 16/32/64 and 180/365-day decay), and two daily Glicko2 channels (initial RD 200/350). Each has a raw and TRAIN-only scalar calibration recipe.
- **60 ridge-logistic recipes:** five explicit predictor groups, four regularization strengths, three TRAIN recency policies. Scaling and coefficients fit TRAIN only; no intercept; mirrored training examples.
- **24 shallow histogram-boosting recipes:** two predictor groups, three leaf caps, two regularization values and two minimum-leaf supports. Early stopping is disabled; no hidden random validation split or TEST-driven iteration count.
- **Four predeclared probability blends:** fixed components and weights, not weights optimized on the reserved period.

The 83-predictor bank covers strengths, uncertainty, past 5/10/30-series form, series-map margins, opponent-adjusted residual form, experience/rest/activity, prior team measurements and five VRS covariates. Unknown teams, missing measurements and ranking coverage are explicit. Literal organization names are not silently alias-merged. Every predictor has declared side-swap semantics. Learned predictions use mirrored probability projection; reversal complements the probability.

Map-to-series conversion uses the conditional-independent, exchangeable-map majority formula for BO3/BO5. It is an approximation, not a reconstructed veto model. No played future map or current starting roster is a predictor. A map-score update also remains a heuristic strength channel; score fractions from stopped series are not independent fixed-length binomial samples. The actual series probability is what is evaluated.

## Historical Valve standings are covariates, not calibrated odds

Valve rankings contain achievement/opponent information beyond simple match-count Elo; ranking quality does not imply calibrated probabilities. [3] The collector reads the **first-introduction blob**, not today's corrected text bearing an old filename. The local availability convention is

`max(snapshot midnight UTC + 7 days, first-introduction committer UTC clock)`.

For example, the 2024-08-06 snapshot was first introduced on 2024-08-21 at 22:31:05 UTC; it is not backdated to August 6. The 2026-10-05 snapshot is unavailable before October 12 under the conservative policy, so it cannot inform the October 7 reserve endpoint. These are documented retrospective proxies, not independent proof of when a bettor could download the file.

For each source day, lookup chooses the latest globally eligible snapshot strictly before midnight UTC. If a team is missing from that snapshot, it is not backfilled from an older one. Only casefold/whitespace normalization is used. Multiple ranked rosters sharing an organization name use the fixed highest-points aggregate with a roster-support indicator; that does not assert the actual match starters. Features are point difference, log-rank strength difference, availability, snapshot age and minimum roster count.

A parser error found by the actual public acquisition was corrected: official tables use single-dash alignment cells, not only three-dash separators. Its regression failed before the fix; the collector and hash-verified offline loader then ran successfully. No current-snapshot fallback was introduced.

## Chronology, calibration and selection

All queries on a source day see the same opening state. Results, maps and measured box scores enter only on later days; same-day order cannot affect a forecast. State queries do not mutate or age saved history. The final persisted history includes the completed tail of replay, while coefficients stay TRAIN-fitted. A separate two-day result-history replay checks sensitivity to coarse source-day timing.

Rolling-origin evaluation and calibration require separating fitted outcomes from later forecasts. [4][5] Scalar calibration is fitted only to already prequential TRAIN rating forecasts; this is not calibration of a learned classifier's in-sample predictions. No learned probability stack is fitted to its own training forecasts. Three expanding diagnostics lie entirely inside TRAIN.

Selection minimizes model log loss on the **fixed 545-match one-hour validation union**. Available bookmaker probabilities determine the comparator, not a feature or fitting target. Full-validation scores and individual-book comparisons remain diagnostics. No TRAIN+VALIDATION coefficient refit occurs. The winner, recipes, input/output hashes and source hashes are saved before confirmation. Repeated development choices cannot turn validation quality into an unbiased estimate; the reserved comparison remains the decision evidence.

## Fixed-time bookmaker reconstruction

Each fixture/book chooses its latest relevant cached receipt that explicitly requested that book. A later no-history response, missing book, invalid raw hash, failed receipt or ambiguous equal-time response cannot revive an older positive response. Raw gzip paths are contained under the database root; compressed and decoded hashes are checked. Both outcomes come from one complete raw response, not a cross-response synthetic pair.

Within that history, select the latest side state strictly before the origin **before** checking activity, price and scope. A last suspended/invalid state therefore cannot expose an older valid price. Original scheduled and actual starts remain unchanged. The primary origin is one hour before the earlier provider start; six-hour and source-day origins are fixed sensitivities. Unknown or too-early origins are not fabricated. Source clocks and retrospective retrieval receipts still do not certify bookmaker publication, access or execution. [6]

Proportional de-vig is primary; price-only power de-vig is a sensitivity, not fitted to outcomes. [7] The primary comparator averages available valid book probabilities once per native match. It is **not an executable price**. Per-book populations and the all-book-common intersection are separate comparisons; bookmaker rows are never counted as independent matches. Proper scores must compare the same situations. [8]

## Frozen competitiveness decision

At least 200 paired cases and 30 source days are required. Model-minus-consensus log loss must have point difference <= 0.005 and a 95% one-sided paired-bootstrap upper bound <= 0.01. Brier limits are 0.0025 and 0.005. Both source-day and fixed seven-calendar-day resampling must meet the upper-bound rules (5,000 draws, seed 42). Each individual book with at least 50 paired cases must also have point log-loss difference <= 0.01. Sparse-book findings remain inconclusive.

A wide confidence interval crossing zero is **not** evidence of equivalence. These thresholds are operational retrospective decision rules, not guarantees that the bootstrap is valid under arbitrary event dependence or future drift. Passing them would not prove profitable bets, executable prices, unlimited staking, future superiority or a universal model optimum. Failure does not permit tuning on the reserved cases or changing the thresholds afterward.

## Reproducible evidence

The root `artifacts/bookmaker_challenge_20261009/` preserves the frozen protocol inputs and the first development-only run. **That first development run is superseded, not confirmation evidence.** A source review found input-binding and in-flight source-freeze weaknesses, an empty-timed-cohort failure and incomplete individual-book checks in the two-day sensitivity. Behavioral regressions reproduce those boundaries; the workflow now validates the actual consumed input files, binds code before fitting, reports unsupported timing without inventing origins, and re-scores every individual book for the sensitivity.

The reviewed rebuild uses `artifacts/bookmaker_challenge_20261009/verified/`; it retains the same frozen cohorts, candidate grid, selection rule and acceptance thresholds. The earlier run did not generate reserved-TEST forecasts or losses.

- Root `frozen_cohorts.json`: exact native IDs and days.
- Root `previously_evaluated_ids.json`: 8,318-ID exclusion union and input provenance in `coverage_scope.json`.
- `verified/prepared_manifest.json`: frozen files, source hashes, coverage-only counts and feature support.
- `verified/history_audit.json`, `market_audit.json`, `market_snapshot.json.gz`, `market_cases.json`: intake and immutable reconstructed-market evidence.
- Root `foundation_verification.json`: the earlier acquisition/intake/prepare execution and 123 passing tests plus 235 subtests. It predates the final workflow corrections and is not their verification.
- `scripts/evaluate_bookmaker_challenge.py`: separate `prepare`, `develop`, `confirm` commands; existing frozen evidence is protected against accidental overwrite/reselection.

## Sources

[1] Király & Qian, Bradley–Terry–Elo paired-competition models: https://arxiv.org/abs/1701.08055

[2] Thorrez, EsportsBench: https://claytonthorrez.com/papers/esportsbench/EsportsBench_preprint.pdf

[3] Valve's official standings and methodology: https://github.com/ValveSoftware/counter-strike_regional_standings ; inspected original January 2026 snapshot: https://github.com/ValveSoftware/counter-strike_regional_standings/blob/0b2154eacafe50abd2ec251ed78d460d8f833088/invitation/2026/standings_global_2026_01_05.md

[4] Hyndman & Athanasopoulos, time-series cross-validation: https://otexts.com/fpp3/tscv.html

[5] scikit-learn, probability calibration: https://scikit-learn.org/stable/modules/calibration.html

[6] OddsPapi historical-odds clock documentation: https://oddspapi.io/en/docs/get-historical-odds

[7] Clarke et al., adjusting odds for overround: https://www.sciencepg.com/article/10.11648/j.ajss.20170506.12

[8] Gneiting & Raftery, proper scoring rules: https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf
