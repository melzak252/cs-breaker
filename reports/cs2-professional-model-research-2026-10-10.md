# Professional CS2 forecasting: public evidence and the next model

Research date: 2026-10-10. Scope: pre-match ordinary BO3/BO5 winner probabilities, sports-only inputs, and a fair comparison with bookmakers. Live-market techniques are included only where their mechanisms transfer. This report does not claim access to proprietary production models.

## Decision

Test a forward-only Bayesian latent map-strength model with a stopped-series score likelihood, uncertainty integration and time-varying strength. Compare it with the existing model on the same development matches. Keep the previously exposed July–October TEST closed.

This is a research-derived hypothesis, **not a disclosed PandaScore, Oddin.gg or Abios algorithm**. Professional sources support coherent distributions, richer official data, adaptation and disciplined operations; they do not establish that our particular approximation will beat a bookmaker. The bounded experiment is fixed in `cs2-professional-model-protocol-2026-10-10.json` before new challenger forecasts or quality scores.

## What professionals actually disclose

| Evidence | Documented practice | What it does not establish | Transfer to this project |
|---|---|---|---|
| PandaScore, 2022 technical article [1] | One round-state distribution supports multiple consistent markets; a Markov-binomial proof of concept uses stopping rules and halftime transitions. | The article expressly says production became more complex. It describes old MR15 CS:GO, not a current CS2 production recipe or a bookmaker benchmark. | Model valid outcomes jointly; use score information with the correct stopping rule. Do not copy its round constants. |
| Oddin.gg trading-desk article, 2025 [2] | Title experts and mathematicians recalibrate models for patches; machine learning flags performance/tempo shifts; pre-match and live teams operate alongside risk specialists. | No public feature weights, training cadence, calibrated probabilities or independent edge measurement. | Separate data/identity operations, forecasting, evaluation and trading. Time adaptation is useful but cannot manufacture missing roster or patch information. |
| Oddin.gg official-data product [3] | Licensed event data delivered into customers' proprietary pricing models, separately from a priced odds-feed product. | Vendor latency/quality claims are not independently measured here; buying odds is not training a sports-only model. | Acquire event-level inputs with rights and publication clocks, rather than treating bookmaker prices as our discovery. |
| Abios/Kambi bet-builder announcement, 2024 [4] | Proprietary automated CS2 modelling supports related markets including player markets. | Exact dependence models and performance are undisclosed. | Shared underlying match states matter when pricing correlated outcomes. |
| GRID plans [5] | Access arrangements depend on use case and commercial terms. | Public/media access does not establish permission for betting use. | Verify historical access, redistribution and betting rights before committing to a provider. |

These are first-party descriptions. They establish what the companies say they do, not superiority. Their operational/data advantages are broader than selecting a classifier.

PandaScore supplies useful counterevidence against oversimplification: its public model overestimated close games and underestimated stomps between similarly rated teams. Economy, map, pistol-round and other mechanisms were omitted; the author says those limitations motivated a more complex production model. A coherent model can still be wrong. [1]

## Relevant sports and skill-rating research

### Dynamic strength and uncertainty

Glickman's dynamic paired-comparison model represents ability probabilistically, increases uncertainty through time, and updates from results. Its predictive framework distinguishes current/prior information from retrospective smoothing. The practical independent-marginal approximation also loses some information shared through opponents. We already have Glicko-style channels, so merely adding another rating brand is not the proposed improvement. [6]

Whole-History Rating jointly fits evolving Bradley–Terry strengths. A newcomer's later games can inform how strong earlier opponents were, which an independent online update can miss. However, a history smoother fitted through October cannot legitimately generate a May pre-match feature. A prefix refit would be valid; smoothing the full archive and backtesting historical predictions would leak. Coulom's published comparison uses Go data, not CS2. [7]

Flexible-time pairwise-comparison work shows how temporal latent-strength models can exploit structured inference rather than rebuilding a generic large classifier. This supports a dynamic-strength design, not a claim of esports profitability. [8]

TrueSkill 2 uses additional player/game information, including kills/deaths and squad-related effects. Its reported Halo results do not transfer numerically to CS2. More importantly, its mechanism needs identities and observed player context that our full historical cohort does not contain. [9]

### Maps, players and vetoes

The CS:GO veto-bandit paper studies strategic map choices. Better simulated veto strategy is not the same task as predicting which maps teams actually selected, nor is it evidence of lower pre-match bookmaker log loss. We would need historical legal pools, ordered actions, teams' available information and forecast-time availability to validate a veto predictor. [10]

The public map-level Pick'Em project provides a concrete design example: map-level learning, calibrated probabilities, recent team-map features and a veto/series layer. Its unpublished crawler and example data prevent treating its public description as a reproducible professional benchmark. [11]

A second public forecasting project describes combining shorter organization histories with longer player histories. This is a reasonable architectural lead, not proof of edge. Historical announced starters, substitutes and identity continuity must be reconstructed before using that mechanism honestly. [12]

HLTV's Rating 3.0 article describes contextual performance measurement involving round state and economy. That is evidence that raw kills/deaths omit important context. It is not permission to use a match's eventual rating as a pre-match input for that same match. [13]

## What data we really have

Measured directly from local archived CSVs; the evidence ledger retains the counts and scope. This was a coverage audit, not a new TEST evaluation.

| Local resource | Observed usable scope | Limitation |
|---|---|---|
| Existing broad challenge history | 23,568 eligible complete BO3/BO5 series proxies and 4,420 map-coded observations from 2024 onward. Coefficient fitting uses 16,164 series; fixed validation has 1,487, of which 545 have the primary one-hour market comparison. | Literal organization names and source-calendar-day timing; not a complete historical lineup/veto/event-state archive. |
| Verified aggregate measured statistics already in development features | 301 TRAIN/history series and 133 validation series. | Sparse relative to the results cohort; those measured participants do not establish when future starters were announced. |
| OpenCS2 exported map rows | 1,988 rows, all before 2026-05-01; native-ID joins reach 860 distinct series, all in frozen TRAIN/history. | These are raw export rows, not 1,988 independently certified organization-labelled map outcomes. |
| StrandedPond raw pre-May map rows | 1,044 rows; 721 rows relate to 301 already verified measured series. | Raw rows include duplicates. Exact native-ID, orientation, score reconciliation and roster checks remain necessary. |
| Ektarr archive | Separate match/team/player identifier namespace. | No verified crosswalk to native HLTV identities was established. Do not join by numeric ID or fuzzy name as if verified. |

**Important label correction:** all 1,988 OpenCS2 export rows have `score1/score2` equal to the native **series** score. Repeating those values across map rows does not create map results. `analysis_score_ct/t` are side counts, not stable organization scores. The existing demo reconstruction produces anonymous roster-group outcomes; assigning them to organizations still requires an independently validated orientation. The dataset's published schema/demo is a data lead, not a guarantee of those downstream joins. [14]

No inspected local source established a complete ordered veto history, certified pre-match lineup-announcement times, or verified cross-source organization aliases. Therefore the current experiment uses broadly available results rather than inventing granular features. A future extraction can promote reconstructed map labels only after independent orientation and native series-score reconciliation.

## Proposed model and why it differs from the incumbent

The current ladder already contains series Elo, map-score-fraction Elo, daily Glicko, form/statistical channels, standings, regularized logistic regression and gradient boosting. The earlier selected ridge model failed the reserved bookmaker comparison. More capacity alone is not evidence of progress.

The new hypothesis changes the observation model and uncertainty propagation:

1. Each literal organization has a Gaussian latent map-strength marginal, with mean and variance.
2. Random-walk or mean-reverting time dynamics change the prior between observed source days.
3. A single latent strength difference is shared across the maps of a series. Maps are conditionally exchangeable given that difference; integrating it creates predictive dependence. This is **not** map-specific strength or a learned veto policy.
4. Observe either the winner alone or the complete legal terminal series score. The winner-only channels are explicit ablations.
5. Integrate the series probabilities over latent uncertainty instead of applying the series formula to only the mean strength or mean map probability.
6. Update all results on a source day from the same opening state; do not let an arbitrary within-day row order supply unavailable information.

For first-to-\(r\) maps and loser score \(l<r\), our conditional terminal-score probability is

\[
P(r,l\mid p)={r+l-1\choose l}p^r(1-p)^l.
\]

The final map must be the winner's, so a fixed-length binomial coefficient would be wrong. A 2–0 result and a 2–1 result carry different likelihoods about latent map strength. For map-coded history, one completed map contributes one binary map outcome—not thirteen independent round outcomes.

The implemented design uses fixed 64-point Gauss–Hermite quadrature and Gaussian marginal moment matching. Per-match posterior-to-prior Gaussian site increments are summed within the day. This is an assumed-density/day-site approximation, not exact joint Bayesian inference, exact Whole-History Rating, or a posterior confidence guarantee. Cross-team covariance is discarded. Fixed quadrature has finite accuracy, especially for unusually broad inactivity variances or extreme surprises. These limitations must remain visible when interpreting results.

An independent 108-point numerical check compared the quadrature with adaptive integration. The largest sampled series-win probability error was 0.002273 for variance at most 32; it reached 0.022514 at variance 128. This is a sampled accuracy diagnostic, not a uniform bound. Full inputs/results are retained in the experiment's `verification.json`.

The selected ridge model forecasts the series winner from these channels. Its match probability must not be combined with one Bayesian expert's terminal-score distribution and advertised as one coherent set of cross-market prices.

### Frozen candidate set

| Candidate family | Count | Fitting/selection rule |
|---|---:|---|
| Existing selected ridge baseline | 1 | Identical TRAIN boundary and original recipe. |
| Raw Bayesian channels | 16 | Winner/score likelihood × prior SD 1/2 × four declared dynamics. |
| Slope-calibrated Bayesian channels | 16 | Calibration fitted on TRAIN only. |
| Ridge combinations | 8 | Four regularization strengths on Bayesian-only and existing ratings/form plus Bayesian channels. |
| Fixed blends | 8 | Existing baseline plus four designated TRAIN-calibrated score channels, at two predeclared weights. |

Random-walk variance rates are 0.0025 and 0.01 per day in natural-log-strength units; mean-reversion half-lives are 90 and 365 days. These are bounded experimental hypotheses, not values claimed by a professional provider. No odds enter predictor columns. No TRAIN+validation coefficient refit is allowed.

## How to compare with bookmakers without fooling ourselves

The prior reserved comparison used 419 matched, clock-eligible one-hour consensus cases: model log loss **0.632088**, consensus **0.600528**, difference **+0.031560**. Its day-block 95% interval was **[+0.015649,+0.048596]**. That is evidence of underperformance, not bad luck established by an accuracy headline. See the immutable prior `verified/confirmation.json` and `cs2-bookmaker-challenge-results-2026-10-09.md`.

For the new research experiment:

- Fit coefficients only through April 30; select on the same May–June development cases, with the same team orientation, results, market horizon and de-vig convention.
- Report proper scoring rules: log loss and Brier score. Accuracy/AUC are diagnostics, not the selection objective.
- Compare selected model, incumbent and consensus on **identical IDs**, not the model's broad 1,487-row population against the bookmaker's 545-row subset.
- Retain every candidate's score and use TRAIN-only chronological diagnostic folds. Repeated development selection still creates optimism; those folds and validation uncertainty are not a fresh final test.
- Do not score the new candidate on the already exposed July–October TEST. Its labels/aggregate failure informed this research direction, so relabelling it as unseen would be dishonest.
- Advancing an inference state through completed archived matches is legitimate for a future scenario; computing a new historical TEST leaderboard is not part of this experiment.
- Require newly reserved later outcomes and contemporaneous quotes for independent confirmation. Net betting profitability additionally requires executable prices, limits, rejects, settlement rules and costs; a de-vig probability comparison does not establish ROI.

## Prioritized next data work

1. **Historical/current identity and availability ledger:** native match, event, team and player IDs; source publication times; announced starters/substitutes; map-pool/rule era. This is a missing prerequisite for honest roster/context features, not a request to guess lineups.
2. **Verified organization-labelled map outcomes:** preserve map names and exposure; validate joins/orientation/duplicate handling before shrinking toward team-wide strength.
3. **Observed veto sequences:** learn actual actions conditioned on legal pools and the forecast horizon; do not assume the realized future veto was already known.
4. **Official round/economy data where justified by target and access rights:** useful for coherent live/round/player markets, but not a substitute for a fair pre-match experiment. [1, 3, 5]
5. **Prospective evaluation:** reserve future dates and log immutable predictions/quotes before outcomes. Stop adaptive reuse of exposed test sets.

## Evidence limits and saved provenance

`cs2-professional-model-research-sources-2026-10-10.json` preserves the three research tracks, exact quoted evidence, source conditions and the local coverage audit. Abstract-only and search-only leads remain labelled as such. The unresolved Valve Steam news-feed lead is not used as a verified citation here.

The research provides mechanisms to test, not a guarantee of becoming competitive. The separate experiment artifacts and results report record what actually ran and whether the new model improved development performance. No vendor, Go, Halo, tennis, basketball or public Pick'Em score is presented as our CS2 bookmaker performance.

## Bibliography

[1] Jordan Peltier, PandaScore (2022-09-16). **CS:GO Round Modeling**. Public proof of concept and its limitations. https://www.pandascore.co/blog/cs-go-round-modeling

[2] Ota Špecinger, Oddin.gg (2025-09-10). **Inside Oddin.gg: What It Really Takes to Run a 24/7 Esports Trading Desk**. First-party operational description. https://oddin.gg/knowledge-hub/esports-trading-desk

[3] Oddin.gg (undated; accessed 2026-10-10). **Official Esports Data Feed**. First-party product description; performance claims not independently verified. https://oddin.gg/official-esports-data

[4] Abios (2024-09-25). **Abios launches Counter-Strike 2 Bet Builder on the Kambi network**. First-party product announcement. https://abiosgaming.com/press/esports-bet-builder-kambi/

[5] GRID (undated; accessed 2026-10-10). **Plans & Pricing — Official Esports Data Access**. Access/use-case information. https://www.grid.gg/plans/

[6] Mark E. Glickman (1999). **Parameter estimation in large dynamic paired comparison experiments**. Applied Statistics 48(3), 377–394. https://www.glicko.net/research/glicko.pdf

[7] Rémi Coulom (2008). **Whole-History Rating: A Bayesian Rating System for Players of Time-Varying Strength**. Original method and Go experiment. https://www.remi-coulom.fr/WHR/WHR.pdf

[8] Lucas Maystre, Victor Kristof and Matthias Grossglauser (2019). **Pairwise Comparisons with Flexible Time-Dynamics**. Original temporal skill-model paper. https://arxiv.org/pdf/1903.07746

[9] Tom Minka, Ryan Cleven and Yordan Zaykov (2018). **TrueSkill 2: An improved Bayesian skill rating system**. Microsoft Research report; original Halo population. https://www.microsoft.com/en-us/research/wp-content/uploads/2018/03/trueskill2.pdf

[10] Guido Petri et al. (2021). **Bandit Modeling of Map Selection in Counter-Strike: Global Offensive**. Veto-strategy study; not a bookmaker probability benchmark. https://arxiv.org/pdf/2106.08888

[11] holygodly (repository accessed 2026-10-10). **CS2 Major ML Pick'Em Predictor**. Public implementation description, incomplete data reproducibility. https://github.com/holygodly/CS2-Major-ML-PickEm-Predictor

[12] AmitYitshaki (repository accessed 2026-10-10). **cs2-ai-forecasting**. Public project description, not a verified professional performance claim. https://github.com/AmitYitshaki/cs2-ai-forecasting

[13] HLTV (2025-08-20). **Introducing Rating 3.0**. Contextual performance-measurement explanation. https://www.hltv.org/news/42485/introducing-rating-30

[14] blanchon (dataset page accessed 2026-10-10). **OpenCS2 dataset demo**. Published schema/sample documentation; local downstream labels independently audited. https://huggingface.co/datasets/blanchon/opencs2_dataset_demo
