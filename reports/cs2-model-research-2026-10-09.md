# Building credible Counter-Strike and sports probability models

**Research snapshot: 2026-10-09.** This report supersedes model-quality, universal-floor and cross-sport-ceiling claims in the three older October 5 reports. Their bodies remain archival, not current evidence. Companion deliverables: [source/claim ledger](cs2-model-research-sources-2026-10-09.json), [detailed local results](cs2-model-results-2026-10-09.md), and [frozen experiment protocol](cs2-model-experiment-protocol-2026-10-09.json).

## Executive conclusion

The inspected literature supports a disciplined construction process, not a universal CS2 log-loss floor or a ready-made unbeatable architecture. Start with past-only paired strength ratings; add low-capacity, symmetry-preserving features and calibration; evaluate probability quality chronologically on identical situations. Published scores differ in population, target, horizon and access to information. Proper-score theory explicitly warns against interpreting scores from different situations as direct model-quality comparisons. [1]

The actual local attempt selected a **fixed equal-probability blend of player Elo and broader-archive team Elo** on VALIDATION, without using market prices or learning blend weights. On the unchanged 164-series TEST, log loss fell from **0.644669 to 0.640090**, but the paired descriptive interval includes zero; calibration error and the last chronological block worsened. On the 33 all-book-common cases, the new model was worse than the old model and each book by log loss. This is an exploratory historical improvement attempt, not a fresh confirmation, universal optimum, market edge or wagering result. Exact evidence is in `../artifacts/model_research_20261009/after/summary.json`, `../artifacts/model_research_20261009/markets/proportional_pre_start/bookmaker_comparison.json`, and the detailed results report.

## 1. Scope and evidence standards

Three research tracks inspected CS literature/public implementations, general sports forecasting methods and bookmaker evaluation. The accompanying ledger merges their claim excerpts, deduplicates sources and records corrections and access limits. A full inspected paper, an inspected abstract, mutable project documentation and a located-but-blocked publication are not equivalent evidence. Quantitative results below are source-author reports unless explicitly described as local saved observations. Abstract-only sources support only the claims visible there; blocked full texts supply no hidden performance claims.

The local target is one complete BO3/BO5 series with exact played-lineup eligibility, not one map or one round. The cohort contains 675 series, split into 396 TRAIN, 115 VALIDATION and 164 TEST observations. Historical played rosters condition the forecasts; their pre-event release times are not certified. Source-calendar-day replay conservatively freezes rating features before every result update on that day. This establishes a reproducible **retrospective actual-lineup-conditional** experiment, not proof that every feature was obtainable by a bettor at kickoff. See `../data/audits/modeling_data_audit.json` and the frozen protocol.

## 2. What actual Counter-Strike evidence establishes

| Source or public method | Inspected evidence | Relevant comparison limit |
|---|---|---|
| EsportsBench | Counter-Strike: 170,003 TRAIN and 19,302 TEST results; Elo log loss 0.6473, Glicko/Glicko-2 0.6375. Accuracies 62.90%, 64.17%, 64.18%. [2] | Mixed Counter-Strike since 2001, formats/levels and draws; not the local binary professional CS2 cohort. |
| Lund thesis | 3,788 professional CS:GO matches from 2016–2020, randomly divided into thirds; separate nine-match BLAST application did not surpass bookmaker benchmarks. [3] | Random split is not forward deployment; nine tournament matches cannot establish a stable market ranking. |
| Petri et al. | Contextual-bandit veto analysis on over 3,500 matches and 25,000 choices; estimated policy benefits up to 11% map and 19.8% series win probability for evenly matched teams. [4] | Counterfactual policy value, not observed predictive uplift or known future vetoes. |
| Xenopoulos et al. | Action valuation through changes in live win probability, using over 70 million in-game events. [5] | In-game state/action target, not a pre-match series benchmark. |
| Makarov et al. | Publisher abstract describes winner prediction and TrueSkill player-impact ratings for CS:GO and Dota 2. [6] | Full text gated; no comparable modern CS2 score adopted. |
| Public CS2 Match Analytics | README reports LR+roster log loss 0.6377 on 1,811 chronological test series; its DATA.md flags label/schema problems. [7] [8] | Project-authored, not independently reproduced; data-contract caveats prevent treating it as certified SOTA. |

EsportsBench is the strongest inspected large rating comparison, but still not a local performance target. Its data run through **March 2024**, with the latest year held out. Experiments use **seven-day rating periods**; the separate data-quality recommendation is periods no shorter than a day. Its Counter-Strike draw rate is 0.0177: accuracy excludes draws, whereas log loss accommodates fractional draw targets. Community-entered entity/date errors and mixed historical populations further limit transfer. Its results justify testing Glicko alongside Elo, not assuming Glicko always wins. [2]

The Lund thesis illustrates the difficulty of market comparison. The nine-match application gives the models five correct predictions, 55.6%, while listed books achieve 66.7–77.8% accuracy. Bookmaker log losses around 0.299–0.330 on those nine selected games are neither a general CS floor nor comparable to our 164 cases. Day-of-match odds and historical/career player summaries do not automatically establish a timestamp-matched forecast experiment. [3]

### Ranking is not probability calibration

Valve Regional Standings incorporate prize winnings, defeated-opponent achievement/network and head-to-head information. Valve evaluates matches against preceding-week standings and reports a binned Spearman correlation of 0.98, while explicitly acknowledging underestimation at the low end and overestimation at the high end. **0.98 is not match accuracy** and does not establish calibrated probabilities. [9] A dated roster calculation shows age-weighted achievement components and ten-best-results rules over six months, illustrating that VRS is more than ordinary Elo. [10]

A released historical VRS snapshot could therefore be a ranking comparator or explanatory covariate. A strength-to-probability mapping would need estimation from earlier outcomes and evaluation on later ones. Today's repository state cannot replace the version actually released before an old match. Snapshot date, publication clock and roster correspondence are prerequisites, not assumptions.

## 3. A reusable model-construction recipe

### Transparent strength, player and uncertainty baselines

Bradley–Terry models express `logit P(A beats B) = strength(A) - strength(B)`; structured versions add explanatory and contest-specific effects. Elo belongs to the related online paired-comparison family. These are useful starting points because strength differences are interpretable and side reversal naturally complements the probability. [11] [12]

For team Elo, decide what constitutes a continuing entity and when an observation updates state. Literal names are reproducible but can split aliases or merge organizations incorrectly. For player Elo, define lineup aggregation, cold-start priors and attribution of a team outcome to players. Neither a player model nor a team model automatically handles roster continuity well. Glicko's uncertainty-aware alternative should be measured under the same update schedule and scale; the empirical EsportsBench advantage is corpus-specific. [2]

Small-data design recommendation: shrink rare team/player/map effects toward a global or team prior, keep a small regularization grid, and report support counts. Do not estimate hundreds of unconstrained coefficients merely because five players are named. A roster-switch study reports initial individual and collective performance decline followed by recovery, supporting a continuity hypothesis—not a universal causal penalty or optimal half-life. [13] Past action-value summaries are possible features only from already completed games; live action valuation itself is not a pre-match input. [5]

FiveThirtyEight's first-party NBA data documentation explicitly distinguishes Elo entering a game, Elo after the game and a forecast based on Elo and game location. That transparent separation is a useful engineering pattern: persist both pre-observation forecasts and subsequent state updates. The inspected README is not an independently reproduced algorithm or evidence for the older report's cross-sport performance table. [14]

### Maps, vetoes, series and era

A pre-veto forecast integrates uncertainty over legal future map choices; a post-veto forecast conditions on announced maps. A played-map record cannot silently become a pre-veto feature. Petri et al.'s pick/ban analysis supports modeling that decision process, but its estimated policy improvements do not certify a forecasting gain. [4]

For a known BO3 sequence, an illustrative conditional-independence construction is `p1*p2 + p1*(1-p2)*p3 + (1-p1)*p2*p3`. This is an algebraic modeling assumption, not evidence that direct series classification is inferior. Shared strength uncertainty and correlated map outcomes can invalidate naive independence. A stopped-series dynamic program or shared latent-strength simulation is an option only after effective pool, series rules and forecast-time map information are captured. Score the final series target; do not inflate the sample by treating maps from one series as independent opportunities.

Game-version, rule, map-pool and roster eras should be explicit. The local broader archive starts at 2024-01-01 as a **modern-era proxy**, not independently verified CS2 version evidence. Older results might initialize a shrinkage prior, but a transfer assumption requires a separate chronological sensitivity comparison. Neither broad historical EsportsBench results nor a continuity paper proves that a chosen decay constant works for modern professional CS2. [2] [13]

### Point-in-time feature contracts and prequential evaluation

For every field, retain its entity key, event time, capture/release time, origin and eligibility rule. “Historical” is insufficient: an eventually corrected result or played roster can still contain information unavailable at prediction time. Fit imputers, scalers, feature selection and classifier coefficients only within the preceding training window. Rolling-origin evaluation trains only on observations before the forecast observation. [15]

Ratings can update online after evaluation-day outcomes to predict later days; fitted coefficients must remain under the declared frozen policy. Distinguish this prequential state replay from batch retraining. Repeated earlier origins expose sensitivity to drift; an empirical time-series study favors repeated out-of-sample evaluation under nonstationarity while allowing other methods under stationary conditions. Chronological deployment simulation is a task-specific requirement, not a theorem that every random cross-validation design is invalid. [16]

A durable experiment record specifies target, orientation, inclusion rules, split dates, feature availability, forecast horizon, update order, candidate recipes, selection rule, refit policy, hashes and saved per-case predictions. The generic local `pipeline.cli` training path selects on TEST and is outside this audited Kedro experiment; its outputs must not be mixed with VALIDATION-selected evidence.

### Proper scoring, calibration and limited calibration data

For a binary outcome, log loss is `-y*log(p) -(1-y)*log(1-p)` and Brier loss is `(p-y)^2`. Strictly proper rules reward truthful probability reporting in expectation. Accuracy only tests the side of a threshold; 0.501 and 0.99 make the same hard decision but very different errors after an upset. Proper scores combine aspects of calibration and discrimination; a better aggregate score need not imply better calibration alone. [1] [17]

A constant 0.5 forecast has log loss `ln(2)`, approximately 0.6931. For true conditional probability `p`, conditional entropy is `-p*log(p)-(1-p)*log(1-p)`; its average depends on the information set and population. It is not a sport-wide constant and there is **no unique accuracy-to-log-loss conversion**. The older 0.630–0.650 CS2 floor and supposed accuracy requirement for log loss 0.600 are unsupported deductions, not established bounds. [1]

Calibration must use forecasts generated independently of the fitted classifier's own training cases. Official calibration guidance warns that training-output calibration is biased and isotonic can overfit small samples. [17] For an already prequential rating channel, a TRAIN-only scalar `sigmoid(a*logit(p))`, with nonnegative slope and no intercept, offers low capacity and preserves side complementarity. That differs from calibrating a classifier's in-sample probabilities. In a learned ensemble, past-only base forecasts do not automatically make the meta-learner's fitted forecasts out-of-sample: stacking needs leakage prevention at both stages. [18]

Use reliability diagrams, probability distributions and support counts alongside scores. CORP documents instability from arbitrary bin choices and offers a principled reliability diagnostic; it does not make tiny dependent sports samples precisely calibrated. [19] A diagnostic isotonic curve is also distinct from fitting an isotonic calibrator for deployment.

### Selection and uncertainty

Declare candidates before scoring, select only in development, and reserve a genuinely future final period. A minimum over many validation candidates is optimistically selected; 115 validation cases and 16 candidates warrant restraint. Use paired per-case loss differences, repeated chronological origins and descriptive dependence-aware intervals rather than separate-cohort mean subtraction. Same-case comparison follows proper-score theory directly. [1]

The present source-day bootstrap samples whole days, 5,000 replicates with seed 42; it is a recorded local procedure, not a claim of valid inference under arbitrary tournament dependence or drift. Whole-day resampling need not preserve correlations across days. A future stronger design can preregister event/contiguous-block sensitivities when identifiers and sample size support them. Neither an interval crossing zero proves equivalence nor a selected point gain proves durable superiority.

## 4. Comparing bookmakers without confusing skill and execution

For decimal prices `O_i`, define `q_i=1/O_i` and overround `sum(q_i)-1`. Proportional de-vig sets `p_i=q_i/sum(q)`. The power alternative solves `sum(q_i^a)=1` **separately from each quote's prices**, then uses `p_i=q_i^a`. The exponent is not fitted to outcomes or chosen on TEST. Clarke et al. report dataset-specific power-method advantages in three sports and equivalence of Shin/additive for binary markets; this does not establish a uniquely correct CS2 fair probability. [20]

A fair comparator matches the exact series market, native event, outcome orientation, opportunity set and time horizon. Publish each book/model paired cohort, exclusions and a separate all-book intersection; never rank books from differently covered populations. Overround is not realized bookmaker profit. Better log loss is not a betting return. Evidence from football separates overall odds/outcome bias from individual books' failure to use competitors' information, illustrating that calibration and informational efficiency are different tests. [21]

OddsPapi documents `createdAt` as when a historical odds entry was created. That does not certify original bookmaker publication, local receipt, bettor access or an accepted price. [22] Latest stored pre-start quotes are therefore not automatically verified closing lines or equal-information comparisons. A source-day midnight sensitivity is a convention, not a verified release clock or reconstruction of T-minus snapshots.

CLV needs an entry price and a clearly defined later benchmark; executable ROI additionally needs accepted prices, stake rules fixed without outcome selection, limits, settlement/void rules, costs and complete eligible opportunities. These are evidence prerequisites, not implemented claims. The current experiment contains no provider requests, automatic bets, market blending, actionable ROI or CLV result.

## 5. Why this frozen ladder, and what actually happened

The protocol retained six original models: player Elo, Glicko, TrueSkill, OpenSkill, logistic regression and gradient boosting. It added two raw archive Elo channels (no decay and 180-day decay), four TRAIN-only slope calibrations, two compact ridge models and two fixed 50/50 player/archive probability blends: **16 candidates plus two diagnostic controls**. Ridge recipes use TRAIN-only scaling, `C=0.1` and no intercept. The controls are constant 0.5 and a TRAIN class prior, the latter explicitly orientation-asymmetric rather than a deployed symmetric model.

This bounded ladder tests available history and confidence scaling without inventing map/roster release clocks or fitting a flexible stack to a few hundred series. Three expanding-origin folds lie entirely inside TRAIN and are diagnostic, not an extra selection mechanism. Selection is minimum VALIDATION log loss, with deterministic tie rules and **no TRAIN+VALIDATION refit**. Exact recipes and split dates are in the protocol.

The archive adds 23,568 valid modern BO3/BO5 results covering 1,768 literal team names, only as past history. It does not add target labels to the frozen 675-series evaluation cohort. Seven archive fields increase the schema from 31 to 38 predictors. Saved full-cohort validation preserves all original IDs, dates, orientation, targets, split membership and original predictor values. This establishes paired experiment comparability, not certified entity identity or pre-event lineup availability.

| Local observed metric | Original selected player Elo | Selected fixed player/archive blend |
|---|---:|---:|
| VALIDATION log loss | 0.684419 | 0.673296 |
| TEST log loss | 0.644669 | 0.640090 |
| TEST Brier | 0.226488 | 0.224272 |
| TEST accuracy | 60.9756% | 64.0244% |
| TEST ECE, same diagnostic bins | 0.087954 | 0.102596 |

The paired TEST log-loss delta is -0.0045795, descriptive 95% source-day interval **(-0.0214665, +0.0127462)**; the Brier interval also includes zero. The final chronological block worsened. TEST had already been observed before this attempt, so recipe freezing and label-isolated selection do not turn it into fresh independent confirmation. See saved `after/summary.json` and `after/metrics.json`; `after/manifest.json` records evidence hashes, not independent proof of opaque pickle training provenance.

Primary market coverage is unchanged: 203 book/match quote pairs, 92 games and 33 all-book-common games. On those common 33, new-model log loss is **0.664955**, old-model **0.641430**, Pinnacle **0.625063**, Bet365 **0.629480**, Unibet **0.607781**. Per-book samples are 67/74/62; new-model-minus-book log-loss deltas are +0.017558/+0.014418/+0.027336, all primary intervals crossing zero. Thus the overall gain did **not** carry over to this odds-covered intersection; neither model establishes an edge.

Power sensitivity preserves cohorts and does not establish an advantage. The source-day filter leaves just 13 games, book counts 1/0/12 and no all-book intersection; singleton/very sparse intervals are not substantive confirmation. `../artifacts/model_research_20261009/markets/validation.json` certifies identical original/new primary quote IDs, raw prices, clocks, orientation and bookmaker probabilities. The four exact market-mode reports and detailed results retain the full comparisons.

Serialized/CLI inference reproduces all 164 saved forecasts and complements reversed teams; evidence is `../artifacts/model_research_20261009/inference_smoke.json`. Its Legacy–M80 example uses supplied historical roster evidence, **not verified current starters or a live prediction**. Model pickles require trusted inputs. Local runtime execution and engineering checks are distinct from statistical confirmation.

Durable parent verification is recorded in `../artifacts/model_research_20261009/verification.json`: 11 input, 38 source and seven archived-output hashes checked; the six original candidate scores/probabilities unchanged; 398 tests and 218 subtests passed. `../artifacts/model_research_20261009/environment.json` records Python 3.12.3 and actual numerical-library versions. Citation linkage passed: 38 inline markers and 22 referenced bibliography entries; this is a linkage/URL-syntax check, not independent factual replication. The selected model is the stronger default **under the predeclared VALIDATION criterion only**; its future or market-relative superiority is not established.

## 6. Evidence-gated design menu

These are conditional design options, not promised unfinished features:

- **Map/veto/series modeling:** requires captured effective pools, rules and timestamped veto publication; distinguish pre-veto from post-veto forecasts. [4]
- **Roster continuity and hierarchical player effects:** requires trustworthy pre-event starters, identity mapping and prior shared-history support; observational switching effects are a hypothesis, not a fixed penalty. [13]
- **VRS covariates:** require the released historical snapshot and a past-only calibrated mapping, not today's rank or a correlation misread as accuracy. [9]
- **Market consensus/blends or fixed-horizon snapshots:** require synchronized quote/receipt clocks and development-period coverage; never choose weights/horizons from this TEST. [18] [22]
- **Deployment/ROI:** requires prospective forecasts with recorded information cutoffs, execution and settlement evidence plus a new future holdout.

The current defensible outcome is a reproducible, modest model-improvement attempt with an honest negative market-relative finding and unresolved timing constraints—not a ceiling, SOTA title or actionable betting recommendation.

## Bibliography

[1] Gneiting & Raftery, Strictly Proper Scoring Rules, Prediction, and Estimation — https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf

[2] Thorrez, EsportsBench: A Collection of Datasets for Benchmarking Rating Systems in Esports — https://claytonthorrez.com/papers/esportsbench/EsportsBench_preprint.pdf

[3] Broms & Nordansjö, Predicting Counter-Strike Matches using Machine Learning Models — https://lup.lu.se/student-papers/record/9145457/file/9145459.pdf

[4] Petri et al., Bandit Modeling of Map Selection in Counter-Strike: Global Offensive — https://arxiv.org/abs/2106.08888

[5] Xenopoulos et al., Valuing Player Actions in Counter-Strike: Global Offensive — https://arxiv.org/abs/2011.01324

[6] Makarov et al., Predicting Winning Team and Probabilistic Ratings in Dota 2 and Counter-Strike: Global Offensive Video Games — https://link.springer.com/chapter/10.1007/978-3-319-73013-4_17

[7] CS2 Match Analytics, project README — https://github.com/IlliaDol/cs2-match-analytics

[8] CS2 Match Analytics, DATA.md — https://github.com/IlliaDol/cs2-match-analytics/blob/main/DATA.md

[9] Valve, Counter-Strike Regional Standings — https://github.com/ValveSoftware/counter-strike_regional_standings

[10] Valve, WW roster calculation report, 2026-09-07 — https://github.com/ValveSoftware/counter-strike_regional_standings/blob/main/invitation/2026/details/2026_09_07/0064--ww--ct0m-deko-kelien-m3wsu-strogo.md

[11] Turner & Firth, Bradley-Terry Models in R: The BradleyTerry2 Package — https://www.jstatsoft.org/article/view/v048i09

[12] Király & Qian, Modelling Competitive Sports: Bradley-Terry-Elo Models for Supervised and On-Line Learning of Paired Competition Outcomes — https://arxiv.org/abs/1701.08055

[13] Zhang, Muric & Ferrara, Individual and Collective Performance Deteriorate in a New Team — https://arxiv.org/abs/2205.09693

[14] FiveThirtyEight, Historical NBA Elo data README — https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/README.md

[15] Hyndman & Athanasopoulos, Forecasting: Principles and Practice, Time series cross-validation — https://otexts.com/fpp3/tscv.html

[16] Cerqueira, Torgo & Mozetič, Evaluating time series forecasting models: an empirical study on performance estimation methods — https://kt.ijs.si/igor_mozetic/papers/CerTorMoz-TimeSer-MACH-20.pdf

[17] scikit-learn, Probability calibration — https://scikit-learn.org/stable/modules/calibration.html

[18] Schaffer et al., Stacking Regression for Time-Series — https://pure.hw.ac.uk/ws/portalfiles/portal/91472074/Schaffer_Stacking_paper_ECONVN2023.pdf

[19] Dimitriadis, Gneiting & Jordan, Stable reliability diagrams for probabilistic classifiers — https://pmc.ncbi.nlm.nih.gov/articles/PMC7923594/

[20] Clarke, Kovalchik & Ingram, Adjusting Bookmaker’s Odds to Allow for Overround — https://www.sciencepg.com/article/10.11648/j.ajss.20170506.12

[21] Elaad, Reade & Singleton, Information, prices and efficiency in an online betting market — https://centaur.reading.ac.uk/86111/1/betting_efficiency_elaad_reade_singleton.pdf

[22] OddsPapi, GET historical-odds documentation — https://oddspapi.io/en/docs/get-historical-odds
