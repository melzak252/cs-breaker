# cs-breaker: a Counter-Strike 2 forecasting and research platform

**Research date:** 2026-10-04  
**Scope:** a CS2 counterpart of EnsembleLegends: prematch map/series probabilities, tournament forecasts, and an optional odds-analysis/manual-ledger application.  
**Method:** four parallel research tracks, primary-source inspection, literature comparison, local inspection of EnsembleLegends contracts and services, and executable checks of the series/payout mathematics.  
**Status at research time:** design and research, not then an implemented or trained CS2 system. [Organization-series and player-roster map Elo experiments have since been implemented and evaluated](../README.md); §13 records player coverage and negative forecasting comparisons. The broader learned-history/veto/tournament/dashboard design remains research. No provider account, paid entitlement, production data coverage, or profitable betting strategy was verified.

**Source-provenance follow-up:** [Counter-Strike statistics sites, original data, and collection options](cs2-statistics-sites-and-data-sources.md).

> **2026-10-09 supersession notice:** This report preserves historical design and older experiments. Current audited selection, scores, artifact versions and market comparisons: [observed results](cs2-model-results-2026-10-09.md), [research](cs2-model-research-2026-10-09.md), [source ledger](cs2-model-research-sources-2026-10-09.json). The validation-selected player/archive Elo blend has only an exploratory overall TEST gain; observed ten-bin ECE, the final TEST block and bookmaker-common scores regress. No demonstrated market or ROI edge. Unimplemented map/veto/tournament/market-blending ideas remain conditional research, not promised or completed features. Standalone player-map studies use separate cohorts.

## Executive recommendation

Build **a map-aware, roster-aware, time-auditable CS2 forecasting platform**. Start with professional match-winner forecasts and an explanatory dashboard. Add tournament simulation using event-specific rules. Treat betting analysis as a separate, rights-gated extension rather than the reason to assume a statistics feed is usable.

The central modeling difference from League of Legends is that a CS series is composed of different map matchups chosen through a veto. The Valve Major rulebook specifies sequential bans/picks and map-specific starting-side choices; other organizers use different side-selection procedures. Consequently, distinguish a **pre-veto forecast**, integrating possible legal map sequences, from a **post-veto forecast**, conditional on the published sequence. Neither should use a lineup, veto, or statistic that became available after its forecast cutoff. [14-16]

The main project risk is **data access and permitted use**, not whether Python can fit a classifier. PandaScore advertises free fixtures, historical statistics from €400/game/month, and live statistics from €1,000/game/month, excluding tax; its pricing explicitly restricts those statistics plans to non-betting use. Its subscription terms prohibit developing odds or odds-related products using subscription data. HLTV's terms also prohibit scraping/data mining and competitive-product use. A public dataset or parser does not automatically confer rights to the original data. [3,4,8,10]

Use chronological global/team-map ratings and regularized tabular models before neural architectures. Existing CS literature supports examining maps, player effects, roster continuity, and veto behavior, but it does **not** establish a reproducible contemporary CS2 betting edge. A 2026 professional-CS2 undergraduate thesis reports AUC 0.6828 and accuracy 0.6293 for its selected logistic model; a separate professional-tournament study failed to beat its bookmaker benchmark. These are heterogeneous studies, not promised performance targets. [20,26]

The strongest contribution is an honest answer to: **How much do map specialization, veto uncertainty, and roster continuity improve forward, calibrated CS2 predictions over a global-strength baseline?** A good sports forecast remains useful even if it cannot beat executable bookmaker prices.

| Finding | Evidence | Decision |
|---|---|---|
| Map selection is part of the prediction problem | Official veto rules; research on historical veto choices [14,21] | Separate pre-veto and post-veto targets |
| Rules and map pools change | ESL's July 2026 pool update; organizer-specific handbooks [15-18] | Version rules by event and effective date |
| Free fixtures are not rich historical statistics or betting rights | Provider plan reference, pricing, terms [1-4] | Audit coverage and license before integration |
| Roster/player effects are plausible but hard to identify | Plus/minus and roster-transfer studies [22,23] | Shrink estimates; expose uncertainty |
| Accuracy alone is an inadequate probability-model objective | Proper-scoring and calibration literature [28,30] | Lead with log loss, Brier score, calibration |
| Betting backtests depend on quote time and settlement | Odds-feed documentation and bookmaker rules [35,36,39] | Store raw quote state and actual market rules |

## 1. Scope, evidence, and research questions

### What “the same project” should mean

Retain the useful product structure of EnsembleLegends, not its sport-specific assumptions:

1. Reproducible offline model comparison on chronological data.
2. Prematch probabilities for named maps and entire series.
3. Tournament qualification/championship probability forecasts.
4. A web dashboard showing forecasts, provenance, uncertainty, and results.
5. Optional comparison with market probabilities and manual reserve/settle accounting.

Research boundaries: no automated wagering, paid purchases, mass scraping, model training, or deployment in this investigation. Live round prediction, player props, and map-round totals are researched as possible extensions; they require separate targets, source entitlements, and settlement contracts.

### Four parallel tracks

- **CsDataAccess:** fixtures, statistics, demos, licenses, coverage, access cost.
- **CsSportsModels:** map/team/player/veto models and published empirical results.
- **CsRulesTournaments:** map pools, series rules, Swiss/playoff formats, rankings.
- **CsOddsValidation:** historical quotes, evaluation leakage, settlement, betting feasibility.

Sources were opened rather than relying only on search snippets. High-materiality findings were checked against official documentation where available. Some claims remain single-provider specifications or single-study findings; the report labels their limits rather than pretending they are independently replicated. Public documentation does not establish an account's actual entitlement or a historical dataset's completeness.

Local inspection covered EnsembleLegends' model contract, forecast-evidence service, qualification service, wallet operations, API entry point, and benchmark structure. The new `cs-breaker` workspace had only a README before this report. All architecture below is **proposed**, not existing code.

### Questions that the implementation should answer

- Do map-specific ratings improve series probability forecasts beyond global team strength?
- Does a published veto improve forecasts, and how much is due to additional information rather than a better model?
- Do player and roster-continuity features improve forecasts after accounting for opponent strength and map?
- Does older CS:GO data help as a transfer prior, or hurt through regime mismatch?
- How does the sports-only model compare with a same-cutoff market-only model?
- Can a rule-aware simulator produce calibrated tournament forecasts without future-bracket information?
- If betting use is authorized, does a frozen decision policy survive prospective executable-price, tax, and settlement accounting?

## 2. Data access: choose the lawful route before the model

### Source comparison

| Source | What is supported by inspected material | Access / constraint | Recommended role |
|---|---|---|---|
| PandaScore fixtures | Schedule, results, basic competition context [1,2] | Advertised €0/game/month; 1,000 requests/hour [3] | Candidate sports-only fixture backbone, subject to terms |
| PandaScore historical/live stats | Historical match/game/team/player endpoints; round/event access varies by tier [2] | Historical from €400; live basic from €1,000/game/month; non-betting stats plans [3,4] | Rich statistics only after coverage and use-right checks |
| GRID Open Access | Central Data and Series State [5] | Events WebSocket and File Download are explicitly excluded [5] | Evaluate available series state; do not promise a demo/event archive |
| GRID historical/commercial products | Historical state/event products documented; betting commercial offering [6,38] | Historical reach is title-dependent; commercial scope/territory negotiated | Candidate negotiated feed, no public price or CS2 coverage established here |
| Liquipedia API | Tournament/participant/context information | API policies, attribution, caching, rate limits [7] | Supplement event metadata through permitted API access |
| HLTV | Valuable results, roster and map context | Terms prohibit scraping/data mining and competitive use; business licensing required [8] | Licensed source or manual research reference, not a scraper plan |
| Licensed CS2 demos + Awpy | Rounds, kills, damage, shots, bombs, utility, footsteps, ticks [9] | Demo acquisition/redistribution rights separate from parser code | Rich retrospective feature extraction when lawful |
| OpenCS2 dataset card | Per-map parquet and demo data with match/player/round context [10] | Publisher labels CC BY 4.0 but describes HLTV-derived material and downstream obligations | Evaluation lead, not automatically cleared production data |
| Valve VRS | Dated rankings, roster/match details, rating-design explanation [11,12] | Ranking release time and upstream-data rights matter | Transparent dated baseline/covariate, not ready-made probabilities |

### Material licensing constraints

PandaScore's ordinary subscription terms state:

> “Under no circumstances shall the customer develop, attempt to develop, distribute, supply, or commercialize odds or odds-related products or services … using the data and services provided through the Subscriptions.” [4]

Its pricing independently says:

> “These stats plans are only available to customers with non betting-related usage.” [3]

Therefore, **do not select an ordinary PandaScore statistics subscription for a betting-probability application**. Even internal odds-related development may fall within that language. A separate written betting-use agreement/product is necessary if pursuing that route; this report is not legal advice. PandaScore advertises a separate odds product, but that advertisement does not waive the statistics subscription restrictions. [37]

GRID says Open Access includes Central Data and Series State, while Series Events, File Download, and other APIs are not included. Generic historical-data documentation does not prove that Open Access exposes populated historical CS2 veto sequences, complete rosters, or downloadable demos. Its commercial betting page leaves scope and territory to each agreement. [5,6,38]

Liquipedia's API terms specify no more than one MediaWiki request every two seconds, with `action=parse` limited to one request every 30 seconds; identifiable contact-bearing User-Agent, caching, and compressed requests are required. Automated non-API HTML access is not allowed. Content uses CC BY-SA 3.0, with attribution and applicable content-sharing obligations; that is not a blanket statement that every consuming application's code inherits that license. Approved LiquipediaDB access has its own limit of 60 requests/hour. [7]

HLTV's terms are not permission to build a commercial or competing archive through scraping. Do not design CAPTCHA bypasses or unofficial scraper wrappers into the project. [8]

The OpenCS2 card's permissive publisher license is not enough to conclude that all original tournament/HLTV material is licensed for every downstream use. The card explicitly says downstream users must respect original tournament terms. A lawful research corpus and a redistributable production corpus may be different artifacts. [10]

### Minimum useful data, in priority order

**First baseline:** stable match ID; organization/team IDs; participants; scheduled and actual start where known; competition/event; BO format; played map results; source publication/availability and retrieval timestamps.

**Map-aware system:** the historical active pool; map/build identity; played map sequence; legal veto steps and publication time; starting sides; rulebook version; forfeits, advantages and replay/administrative outcomes.

**Roster layer:** stable player IDs; expected versus confirmed starters; stand-ins; roster announcements and when they were known; prior player appearances; historical team membership. An organization name alone is not a stable roster identity.

**Demo/statistics layer:** earlier map/player/round performances; map and side exposures; economy context; missing-data markers; demo checksum and acquisition provenance.

**Optional betting layer:** named bookmaker, jurisdiction/account context, exact market/outcome IDs, line and map number, decimal price, quote state, timestamps, stake limits where supplied, and applicable settlement rules.

### Two workable acquisition strategies

**Lower-cost research route:** use a permitted fixture/result source plus a bounded, rights-cleared event/demo corpus. Narrow the initial population to the coverage actually obtained. If pre-veto logs or historical lineup-publication times are unavailable, report that the study cannot honestly measure those targets yet. A direct series baseline is still useful; it is not a substitute for claiming a completed veto-aware product.

**Commercial route:** obtain written CS2 rights covering the intended sports/odds use, territories, retention, derived outputs, and redistribution. Request a sample export spanning map-pool changes, stand-ins, reschedules, forfeits, and multiple event tiers. Verify coverage before committing to a schema or paid plan.

No source in this research establishes a complete, free, rights-cleared corpus for all professional CS2 matches, demos, vetoes, and executable historical odds.

## 3. What the literature supports—and what it does not

| Work | Setting / result | Useful takeaway | Transfer limitation |
|---|---|---|---|
| Schmidt, 2020 thesis [19] | 6,693 FACEIT CS:GO games; reported 54.38% accuracy with a random 70/30 split | Explore map/player inputs | Amateur platform, older game, random split; not forward pro-CS2 proof |
| Rocha, 2026 thesis [20] | Professional CS2; chronological development and 2026 first-semester holdout; selected L2 logistic `C=0.3`, 11 variables; AUC 0.6828, accuracy 0.6293 | A serious tabular/chronological baseline is viable | Repository abstract inspected; not independently reproduced or a market-edge result |
| Petri et al., 2021 [21] | 3,500+ matches, 25,000+ veto choices, April 2020–March 2021 | Vetoes are sequential contextual choices | Counterfactual policy evaluation is not a demonstrated live win/profit improvement |
| Xu & Moka, 2024 [22] | Regularized/Bayesian player plus/minus on 2018–2023 professional CS:GO | Partial pooling for players and teammates | Correlated rosters; ranking/plus-minus results do not establish calibrated prematch forecasts |
| Zhang, Muric & Ferrara, 2022 [23] | Observational study of performance after roster changes | Continuity and adaptation deserve features | No universal causal transfer penalty or fixed recovery period |
| Xenopoulos et al., 2020 [24] | In-play win-probability/action valuation | Richer historical player contribution than kills alone | Future in-match state is unavailable at prematch cutoff |
| Optimal Team Economic Decisions, 2021 [25] | Money/equipment and round-state modeling | Economy and rounds are dependent | Live state belongs to a distinct live target |
| Broms & Nordansjö, 2024 thesis [26] | Logistic/RF study with bookmaker benchmark; professional tournament model did not surpass benchmark | Include the market as a serious comparator | Limited sample and large uncertainty; not proof all CS2 models must fail |
| du Plessis, 2024 thesis [27] | 11,271 matches, 142 features; best reported accuracy 62.7%; abstract reports a profitable six-month backtest | Examine ratings, market relation, and return evaluation | Quote-time selection, costs and untouched confirmation not independently audited here |

Do not rank these papers by raw accuracy. Cohort, target, split, game version, class balance, available information, and market comparator differ. The two betting theses are not contradictory evidence about the same experiment. Neither is a defensible expected ROI for this project. [26,27]

The 2026 thesis performed substantial model/feature selection. Its final chronological holdout is an important distinction, but thousands of tried combinations reinforce why the new project's selection protocol must be explicit. Repeated adaptive selection can overfit an evaluation criterion by an amount comparable to apparent algorithm improvements. [20,29]

### Baseline families worth using

- **Elo / logistic strength:** simple replayable team baseline. Fit probability scale and time handling rather than inheriting League parameters.
- **Glicko-style uncertainty:** rating, deviation, volatility, and inactivity handling can motivate an uncertainty-aware team model. Adapting this to team/map components is a new design, not a result validated by the original Glicko-2 document. [32]
- **Player-skill models:** useful for changing rosters if team/player effects are identifiable and strongly regularized. TrueSkill2's Halo results are not CS2 performance targets. Its online filtering versus batch smoothing distinction is important: a future-informed smoother is inappropriate for a historical prematch forecast. [33]
- **Regularized logistic regression and a modest tree booster:** credible tabular contenders. Broad tabular benchmarks motivate trying trees, not declaring them universally superior for sequential CS2. [34]
- **Neural models:** possible later for set-based rosters or sequence history, after simpler controls show what information helps. No justification for porting a fixed LoL-role network or choosing complexity before coverage is known.

### Valve rankings: baseline, not calibrated oracle

Valve explains its rankings in terms of team/opponent accomplishments and head-to-head results, and publishes dated standings. Its inspected README explicitly acknowledges probability miscalibration: the expected-versus-observed relationship has a shallower slope than desired, underestimating low-end win rates and overestimating high-end win rates. A high binned correlation is not equivalent to calibrated probabilities. The README is historical explanatory material, not a current independently validated CS2 forecasting benchmark. [11,12]

Use only a ranking snapshot published before the forecast cutoff. Learn a rank/point-to-probability mapping from earlier outcomes, and compare it with the global rating baseline. Do not use final event rankings to predict the event that created them.

## 4. Proposed prediction system

### Forecast contracts

Define four distinct outputs:

1. **Series winner, pre-veto:** integrate legal map/veto uncertainty.
2. **Series winner, post-veto:** condition on the published map sequence and available side information.
3. **Named-map winner:** identify the map number/name and whether starting side is known.
4. **Tournament scenario:** condition on event rules and information available at a declared origin.

Each record should contain target, horizon, match/team orientation, predicted probability, model/calibrator versions, information cutoff, generation time, supporting source-capture IDs, roster status, pool/rules version, and coverage/uncertainty diagnostics.

An expected roster is not a confirmed roster. An inferred veto is not a published veto. Record those distinctions instead of silently turning unknowns into facts. A refusal/abstention is also an outcome that belongs in the opportunity denominator.

### A compact model ladder

**M0 — global team strength.** Chronological Elo/logistic baseline, with a separately calibrated dated-VRS comparator. Establish cold-start behavior and release-time updates first.

**M1 — hierarchical map strength.** Model team-map strength as a global component plus a shrunk map residual:

`strength(team, map, time) = global_strength(team, time) + map_residual(team, map, time)`.

Sparse/new maps shrink toward global strength; no independent unrestricted rating for every rare pairing. Side effects should be map- and regime-aware. If side is unknown, marginalize it; do not apply an eternal universal CT bonus.

**M2 — roster/player contribution.** Add prior player performance, roster continuity, shared experience, stand-in status, and lineup uncertainty. Use regularization because players often appear together and attribution is confounded. Avoid double-counting a team rating and the same players' unadjusted rating. Test roster features as an ablation rather than assuming a benefit. [22,23]

**M3 — tabular classifier.** Compare a regularized logistic model and a limited tree booster using the same historical state: rating differences, map specialization, opponent-adjusted form, roster continuity, rest, LAN/online context where documented, and missingness. Freeze a small candidate set; no uncontrolled combinatorial search.

**M4 — veto distribution.** Estimate observed legal choices conditional on team/opponent, event, available pool, BO format, prior picks/bans, and known information. A simple historical-choice baseline comes before a sequential learned model. The first goal is predicting what teams do, not claiming an optimized counterfactual policy. [21]

**M5 — optional richer history/set model.** Only proceed after a matched comparison demonstrates a limitation of M0–M4. Roster inputs should be permutation-aware; CS does not have LoL's universal five fixed positional slots.

### Feature rules

Compute features from released earlier matches only. Weight recent results through a development-selected decay scheme rather than porting a fixed W20 window. Account for opposition and map exposure.

Potential richer features include prior ADR/KAST, opening-duel involvement and success, trades, survival, utility damage, flash assistance, and economy-conditioned contributions. Distinguish opportunities/exposures from raw counts; raw kills can reflect role and map circumstances. These are proposed candidates, not findings that each improves forecast quality. Awpy documents the demo-derived event tables needed to construct many of them. [9]

Do not include target-map final statistics, rounds, score, final demo lineup, postmatch rankings, or published-after-cutoff roster corrections. Parsing a demo proves what happened; it does not prove what was known before it happened.

### CS:GO to CS2 transfer

Start evaluation on CS2. Tag older CS:GO data by game version, round-limit/rules regime, map version and event context. The inspected EWC 2026 rules use 24 regulation rounds and six-round overtime periods with $12,500 starting money; older datasets cannot simply be treated as the same environment. [15]

Test older data as an initialization/transfer prior against a CS2-only baseline. Do not pool MR15 and MR12 observations unconditionally or reset all map knowledge merely because the display name stayed the same.

## 5. Map-to-series mathematics and veto uncertainty

For a fixed BO3 sequence with conditionally independent map-win probabilities `p1, p2, p3`:

`P(A wins) = p1*p2 + (p1*(1-p2) + (1-p1)*p2)*p3`.

Equivalent: `p1*p2 + p1*p3 + p2*p3 - 2*p1*p2*p3`.

Stopped scorelines are:

- `P(2-0) = p1*p2`
- `P(2-1) = (p1*(1-p2) + (1-p1)*p2)*p3`
- `P(0-2) = (1-p1)*(1-p2)`
- `P(1-2) = (p1*(1-p2) + (1-p1)*p2)*(1-p3)`

**Why not use the average map probability?** With `(0.9, 0.9, 0.1)`, exact BO3 win probability is **0.828**. Plugging the mean map probability into an identical-map binomial produces **0.695259**, a **13.274 percentage-point** difference. That is a mathematical counterexample, not a measured CS2 improvement.

Important nuance: under this fixed independent model the winner probability is symmetric in the three probabilities. Map order changes stopped scorelines and length; it can change winner probability when probabilities depend on side, earlier outcomes, fatigue, or shared state. Do not falsely claim that simply permuting three fixed independent probabilities always changes the series winner probability.

Before the veto, sum over legal sequences rather than inventing a single map:

`P(series win | information) = sum_v q(v | information) * integral P(series win | v, z) dP(z | information)`.

Here `v` is a legal veto/map/side path and `z` represents uncertain shared strengths/form. A 50/50 mixture of the 0.828 example and an all-maps-0.4 scenario (series probability 0.352) gives **0.59**, not a binomial applied to the mixed average map probability.

Conditional independence is an explicit baseline assumption. Shared team form and uncertain ratings can induce dependence across maps. A proposed extension draws latent strengths once per simulated series/event and uses them for all its maps. Validate whether the richer dependence changes calibration or scoreline quality; do not add it merely for realism.

For BO5, use a stopped dynamic program or exact path enumeration with map-specific probabilities and any event-authorized initial map advantage. A series-win probability does **not** identify a map-round-total distribution or player-kill distribution; those require separate outcome models.

### Executed mathematical check

A throwaway Python calculation used exact rational arithmetic on 125 BO3 input triples with each probability drawn from `{0, 0.2, 0.5, 0.8, 1}`. Stopped scoreline mass summed to one and series-win probability matched an independent exhaustive-majority oracle for every triple. It reproduced the 0.828/0.695259 counterexample and the 0.59 veto mixture. This verifies the illustrative math, not any fitted CS2 model.

## 6. Tournament forecasting must execute the right rulebook

### Version rules, not just tournament names

The Valve Major supplemental rulebook says it is not a complete organizer rulebook. Its inspected text describes 32 teams, three 16-team Swiss stages, and an eight-team playoff. It also contains a Winter 2025 regional-allocation table; do not reuse those allocations as a 2026 fact. Major Swiss advancement/elimination matches are BO3 and other matches BO1; advancement is three wins and elimination three losses, with detailed pairing/seeding constraints. [14]

Tournament-operation requirements attach rules to an event's applicable information deadline and distinguish invitation ranking snapshots from live rankings. Freeze the applicable document/commit and organizer notices for the event, not whichever `main` revision exists when a backtest runs. [13]

The inspected Major BO3 veto has A/B bans, A's pick with B choosing side, B's pick with A choosing side, then B/A bans, with B choosing the decider's starting side. The inspected BLAST/EWC materials contain their own side/knife procedures. **A knife-round decider is not a universal Valve Major rule.** [14-16]

The July 2026 ESL changelog and Valve update document Cache replacing Overpass. The inspected pool is Cache, Inferno, Mirage, Nuke, Dust2, Ancient, Anubis; this is an observed update, not an eternal constant or permission to rewrite earlier match histories. [17,18]

### Proposed event specification

Represent an event as a versioned configuration/state machine containing:

- entries, initial seeding and the published-at time of each;
- stage graph and advancement criteria;
- Swiss pairing priorities, repeat-match restrictions, tie-break calculations;
- BO format, map pool, veto and side-choice procedures;
- regulation/overtime and administrative-result policies;
- any explicit map advantage, forfeits, walkovers, replacement teams;
- transition rules and organizer overrides with provenance.

Each simulation starts from the declared forecast origin. Draw uncertain strength once per path when using the shared-strength model. Produce future opponents from simulated outcomes; never inject the real later bracket or final Buchholz into earlier forecasts. If simulated performance affects temporary simulated ratings, isolate that state from the real historical rating store.

Report qualification, elimination, finalist/champion probabilities and selected scenario distributions. Preserve unresolved seeds/slots as explicit scenarios when not yet known. End-of-event forecasts should not masquerade as start-of-event predictions.

**Verification required at implementation:** exact tiny-bracket oracle, probability-mass checks, legal pairings, correct Swiss stopping/advancement, versioned side selection, no real-state mutation, and a manually audited complete edition. Evaluate across tournament editions; team outcomes from one edition are not independent observations.

## 7. Validation: an honest forward experiment

### Information time is part of the model

For each source, distinguish:

- source-published/source-event time, with documented semantics;
- local retrieval time;
- model information cutoff;
- forecast generation time;
- result availability/release time.

A convenient historical timestamp is not proof of prematch availability. If historical publication times are missing, explicitly downgrade the claim and collect prospective evidence. A hash protects captured-byte integrity; it does not authenticate the publisher or establish the original publication time.

### Chronological protocol

Use earlier development data for fitting and model choice, a later calibration interval, and a still-later untouched test interval. For rolling evaluation, freeze the retraining/calibration recipe before entering the later period. Keep every map of one series together. Update ratings when results become available, not merely at scheduled start or by their final database order.

Avoid random map/player-row splits, future-smoothed skills, full-dataset preprocessing, current rosters substituted into old matches, or calibrators refitted on the final test period. Selection overfit is a separate risk from ordinary parameter overfit. [29,33]

Define before testing:

- target population and forecast horizon;
- cancellation/forfeit/administrative-result treatment;
- model candidate set and feature ablations;
- score aggregation and uncertainty procedure;
- coverage/abstention handling;
- market de-vig method and quote-cutoff policy, if applicable;
- promotion/acceptance decision rule.

### Metrics and uncertainty

**Primary:** log loss and Brier score on the same eligible opportunity cohort. These are proper scoring rules: they reward honest probability forecasts rather than only correct thresholded decisions. [28]

**Secondary:** accuracy, ROC AUC, calibration plots with uncertainty, sharpness, subgroup results, coverage and rejection reasons. Calibration assessment is sensitive to grouping and sample size; a single ECE number does not establish reliable conditional probabilities. [30]

If calibration is needed, compare a simple intercept/slope adjustment or another development-selected method using only permitted calibration data. Temperature scaling is a candidate motivated by other domains, not a guaranteed CS2 remedy or an antidote to betting-selection bias. [31]

For scorelines use a separate multiclass proper score or explicitly defined ordered score; do not pretend winner-only accuracy validates a scoreline distribution. Tournament predictions need edition-level outcome scoring and uncertainty.

Pair models on the same opportunities. Use time-block resampling and an event/edition-cluster sensitivity analysis; report instability when there are too few independent blocks. Stratify by map/pool regime, BO format, LAN/online, roster certainty, event tier, and data coverage. Many maps from one match or teams from one tournament do not create many independent experiments.

### Required controls and ablations

1. Global strength versus hierarchical map strength.
2. Team-only versus roster/player augmentation.
3. CS2-only versus older-game transfer priors.
4. Pre-veto versus post-veto **as different horizons**, not a fair same-information model contest.
5. Within each horizon, identical-information model comparisons.
6. Sports-only versus market-only versus a separately selected hybrid.
7. Calibration/no-calibration comparison on the permitted cohort.
8. Missing-statistics coverage and cold-start teams/maps.

Useful implementation checks: team-swap equivariance, reproducible replay from the same source captures, and changing future-only data without changing earlier forecasts. Permanent tests should protect consumer-visible invariants, not assert incidental wording or copied parameters.

## 8. Optional odds analysis and manual ledger

### Provider leads, not verified access

OddsPapi documents fixture-based historical odds with bookmaker IDs, price, active state, limits and a `createdAt` field. Its notes say history is available since January 2026, while examples contain 2025 timestamps. Treat this as an internal documentation inconsistency and confirm actual CS2/book coverage with a sample; do not infer older archive availability from the example. `createdAt` denotes entry creation, not independently proved bookmaker publication time or a price this user could execute. [35]

EsportsOdds distinguishes price-change history from feed liveness and aggregate computation time from its oldest input. It allows unknown `in_play` state; closing flags can depend on actual start or scheduled fallback. An aggregate reference price is not a named executable bookmaker offer. Its documented truncation/closing semantics must be checked before defining a historical benchmark. [36]

PandaScore's odds product and GRID's negotiated betting offering are additional commercial leads, not evidence that their ordinary statistics/open-access tiers cover the intended betting use. [4,5,37,38]

### Market normalization

Key every observation by bookmaker, match/series identity, exact market, outcome orientation, map number, line, jurisdiction, and settlement contract. Store source and retrieval clocks plus raw source bytes.

For two complementary decimal prices `oA, oB`, a transparent proportional de-vig baseline is:

`p_market_A = (1/oA) / (1/oA + 1/oB)`.

This is a modeling convention, not a uniquely correct bookmaker probability. Freeze the chosen method. Do not combine prices from different times, markets, or books and call them one executable offer.

Select the last eligible quote state at the decision cutoff. If the most recent state is suspended/inactive, do not skip it and revive an older active price. Null/unknown prematch status requires documented treatment. Staleness needs source-heartbeat context because an unchanged price is not inherently a disconnected feed. [35,36]

A sports-only forecast uses no prices. A market-only baseline uses the same horizon. A hybrid's weight must be selected/calibrated on earlier data; do not inherit the LoL app's fixed mixing weight or evaluate against later closing prices as if they were available earlier.

### Expected value and Polish payout example

Without deductions, unit-stake expected net return at decimal odds `o` is `p*o - 1`.

If the actual payout contract reduces the stake entering the odds calculation by a factor `1-tau`, a simplified formula is:

`EV = p*(1-tau)*o - 1`, and `p_break_even = 1/((1-tau)*o)`.

Poland's Ministry of Finance lists a 12% rate for ordinary mutual betting. That statutory operator tax alone does not prove every bookmaker/account/promotion deducts exactly 12% from the user's payout calculation. STS's inspected help article explicitly illustrates using a 0.88 stake multiplier and describes additional high-win treatment. [41,42]

For the **STS-style simplified stake-deduction example**, `p=0.60`, `o=2.00`, `tau=0.12` gives EV **0.056**, not 0.20, and break-even probability **0.568182**. The Python check reproduced both numbers. Additional tax, promotions, fees, limits or account-specific payout treatment can change the calculation. The help article's threshold wording is internally inconsistent at the boundary; implement current authoritative bookmaker/account rules rather than pinning an ambiguous example threshold. [42]

### What would count as betting evidence?

Not merely high accuracy, low aggregate calibration error, positive retrospectively selected ROI, or a correlation with closing odds. The edge-selected subset can be miscalibrated even when the full cohort looks good.

Use a frozen opportunity/decision policy and record **every eligible opportunity**, including missing forecasts, stale quotes, denied stakes, rejections, and no-bet decisions. Evaluate realized net settlement after taxes/fees under actual execution assumptions. Report odds-weighted error and return uncertainty, not just win rate. Choose flat notional paper stakes initially; Kelly sizing is not a cure for uncertain probabilities or correlated exposures.

A manual ledger should reserve free capital atomically, enforce correlated same-series/map exposure limits, and settle/reverse according to auditable rules. Use decimal monetary arithmetic and database numeric fields; do not copy floating-point financial arithmetic merely because it exists in the LoL application.

Bookmaker rules differ for overtime, abandoned matches, forfeits, format changes, map advantages, and player participation. The inspected bet365 jurisdiction-specific rules give concrete examples; they are not universal settlement rules for every account. [39]

ESIC prohibits covered participants from betting on games in which they are professionally involved and from misuse of inside information. Use public/authorized inputs and do not design a project around insider access; the disciplinary code is not a claim that all lower-tier matches are corrupt or a substitute for local law. [40]

## 9. Architecture: reuse engineering patterns, replace sport assumptions

### What to reuse from EnsembleLegends

Local inspection identified useful patterns in:

- `src/analysis/model_benchmark.py`: locked benchmark artifacts, paired comparison structure.
- `betting_app/services/forecast_evidence.py`: raw source captures, hashes, explicit clocks, discovery/decision/correction/outcome events.
- `betting_app/core/models/contract.py`: versioned specifications and distinct prediction targets.
- `betting_app/services/wallet_service.py`: transactional reserve/settle flow.
- `betting_app/api/main.py` and the React client: API/dashboard organization.

Reuse concepts or extracted generic utilities only after inspecting their CS requirements. Do not wholesale copy the old research champion, W20 rules, champion features, TOP/JG/MID/ADC/SUPPORT slots, qualification constants, fixed market mixing, LoL tournament assumptions, or historical profitability language. Research and production artifacts should remain distinct and identifiable.

### Proposed stack and boundaries

A boring initial stack is sufficient: Python, pandas/NumPy, scikit-learn plus a tabular booster; FastAPI; PostgreSQL; parquet for feature tables; content-addressed files/object storage for permitted raw payloads and demos; React/TypeScript for the dashboard. Retain familiar deployment patterns where useful. Add no Kubernetes, distributed training, or separate feature service without a measured need.

Suggested new layout, **not existing files**:

```text
src/cs_breaker/
  ingestion/     # authorized provider adapters, raw capture, replay
  domain/        # identities, rosters, maps, clocks, rulesets
  features/      # availability-aware historical state
  ratings/       # global/map/player state and chronological updates
  models/        # map, series, veto and calibration artifacts
  simulation/    # versioned event execution
  evaluation/    # locked cohorts, metrics and paired comparisons
  api/           # forecast/results/provenance/tournament endpoints
client/          # React forecast and tournament surfaces
```

Do not introduce aliases/re-exports to emulate LoL package names. It is a new sport repository with clean CS contracts.

### Storage model

| Entity | Essential fields / invariant |
|---|---|
| `source_capture` | URI/provider, exact bytes/hash, retrieval time, reported source time, rights/provenance metadata |
| `team_identity` / `player_identity` | Stable canonical IDs; aliases with validity intervals; avoid fuzzy merging academy teams |
| `roster_snapshot` | Players, expected/confirmed status, organization, valid/available times, source capture |
| `match_revision` | Participants, schedule changes, actual start, event/BO/status, source availability |
| `map_instance` / `veto_action` | Map/build/pool, order/side, legal action sequence, publication clock |
| `ruleset_version` | Frozen source document/commit, effective event, parsed policy and exceptions |
| `result_release` | Sporting versus administrative result, availability, corrections |
| `feature_snapshot` | Cutoff, source/artifact hashes, missingness and replay version |
| `forecast` | Target/horizon/orientation, probabilities, uncertainty, artifact versions, supporting evidence |
| `opportunity_event` | Discovery, revision, forecast/abstention, rejection, decision, correction, outcome |
| Optional `quote` / `bet` / `wallet_entry` | Exact market/settlement identity, source clocks, state, decimal amounts and transactions |

Forecasts should be append-only with explicit corrections, not silently overwritten when later data appears. Raw rights-restricted payloads need restricted storage/export rules; “open-source application” does not mean “publish every purchased demo/feed payload.”

### Useful UI surfaces

- Upcoming series: pre/post-veto label, roster certainty, map probabilities, timestamp.
- Match detail: known picks/bans, map-specific strengths, scoreline probabilities, provenance.
- Team page: global/map form and roster history without overstating feature importance as causality.
- Tournament page: origin, rule version, qualification/champion probabilities and scenario paths.
- Results/evaluation: frozen model IDs, reliability diagrams, coverage, errors, subgroup performance.
- Optional odds/manual-ledger page: exact book/market price, age/state, payout assumptions, paper decision and settlement.

Explanations should identify input contributions and uncertainty, not invent causal stories such as “the roster change caused a five-point loss” from an observational coefficient.

## 10. Implementation roadmap with observable proofs

No fixed delivery-time or accuracy promise is defensible before access/coverage is known. Sequence by dependencies and keep each milestone runnable end-to-end.

| Milestone | Deliverable | Proof required before proceeding |
|---|---|---|
| A. Access and identity audit | Rights/coverage manifest and sampled canonical fixtures/results/rosters/maps | Representative real records, duplicate/alias reconciliation, availability-clock audit, written intended-use entitlement |
| B. Global forecast vertical slice | Replayed historical state → global model → persisted forecast → CLI/API result | Frozen chronological cohort; deterministic replay; actual held-out forecast and observed log loss/Brier/coverage |
| C. Map and roster study | Hierarchical map model, roster ablations, explicit pre/post-veto contracts | Same-cohort paired comparisons; sparse-map/cold-start checks; no future-data influence; legal veto distributions |
| D. Operational dashboard | Upcoming/results/provenance UI using the real forecast path | Launch the application and inspect actual displayed orientation, horizon, clocks, updates and unknown-state labels |
| E. One audited tournament | Executable versioned format and full forecast surface | Exact small-event oracle plus complete real-edition manual rule audit; legal transitions and mass conservation |
| F. Rights-gated odds shadowing | Raw named-book quotes, sports/market/hybrid comparison, paper decisions, manual ledger | Eligible opportunity denominator; no suspended-price resurrection; prospective frozen shadow run and real settlement reconciliation |
| G. Separate richer targets | Round totals, kill props, live rounds or broader event coverage | Dedicated target/settlement contract, additional access, matched forward benchmark and actual scenario smoke |

For a thesis/research project, B–E form a complete substantive deliverable without wagering: reproducible model comparison, map/veto analysis, uncertainty, rule-aware simulation, and a usable dashboard. F is an additional empirical study, not a requirement for calling the sports project successful. G is possible future scope, not hidden unfinished work in the initial deliverable.

## 11. Failure modes, disputed claims, and decision gates

| Risk / apparent contradiction | Why it matters | Decision |
|---|---|---|
| Free fixtures described as “data access” | Rich stats/rounds/odds have different entitlements [1-4] | Separate capability and license matrices |
| Open Access assumed to include demos/events | GRID explicitly excludes File Download and Series Events [5] | Confirm actual payloads; no archive promise |
| Permissive dataset card versus source restrictions | Publisher license does not erase upstream obligations [8,10] | Verify acquisition and downstream rights |
| Today's roster/veto used in past forecasts | Final/as-played evidence can be later than origin | Availability-aware replay; prospective capture where absent |
| High VRS correlation treated as calibration | Valve's README explicitly identifies calibration problems [11] | Fit and evaluate a dated probability mapping |
| Positive and negative betting theses | Different cohorts/protocols; no replicated common test [26,27] | Benchmark on one locked cohort; no expected-profit claim |
| Historical odds examples predate stated availability | OddsPapi docs are internally inconsistent [35] | Confirm a sample archive and timestamps |
| “Closing” price assumed available before start | Feed closing semantics can use scheduled fallbacks [36] | Use actual cutoff eligibility, report uncertainty |
| Map order always assumed to change BO3 winner odds | False for fixed independent probabilities | Separate winner, stopped scoreline, and state dependence |
| Generic knife-round decider hardcoded | Official Major/organizer procedures differ [14-16] | Execute event-specific side rules |
| Successful global calibration assumed to imply betting edge | Selected odds/edge cohort differs | Evaluate the frozen decision subset and realized payout |
| Low-tier inclusion justified only by sample growth | Coverage, roster certainty and outcome integrity can differ | Predeclare cohorts and report strata; avoid unsupported corruption claims |

**Decision framework:**

- Choose **sports-only research/dashboard** when lawful fixture/results access exists but betting permissions or executable odds history do not.
- Choose **map/veto-rich research** when historical action sequences and their availability are auditable; otherwise explicitly limit the initial model claims.
- Choose **player/demo-rich modeling** when licensed demo/stat coverage is sufficient and matched ablations justify its ingestion cost.
- Choose **betting shadow analysis** only after written intended-use rights and trustworthy named-market snapshots exist.
- Choose **neural/live/props extensions** only when the simpler model's limitations and the new target's data contract are demonstrated.

## 12. Verification and evidence ledger

The completed research includes primary-source inspection and four parallel literature/provider/rules tracks. It includes no empirical reproduction of published model results and no verification of a provider account or real executable price. The citation audit checks marker-to-bibliography linkage and valid non-placeholder URL structure, not legal correctness, live URL permanence, or factual truth by itself.

Selected material evidence:

| Source | Extract / quantitative anchor | Bound on interpretation |
|---|---|---|
| PandaScore pricing [3] | “only available to customers with non betting-related usage”; €400 historical / €1,000 live-basic starting prices | Advertised tier prices, not total negotiated betting cost |
| PandaScore terms [4] | “Under no circumstances” develop odds/odds-related products through Subscriptions | Ordinary subscription constraint; obtain separate written terms |
| GRID Open Access [5] | Central Data and Series State; no Series Events/File Download | Not historical CS2 completeness proof |
| OpenCS2 card [10] | “Downstream users are responsible for respecting the original terms from those tournaments.” | Not blanket production clearance |
| Valve VRS README [11] | “the slope is shallower than we'd like” | Historical self-assessment, not current calibrated probability guarantee |
| Rocha thesis abstract [20] | AUC 0.6828, accuracy 0.6293 | One selected model/holdout; not independently reproduced |
| Lund thesis [26] | Model did not surpass bookmaker benchmark on the professional tournament | Cohort-specific result with uncertainty |
| OddsPapi docs [35] | Notes specify January 2026 historical availability; examples include 2025 | Availability contradiction requires vendor/sample confirmation |
| STS help [42] | Uses 0.88 stake multiplier | Specific payout illustration, not every account's net-return contract |
| Executed BO3 arithmetic | 125 exact cases passed; 0.828 vs 0.695259 example | Mathematical correctness only, not model accuracy |

Research correction audit: an unrelated spreadsheet product's GRID evaluation-license page was discarded; no esports access conclusion is based on it. A generic knife-round decider attribution was corrected against the Major rulebook. An unrelated Valve update URL was excluded from the MR12 evidence. These corrections matter because similarly named products and mutable rule pages can create convincing but wrong research conclusions.

## 13. Executed player-rating comparison and data sufficiency

**Execution date:** 2026-10-04. **Decision:** there are enough verified observations to compute and audit exploratory player Elo, but not enough evidence to claim a LoL-equivalent player-history predictor or a predictive improvement. Both player experiments below were implemented, run on the actual downloaded records, and retained with their negative probability-scoring results. No reference-project files were changed.

### How player ratings and models work in `~/dev/inzynierka`

These are separate systems, not interchangeable names for one predictor:

1. **Legacy Elo controls** — [`src/ratings/elo.py`](../../inzynierka/src/ratings/elo.py), `RatingManager` and `scripts/05_ratingi_baseline/03_generate_ratings.py`. Player and organization ratings are independent states initialized at 1500. A player-roster forecast uses the arithmetic mean of its five player ratings and the usual base-10 Elo logistic with a 400-point scale. The legacy player update is `K × (map_result − expectation) × (roster_mean / player_rating)` for positive ratings. Class defaults are player K=32/team K=64; the baseline generator uses player K=48/team K=64. Elo updates on individual maps, not one final BO3 result. Player IDs persist when organizations change. The CS2 experiments retain roster means and per-map result updates, but use a uniform result residual rather than copying the legacy mean/player-rating multiplier.
2. **Glicko-2 controls** — [`src/ratings/glicko2_core.py`](../../inzynierka/src/ratings/glicko2_core.py) and [`FamilyCalibratedGlicko2`](../../inzynierka/src/ratings/family_calibrated_glicko2.py). Player state includes rating, deviation and volatility; inactivity increases uncertainty. The research engine freezes the prior for an entire date, collects map observations, and commits player updates simultaneously. Its roster mean is the average player rating; variance of that independent mean is the sum of player RD-squared divided by roster size squared. Family/tier offsets and affiliations are separate mechanisms, not organization Elo. The older `glicko.py` wrapper has different roster-uncertainty aggregation and update timing. None of these Glicko engines was claimed as implemented in this CS2 slice.
3. **Native research A0** — player rating representations plus masked histories of up to **16 earlier maps per player**, attention pooling, learned expert assembly and chronological calibration. The native history representation has 33 channels, including statistics, missingness, past result/recency/opponent strength and LoL role indicators. Sixteen is a maximum history window, **not** a minimum-game eligibility rule; empty histories are masked. See `src/models/prematch_features.py`, `src/models/a0_native_refit.py` and the existing native-history archive described in `docs/RESEARCH.md`. CS2 does not have LoL's five universal lane-role slots, so those features cannot simply be copied.
4. **Configured application model** — [`betting_app/core/models/registry.py`](../../inzynierka/betting_app/core/models/registry.py), lines 81–101, specifies EXP081 sports predictions with a 0.50 model-weight market logit blend. It is separate from native A0. This is local configuration evidence, not a fresh inspection of a remote running deployment.

Reference data volumes are different units: **41,915 historical series, 69,370 maps, 40,636 feature-bank targets, and 11,550 canonical assessment matches**. Sources: `docs/RESEARCH.md:2671–2678` and `reports/lol_model_improvement_research_20260927.md:275`. The ten player records in a map are not ten independent outcome labels. The inspected manifests did not establish a comparable unique-player total.

### What the downloaded CS2 records actually support

Machine-readable evidence: [`data/audits/player-rating-feasibility.json`](../data/audits/player-rating-feasibility.json). Source-specific results: [`artifacts/player_ratings/metrics.json`](../artifacts/player_ratings/metrics.json) and [`artifacts/steam_player_ratings/metrics.json`](../artifacts/steam_player_ratings/metrics.json).

| Cohort | Accepted maps | Distinct source series | Player identities | Period |
|---|---:|---:|---:|---|
| GitHub map/player joins | 1,823 | 824 | 651 exact nickname handles | 2025-12-08 to 2026-10-03 |
| OpenCS2 event-reconstructed rosters | 907 | 615 | 1,490 SteamIDs | 2026-03-23 to 2026-04-20 |

The two accepted corpora overlap on **103 exact series-ID/map-name keys**. They are evaluated separately and player identities are not merged across sources.

**Kaggle is not a historical player-lineup bank.** Its raw file contains 1,564 distinct player strings. Of 331 team names, 287 have exactly one distinct five-player set across the entire historical export; Spirit always has the set `chopper, donk, sh1ro, tn1r, zweih`. These are observed snapshot properties, not evidence that those players competed in every historical fixture. Player names and scrape-time statistics are not used to reconstruct old lineups.

**GitHub cleaning and joins:** exact duplicates are removed without changing source files. Maps are joined to player groups on exact event/date/map/team, and both sides must have five unique, non-overlapping names and an unambiguous match association. All 107 series spanning more than one source calendar date are quarantined; no timezone correction is guessed. Conflicting map representations, ambiguous roster joins and the six-player group are excluded. The ledger has **304 excluded map keys**; exclusion reasons overlap. No stable person/SteamID identity is asserted for nicknames.

**OpenCS2 now has a verified outcome route that does not need organization attribution.** The original Parquet, unlike the flattened player CSV, contains two roster groups plus round and kill events. For each round, participant SteamIDs and their recorded current CT/T sides identify which roster occupied the winning side. Repeating this across all rounds handles side switches without assuming that starting CT/T names an organization or that aggregate CT/T wins are organization scores. Where one roster's current side is observed, the opposing roster has the other side; no missing round is filled from a guessed halftime schedule.

For example, map `2393397-m1` reconstructs **roster-group scores 4–13**, while its aggregate CT/T totals are **3–14**. Directly reading 3–14 as a team score would be wrong.

All 137 pinned metadata shards were checksum/size-verified before parsing. Each accepted map requires two disjoint five-ID rosters, per-round side/winner evidence, and a standard complete MR12 terminal score. The **1,081 rejected maps** comprise:

- 832 with an event participant/side outside the declared roster contract;
- 235 without two disjoint five-ID groups;
- 11 missing per-round winner/side evidence;
- 3 with nonstandard/incomplete reconstructed terminal scores.

These reasons establish failed data contracts, not explanations such as observers or warmup that were not independently verified. Accepted labels and rosters are saved in `artifacts/steam_player_ratings/reconstructed_labels.csv`. Named organizations remain unmapped; the comparator is **exact-lineup Elo**, not organization Elo.

### Do individual players have enough history?

Counts below are **accepted maps per player at corpus end**, not prematch support:

| History measure | Nickname cohort | SteamID cohort |
|---|---:|---:|
| Median maps per identity | 6 | 5 |
| At least 5 maps | 385 | 797 |
| At least 10 maps | 259 | 278 |
| At least 20 maps | 203 | 25 |
| At least 50 maps | 143 | 0 |

The raw OpenCS2 `all` scope has 20,817 rows but only 20,287 distinct player-map memberships across 1,691 SteamIDs. Those raw exposures are not all eligible outcome-rating observations; the event-backed five-player cohort is substantially smaller.

Strictly earlier-date support at each **test forecast**:

| All ten players have this much earlier map history | Nickname test, n=644 | SteamID test, n=158 |
|---|---:|---:|
| At least 1 map each | 568 | 99 |
| At least 5 maps each | 464 | 38 |
| At least 10 maps each | 376 | 8 |
| At least 20 maps each | 257 | 0 |

There are 76 nickname-test maps and 59 SteamID-test maps with at least one cold player. Observing a player under multiple team-name strings is not independent proof of an actual transfer; 105 nickname handles have that property. In the SteamID cohort, 267 IDs occur in multiple exact-lineup fingerprints, which is lineup variation rather than organization-transfer evidence.

**Sufficiency decision:** outcome-based Elo is runnable; reliable individual-impact ranking and a comparable learned player-history model are not established. Stable-ID history covers only four weeks, with sparse per-player support and a 158-map final comparison. A0's masked architecture can technically accept short histories, but that does not supply enough independent training/evaluation evidence. No universal minimum-game count or formal neural sample-size guarantee is claimed.

### Executed player Elo and matched controls

Implementation: [`pipeline/ratings/player_elo.py`](../pipeline/ratings/player_elo.py), [`pipeline/ratings/opencs2.py`](../pipeline/ratings/opencs2.py). Shared probability metrics reuse `pipeline/models/metrics.py`.

Player ratings start at 1500. A roster's strength is its arithmetic mean. Each accepted map gives every player on a side the same `K × (binary_map_result − roster_expectation)` increment. K is **per map, not per round**; 13–4 is converted to a binary map winner, not a 13/17 fractional skill target. All predictions for one source date precede all updates for that date. Player state follows its identity across different rosters.

K candidates are frozen at `0, 8, 16, 24, 32, 48, 64`; player and comparison-entity K are selected independently by validation log loss. No test-dependent retuning or calibrator was added after the negative results. Ratings continue updating after each test date using the frozen recipe.

| Cohort | Train maps | Validation maps | Test maps | Selected player/control K |
|---|---:|---:|---:|---|
| Nicknames: boundaries 2026-04-01 / 2026-07-01 | 585 | 594 | 644 | 48 / 48 |
| SteamIDs: boundaries 2026-04-01 / 2026-04-11 | 317 | 432 | 158 | 24 / 32 |

Each row of the following table is scored on the same map IDs as its controls. The two cohorts and the older Kaggle **series-winner** test are not directly comparable.

| Cohort / model | Test log loss ↓ | Brier ↓ | Accuracy ↑ | ROC AUC ↑ |
|---|---:|---:|---:|---:|
| Nickname player Elo | 0.69618 | 0.24847 | 57.45% | 0.60328 |
| Nickname organization-name Elo | 0.69766 | 0.24838 | 58.85% | 0.60954 |
| Nickname constant training prior | **0.68638** | **0.24662** | 55.90% | 0.50000 |
| Nickname constant 0.5 | 0.69315 | 0.25000 | 55.90% | 0.50000 |
| SteamID player Elo | 0.70104 | 0.25392 | 50.00% | 0.47169 |
| SteamID exact-lineup Elo | 0.70255 | 0.25461 | 46.84% | 0.46670 |
| SteamID constant training prior | 0.69428 | 0.25055 | 53.16% | 0.50000 |
| SteamID constant 0.5 | **0.69315** | **0.25000** | 46.84% | 0.50000 |

Accuracy uses `p >= 0.5` for side one; constant 0.5 is a probability control, not simulated random guessing. Player Elo slightly improves log loss over its entity control in each cohort but **does not beat the simple probability controls**. It is not adopted as a demonstrated forecasting improvement. These are point estimates, not significance or profitability claims; this comparison does not prove sparse data is the sole cause of the negative result.

Rankings are outcome-based roster ratings, **not isolated individual contribution**. Players with identical histories and initial values receive identical updates. The exercised SteamID top-five ranking with a 20-map display minimum consists of five players from the same unchanged lineup at the same rating; presenting that as five independently estimated individual skills would be misleading.

### Reproduce and inspect

The existing project `.venv` was used: Python 3.12 and PyArrow 25.0.1. The plain-system full suite initially could not import the separately added pipeline's `joblib`; using the existing project environment resolved that dependency, without installing new packages or changing that pipeline.

```bash
.venv/bin/python -m pipeline.ratings.player_elo backtest
.venv/bin/python -m pipeline.ratings.opencs2 backtest

.venv/bin/python -m pipeline.ratings.player_elo rankings --min-maps 20 --limit 20
.venv/bin/python -m pipeline.ratings.player_elo rankings \
  --model artifacts/steam_player_ratings/model.json --min-maps 20 --limit 20

.venv/bin/python -m unittest discover -s tests -v
```

`predict` accepts two explicit five-player rosters using exact saved nicknames or SteamID identifiers and `--model PATH`. It returns a **map-winner historical roster scenario**, not a BO3 winner, live forecast or proof the lineup was announced at a historical prediction time.

Each source writes `metrics.json`, `model.json`, `predictions.csv`, `rankings.csv` and `excluded_maps.csv`; SteamID execution additionally writes `reconstructed_labels.csv`. Stored rankings include map/series counts, last-played dates and identity namespace. A display minimum filters the output only, not the training/evaluation population.

**Verification:** 47 tests passed, including the existing pipeline suite and new behavioral guards for transfers, same-date predictions, ambiguous joins, roster overlap, side switches, missing round evidence and outcome reconstruction. Both actual-data backtest CLIs, both ranking paths, and both explicit-roster forecast/reversal/unknown-player paths ran. Second executions reproduced all five nickname artifacts and all six SteamID artifacts byte-for-byte. Temporary replay directories were removed automatically.

**Known limits:** no verified nickname-to-SteamID cross-source registry, complete circuit coverage, roster announcement clocks, exact result-publication clocks, person-level Glicko uncertainty, performance-aware rating, neural CS2 history training, veto-aware series projection, market comparison or profitability result. These were not silently filled with current rosters, fake labels or guessed organization mappings.

## Bibliography

Dates below identify publication/version where established; otherwise documents are mutable and were inspected during this research. Freeze source captures/commits before an actual historical study. Vendor documentation is evidence of advertised capability and contractual wording, not independent delivery validation.

| ID | Title / document | Author / organization; date | URL |
|---|---|---|---|
| [1] | API introduction and data products | PandaScore; mutable documentation | https://developers.pandascore.co/docs/introduction |
| [2] | Plan reference and endpoint entitlement | PandaScore; mutable documentation | https://developers.pandascore.co/docs/plan-reference |
| [3] | Pricing | PandaScore; mutable pricing page | https://www.pandascore.co/pricing |
| [4] | Terms and conditions, Article 2.8 | PandaScore; mutable terms | https://www.pandascore.co/terms-and-condition |
| [5] | What APIs does an Open Access developer have access to? | GRID; mutable help article | https://grid.helpjuice.com/client-help/what-apis-does-an-open-access-developer-have-access-to |
| [6] | Historical data scope | GRID; mutable help article | https://grid.helpjuice.com/en_US/client-help/what-historical-data-does-grid-offer-and-how-far-back-does-it-go |
| [7] | API terms of use | Liquipedia; mutable policy | https://liquipedia.net/api-terms-of-use |
| [8] | Terms of use | HLTV; effective 2025-01-27 | https://www.hltv.org/terms |
| [9] | Parsing a Counter-Strike 2 demo | Awpy; mutable documentation | https://awpy.readthedocs.io/en/latest/examples/parse_demo.html |
| [10] | OpenCS2 dataset card | blanchon; mutable dataset README | https://huggingface.co/datasets/blanchon/opencs2_dataset_demo/blob/main/README.md |
| [11] | Counter-Strike Regional Standings explanation | Valve; historical explanatory README, mutable repository | https://raw.githubusercontent.com/ValveSoftware/counter-strike_regional_standings/main/readme.md |
| [12] | Counter-Strike Regional Standings repository and dated reports | Valve; ongoing releases | https://github.com/ValveSoftware/counter-strike_regional_standings |
| [13] | Tournament Operation Requirements | Valve; mutable main revision | https://raw.githubusercontent.com/ValveSoftware/counter-strike_rules_and_regs/main/tournament-operation-requirements.md |
| [14] | Major Supplemental Rulebook | Valve; mutable main revision | https://raw.githubusercontent.com/ValveSoftware/counter-strike_rules_and_regs/main/major-supplemental-rulebook.md |
| [15] | Counter-Strike 2026 rulebook | Esports World Cup; inspected July 2026 update | https://cdn.esportsworldcup.com/resources/uploads/EWC_CS_2026_Rulebook_fe6e24ed0b.pdf |
| [16] | BLAST Premier Handbook 2026 | BLAST; 2026 | https://assets.blast.tv/rulebook/BLAST_Premier_Handbook_2026.pdf |
| [17] | ESL Pro Tour Counter-Strike Rulebook 2026 Update #2 changelog | ESL; 2026-07-07 | https://pro.eslgaming.com/tour/2026/07/esl-pro-tour-counter-strike-rulebook-2026-update-2-changelog/ |
| [18] | Cache active-duty map-pool update | Valve; 2026 update | https://www.counter-strike.net/newsentry/701021228894257508 |
| [19] | CS:GO match-prediction master's thesis | Zachary Schmidt; 2020 | https://theses.liacs.nl/pdf/2019-2020-SchmidtZachary.pdf |
| [20] | Professional CS2 prematch prediction undergraduate thesis, repository abstract | Gabriel Herdy Rocha, UFSCar; 2026-07-01 | https://repositorio.ufscar.br/items/05d81da4-11fa-4a99-a98c-2935fbead599 |
| [21] | Bandit Modeling of Map Selection in Counter-Strike: Global Offensive | Petri et al.; 2021 | https://arxiv.org/abs/2106.08888 |
| [22] | Regularized player plus/minus analysis in professional CS:GO | Xu and Moka; 2024 | https://arxiv.org/pdf/2409.05052 |
| [23] | Individual and Collective Performance Deteriorate in a New Team | Zhang, Muric and Ferrara; 2022 | https://arxiv.org/abs/2205.09693 |
| [24] | Valuing Player Actions in Counter-Strike: Global Offensive | Xenopoulos et al.; 2020 | https://arxiv.org/pdf/2011.01324 |
| [25] | Optimal Team Economic Decisions in Counter-Strike | Research paper; 2021 | https://arxiv.org/abs/2109.12990 |
| [26] | CS:GO prediction and bookmaker comparison thesis | Broms and Nordansjö, Lund University; 2024 | https://lup.lub.lu.se/student-papers/record/9145457 |
| [27] | CS:GO machine-learning prediction and betting thesis | du Plessis, University of Cape Town; 2024 | https://open.uct.ac.za/items/26d92119-6b96-4af2-8408-232cb8f62200 |
| [28] | Strictly Proper Scoring Rules, Prediction, and Estimation | Gneiting and Raftery; 2007 | https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf |
| [29] | On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation | Cawley and Talbot; 2010 | https://www.jmlr.org/papers/v11/cawley10a.html |
| [30] | Evaluating Model Calibration in Classification | Vaicenavicius et al.; 2019 | https://proceedings.mlr.press/v89/vaicenavicius19a.html |
| [31] | On Calibration of Modern Neural Networks | Guo et al.; 2017 | https://proceedings.mlr.press/v70/guo17a.html |
| [32] | Example of the Glicko-2 system | Mark Glickman; 2022 document | https://www.glicko.net/glicko/glicko2.pdf |
| [33] | TrueSkill 2: An improved Bayesian skill rating system | Tom Minka, Ryan Cleven and Yordan Zaykov; 2018 | https://www.microsoft.com/en-us/research/wp-content/uploads/2018/03/trueskill2.pdf |
| [34] | Why do tree-based models still outperform deep learning on typical tabular data? | Grinsztajn, Oyallon and Varoquaux; 2022 | https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html |
| [35] | Get historical odds | OddsPapi; mutable documentation | https://oddspapi.io/us/docs/get-historical-odds |
| [36] | Odds movement concepts | EsportsOdds; mutable documentation | https://docs.esportsodds.gg/docs/concepts/odds-movement |
| [37] | Esports odds product | PandaScore; mutable product page | https://www.pandascore.co/odds |
| [38] | Betting solutions | GRID; mutable product page | https://www.grid.gg/solutions/betting/ |
| [39] | Esports sports rules | bet365, North Carolina help site; jurisdiction-specific, mutable | https://help.nc.bet365.com/s/en-us/sportsrules/esports |
| [40] | Anti-Corruption Code | Esports Integrity Commission; mutable code | https://esic.gg/codes/anti-corruption-code/ |
| [41] | Gaming-tax rates and limits | Polish Ministry of Finance; 2025-12-15 page | https://podatki.gov.pl/pozostale/podatek-od-gier/stawki-i-limity |
| [42] | Jakie podatki są pobierane? | STS; article updated 2024-06-25 | https://pomoc.sts.pl/jakie-podatki-sa-pobierane/ |
