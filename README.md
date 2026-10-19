# cs-breaker

CS2 forecasting research using downloaded public results. Implemented rating
experiments include organization-name series Elo and two source-isolated,
player-roster map-winner Elo studies. They compute historical strength from
earlier source-date results, not scrape-time statistics or market odds.

## Local odds and historical-match application

`cs_app/` is a runnable research workspace, separate from the forecasting
experiments below. Open **http://127.0.0.1:8017/** for fixtures, historical
matches, full Winner timelines, mapping review, durable jobs, settings, and
research metrics/CSV exports.

```bash
# Reuse the existing Python 3.12 .venv, or create one on a fresh checkout.
.venv/bin/pip install -r requirements-app.txt
.venv/bin/python -m cs_app init --seed-existing
.venv/bin/python -m cs_app serve --env-file .env
```

The prepared local database has already been seeded. `init --seed-existing` is
for a fresh database or explicit re-import; it needs no provider credential or
network access. Without the downloaded archives, use `init`, import supported
historical CSVs, and explicitly queue an odds backfill instead.
Credentials come from `ODDSPAPI_API_KEY` in the process environment, or an
explicit `--env-file`; the application does not implicitly load `.env`.
The server includes one serialized worker. Opening the browser does not submit
jobs. Previously authorized queued jobs survive a restart.

### Collection and research workflow

1. **Import histories:** import OpenCS2 `maps.csv`, Kaggle `results_only.csv`, or
   StrandedPond `matches.csv` from the Historical matches screen. Original CSV
   evidence is retained; conflicting existing source IDs reject the entire
   batch. The web upload limit is **16 MiB**.
2. **Backfill historical odds:** give aware UTC bounds in Collection jobs.
   Discovery is split into at-most-seven-day windows. Native IDs and bookmaker
   scope are frozen; successful histories and exact explicit no-history
   responses are reused. Failed requests remain evidence, not absence.
3. **Collect upcoming odds:** explicitly queue current pre-match snapshots.
   Later missing, suspended, or unusable observations supersede earlier
   availability. Current source clocks are `changedAt`; local receipt clocks
   are recorded separately. Other markets remain in the full raw envelopes.
4. **Refresh finished matches:** the finished job discovers recent completed
   fixtures and collects their histories. The results job acquires explicit
   native `scores.periods.result` series scores, prioritizing priced and newly
   finished fixtures. Default result jobs fill missing labels; the explicit
   existing-score refresh checkbox/CLI flag also checks for corrections.
   Strictly newer native result clocks permit audited revisions; map scores
   and numbered periods are not guessed into series labels.
5. **Review identities:** exact native fixture IDs take precedence. OpenCS2
   automatic links reuse the conservative team/event/time matcher described
   below. Manual accept/reject/clear actions require reasons and preserve an
   audit trail, with explicit same/reversed team orientation and no duplicate
   assignment of one historical result to two active fixtures.
6. **Export for later research:** `fixtures`, `history`, `quotes`, `states`,
   `mappings`, `jobs`, and oriented `mapped_quotes` CSVs retain stable IDs and
   provenance references. Large state exports stream from SQLite rather than
   loading the whole history into memory. Metrics distinguish available,
   pairwise-common, and all-book-common cohorts; missing labels are not zeros.

CLI equivalents:

```bash
.venv/bin/python -m cs_app collect backfill --env-file .env \
  --from 2026-10-01T00:00:00Z --to 2026-10-06T00:00:00Z
.venv/bin/python -m cs_app collect upcoming --env-file .env --max-fixtures 1
.venv/bin/python -m cs_app collect finished --env-file .env
.venv/bin/python -m cs_app collect results --env-file .env
.venv/bin/python -m cs_app collect results --env-file .env --refresh
.venv/bin/python -m cs_app import-history --source kaggle \
  --file data/exports/kaggle/results_only.csv
.venv/bin/python -m cs_app remap
.venv/bin/python -m cs_app export mapped_quotes --output /tmp/cs2-research.csv
.venv/bin/python -m cs_app job JOB_ID --retry --env-file .env
.venv/bin/python -m cs_app status
```

When the web worker is running, CLI submissions are queued for that worker.
Otherwise the command executes the queue itself. `worker --env-file .env`
runs a headless worker; `worker --once` processes one queued job. Retries are
explicit, retain progress, and do not discard the frozen scope. CSV exports
refuse to overwrite an existing target.

### Budgets, persistence and deployment

Scheduling is **off by default**. Enabling it explicitly authorizes future
requests, cycling upcoming/finished/results jobs at the saved interval
(default **360 minutes**, one kind per interval). Defaults: **4 billable
attempts/run**, **8/day**, **100 HTTP attempts/run**, **100 history calls/run**,
**20 fixtures/run**, and at least **60 minutes** between current snapshots.
Upcoming discovery spans seven days; finished discovery spans two.
Budgets pause jobs rather than silently dropping pending fixtures. Failed paid
attempts count conservatively. Each network run checks a fresh sanitized
account snapshot; the provider key is shared with other projects, so local
caps cannot reserve external account quota. Auth, quota, rate and secret-echo
stops do not trigger automatic retries.

Owned storage is **`data/app/`**, ignored by Git: SQLite WAL, immutable gzip
raw evidence and hashes, states, historical results/revisions, mappings,
settings and durable job progress. Existing archives, manifests, model inputs,
and the reference `inzynierka` project are not modified. Stop all workers
before copying the entire directory for a complete database-and-evidence backup.
Use `--data-dir` for a separate workspace.

```bash
# Alternative to the native server; both use port 8017.
docker compose -f compose.app.yaml up -d --build
docker compose -f compose.app.yaml exec workspace \
  python -m cs_app init --seed-existing
```

Docker uses a named writable application volume and read-only source mounts;
the image excludes credentials and data. Compose passes the key from the
environment/local `.env` without copying it into the image. Published access
is loopback-only. There is no remote-user authentication: do not expose this
server publicly. Mutations require the same-origin application header, and
foreign Host/Origin requests are rejected.

The initial seed imported **5,449 fixtures**, **8,833 historical records**
(including nine native results), **2,587,677 Winner states**, and **16,347
quote/availability observations**, with **346 active links** and **308 scored
quote pairs**, using **zero network calls** and preserving source manifests.
Expanding historical series coverage with the modern 2023–2026 Kaggle dataset
(`ektarr/counter-strike-pro-matches`, 11,054 series rows) brought the retained
historical match archive to **19,920 records**, **1,536 mapped fixtures**, and
**1,631 scored quote pairs** across **1,032 distinct matches with odds**. Live smoke
and bounded results jobs acquired native completed-series results using **eleven
billable attempts total** within daily quota headroom.
The Docker image, offline API startup and named-volume persistence across a
container restart were exercised.

Verification: **149 project tests passed**. Browser smoke exercised fixture
search/details, inactive-state timelines, audited mapping exclusion/restoration,
CSV import, settings, cost-consent cancellation and cohort metrics. The full
**2,587,679-state / 524 MB** CSV export streamed in approximately **28 MB**
sampled resident memory, with every row counted; source gzip evidence remains
separate from these normalized states.
SQLite integrity and foreign-key checks were clean; all **3,498 retained raw
bodies** passed encoded-file and decoded-body SHA-256 verification. The current
provider credential was absent from the persisted database and decoded raw
evidence; account receipts retained neither body files nor body digests.


## Run

The series-only commands use Python 3.12's standard library. SteamID parsing
uses PyArrow 25.0.1 in the existing project `.venv`; the full suite also covers
the separate pipeline and its installed dependencies. No credentials are needed.
Run commands from the repository root:

```bash
python3 -m cs_breaker backtest
python3 -m cs_breaker predict Spirit Vitality
.venv/bin/python -m unittest discover -s tests -v
```

Backtesting reads `data/exports/kaggle/results_only.csv` and writes:

| Artifact | Contents |
|---|---|
| `artifacts/baseline/metrics.json` | Source checksum, frozen splits, candidate selection, aligned model comparisons, calibration bins, monthly and cold-start results |
| `artifacts/baseline/predictions.csv` | Every eligible match's pre-update probabilities, target label, cohort, prior match counts, and historical date cutoff |
| `artifacts/baseline/excluded.csv` | Original rows excluded for unsupported format/score combinations, with reasons |
| `artifacts/baseline/model.json` | Selected K, historical team ratings, match counts, last-played dates, and source checksum |

Alternate input/output paths and chronological boundaries:

```bash
python3 -m cs_breaker backtest \
  --data data/exports/kaggle/results_only.csv \
  --output artifacts/baseline \
  --train-end 2025-01-01 \
  --validation-end 2025-07-01
```

`predict` also accepts `--model PATH`. Exact stored team names take precedence;
otherwise only an unambiguous case-insensitive/Unicode-normalized lookup is
accepted. Unknown teams, ambiguous names, and self-matchups fail rather than
silently receiving invented ratings.

## Data eligibility

The source has **7,033 unique match IDs**, all labelled `bo3`. Inspection found
**1,126 scores incompatible with a completed ordinary BO3**, including round-like
13–6 results, 3–2 results, and 1–0 results. The format label alone is unreliable.

The model uses **5,907 BO3-score-compatible matches**: 2–0, 2–1, 0–2, or 1–2.
These are score-compatible records, not independently verified BO3 identities.
Excluded records remain in the ledger and raw exports; their actual formats are
not guessed. No source file is modified by a backtest.

Input uses HLTV match ID, UTC date, literal team names, and winner. Scores are
used only for eligibility and winner consistency, never as prediction features.
Current snapshot statistics and outcome-oriented `winner_*`/`loser_*` metrics
are deliberately not loaded.

## Evaluation protocol

| Cohort | UTC-date interval | Matches |
|---|---|---:|
| Training/history | Before 2025-01-01 | 1,796 |
| Validation/model selection | 2025-01-01 to before 2025-07-01 | 1,873 |
| Final test | 2025-07-01 through 2025-10-16 | 2,238 |

- New teams start at 1500. Probability is the logistic Elo difference with a
  400-point base-10 scale, directly targeting the series winner.
- All matches assigned the same UTC date are predicted before applying any
  outcomes from that date. Rating changes are summed from the shared pre-day
  state, including when a team plays multiple matches.
- K candidates are fixed at `0, 8, 16, 24, 32, 48, 64`. Minimum validation log
  loss selects K, with smaller K breaking exact ties. Test outcomes do not select
  the recipe.
- Ratings continue updating after each test date. This is walk-forward
  evaluation, not a rating snapshot frozen for the entire test period.
- Both controls use exactly the same validation/test match IDs: constant 0.5,
  and the constant team1 win frequency estimated from training only.

## Observed test results

Validation selected **K=32**. On the 2,238-match test cohort:

| Model | Log loss ↓ | Brier ↓ | Accuracy ↑ | ROC AUC ↑ |
|---|---:|---:|---:|---:|
| Elo | **0.6660** | **0.2368** | **59.79%** | **0.6253** |
| Constant training-frequency prior | 0.6881 | 0.2475 | 55.23% | 0.5000 |
| Constant 0.5 | 0.6931 | 0.2500 | 55.23% | 0.5000 |

Accuracy predicts team1 when probability is exactly 0.5; the coin-flip
probability control therefore has the always-team1 accuracy, not a simulated
random-guess accuracy. Log loss and Brier are the main probability comparisons.
Calibration bins and ten-bin ECE are saved; no calibrator was fitted and no
statistical-significance or profitability claim is made.

The CLI example returned Spirit **35.73%**, Vitality **64.27%**, using history
through **2025-10-16**. These are historical snapshot probabilities, not today's
matchup forecast. Ratings represent **315 literal team-name identities**.

Verification: 13 regression tests passed. The actual downloaded-data CLI ran;
a second run reproduced all four artifacts byte-for-byte. Reversed matchup
orientation produced complementary probabilities; an unknown team exited with
a clear error.

## Player-level ratings

```bash
.venv/bin/python -m pipeline.ratings.player_elo backtest
.venv/bin/python -m pipeline.ratings.opencs2 backtest

.venv/bin/python -m pipeline.ratings.player_elo rankings --min-maps 20 --limit 20
.venv/bin/python -m pipeline.ratings.player_elo rankings \
  --model artifacts/steam_player_ratings/model.json --min-maps 20 --limit 20
```

| Source-specific player cohort | Accepted maps | Series | Identities | Median maps/player |
|---|---:|---:|---:|---:|
| Exact GitHub nickname joins | 1,823 | 824 | 651 handles | 6 |
| Event-reconstructed OpenCS2 | 907 | 615 | 1,490 SteamIDs | 5 |

The sources are not merged. SteamID map winners are reconstructed using actual
round winners and kill-event identity/current-side evidence, including side
switches; aggregate CT/T totals are not treated as organization scores.
Unlisted participants, missing evidence and incompatible rosters are rejected.
Nickname dates have unknown timezone; cross-date series are quarantined.

Player strength is the mean of the five player ratings. Every map updates its
participants after all forecasts for that source date. IDs carry state across
rosters; no LoL lane slots or current snapshot lineups are imported.

| Final test cohort | Player Elo log loss | Matched entity Elo | Simple probability control |
|---|---:|---:|---:|
| Nicknames, 644 maps | 0.69618 | Organization: 0.69766 | Training prior: **0.68638** |
| SteamIDs, 158 maps | 0.70104 | Exact lineup: 0.70255 | Constant 0.5: **0.69315** |

Neither experiment establishes a predictive improvement over the simple
probability controls. These map results are not comparable with the older
series-winner table. Rankings measure roster win/loss history, not independently
identified individual impact: players with identical histories can tie.

Only **25 SteamIDs have 20 accepted maps**. Just **8/158 SteamID test maps**
have all ten players with at least ten earlier accepted maps. The data supports
exploratory rating replay, not a demonstrated LoL-equivalent neural/history model.

Each run saves model, predictions, rankings, metrics and exclusions under
`artifacts/player_ratings/` or `artifacts/steam_player_ratings/`; the latter also
includes reconstructed map labels. `predict --roster1 P1 P2 P3 P4 P5 --roster2
P6 P7 P8 P9 P10 --model PATH` accepts exact source identities and returns a
historical **map-winner roster scenario**, not a current forecast or BO3 result.

Full implementation/data comparison with `~/dev/inzynierka`:
[research §13](reports/cs2-project-research.md#13-executed-player-rating-comparison-and-data-sufficiency).
Machine-readable coverage: [player-rating audit](data/audits/player-rating-feasibility.json).
Verification: **47 tests passed** in the project environment; both real backtests,
rankings and roster forecast paths ran. Both sources reproduced their artifacts
byte-for-byte on second execution.

## Audited Kedro player-series model (2026-10-09)

Earlier player-cohort research: [synthesis](reports/cs2-model-research-2026-10-09.md),
[source ledger](reports/cs2-model-research-sources-2026-10-09.json), and
[complete observed results / reproduction](reports/cs2-model-results-2026-10-09.md).
The previous 7,038-row player benchmark was invalidated by scrape-time squads,
same-day chronology, rating-state and TEST-selection defects; it is not a valid
before/after comparator. The preserved corrected incumbent is `pure_elo`.
Parent-executed `artifacts/model_research_20261009/verification.json` also records
validity of all 11 input hashes, 38 source hashes and seven archived output hashes,
and exact preservation of the six original candidates' validation/TEST metrics
and all 164 per-candidate probabilities. Installed exercised versions are pinned.

### Audited cohort and model

The 122,038-ID master supplies **675 eligible lineup-conditional series**:
670 historical map-derived lineups and five match-page captures; **621 observed
player identities**. Native player IDs and source-scoped historical nicknames
are not merged. Kaggle scrape-time squads/stat snapshots are excluded. Whole
source-calendar days are frozen before all same-day outcomes/measurements;
earlier evaluation days update later rating state, but estimators remain
TRAIN-fitted. Source timezone, prematch lineup availability and substitutions
are not independently certified: these are retrospective actual-lineup scenarios.

| Partition | Dates | Series |
|---|---|---:|
| TRAIN | 2025-12-08–2026-06-07 | 396 |
| VALIDATION | 2026-06-09–2026-08-13 | 115 |
| TEST | 2026-08-14–2026-10-07 | 164 |

All 675 IDs/dates/orientations/labels and **31 original predictors** are unchanged.
Seven archive features give **38 predictors**. Past-only archive history adds
23,568 complete modern-proxy BO3/BO5 results over 1,768 literal team names;
it does not add target labels or certify organization aliases/CS2 versions.
Minimum archive source date is 2024-01-01. Elo has K=32 and a 400-point scale;
the alternative archive decay has fixed 180-day half-life. Player performance
features use preceding-day measurements with explicit missing-support flags.

**VALIDATION selected `blend_player_archive_elo`** among 16 predeclared candidates
(two controls separate): exactly half player Elo and half no-decay archive Elo,
not learned weights and not odds-informed. Selection minimizes validation log
loss; no TRAIN+VALIDATION refit. Three expanding TRAIN-only folds are diagnostic.
Generic `pipeline.cli train` / `run` instead use a separate **TEST-selected
experimental path** and are not audited modeling reproduction. Standalone
organization/player-map studies above and application models remain unchanged.

| Model | VALIDATION LL (115 series) | TEST LL (164 series) | TEST Brier | TEST accuracy | TEST AUC |
|---|---:|---:|---:|---:|---:|
| Before: player Elo | 0.684419 | 0.644669 | 0.226488 | 60.98% | 0.676777 |
| After: selected fixed blend | 0.673296 | 0.640090 | 0.224272 | 64.02% | 0.681219 |

Paired after-minus-before log loss is **−0.004580**, 95% source-day bootstrap
interval **(−0.021467, +0.012746)**; Brier −0.002216, interval
(−0.010352, +0.006043). Both include zero (5,000 draws, seed 42).
**Not a demonstrated statistical improvement or fresh TEST confirmation:** TEST
was already observed and validation has only 115 cases. **ECE worsens
0.087954→0.102596**, and the last 46-series chronological TEST block worsens
LL **0.644277→0.665819** and Brier **0.227674→0.237983**.
All 16 candidate scores, controls, folds and descriptive blocks are in the results report.

### Offline bookmaker comparisons

Comparison uses a read-only cached SQLite snapshot, native HLTV/result/orientation
checks, latest stored observations when valid and pre-start, and **zero provider requests**.
It does not change app mappings, refit/reselect the model or fall back after an
invalid latest observation. Current coverage: 5,465 fixtures, 3,467 native links,
3,058 linked played BO3/BO5 results; 97 TEST matches linked, **92 with usable
quotes, 203 quote pairs and 33 common to all three books**. The remaining
1,998 fixtures retain unmatched/ambiguous/conflicting/not-finished statuses.

Before/after primary quote IDs, raw prices, clocks, orientation, labels and book
probabilities are identical (`markets/validation.json`). **The new model regresses
on the bookmaker-common cohort despite its overall TEST gain.**

| Same 33 TEST matches, proportional pre-start | Log loss | Brier | Accuracy | AUC |
|---|---:|---:|---:|---:|
| Previous player Elo | 0.641430 | 0.224437 | 66.67% | 0.681481 |
| Selected blend | 0.664955 | 0.235420 | 63.64% | 0.640741 |
| pinnacle | 0.625063 | 0.214726 | 63.64% | 0.744444 |
| bet365 | 0.629480 | 0.215742 | 66.67% | 0.740741 |
| unibet | 0.607781 | 0.208049 | 63.64% | 0.772222 |

| Per-book matched cohort | N | Selected-model LL | Book LL | Model − book LL (95% source-day interval) |
|---|---:|---:|---:|---|
| pinnacle | 67 | 0.636018 | 0.618459 | +0.017558 (-0.052805, +0.081612) |
| bet365 | 74 | 0.624941 | 0.610523 | +0.014418 (-0.037696, +0.067997) |
| unibet | 62 | 0.683737 | 0.656401 | +0.027336 (-0.054237, +0.124253) |

All primary paired intervals include zero: **no demonstrated bookmaker edge**.
Different per-book populations must not rank books against each other; the
33-game intersection is also too small for confident ranking. Power de-vig uses
a price-only exponent solved separately for each quote and preserves cohorts;
it does not establish an edge. Nominal source-day-midnight sensitivity leaves
13 games (Pinnacle/Bet365/Unibet 1/0/12), no common intersection, and sparse or
degenerate intervals. It filters latest stored observations, not reconstructed
historical snapshots. Day-frozen features and latest pre-start odds do not share
certified information cutoffs. No closing-price, executable-price, ROI or CLV
claim, automatic bets, market blend or provider acquisition was made.

### Rebuild, evidence and explicit-roster inference

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

The Legacy/M80 keys above reproduce a **historical supplied-roster scenario**,
not verified current starters or a live prediction. Saved history ends
2026-10-07; the query day must be strictly later than both player/archive history.
Observed scenario probabilities: Legacy 57.3871%, M80 42.6129%, with minimum one
prior player match and no measured-performance availability. **No automatic
rosters:** supply explicit disjoint five-player identity lists and team names.

Load **trusted local pickles only**. Model bundle **version 4** and rating state
**schema 2** are a clean cutover: obsolete versions are rejected; rebuild with
Kedro rather than relying on compatibility shims. The canonical outputs are
`data/processed/match_features_with_ratings.csv`,
`data/processed/player_rankings_master.csv`, `data/audits/modeling_data_audit.json`,
`artifacts/baseline_model_metrics.json`, `artifacts/baseline_test_predictions.csv`,
`artifacts/baseline_model.pkl` and `artifacts/player_rating_state.pkl`.
The predictor accepts both saved pickle artifacts.

Frozen experiment evidence lives in `artifacts/model_research_20261009/`:
`before/`, `after/summary.json`, `after/metrics.json`, `after/manifest.json`,
four `markets/` mode directories, `markets/validation.json`, `environment.json`
and `inference_smoke.json`. The SHA256 manifest links exact input/code/output
bytes; the evaluator checks declared recipe provenance but opaque pickle bytes
alone do not prove training. Independent runtime execution does: actual feature
and training nodes, evaluator, all four comparison commands, saved/API/CLI
inference. All 164 persisted forecasts reproduce to maximum absolute error
8.326672684688674e-17; reverse probabilities complement and queries leave state
and bytes unchanged. Full implementation gate: **398 tests + 218 subtests passed**.
No LSP/lint/type-check claim. See the results report for exact evidence and limits.


## Opt-in OddsPapi CS2 history probe

`scripts/collect_cs_oddspapi.py` is a separate standard-library v4 collector.
It does not touch forecasting, canonical match joins, the pipeline CLI, application
databases, schedulers, baseline artifacts, or `data/source-manifest.json`.

The collector reads `ODDSPAPI_API_KEY` from its **process environment only**; it
does not automatically load `.env`. At the user's request, both project-local
`.env` files now hold the credential, are Git-ignored, and have mode `0600`.
Keep these files private; do not put the key in source, arguments, or logs.
Rotate any key disclosed in chat. The account gate
requires active sport-17 access, at least one requested bookmaker, and quota
remaining strictly greater than uncached billable discovery calls. A fresh probe
needs at least three credits for its two billable calls. History is documented
as free but still requires quota headroom; shared-account consumption is not
attributed to this collector.

```bash
set -a; . ./.env; set +a  # Explicitly export the private local credential.
.venv/bin/python scripts/collect_cs_oddspapi.py \
  --from 2026-04-14T00:00:00Z --to 2026-04-20T23:59:59Z \
  --bookmakers pinnacle,bet365,unibet --max-history 10 \
  --max-billable-requests 2 --max-requests 14 \
  --output-dir data/raw/oddspapi/cs2_20260414_20260420_probe
```

The single requested interval must be timezone-aware, positive, and shorter than
ten days. Finished fixtures are frozen by `(startTime, fixtureId)` before history
requests; unavailable books remain `not_entitled`, never substituted.
Provider-supported domain-style slugs such as `bookmaker.eu` are accepted;
path separators and empty domain components remain invalid.
Defaults bound acquisition to ten history attempts, two billable attempts, and fourteen
total HTTP attempts including two account checks. There are no automatic retries;
access, quota, rate-limit, redirects, and unsupported successful contracts stop
new acquisition. History request starts are paced at least 5.1 seconds apart.

The run directory contains `protocol.json` (sanitized account snapshots and frozen
scope), exact data-response bytes in `markets.json.gz`, `fixtures.json.gz`, and
`history/<fixtureId>__<sorted-book-slugs>.json.gz`, plus `receipts.jsonl`,
`coverage.csv`, `selected_pre_match_quotes.csv`, and `summary.json`. Failed response
bytes, when safe to retain, are kept separately in `responses/`; they are not
terminal caches. Receipts distinguish request-start, local receipt, decoded-body
hash, and saved-file hash. Raw account bodies and secret-echo bodies are never saved.
Account-scoped raw data are ignored by Git, not offered for redistribution.

A nonempty directory requires the same command with `--resume`. Scope and raw
receipt hashes are verified first; corrupt, partial, or incompatible evidence
fails explicitly. Only matching successes and documented no-history responses
are reused. A fully terminal cache resumes offline without a key or account call;
incomplete runs require the current account gate before new acquisition. Derived
CSV rows and original retrieval clocks remain stable on completed resume.
Coverage `origin` records the original observation origin; the summary separately
counts cache reuse and actual HTTP attempts for the current invocation.

Quotes require both latest usable source outcome states **strictly before the
earlier scheduled/actual start**. Missing clocks, latest suspensions, invalid
prices, tied conflicts, and explicit live/map/player scope block extraction;
older active values are not revived. Output includes proportional no-vig
probability, not a canonical-team mapping. `createdAt` is not local receipt time
or proof of publication/forecast availability; provider outcome convention is
not independent bookmaker participant binding. Every quote is
`provider_catalog_convention_only` and `not_certified` for execution.

**Observed authenticated execution:** account preflight confirmed sport 17,
Pinnacle, bet365, and Unibet, with a 250-request limit and 0 used at the gate.
A validator bug rejecting other entitled domain-style bookmaker slugs was fixed
before continuing. Discovery returned **94 finished CS2 fixtures** in the requested
April 14–20 interval; the earliest **10** were frozen. Catalog mapping is series
Winner market **171**, with provider-labelled outcomes **171 / 172**.

The initial second history request hit **HTTP 429 `RATE_LIMITED`**, despite
5.1-second request-start pacing. That invocation stopped without automatic retry.
After a later explicit API check confirmed historical CS2 prices, an explicit
resumed acquisition used **6.5-second** request-start pacing and completed the
same frozen sample. No fixtures, dates, or bookmakers were substituted.

**Completed coverage:**

| Bookmaker | Admitted pre-start pairs | Book absent | Fixture has no history |
|---|---:|---:|---:|
| Pinnacle | 4 | 5 | 1 |
| bet365 | 0 | 9 | 1 |
| Unibet | 9 | 0 | 1 |

Nine price-history responses and one documented no-history response are cached.
The extract contains **13 two-sided quote pairs**. The official
[CS2 coverage page](https://oddspapi.io/sports/esports/cs2) identifies sport 17;
the [history contract](https://oddspapi.io/en/docs/get-historical-odds) advertises
history from January 2026. These ten early provider fixtures do not establish
full circuit or older CS:GO coverage.

**Bookmaker log loss:** `/settlements` returned HTTP 500 `INTERNAL_ERROR` for all
nine fixtures with quotes. The separate documented `/scores` endpoint returned
matching fixture IDs and explicit `scores.periods.result` final scores for all
nine. Those results supplied binary outcome-1 labels; optional `fulltime` scores
were also present and agreed on seven fixtures. No anonymous-roster or canonical
match join, guessed numeric period ID, or odds-derived winner was used.

| Bookmaker | Scored available matches | Available-cohort log loss | Log loss on the same 4 shared matches |
|---|---:|---:|---:|
| Pinnacle | 4 | 0.694582 | 0.694582 |
| bet365 | 0 | — | — |
| Unibet | 9 | 0.616802 | 0.651471 |

Scores use natural logarithms and proportional no-vig probabilities from the
latest usable states strictly before the earlier scheduled/actual start. Only
the shared four-match column is a matched bookmaker comparison. Four fixtures
are too few for a stable bookmaker-quality conclusion; there is no common
three-book cohort because bet365 is missing throughout this sample. Constant
50/50 predictions score 0.693147, a reference rather than another bookmaker.

Raw odds and extracts are under
`data/raw/oddspapi/cs2_20260414_20260420_probe/`. `bookmaker_logloss.json` contains
cohorts and metrics; `scored_bookmaker_quotes.csv` contains per-pair labels/losses.
The sibling `cs2_20260414_20260420_probe_evaluation/` retains exact scores and
failed settlement responses, its own protocol/receipts, and
`verified_result_labels.csv`. `run_execution_notes.json` records pacing, attempts,
and verification; `diagnostics/account_gate.json` retains only the sanitized
account diagnostic, never the raw credential-bearing account body.

Recorded attempts: **18** for odds acquisition, **22** for result retrieval, and
**1** unmetered account diagnostic: **41 HTTP calls**, including **20 billable
attempts** (2 discovery, 9 failed settlements, 9 successful scores). Shared-account
quota-counter changes are not attributed wholesale to this collector.

Independent reconstruction from saved history and result bytes reproduced all
**13** quote probabilities, targets, individual losses, and bookmaker averages.
All **31** raw/file receipt hash checks and actual-credential scans passed.
Completed actual CLI resume, with no key and zero request allowance, made **zero
HTTP calls** and preserved both CSVs, original receipts, and the protocol
byte-for-byte. Focused regressions: **26 passed**; full suite: **84 passed**.

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_oddspapi_cs.py' -v
```

## Full 2026 OddsPapi download

`data/raw/oddspapi/cs2_full_2026/` contains the frozen **2026-01-01
00:00:00 UTC–2026-10-05 23:44:38.840857 UTC** acquisition for **Pinnacle,
bet365 and Unibet**. All **5,449 finished sport-17 fixtures** returned across
**40 contiguous seven-day discovery windows** were queried, without sampling or
a `hasOdds` filter. **3,434** returned historical envelopes; **2,015** returned
the documented explicit no-history response for the requested bookmaker group.
Completion does not mean every fixture or bookmaker has usable prices.

The gzip evidence preserves the entire returned history: all markets, outcomes,
players and active/inactive updates. `coverage.csv` exhausts all **16,347**
fixture/book combinations. `selected_pre_match_quotes.csv` contains **4,613**
strictly pre-start two-way series Winner pairs: Pinnacle **2,068**, bet365
**507**, Unibet **2,038**. These remain research quotes, not certified executions.
`manifest.json` freezes the scope and fixture IDs; `fixture_index.json` preserves
native metadata; `summary.json` records coverage and acquisition attempts.

Acquisition used **41 billable fixture-discovery attempts**, including one
rate-limited attempt; historical requests were free. Verified probe cache was
reused. Later history requests were paced **6.5 seconds after HTTP receipt**.
Provider HTTP 500 failures were recovered through explicit resumed invocations;
failed raw responses remain evidence, never cached no-history.

`verification.json` records **5,596 raw receipt hash checks**, independent
reconstruction of all **4,613** quote pairs, a clean credential scan, and actual
zero-HTTP collector CLI resume for **all 40 windows**, preserving their CSVs,
receipts and protocols byte-for-byte. No result labels or full-cohort log losses
were acquired in this download; the probe evaluation above remains separate.

## Offline odds-to-historical-series linkage

`scripts/map_cs_oddspapi_history.py` links native odds fixtures to explicit
OpenCS2 BO3 series results without API requests or changes to model intake.
Use a new, empty output directory; previous audits are not overwritten:

```bash
.venv/bin/python scripts/map_cs_oddspapi_history.py \
  --output-dir data/raw/oddspapi/cs2_full_2026_historical_mapping
```

The observed run accepted **346** historical identities, including **222**
fixtures with usable odds and **308** bookmaker quote pairs. Scored fixtures
span **2026-03-23–2026-04-20**, not the full January–October odds range. The
2024–2025 Kaggle series source has no date overlap with these odds.

Acceptance requires mutually unique team-name pairs, compatible event metadata,
and an aware historical source timestamp **0–6 hours after scheduled start**.
Normalization handles case/whitespace, provider full/short names, leading
`Team`, and trailing `esports`/`e-sports`/`gaming`/`clan`; academy, junior and
former-team qualifiers are retained. Explicit regional event conflicts block
links. Neither nearest-time selection nor result values break identity ties.
This is declared natural-key linkage, not independently certified entity binding.

The wider StrandedPond source contributes **602 provisional candidate rows**
and **55 references anchored by an accepted OpenCS2 HLTV ID**. Its naive dates,
map-only scores and absent BO format do not supply new series-winner labels.
All 5,449 fixtures have explicit statuses in `mapping_coverage.csv`; accepted
links, provisional candidates and oriented scored quotes are separate CSVs.

| Bookmaker | Mapped matches | Available-cohort log loss |
|---|---:|---:|
| Pinnacle | 138 | 0.606227 |
| bet365 | 1 | 0.323787 |
| Unibet | 169 | 0.618497 |

These cohorts differ. On **85 shared Pinnacle–Unibet matches**, losses are
**0.606299 / 0.617806**, respectively. bet365 has only one mapped result;
no bookmaker-quality ranking is established. `bookmaker_logloss.json` records
all available, pairwise-common and all-book-common cohorts.

Verification independently reconstructed all **346** oriented series labels
and all **308** raw odds/loss rows. All **nine** cached native provider result
responses agreed with their mapped historical scores. A second actual CLI run
with sockets forbidden reproduced every CSV and metric byte-for-byte; source
and acquisition manifests stayed unchanged and credential scans passed.
`verification.json` retains this evidence. Focused linkage tests: **12 passed**;
full project suite: **96 passed**.

## Limits

Historical result-release/end timestamps are unavailable. UTC-day batching
prevents same-date result influence but does **not** prove every earlier-dated
result was published by the next date. This is retrospective research, not an
execution-grade historical betting backtest.

Model team names are not stable roster IDs. Model case variants are not silently
merged; organization renames, roster turnover, inactivity and event coverage
remain unresolved. The application now stores observed current research odds;
executable historical odds remain unavailable. Saved histories and natural-key
links remain separate research evidence: neither retrospective source clocks
nor declared participant binding certify executable historical markets.

The map/player experiments above remain separate from the series baseline.
They do not infer organization names for anonymous OpenCS2 rosters, reconcile
nickname identities across sources, or claim verified historical announcement
and result-release clocks.

## Research and provenance

- [Downloaded exports and source audits](reports/cs2-public-exports-acquired.md)
- [Statistics sites and original data sources](reports/cs2-statistics-sites-and-data-sources.md)
- [CS2 project research and broader design](reports/cs2-project-research.md)
- [Acquisition manifest](data/source-manifest.json)

Publisher licenses and upstream-rights caveats are recorded in the acquisition
report. Anonymous technical access does not independently establish reuse rights.

## Broader sports-only bookmaker challenge (2026-10-09)

**This archived experiment failed the bookmaker-competitiveness rules.**
Its original model remains unqualified. The current experimental CLI uses the
2026-10-10 research challenger documented below; neither experiment replaces
the existing application models or the player-roster predictor.

[Full results and reproduction](reports/cs2-bookmaker-challenge-results-2026-10-09.md),
[methods and research](reports/cs2-bookmaker-challenge-methods-2026-10-09.md),
and [frozen protocol](reports/cs2-bookmaker-challenge-protocol-2026-10-09.json).
Canonical evidence is in `artifacts/bookmaker_challenge_20261009/verified/`.

The workflow uses 23,568 complete modern-proxy BO3/BO5 series and 4,420 map-coded
history observations. Coefficients fit **16,164 series through 2026-04-30**;
May–June validation selects among 128 predeclared rating, ridge, boosting and
blend recipes. The winner, `ridge_C1_ratings_form_hnone`, combines past-only
ratings and recent form. Prices are never predictors. Historical measurements
exist for 670 eligible series, including **661 from 2026**; 27 original Valve
ranking snapshots were also acquired and evaluated as optional covariates.

The reserved TEST excludes earlier prediction-ledger/player-cohort IDs and
contains 2,728 series. This is previously unscored retrospective evidence,
not a prospective trial or wholly unseen raw history. Only the same matched
population may be used to compare with bookmakers:

| Same 419 TEST matches, one-hour origin | Model | Available-book consensus |
|---|---:|---:|
| Log loss | 0.632088 | 0.600528 |
| Brier | 0.221130 | 0.206919 |
| Accuracy | 62.05% | 67.78% |
| AUC | 0.682701 | 0.739706 |

The model-minus-consensus log-loss gap is +0.031560; its 95% day-block interval
is [+0.015649, +0.048596]. Seven-day clustering, each individual bookmaker,
alternative quote horizons/de-vig methods, and a two-day history lag also fail
to establish competitiveness. The 72-day paired cohort meets the minimum
support requirement, but its point-score deficit is not just a lack of power.
No profitable/executable betting claim follows from this study.

### Archived inference artifacts

The original schema-1 state remains immutable historical evidence, not a
current inference input. `predict-series` now requires schema 2; use the
published research challenger and command below. Do not re-confirm the exposed
TEST to rebuild an old state or to tune the new model.

At the original run, **472 tests and 356 subtests passed**. Actual preparation,
128-candidate development, frozen confirmation, all six bookmaker comparison
modes and saved-model CLI scenarios ran. Independent verification reproduced
proper scores from exported predictions, checked all 2,728 native TEST labels
and 3,136 exported quote-clock orderings, and verified source/artifact hashes.
The real CLI rejected reselection/reconfirmation without changing artifacts.
See `verified/verification.json` for exercised evidence; no lint/type-check claim.

The archive has no native results after 2026-10-07. Further development must
use a new independent later confirmation cohort, not retune on this exposed
TEST until it appears to win. Verified as-of rosters and map/veto information
are development hypotheses, not demonstrated remedies for the measured gap.

## Professional-methods research challenger (2026-10-10)

[Sourced research](reports/cs2-professional-model-research-2026-10-10.md),
[measured results and reproduction](reports/cs2-professional-model-results-2026-10-10.md),
and [frozen development protocol](reports/cs2-professional-model-protocol-2026-10-10.json).
Evidence: `artifacts/professional_model_20261010/`.

The experiment adds 16 dynamic Bayesian strength channels with Gaussian
uncertainty, winner-versus-terminal-score likelihoods, time dynamics, and
source-day-atomic updates. BO3/BO5 probabilities account for the series stopping
at the required wins. Maps are conditionally exchangeable; this is not a verified
roster, map-veto or live-economy model. Professional sources do not disclose an
exact production recipe or guarantee a bookmaker edge.

Among 49 predeclared candidates, `ridge_C1_ratings_form_bayesian_hnone`
combined the existing ratings/form with the Bayesian channels. All coefficient
fitting ends on 2026-04-30. The previously exposed July–October TEST was not
forecast or scored again.

| Same 545 May–June development matches | Previous model | Research challenger | Book consensus |
|---|---:|---:|---:|
| Log loss | 0.625585 | 0.624766 | 0.623213 |
| Brier | 0.217833 | 0.217521 | 0.216906 |

This is a small development improvement, **not proven bookmaker competitiveness**.
The challenger-minus-incumbent log-loss difference is −0.000819; its diagnostic
95% day-block interval is [−0.005548, +0.003796]. Repeated development selection
is optimistic, and this interval is not selection-adjusted. The challenger
improves full-validation loss and all three TRAIN-only temporal diagnostics,
but remains worse than each bookmaker's point log loss on its matched
development subset. No independent confirmation or profitability claim.

### Current saved-model inference

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KEDRO_DISABLE_TELEMETRY=1 \
  .venv/bin/python -m pipeline.cli predict-series \
  --team1 Vitality --team2 "Natus Vincere" --day 2026-10-10 --format bo3
```

Defaults load the challenger `selected_model.pkl` and schema-2
`series_inference_state.pkl` from `artifacts/professional_model_20261010/`.
The state includes completed history through 2026-10-07, advanced by updates
only after selection. This is a hypothetical literal-organization scenario,
not a verified upcoming fixture or announced lineup. Output includes training
and history boundaries, coverage, used Bayesian channels and
`competitiveness_qualified=false`. Latent variance is not a calibrated
forecast confidence interval. Only load **trusted local pickle artifacts**.

Verification: **492 tests and 384 subtests passed**; actual develop/publish,
BO3/BO5/reversed/cold-team CLI scenarios and historical-date/overwrite guards
were exercised. Independent CSV arithmetic checked all 49 candidate scores;
1,487 native validation labels/orientations and frozen source/input/artifact
hashes were checked. See `verification.json` for numerical integration limits
and exact evidence. No lint/type-check claim.

## 5-Stage Simulation Pipeline & Kedro 8-Layer Architecture

The repository integrates a Kedro 8-layer data hierarchy (`data/01_raw` through `data/08_reporting`) with a 5-stage simulation pipeline orchestrated via `dvc.yaml`:

```bash
# Run complete 5-stage simulation pipeline end-to-end
.venv/bin/python scripts/run_simulation_pipeline.py --stage all

# Or run specific stages:
.venv/bin/python scripts/run_simulation_pipeline.py --stage verify_inputs
.venv/bin/python scripts/run_simulation_pipeline.py --stage model_benchmark
.venv/bin/python scripts/run_simulation_pipeline.py --stage betting_simulation
.venv/bin/python scripts/run_simulation_pipeline.py --stage econometric_audit
.venv/bin/python scripts/run_simulation_pipeline.py --stage generate_report

# Standard walk-forward model benchmark CLI
.venv/bin/python scripts/run_model_benchmark.py
```

Stages:
1. **`verify_inputs`**: Verifies catalog paths, row counts, and SHA-256 digests against `conf/base/catalog.yml`.
2. **`model_benchmark`**: Evaluates model predictive quality (Log Loss, Brier score, AUC, ECE) and computes Bayesian shrinkage grid ($\alpha \in [0.0, 1.0]$) against market closing lines.
3. **`betting_simulation`**: Simulates Flat and 1/4 Kelly staking under Polish 12% turnover tax and International 0% tax, applying Guard B (Market Divergence Gate) and Guard W (Wasserstein DRO).
4. **`econometric_audit`**: Computes Net CLV and empirical noise decay over varying sample sizes ($\sigma / \sqrt{N}$).
5. **`generate_report`**: Consolidates summary metrics and produces `data/08_reporting/simulation_pipeline/SIMULATION_REPORT.md`.

---

## CS2 C0 Player-History Attention Architecture

An adaptation of the `C0` (Attention-over-Player-History) neural architecture for Counter-Strike 2:

```bash
# Ingest historical match and lineup archives into a rolling player sequence bank
.venv/bin/python scripts/build_player_sequence_bank.py
```

- **Tokenization (`src/models/c0_tokens.py`)**: 16 map positions, 4 side assignments, and 16 continuous performance channels projected to 32 dimensions.
- **Sequence Builder (`src/features/player_sequence_builder.py`)**: Strict chronological FIFO rolling queue (16 maps) with date-cutoff guards preventing lookahead leakage.
- **Set Transformer & Attention (`src/models/c0_architecture.py`)**: Masked temporal attention over map history sequences (zero-filled, no NaNs on empty rosters), 5-node Team Set Transformer with permutation-invariant mean/std pooling, and pairwise antisymmetric scoring head guaranteeing $P(A > B) = 1 - P(B > A)$ to machine precision.
- **Verification**: **520 tests passed** (`PYTHONPATH=. .venv/bin/pytest`).