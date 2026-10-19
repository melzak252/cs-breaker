# Credential-free CS2 database exports: acquired and audited

**Date:** 2026-10-04  
**Answer:** Yes. Public exports exist, and three have now been downloaded into `cs-breaker` without user files, credentials, API keys, paid accounts, or a live-site scraper. This report documents acquisition; the [implemented and evaluated series-winner baseline](../README.md) is a subsequent step.
**Related:** [Project research](cs2-project-research.md) · [Statistics-site provenance](cs2-statistics-sites-and-data-sources.md).

## 1. What is now available locally

These numbers come from parsing the actual downloaded files, not search snippets or dataset descriptions. Counts are not additive: the 2026 sources can overlap.

| Source | Observed records | Observed date coverage | Most useful role |
|---|---|---|---|
| Kaggle / Griffin DesRoches, version 5 [1-3] | **7,033 unique match IDs**, 140 columns, 648 tournaments; all rows marked BO3 | 2024-05-15 through 2025-10-16, timestamp strings ending in Z | Historical series results for a chronological team-strength baseline; quarantine precomputed statistics |
| StrandedPond public GitHub CSV snapshot [7-9] | 2,299 raw map rows, **931 distinct match URLs**; 22,991 raw player rows | 2025-12-08 through 2026-10-03, timezone-unspecified date strings | More recent actual map results and limited player aggregates, after duplicate/date reconciliation |
| OpenCS2 demo-derived metadata [4-6] | **1,988 map rows / 860 match IDs**, 137 Parquet shards, 61,707 scoped player rows | 2026-03-23 through 2026-04-20, UTC metadata timestamps | Stable match/player identities and richer retrospective player/demo analysis |

### File layout

```text
data/
  source-manifest.json
  raw/
    kaggle/cs2_newestcombinedmatches.csv
    strandedpond/matches.csv
    strandedpond/player_stats.csv
    opencs2/metadata-*.parquet       # 137 actual files
    opencs2/manifest.json
  exports/
    kaggle/results_only.csv
    opencs2/maps.csv
    opencs2/player_scopes.csv
  audits/
    kaggle-audit.json
    strandedpond-audit.json
    opencs2-audit.json
```

The GitHub and Kaggle raw CSV copies were checked against the downloaded payload SHA-256 hashes. Every OpenCS2 Parquet file was checked against the Hub's advertised SHA-256 and byte size. No large `.dem` binary was downloaded.

## 2. Best immediate baseline source: the Kaggle match export

Dataset: **CS2 HLTV Professional match statistics dataset**, publisher Griffin DesRoches, version **5**. Its public download endpoint served the CSV anonymously with HTTP 200 and `Content-Type: text/csv`. The main file is **7,387,407 bytes**. [1-3]

Direct version-pinned download:

<https://www.kaggle.com/api/v1/datasets/download/griffindesroches/cs2-hltv-professional-match-statistics-dataset/cs2_newestcombinedmatches.csv?datasetVersionNumber=5>

Observed fields include match ID, HLTV ID/URL, date, tournament, team names, result, series scores, format, event type, map-context fields, named player metrics, team aggregates and derived differences. The publisher says 116 columns; the actual downloaded file has **140**. Use the actual header, not a copied schema claim. [1]

### Executed checks

- 7,033 rows and 7,033 distinct HLTV match IDs; no duplicate match IDs.
- 648 distinct tournament names.
- All 7,033 rows have `match_type=bo3`, but subsequent format validation found **1,126 incompatible scores** such as 13–6, 3–2 and 1–0. The label is unreliable; **5,907 rows are BO3-score-compatible**, not independently verified BO3 fixtures. The baseline preserves exclusions in `artifacts/baseline/excluded.csv`.
- No missing required match ID/date/team-name fields.
- All scores parsed as integers and agreed with the recorded winner orientation.
- Event types mix `LAN`/`lan` and `Online`/`online`; normalization is necessary.
- Observed date range: `2024-05-15T06:00:00Z` to `2025-10-16T13:00:00Z`.

These are internal consistency checks, not independent verification of every result against the original tournament.

### The important problem is leakage, not download access

The publisher explicitly warns:

> “THE FEATURES PERTAINING TO TEAM LEVEL STATISTICS SCRAPE CURRENT TEAM STATISTICS … THE DATA MAY NOT REPRESENT THE TEAM ON THE DATE THAT THE GAME WAS SCRAPED IN” [1]

Therefore, do **not** train a prematch model on all 140 columns. Current team/player averages, rolling claims, head-to-head summaries, rankings and derived differences lack the required historical availability guarantee. `winner_*`/`loser_*` fields are also outcome-oriented; their names/semantics must not be mistaken for independent prematch features. Map-context fields do not establish played-map outcomes or publication timing.

I created **`data/exports/kaggle/results_only.csv`** with these 13 fields:

```text
match_id, hltv_match_id, date, tournament,
team1_name, team2_name, match_type, event_type,
score_team1, score_team2, winner, hltv_url, scraped_date
```

This removes the precomputed snapshot statistics from the baseline input table. It does not make every retained field a predictor: **scores and winner are labels, never features**. Compute ratings/form only from earlier available results. The source date and scrape date still do not establish an exact original result-release time; avoid claims of precise historical execution or betting eligibility from this export alone.

The publisher labels the dataset **CC0: Public Domain**. That is its license assertion, not independent verification that it can waive all upstream HLTV/tournament rights. [1]

## 3. More recent map results: the public GitHub snapshot

Repository: **StrandedPond/hltv_scraper**. The repository actually commits result/player CSVs; it is not merely scraper code. Its README attributes the acquisition to HLTV scraping. We downloaded the existing public data files but **did not execute its scraper or bypass tooling**. [7]

Pinned revision:

`a2e4d26694197451e33e0fe2b64d813bdafca446`

Direct downloads:

- Map-results CSV: [8].
- Player-statistics CSV: [9].

The map table contains event/date/map, team orientation, round scores, half scores, total rounds, winner, starting side and match URL. The player table contains names and K/D/A, ADR, KAST, rating, opening-duel/multikill/clutch-related fields, but lacks match URL and stable player ID.

### Executed quality audit

| Check | Observed result |
|---|---:|
| Raw map rows | 2,299 |
| Distinct exact map rows | 2,140 |
| Exact duplicate map rows | 159 |
| Distinct `(match_url, map)` keys | 2,127 |
| Keys with differing representations | 13 |
| Fields differing in those 13 keys | **Date only** |
| Series with multiple source row dates | 291 |
| Score-total or winner-score inconsistencies | 0 |
| Raw player rows | 22,991 |
| Exact duplicate player rows | 1,590 |
| Player rows without a matching event/date/map/team key | 0 |
| Ambiguous match-team join keys | 1 |

Example: the same Vitality–The MongolZ series has source row dates `2025-12-11 20:10` and `2025-12-12 03:10`. Do not silently reinterpret them as separate matches or assume a timezone correction. The source timezone is unspecified.

Only one distinct player-team group had six names rather than five after exact deduplication. That and the ambiguous join require inspection before longitudinal player modeling.

**Decision:** preserve the raw files. Exact duplicates can be deduplicated in a downstream audited import, but conflicting date representations must remain explicit. Map order and BO format are absent; the number of exported maps alone does not prove series completeness. This is useful recent map-result material, not a fully reconciled production database.

No explicit data-license grant was established for this GitHub export. An educational disclaimer is not such a grant. That is a reuse caveat, not an assertion that the public download is technically unavailable.

## 4. Richer demo-derived export: OpenCS2 metadata

Repository: **`blanchon/opencs2_dataset_demo`**, observed as public and ungated. [4]

Pinned revision:

`912a363098feec3f220e41d68634d2f71676c28b`

Its card specifies **`data/metadata-*.parquet`**; some README example paths use a different repository name, so acquisition followed the actual Hub API/file tree. [4-6]

All **137** listed metadata shards were acquired, totaling **13,071,350 bytes**, without downloading the much larger demo binaries. The file listing had no remaining pagination link. Each file's actual length and SHA-256 matched the advertised metadata. [6]

The actual Parquet schema includes stable `match_id`, `row_id`, map index/name, build, event/teams, UTC match date, source paths, uploaded time, and nested player/team/round/kill structures. SteamIDs are available for longitudinal identity. [5]

### Actual coverage and limitations

- 1,988 rows and 1,988 distinct row IDs.
- 860 distinct match IDs; no repeated `(match_id, map_index)` keys.
- All rows marked BO3.
- UTC dates from **2026-03-23 18:04:09** to **2026-04-20 21:02:59**.
- No missing match dates; no match dates later than their recorded upload time.
- 53 series have an exported distinct-map count different from the sum of recorded series map-win scores: **the corpus is incomplete for those series under that check**.
- Top events include ESL Challenger League/CCT and PGL Bucharest/IEM Rio; it is not exclusively tier-one coverage.

### Do not misread the nested statistics

**Top-level `score1`/`score2` are series map wins**, not per-map round scores. The inspected first series is 1–2 at this level. [5]

Nested `score_ct`/`score_t` cannot be assumed to be round scores for organization A/B: 501 records have a CT/T total above 13 with fewer than 25 rounds. The totals sum to the parsed round count, but side/restore semantics and roster orientation need auditing. One observed example is 3 CT versus 14 T over 17 rounds; this is not a valid direct inference of team1 losing 3–14.

Player statistics have **`ct`, `t`, and `all` scopes**:

- 61,707 source player-scope rows.
- 20,817 `all`-scope rows.
- 1,691 distinct SteamIDs.
- 370 maps have more than ten `all`-scope player entries.

Do not sum CT + T + ALL, or assume every scope entry is a distinct match participant. The reasons for extra player identities—such as demo/restore/warmup behavior—are not established; they remain anomalies rather than an invented explanation.

Derived exports preserve these distinctions:

- **`maps.csv`**: source scalar columns, source shard and explicitly named `analysis_score_ct`/`analysis_score_t`.
- **`player_scopes.csv`**: row/match/map identifiers plus player statistics with their source scope intact. No inferred organization assignment or silent participant filtering.

The raw Parquet files retain the original nested information for further audits. CSV readers must preserve SteamIDs as exact integers/strings; floating-point conversion can corrupt large identifiers.

**Player-rating execution update:** [research §13](cs2-project-research.md#13-executed-player-rating-comparison-and-data-sufficiency) now reconstructs roster-oriented winners from original round/kill events for **907 maps / 1,490 SteamIDs**, without assigning anonymous rosters to named organizations. Reconstructed scores are not the aggregate CT/T totals. The remaining 1,081 maps are explicitly excluded by participant/roster/round-evidence/terminal-score checks; no observer or warmup explanation is assumed.

The card declares **CC BY 4.0** and describes HLTV-derived material, while warning that downstream users must respect original tournament terms. Access is verified; upstream permission is not independently cleared. [5]

## 5. Reproducing acquisition without an account

The pinned URLs and checksums are recorded in **`data/source-manifest.json`** and **`data/raw/opencs2/manifest.json`**. Use the stable Kaggle version endpoint, not its temporary signed-storage redirect.

For example, from the repository root:

```bash
curl --fail --location --output kaggle-v5.csv \
  'https://www.kaggle.com/api/v1/datasets/download/griffindesroches/cs2-hltv-professional-match-statistics-dataset/cs2_newestcombinedmatches.csv?datasetVersionNumber=5'

curl --fail --location --output map-results.csv \
  'https://raw.githubusercontent.com/StrandedPond/hltv_scraper/a2e4d26694197451e33e0fe2b64d813bdafca446/hltv_scraper/matches.csv'

curl --fail --location --output example-metadata.parquet \
  'https://huggingface.co/datasets/blanchon/opencs2_dataset_demo/resolve/912a363098feec3f220e41d68634d2f71676c28b/data/metadata-europe-020c2c03.parquet'
```

These are documented repeat-acquisition commands using the URLs actually retrieved; our acquisition used Python's standard HTTP client. Repeat all metadata downloads from the file manifest rather than guessing shard names. The first example Parquet is 100,149 bytes, SHA-256 `f0dd23d8fcc48d293c4659fcc7e43ee7ee1f1440c85556e4d717a03dd1f126c7`. [6]

Parquet inspection used PyArrow **25.0.1**, installed into an isolated temporary location through `uv`. No system Python package installation or project runtime dependency was added. Acquisition times in the manifest are explicitly not original publication/forecast-availability timestamps.

## 6. Scraping and publicness: the practical distinction

A site's terms are not themselves legislation. Nevertheless, “publicly displayed” does not establish unlimited automated collection, redistribution, model-training or commercial-use rights, and “a ban is the only consequence” is not a safe universal assumption. HLTV's terms reserve rights, prohibit scraping/competitive use, and include an indemnification clause; enforcement/applicable law are separate questions this investigation does not adjudicate. [10]

For this task, there is no need to resolve that dispute before obtaining usable material: **public exports were actually downloaded**. Their publisher licenses and upstream caveats are documented, not treated as automatic technical blockers.

This also avoids using scraper code that claims anti-bot bypasses. No live HLTV crawl, paywall access, credential sharing, access-control workaround or provider purchase was performed.

## 7. What we can do now, with nothing more from the user

**The series-winner baseline is implemented and evaluated:** [commands, protocol and measured results](../README.md). It reads the 7,033-row results-only export, excludes 1,126 format-incompatible rows, and computes chronological ratings on 5,907 score-compatible matches. Validation selected Elo K=32; 2,238 later test matches produced log loss 0.6660 and accuracy 59.79%. Current snapshot statistics and target results are not prediction features. Precise original result-release clocks and executable historical odds remain unavailable.

**Player-roster map Elo is also implemented:** exact GitHub map/player joins yield **1,823 maps / 651 nickname handles** after conservative exclusions, and event-backed OpenCS2 reconstruction yields the stable-ID cohort above. Each source has a separate walk-forward comparison, rankings and exclusion ledger; no nickname-to-SteamID identity merge or unsupported organization assignment was made. Both player experiments failed to beat their simple probability controls on later test log loss. Detailed coverage and measured results are in [research §13](cs2-project-research.md#13-executed-player-rating-comparison-and-data-sufficiency).

Missing capabilities are specific: complete event coverage, known pre-veto/roster publication times, trustworthy per-map organization-oriented outcomes for every demo, historical executable odds and operational live updates. None requires the user to supply files merely to begin the baseline; they are later data-contract problems.

## Verification

Actual anonymous downloads succeeded for all three sources. CSVs were parsed, raw-copy checksums matched, all 137 Parquet hashes/lengths matched, and full metadata/player-scope exports were produced. Audits report duplicates, dates, identity/join limitations, result consistency and leakage. The subsequent baseline added score/format consistency checks, passed 13 regression tests, exercised the real backtest/prediction CLI, and reproduced all four artifacts byte-for-byte. No profitability was measured. Citation linkage is verified separately from data correctness and legal rights.

## Bibliography

| ID | Source | What it establishes | URL |
|---|---|---|---|
| [1] | Kaggle dataset metadata/description | Publisher cohort/license claims and explicit snapshot-stat warning | https://www.kaggle.com/api/v1/datasets/view/griffindesroches/cs2-hltv-professional-match-statistics-dataset |
| [2] | Kaggle version-5 file listing | Actual named files and advertised byte sizes | https://www.kaggle.com/api/v1/datasets/list/griffindesroches/cs2-hltv-professional-match-statistics-dataset?datasetVersionNumber=5 |
| [3] | Kaggle version-5 CSV | Actual anonymously acquired historical match records | https://www.kaggle.com/api/v1/datasets/download/griffindesroches/cs2-hltv-professional-match-statistics-dataset/cs2_newestcombinedmatches.csv?datasetVersionNumber=5 |
| [4] | OpenCS2 Hub API | Public/ungated status and observed revision | https://huggingface.co/api/datasets/blanchon/opencs2_dataset_demo |
| [5] | OpenCS2 revision-pinned dataset card | Declared schema, provenance and publisher/upstream-rights notes | https://huggingface.co/datasets/blanchon/opencs2_dataset_demo/raw/912a363098feec3f220e41d68634d2f71676c28b/README.md |
| [6] | OpenCS2 revision-pinned metadata file tree | Metadata paths, lengths and SHA-256 identifiers | https://huggingface.co/api/datasets/blanchon/opencs2_dataset_demo/tree/912a363098feec3f220e41d68634d2f71676c28b/data?limit=1000 |
| [7] | StrandedPond repository | Committed CSV exports and stated HLTV scraping provenance | https://github.com/StrandedPond/hltv_scraper |
| [8] | Revision-pinned map-results CSV | Actual acquired map-score/date/team records | https://raw.githubusercontent.com/StrandedPond/hltv_scraper/a2e4d26694197451e33e0fe2b64d813bdafca446/hltv_scraper/matches.csv |
| [9] | Revision-pinned player-stat CSV | Actual acquired player aggregate records | https://raw.githubusercontent.com/StrandedPond/hltv_scraper/a2e4d26694197451e33e0fe2b64d813bdafca446/hltv_scraper/player_stats.csv |
| [10] | HLTV Terms of Service | Operator's stated restrictions/reserved rights; not a legal judgment | https://www.hltv.org/terms |
