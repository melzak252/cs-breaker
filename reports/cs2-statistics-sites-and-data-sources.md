# Counter-Strike equivalents of Games of Legends: sites, original data, and collection options

**Research date:** 2026-10-04  
**Question:** Is there a Counter-Strike site like Games of Legends, where does its data come from, and should cs-breaker scrape it?  
**Companion report:** [CS2 project/model research](cs2-project-research.md). The original research is already saved in that file; this report adds the source-provenance investigation.  
**Status:** source research and bounded feasibility inspection. No bulk scraper, production feed adapter, training corpus, provider account, or demo parser execution was created or verified.

**Acquisition follow-up:** [Credential-free database exports actually downloaded and audited](cs2-public-exports-acquired.md). The status above describes the earlier site investigation; the follow-up includes three downloaded public datasets and their limitations.

## Executive answer

**Yes. HLTV and bo3.gg are the closest functional equivalents for professional Counter-Strike.** DRAFT5 also publishes professional-match statistics; Skybox EDGE offers deeper demo/tactical analytics. CSStats.gg, Leetify, and Scope.gg primarily analyze users' own matchmaking/platform matches and are not interchangeable with a professional-event database. [1,8,13,15,19,20,21]

**Where the data originates is more useful than which website displays it.** Counter-Strike demos/replays and server-generated records contain the underlying actions. Sites may acquire those records through organizer cooperation, platform integrations, user-authorized match history, demo uploads, or licensed official feeds, then compute their own statistics. There is direct documentation for several of these mechanisms, but not a complete current ingestion specification for HLTV or bo3.gg. [2,3,6,15,16,24,25,27]

**Recommendation:** use HLTV/bo3.gg as reference products; build cs-breaker's dataset from permissioned metadata and legitimately obtained demos or licensed event feeds. Do not make an unlicensed HLTV scraper the foundation. For bo3.gg, request an export/API/research agreement before collecting a model-training corpus: its terms do not clearly grant that use and its robots file expressly signals against AI training. [5,10,11]

The user's “research where they get their data from or do the scraper” is resolved here through the research branch. A scraper was not substituted for an unavailable official source, and technical readability was not treated as permission.

## 1. Which sites actually match the requirement?

| Site | Fit for a gol.gg-like professional-CS product | Publicly documented origin | Access/reuse conclusion |
|---|---|---|---|
| **HLTV** | Strong: professional matches, events, players, teams, statistics, demos [1] | Historical demo parsing, server logs, scorebot integration and organizer cooperation [2,3] | Current terms prohibit scraping/data mining and competitive-product use; business use requires a license [5] |
| **bo3.gg** | Strong: professional results, map/player statistics, rankings, comparison tools [6,8,9] | Claims direct organizer feeds, official APIs, verified databases and analyst review; suppliers unnamed [6,7] | No documented public developer API/bulk-export license found; crawler/training restrictions and intended-use questions remain [10,11] |
| **DRAFT5** | Useful additional professional-event reference [20] | Inspected announcement describes map-by-map statistics, not upstream provider | No documented public statistics API found in reviewed material; coverage and permission must be established |
| **Skybox EDGE** | Rich professional analysis rather than simply a public results table [21] | Product centered on demos, replay and derived tactical analytics | Product access is not bulk export/API or model-training permission; no public EDGE data API found in reviewed pages |
| **CSStats.gg** | Primarily personal matchmaking statistics [13] | User-submitted match share codes; Valve-linked discovery | Terms prohibit robots/data extraction and incorporation into another database without permission [14] |
| **Leetify** | Personal gameplay analytics and match reports [15,16] | Valve share/authentication codes; verified FACEIT demo uploads; other manual uploads | Public API exists, but no-storage/metric-display rules and registered-user scope make it a poor historical training source [17,18] |
| **Scope.gg** | Personal match history, improvement and replay tools [19] | Advertises linked Steam/FACEIT history, API/webhook sync and automatic demo downloads | Product claims, not an independently verified developer feed; no public developer API found in reviewed pages |
| **Liquipedia** | Tournament, participant and event-context supplement [28] | Wiki content through MediaWiki/approved LiquipediaDB access | Use the API, not HTML scraping; attribution and content-license obligations apply |

“Closest equivalent” is a product comparison, not a coverage benchmark. This investigation did not measure completeness across all CS2 tournaments or validate every displayed statistic.

## 2. HLTV: what its own sources actually establish

### Historical ingestion is documented

The clearest primary statement is HLTV's **23 September 2012 CS:GO statistics launch**:

> “our only source of match statistics at the beginning will be server logs” [2]

The same announcement contrasts that with older Counter-Strike:

> “more than 90% of our CS 1.6 statistics were parsed from HLTV or POV demos.” [2]

It also discusses source-access difficulties and organizer cooperation. This is stronger evidence than the common unsupported statement “HLTV gets everything by parsing demos.” It demonstrates that the source mechanism depended on game generation and access.

In September 2016, HLTV documented a livescore bot:

> “automatically setting the score after every round (and map)” [3]

It said its scorebot had been adapted to ESEA's server system, and that server configurations, crashes, and round restores could prevent updates. That establishes an operational integration, not a model-independent always-correct result stream. [3]

In December 2022, HLTV announced removal of demo download limits and a BitSkins demo-hosting partnership, describing demos for over 50,000 matches at that time. **Hosting/delivery is not the same as originating the match data.** BitSkins was described as a demo partner, not demonstrated as the supplier of all statistics. The historical announcement does not override today's terms or establish redistribution rights. [4,5]

### What is not established for current CS2

No complete current public document was found identifying, for every HLTV CS2 event/statistic:

- its original demo/log/live-feed provider;
- whether the statistic is parsed, manually entered, or received already aggregated;
- source-specific availability and correction times;
- exact ingest software and calculation versions;
- partner/API entitlements available to third parties.

**Safe conclusion:** HLTV has documented multiple historical source paths and an extensive curated product. Its complete present CS2 backend remains unverified in this research. Do not convert a 2012 or 2016 statement into a current universal architecture claim.

### Why not implement the obvious HLTV scraper?

Its Terms of Service, effective 27 January 2025, explicitly say users may not:

> “conduct, facilitate or organize data mining or web scraping” [5]

They also prohibit constructing a similar/competitive website, product or service, and require a license for business use of materials. This is not merely a request-rate problem. An unofficial library, scraper repository, browser, proxy, or public demo link does not grant rights the source withholds. [5]

The parent investigation received HTTP 403 on the main statistics page and some older articles. It did not attempt access-control workarounds. Historical statements above were inspected by the source-research track; some parent cross-check requests were blocked. The terms page was directly readable and independently checked.

**Decision:** contact HLTV for a source/export agreement if its curated records are essential; otherwise collect original data through another authorized route.

## 3. bo3.gg: broad source categories, unnamed suppliers

### What it says publicly

Its Stats Guidelines state:

> “Our data collection methods include direct feeds from tournament organizers, official APIs, and verified gaming databases to ensure authenticity.” [6]

Its Why Trust Us page says it works with organizers and official APIs, cross-verifies data, and has analysts review information. These are bo3.gg's own qualitative statements, not an independent audit or a named-supplier inventory. [7]

**What we can say:** it claims a multi-source feed/API/database process with verification.

**What we cannot say:** that GRID supplies it, that it scrapes HLTV, that all statistics are demo-derived, that it uses a particular parser, or that it has complete event coverage. No inspected announcement named a bo3.gg–GRID or bo3.gg–HLTV statistics-supply agreement. Editorial attribution to HLTV would not establish a database-supply relationship.

“Uses official APIs” describes its inputs. It does not mean bo3.gg offers us an official public API.

### A real representative page was checked

The finished-match listing exposed a Spirit–PARIVISION match link. Opening that page through the reader and an ordinary Chromium session showed:

- professional event/stage context and a BO3 format;
- a 2–1 series result;
- Ancient 11–13, Dust II 13–6, Anubis 13–5;
- player scoreboards with K/D/A, ADR, opening-duel and other displayed fields;
- links to performance, aim, grenade, device and economy sections. [8,9]

This demonstrates the kind of product we want. It does not prove all fields exist for every match, nor that the page was available before the match, nor that its values were independently audited.

### Scraping feasibility is not the same as permission

A bounded Python/urllib GET of that **one public match URL** returned HTTP 200 and **13,188 bytes**, but only a JavaScript application shell: no visible Spirit/PARIVISION/map/ADR fields and no HTML table. A normal browser subsequently rendered the statistics page; a screenshot and DOM observation confirmed the actual surface. No authenticated session, direct private-API request, pagination, or bulk collector was used.

Therefore, a basic requests-plus-HTML parser did **not** work on the tested response. Browser rendering worked. This is a one-page finding, not a statement that every route always has the same rendering behavior. The reader's rendered content should not be confused with the bytes a plain HTTP client receives.

bo3.gg's inspected robots file includes:

```text
Disallow: /api/
Disallow: */?
Content-Usage: train-ai=n
Content-Signal: ai-train=no, ai-input=yes, search=yes
```

It also excludes numerous query/filter paths. These are direct evidence of the site's stated crawl/training preferences; they are not a standalone legal opinion on every kind of statistical analysis. Do not ignore them when the planned application is model training. [11]

Its Use of Services says materials are owned or licensed, and:

> “Full or partial distribution or reproduction of site materials is allowed only with an active hyperlink to this website.” [10]

That is not an explicit blanket grant for bulk collection, internal endpoint crawling, long-term ML storage, betting use, or a competing derived data product. A backlink alone does not resolve those questions. Conversely, this report does not claim that its text expressly bans every possible single-page automation; the permission for the intended corpus/use is simply not established.

**Decision:** request permitted API/export access and intended-use clarification through **support@bo3.gg**, which its contact page publishes for partnerships and collaborations. No inquiry was sent on the user's behalf. [12]

## 4. CSStats, Leetify and Scope: the original inputs are easier to trace

### CSStats.gg

Its match-addition documentation tells users to copy a match share code from the game and submit it:

> “Paste the sharecode into the search bar and press enter! This will add the match to our site” [13]

This establishes user-driven match discovery, not a professional-tournament feed. Its terms identify ESL FACEIT Group as the operator, but ownership does not prove that it receives every EFG tournament's detailed data. [14]

Its content license expressly excludes data mining, robots and similar extraction, and restricts putting site content into another database/compilation without permission. The appropriate learning is to reproduce an authorized ingestion method, not scrape its processed profiles. [14]

### Leetify

Its January 2024 support explanation provides a particularly clear pipeline:

1. A user supplies a **match share code**.
2. A user supplies an **authentication code**, authorizing demo requests.
3. Leetify downloads the matching demo and discovers later share codes.
4. It processes the demo into match reports and derived profile statistics. [15,18]

The article says share codes discover newer matches, not arbitrary older history, and describes a 30-day continuity limit. This is a personal-history synchronization mechanism, not a way to enumerate every professional CS2 event. [15]

Its April 2024 FACEIT upload announcement documents a second path: the user downloads a demo from the FACEIT room, uploads it to the corresponding match page, and Leetify verifies the match before adding shared reports/profile statistics. Generic manual uploads have different visibility and aggregation rules. This is a dated description of that product path, not proof that all automatic integrations remain unchanged today. [16]

Leetify **does have a public API**, so it is an important counterexample to “there are no CS analytics APIs.” But its developer guidelines require attribution/backlinks, presentation of metrics without renaming/rescaling, and say **“Do not store data.”** The January 2026 update limits public API data to registered Leetify users. [17,18]

**Decision:** potentially useful for a permitted user-facing integration, not a default source for an immutable professional-model training archive. Seek separate terms if retention/derived model use is required; do not scrape the website to circumvent API limits.

### Scope.gg

Scope's match-history page tells users to link Steam/FACEIT history so it can download matches, and advertises API/webhook synchronization and automatic analysis. This is a documented product claim about account-linked match ingestion. The exact current integration agreements, supported event population and third-party export permissions were not independently verified. No public developer API was found in reviewed pages. [19]

**Shared lesson:** these services show that detailed CS statistics can be constructed from replay/event data without copying another site's finished tables. They do not supply a ready-made complete professional archive for cs-breaker.

## 5. DRAFT5 and Skybox EDGE

DRAFT5's July 2023 announcement says statistics would cover important tournaments, update after each map, and support map filtering in BO3/BO5 series. The inspected announcement does not name the original provider or publish an API/export contract. It is another professional statistics reference, not a verified acquisition route. [20]

Skybox EDGE advertises team/player statistics, public/POV demos, a 2D replay tool, economy/role filters, tactical analysis and veto simulation. Its product table distinguishes professional-team tiers from personal free/pro access; the latter are not equivalent to professional bulk access. [21]

No documented public EDGE data API or blanket export license was found in the inspected product material. Product access does not grant reproduction of proprietary analytics, and the exact demo/feed suppliers remain unverified here. Contact Skybox if licensed exports are necessary; otherwise calculate our own clearly defined metrics from original permitted demos.

Avoid confusing esports Skybox with unrelated products offering skybox image-generation APIs. Similar product names are not source evidence.

## 6. Where original Counter-Strike data comes from

The defensible acquisition paths are distinct:

```text
Authorized organizer/platform records ────────┐
Licensed official event feed ────────────────┤
Legitimately obtained demo/replay ─→ parser ─┤
Permitted event metadata API ────────────────┘
                    ↓
        Identity + time-aware normalization
                    ↓
    Raw captures / rounds / player-map aggregates
                    ↓
       Historical features → model → dashboard
```

This is a proposed architecture, not a verified diagram of any one site's backend.

### Valve user-authorized history

Valve's September 2019 release note explicitly introduced a game authentication code allowing third-party sites/apps to access **a user's match history**. Leetify documents the contemporary CS2 share/auth-code use in detail. Together they support the personal-history pathway, not a token-free professional-results archive. [15,22]

Game State Integration should not be assumed to provide historical professional demos/results. Detailed Valve wiki pages were not reliably readable in this investigation, and no GSI-based archive was established. Do not make it the source plan for this project.

### FACEIT API and demo downloads

FACEIT documents match details and match-statistics resources through its Data API. That scope is FACEIT matches; it is not evidence that every ESL/BLAST/PGL tournament is exposed there. Generic statistics schemas do not guarantee every CS2-specific field on every record. [23]

Its Downloads API documentation states:

> “Downloadable content on FACEIT’s Cloud infrastructure is private by default and can only be accessed with a signed URL.” [24]

Resource URLs may appear in match responses when available. Access to the Downloads API requires an application, and an **exclusive token with Downloads API scope**; the documented response gives a signed download URL. An ordinary Data API key or a visible resource URL does not automatically authorize downloads. The guide advertises a 30-day expected application response, not a promise of approval. [24]

### Official tournament feeds through GRID

GRID and ESL FACEIT Group's July 2025 announcement describes GRID as exclusive distributor, infrastructure and integrity partner for live match data from **EFG-operated CS2 and Dota 2 competitions**, including IEM and ESL Pro League. That is direct evidence of an organizer-to-official-feed path. It is not proof that bo3.gg or HLTV is a GRID customer, nor a CS2-wide license. [25]

GRID's Open Access help explicitly includes **Central Data and Series State**, but excludes **Series Events and File Download**. Historical veto completeness, raw-demo supply and our account's actual entitlements require verification. [26]

If using PandaScore, retain the companion report's warning: ordinary subscription terms prohibit odds-related development through those subscriptions. A sports-only and an odds-related application can require different rights. [29]

### Demo parsing with Awpy

Awpy's documented CS2 example parses a demo into rounds, kills, damages, weapon shots, bomb events, smokes, infernos, grenades, footsteps, and player/tick structures. It uses **demoparser2** as its parsing backend. This establishes available parser capabilities, not permission to acquire any website's demos. [27]

From those records we can define our own reproducible ADR, opening/trade involvement, utility contributions, clutch situations and economy-aware summaries. We should publish exact definitions rather than label them as identical to an opaque HLTV/bo3/Skybox rating. Parser version, game build, warmup/restore handling and partial-demo status must be retained.

No real rights-cleared demo was supplied or parsed during this investigation. Documentation output is not our own parsing smoke. Actual parser acceptance must use a real permitted file, reconcile rounds/scores, and check player/event attribution.

### Event context through Liquipedia

Use Liquipedia's MediaWiki API or approved LiquipediaDB API rather than scraping generated HTML. Its policy requires caching, identifiable contact-bearing User-Agent, gzip support and attribution. MediaWiki limits are one request per two seconds, with `action=parse` no more than one per 30 seconds; approved LiquipediaDB access is limited to 60 requests/hour. Content licensing is CC BY-SA 3.0 with applicable attribution/share requirements. [28]

Liquipedia metadata complements demos; a demo alone does not reliably carry the entire event bracket, veto publication time, expected roster history, or administrative settlement context.

## 7. Concrete decision: collection adapter, not arbitrary website scraper

### Preferred order

1. **Permitted event metadata/results API or export.** Audit stable IDs, history, revision clocks and coverage.
2. **Original permitted demos.** Obtain from organizer/platform or appropriately licensed archive; parse ourselves.
3. **Licensed professional feed.** Use when broader/reliable official coverage justifies cost and contractual access.
4. **Permissioned site extraction only if necessary.** Collect exactly the agreed pages/fields, at agreed rate, with retention/training/derived-output rights.

### If permissioned scraping becomes the chosen source

A proper collector must do more than save a rendered table:

- identify and obey allowed hosts/paths, published policies and granted use;
- use a declared client, bounded rate and caching, not rotating identities;
- stop on 401/403, challenge pages, revocation or disallowed paths;
- treat HTTP 200 with an empty shell/missing required fields as failed extraction;
- capture original payload, source URI/hash and retrieval clock;
- normalize match/team/player/map identities, orientation and rule/version context;
- record missing/partial data explicitly rather than substitute zeros;
- preserve revisions and deduplicate repeated map/match entries;
- separate result availability from scheduled start and scrape time;
- avoid claiming final-lineup/veto knowledge before it was published;
- enforce source retention/deletion/export obligations.

A public browser request made for this investigation is not a permissioned bulk collector. There is no CAPTCHA/proxy/stealth/private-API workaround in the recommended design.

### Source-access request, ready to use

bo3.gg publishes support@bo3.gg for partnership/collaboration inquiries. [12] A useful request would say:

> We are building a CS2 research/forecasting project. Do you offer an API or historical export for professional fixtures/results, map scoreboards, player appearances, vetoes and rosters? Please specify event/date coverage, source provenance, original publication/revision timestamps, rate limits and pricing. We need explicit permission for retention, statistical/ML training, derived probabilities, publication of research results and a dashboard. Please separately confirm whether odds-related use is allowed. We do not intend to scrape undocumented API routes. Can you provide a representative sample and the applicable license?

Adapt that request to HLTV, Skybox, organizers or feed providers. No permission is presumed, and no message was sent.

## 8. Minimum ingestion slice for cs-breaker

**Required inputs:** stable match/player/team IDs; series and map results; format/pool/rules version; a legitimate source; retrieval and available source timestamps. Rich demo data is an optional subsequent input, not a reason to fabricate missing metadata.

**Proposed first deliverable:** one permitted historical event slice → captured raw records → normalized maps/rosters/results → reproducible feature tables → global/map-rating benchmark. Add only sources whose coverage and rights are established.

**Observable proof:**

- compare a representative imported match with its authorized original;
- reconcile series winner from map winners without assuming all matches are BO3;
- reconcile regulation/overtime scores and administrative outcomes;
- demonstrate player identity and team orientation;
- distinguish absent statistics from actual zeros;
- reproduce feature tables from the same captures;
- show that later corrections do not silently rewrite earlier forecast evidence;
- when demos are added, parse a real permitted file and audit round/player/event totals.

A limited permitted corpus can support a valid study. It should not be advertised as complete professional CS2 coverage. Old CS:GO, personal FACEIT matchmaking and top-tier CS2 LAN records are different populations; pooling them simply to increase row count is not an access or modeling solution.

## 9. Evidence boundaries and verification

Three parallel source tracks investigated site origins, alternative analytics platforms, and original official/platform/demo pathways. Primary-source quotations are marked inline; older announcements remain explicitly historical. No API absence is claimed universally: “no documented public API found” is bounded to reviewed material, and private partner APIs may exist.

**Executed inspection:** a single ordinary HTTP GET exposed the empty BO3 JavaScript shell; a normal browser rendered the representative match, its map results and scoreboard controls, with visual confirmation. The tab was closed. No bulk extraction, model training, site-data database or credentials were used.

**Access limits:** some HLTV and CSStats pages returned 403 to parent reads. Readable official terms were checked, and research-track source extracts were used where appropriate. No restricted page was bypassed. Provider capabilities/contracts were researched but not exercised through an approved account.

**Citation verification:** the bundled citation-reference audit passed for both saved reports. This report has 29 cited bibliography entries; the companion has 42. Neither has missing/unused references or placeholder URLs. The audit verifies inline-reference linkage and non-placeholder bibliography URLs; it does not establish contractual enforceability, complete data lineage or operational feed coverage.

## Bibliography

All mutable documents were inspected during this investigation. Historical dates below bound the claim; they do not establish the site's unchanged current backend.

| ID | Document | Publisher / date | URL |
|---|---|---|---|
| [1] | 20 years of HLTV.org | HLTV; 2022-04-11 | https://www.hltv.org/news/33554/20-years-of-hltvorg |
| [2] | CS:GO stats arrive to HLTV.org | HLTV; 2012-09-23 | https://www.hltv.org/news/9101/csgo-stats-arrive-to-hltvorg |
| [3] | Introducing: livescore and filtering | HLTV; 2016-09-19 | https://www.hltv.org/news/18733/introducing-livescore-and-filtering |
| [4] | Download limits for demos removed | HLTV; 2022-12-01 | https://www.hltv.org/news/35174/download-limits-for-demos-removed |
| [5] | Terms of Service | HLTV; effective 2025-01-27 | https://www.hltv.org/terms |
| [6] | Stats Guidelines | bo3.gg; mutable/undated | https://bo3.gg/wiki/stats-guidelines |
| [7] | Why trust us? | bo3.gg; mutable/undated | https://bo3.gg/wiki/why-trust-us |
| [8] | CS2 match results/history | bo3.gg; mutable listing | https://bo3.gg/matches/finished |
| [9] | Spirit vs PARIVISION at ESL Pro League Season 24 | bo3.gg; inspected match page, 2026-10-04 | https://bo3.gg/matches/spirit-vs-parivision-04-10-2026 |
| [10] | Use of Services | bo3.gg; mutable/undated | https://bo3.gg/wiki/use-of-services |
| [11] | robots.txt | bo3.gg; mutable crawl/training directives | https://bo3.gg/robots.txt |
| [12] | Contact us | bo3.gg; mutable contact page | https://bo3.gg/wiki/contact-us |
| [13] | Adding your matches / getting the sharecode | CSStats.gg; mutable support page | https://csstats.gg/getting-the-sharecode |
| [14] | Terms and Conditions | CSStats.gg / ESL FACEIT Group; effective 2025-11-20 | https://csstats.gg/terms-of-use |
| [15] | What are Share Codes? | Leetify; 2024-01-09 | https://leetify.com/blog/share-codes/ |
| [16] | Now you can upload your FACEIT Demos | Leetify; 2024-04-02 | https://leetify.com/blog/faceit-uploads/ |
| [17] | Leetify API – Developer Guidelines | Leetify; published 2025-07-20, updated 2025-12-17 | https://leetify.com/blog/leetify-api-developer-guidelines/ |
| [18] | Privacy updates to our API and Profiles | Leetify; published 2026-01-23, updated 2026-04-16 | https://leetify.com/blog/privacy-updates-to-our-api-and-profiles/ |
| [19] | CS2 Match History | Scope.gg; mutable product page | https://scope.gg/match-history/ |
| [20] | Estatísticas chegaram na DRAFT5! | DRAFT5 / Lucas Spricigo; 2023-07-25 | https://draft5.gg/noticia/estatisticas-chegaram-na-draft5 |
| [21] | Skybox EDGE product and feature tables | Skybox; mutable product page | https://www.skybox.gg/edge/ |
| [22] | Release Notes for 9/17/2019 | Valve; 2019-09-17 | https://blog.counter-strike.net/2019/09/25513/ |
| [23] | Data API (4) | FACEIT; mutable developer documentation | https://docs.faceit.com/docs/data-api/data |
| [24] | Downloads API summary and access guide | FACEIT; mutable developer documentation | https://docs.faceit.com/getting-started/Guides/download-api/ |
| [25] | ESL FACEIT Group partnership for official match-data distribution | GRID / EFG; 2025-07-30 | https://www.grid.gg/resources/newsroom/esl-faceit-grid-esports-data-partnership/ |
| [26] | Open Access API entitlement | GRID; mutable help article | https://grid.helpjuice.com/client-help/what-apis-does-an-open-access-developer-have-access-to |
| [27] | Parsing a Counter-Strike 2 demo | Awpy; mutable project documentation | https://awpy.readthedocs.io/en/latest/examples/parse_demo.html |
| [28] | API Terms of Use | Liquipedia; mutable policy | https://liquipedia.net/api-terms-of-use |
| [29] | Terms and conditions, Article 2.8 | PandaScore; mutable terms | https://www.pandascore.co/terms-and-condition |
