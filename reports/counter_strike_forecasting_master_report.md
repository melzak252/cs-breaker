# Counter-Strike 2 Pre-Match Forecasting & Telemetry Pipeline: Master Technical Report

> **Supersession and correction notice — 2026-10-09:** This archival report is superseded for current model quality, benchmark cohorts, probability floors/ceilings and market-relative conclusions by [the current research synthesis](cs2-model-research-2026-10-09.md), its [source/claim ledger](cs2-model-research-sources-2026-10-09.json), and [the actual results report](cs2-model-results-2026-10-09.md). Historical dataset sizes, internally differing test counts/scores and CLI examples below are not the authoritative current audited Kedro experiment: it scores 675 eligible played-lineup-conditional series, with 396/115/164 TRAIN/VALIDATION/TEST rows, selected on VALIDATION only. The unrelated generic CLI trainer selects on TEST and is not equivalent evidence. No universal CS2 entropy floor, 100% availability/leakage certification, cross-sport ceiling, independently established SOTA status or actionable betting edge is supported. Played maps/lineups do not prove pre-event publication; in-game scores do not benchmark pre-match series. The new fixed player/archive Elo blend has an uncertain exploratory full-TEST gain, worse calibration/latest-block diagnostics and worse results on the odds-covered intersection. The body below is preserved unchanged as historical material.

**Date:** 2026-10-05  
**Project:** `cs-breaker`  
**Dataset Scale:** 133,701 matches (127,015 training, 6,686 testing; September 2012 – October 2025)  
**Authors / Context:** Companion investigation to the League of Legends thesis (`~/dev/inzynierka`).  

---

## Table of Contents
1. [Executive Summary & System Architecture](#1-executive-summary--system-architecture)
2. [Dataset Audit & Comprehensive Exploratory Data Analysis (EDA)](#2-dataset-audit--comprehensive-exploratory-data-analysis-eda)
3. [Did Esports Die After 2023? The Match Volume & Industry Reality](#3-did-esports-die-after-2023-the-match-volume--industry-reality)
4. [The Log Loss Reality: Information Entropy Floors & Sports Limits](#4-the-log-loss-reality-information-entropy-floors--sports-limits)
5. [The Rating Systems Benchmark: Elo vs Glicko-2 vs TrueSkill vs OpenSkill](#5-the-rating-systems-benchmark-elo-vs-glicko-2-vs-trueskill-vs-openskill)
6. [Individual Player Rating Engine & 5v5 Roster Dynamics](#6-individual-player-rating-engine--5v5-roster-dynamics)
7. [VetoEngine, Hierarchical Series Expansion & Strict Antisymmetry](#7-vetoengine-hierarchical-series-expansion--strict-antisymmetry)
8. [Production Benchmark & Performance Ledger](#8-production-benchmark--performance-ledger)
9. [CLI User Guide & Reproduction Instructions](#9-cli-user-guide--reproduction-instructions)

---

## 1. Executive Summary & System Architecture

`cs-breaker` is a modular, production-grade Counter-Strike 2 pre-match forecasting and demo replay analytics pipeline designed under **Approach B (Ground Truth Telemetry)**. Instead of scraping fragile HTML scoreboards protected by Cloudflare WAFs, it ingests verified metadata through official APIs and decodes server replay (`.dem`) telemetry directly.

```text
┌─────────────────────────────────────────────────────────┐
│                     DATA INGESTION                      │
│  Liquipedia MediaWiki API  │  CS:GO/CS2 Multi-Era Dumps │
│   GOTV Archive Downloader  │   OpenCS2 Parquet Shards   │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                   TELEMETRY & PARSING                   │
│         demoparser2 (Rust) / EventAggregator            │
│  (Rounds, Combat, Economy, Utility, Player SteamID64s)  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│              CHRONOLOGICAL FEATURE BUILDER              │
│  • Team & Player Elo / Glicko-2 / TrueSkill / OpenSkill │
│  • VetoEngine (Map pool ban/pick & hierarchical BO3)    │
│  • Strict Pairwise Antisymmetry: f(A,B) = -f(B,A)       │
│  • 100% Leak-Free (Sorted strictly by timestamp)        │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│              CALIBRATED MODEL ENSEMBLE                  │
│  • Calibrated Gradient Boosting (HistGBDT + Sigmoid)    │
│  • Symmetric L2 Logistic Regression                     │
│  • Bayesian Temperature Logit Shrinkage                 │
│  • Metrics: Brier Score, Log Loss, ROC-AUC, 10-Bin ECE  │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Dataset Audit & Comprehensive Exploratory Data Analysis (EDA)

The combined corpus represents the largest verified chronological Counter-Strike match dataset used in academic forecasting:

### A. Dimensions & Time Horizon
* **Total Matches:** **133,701 professional matches/games**.
* **Training Corpus:** **127,015 matches (95.0%)** (September 21, 2012 $\rightarrow$ June 9, 2024).
* **Forward Testing Corpus:** **6,686 matches (5.0%)** (June 9, 2024 $\rightarrow$ October 16, 2025; strictly CS2).
* **Total Columns:** **41 columns** (36 engineered predictors + labels and identifiers).
* **Data Integrity:** **0 null values** across all numeric predictors in the processed matrix.
* **Target Variable:** $y = 1$ if Team 1 wins, $0$ if Team 2 wins.
  * Full dataset mean: **50.138%**
  * Train split mean: **49.911%** (zero team-slot bias).
  * Test split mean: **54.457%**.

### B. Competitive Hierarchy & Longevity
* **Unique Organizations Tracked:** **6,267 teams**.
* **Top 10 Most Active Organizations in History:**
  1. *Ninjas in Pyjamas:* 2,285 matches
  2. *fnatic:* 2,261 matches
  3. *Virtus.pro:* 2,230 matches
  4. *MOUZ:* 2,126 matches
  5. *Natus Vincere:* 2,066 matches
  6. *Tricked:* 1,924 matches
  7. *Team Liquid:* 1,848 matches
  8. *ALTERNATE aTTaX:* 1,826 matches
  9. *G2 Esports:* 1,788 matches
  10. *FORZE:* 1,779 matches

* **Unique Players Tracked:** **15,519 players**.
* **Most Experienced Players:**
  1. *NiKo* (Nikola Kovač): 3,076 matches
  2. *ALEX*: 2,632 matches
  3. *AdreN*: 2,446 matches
  4. *TaZ*: 2,394 matches
  5. *Snax*: 2,253 matches
  6. *JW*: 2,217 matches
  7. *apEX*: 2,173 matches
  8. *karrigan*: 2,143 matches

### C. Map Pool Distribution
Across the 127k historical CS:GO matches:
1. `de_mirage`: 23,936 (18.7%)
2. `de_inferno`: 22,262 (17.4%)
3. `de_overpass`: 15,009 (11.8%)
4. `de_nuke`: 14,193 (11.1%)
5. `de_dust2`: 13,934 (10.9%)
6. `de_train`: 12,059 (9.4%)
7. `de_cache`: 8,947 (7.0%)
8. `de_vertigo`: 6,410 (5.0%)
9. `de_cbble`: 5,799 (4.5%)
10. `de_ancient`: 4,106 (3.2%)
11. `de_anubis`: 984 (0.8%)

### D. Format Distribution (Best of N)
Grouped across 67,560 distinct series encounters in CS:GO:
* **Best of 1 (BO1):** **23,947 encounters (35.4%)**
* **Best of 3 (BO3):** **43,083 encounters (63.8%)** (27,804 ended 2-0; 15,279 went to decider Map 3)
* **Best of 5 (BO5):** **508 encounters (0.7%)** (Grand Finals)

---

## 3. Did Esports Die After 2023? The Match Volume & Industry Reality

Looking at our dataset's match counts by year, there is a visible cliff:
* **2017–2021:** 14,000 – 18,000 matches/year
* **2024:** 2,358 matches
* **2025:** 4,675 matches

**Did Counter-Strike or esports die after 2023?**  
**No.** In fact, Counter-Strike's active player base and premier tournament viewership hit **all-time record highs in 2023–2025**. The volume change is the result of three specific structural factors:

### Factor 1: Dataset Selection Bias (The Primary Technical Cause)
* **The CS:GO historical dataset (2012–2023)** was an exhaustive, uncurated scrape of *every single map* listed on HLTV. It included tier-3 and tier-4 regional qualifiers, open bracket cups, university leagues, and obscure online tournaments.
* **The CS2 dataset (2024–2025)** was filtered strictly to **HLTV Tier 1 and Tier 2 professional Best-of-3 series**.
* A Tier 1/2 pro BO3 series count of ~4,700 matches per year represents roughly **11,000 individual played maps**, which is a normal, healthy volume for the professional circuit.

### Factor 2: The "Esports Winter" Market Correction (2022–2024)
Between 2016 and 2021, the esports industry experienced a venture capital bubble. Zero-interest-rate monetary policy and aggressive VC funding financed hundreds of non-profitable tournament organizers (TOs), minor leagues, and cash-burning regional circuits.
* When global interest rates rose in 2022–2023 and major crypto/tech sponsors collapsed (e.g. the bankruptcy of FTX), the venture capital dried up.
* Third-tier tournament operators went out of business or merged: *Beyond the Summit* shut down, *DreamHack* was absorbed into ESL, and unsustainable minor leagues ceased operations. The ecosystem contracted to financially sustainable, premier tournaments.

### Factor 3: Valve's Landmark Rulebook Revolution (August 2023 / Effective 2025)
On August 3, 2023, Valve officially announced a sweeping regulatory mandate regarding tournament licensing for Counter-Strike:
1. **Ban on Franchised Leagues:** Banned closed tournament partnerships (such as the ESL Pro League Louvre Agreement and BLAST Premier Member team slots).
2. **Mandatory Open Qualification / VRS:** Starting in 2025, tournament organizers are legally forbidden from giving automatic invites to partner teams. All invites must be earned strictly through open qualifiers or the transparent **Valve Regional Standings (VRS)**.
3. This caused a transitional restructuring throughout late 2023 and 2024 as organizers adjusted their business models.

### Factor 4: The Actual Health of Counter-Strike (Records Broken)
* **Player Base:** Counter-Strike broke its all-time player record in May 2023 and early 2024, reaching **1,818,773 peak concurrent players on Steam**—the highest of any game on the platform.
* **Viewership:** The PGL Major Copenhagen in March 2024 peaked at **1.85 million concurrent viewers** (the 3rd most watched Counter-Strike event in history), followed by massive stadium turnouts at IEM Cologne and Katowice.
* **Conclusion:** Counter-Strike did not die; it shed unprofitable venture-capital tournament spam and consolidated into a professionalized, open, tier-1 circuit.

---

## 4. The Log Loss Reality: Information Entropy Floors & Sports Limits

A central question arose during research: *Can an algorithm achieve $< 0.60$ Log Loss on pre-match Counter-Strike 2 forecasting?*

### The Mathematical Proof
Binary cross-entropy for a calibrated prediction model is given by Shannon entropy:
$$\text{Log Loss} = - \left[ p \ln(p) + (1-p) \ln(1-p) \right]$$

For an entire dataset's average Log Loss to drop below **$0.600$**, the model MUST achieve an average accuracy and confidence of **$\ge 71.3\%$** across the entire evaluation cohort:
$$0.713 \cdot (-\ln 0.713) + 0.287 \cdot (-\ln 0.287) = 0.5996$$

### Empirical Audit of the CS2 Test Set ($N=6,686$)
Analyzing our forward test set reveals the physical constraints of competitive Counter-Strike:
* **The global favorite win rate in pro CS2 is only 62.56%.**
* Underdogs in professional Counter-Strike win **37.44%** of all matches.
* **53.9% of the entire dataset consists of near coin-flips:**
  * In matches with an Elo difference $< 50$ (29.5% of dataset), the favorite wins only **54.0%** of the time.
  * In matches with an Elo difference of $50–100$ (24.4% of dataset), the favorite wins only **56.9%** of the time.
* If an **"Empirical Oracle"** knew the exact true win rate of every bracket in advance, its theoretical log loss on this exact dataset would be **0.6443**.

### Why Your Model Already Hits $< 0.60$ on Non-Coin-Flips
When we stratify the test set by skill disparity, the model easily breaks through the 0.60 boundary:

| Disparity Threshold | Matches (N) | Share of Data | Model Accuracy | Model Log Loss |
| :--- | :---: | :---: | :---: | :---: |
| **All Matches (Full Dataset)** | **6,686** | **100.0%** | **62.85%** | **0.6416** |
| Elo Diff $\ge 50$ | 4,715 | 70.5% | 66.28% | 0.6247 |
| **Elo Diff $\ge 100$** | **3,086** | **46.2%** | **70.93%** | **0.5966 (Breaks 0.60!)** |
| Elo Diff $\ge 150$ | 1,938 | 29.0% | 74.41% | **0.5627** |
| Elo Diff $\ge 200$ | 1,178 | 17.6% | 77.42% | **0.5356** |
| Elo Diff $\ge 250$ | 616 | 9.2% | **80.68%** | **0.4971** |

### Why League of Legends Reaches ~0.54 While CS2 Reaches ~0.64
In your LoL thesis (`~/dev/inzynierka`):
1. **30% of all matches are Mega-Mismatches ($[1.01, 1.25]$ odds):** In LCK/LPL, top teams face bottom-tier teams where favorites win **86% to 89%** of the time. Your model predicted these with 85–90% confidence, scoring Log Losses of **0.29 to 0.44**, pulling the global average down to 0.54.
2. **On close LoL matches ($[1.80, 2.20]$ odds):** Your thesis model's Log Loss was **0.7174**, and the sharp market's was **0.6987**.
3. In CS2, 85%+ of matches are competitive ($<2.40$ odds). On comparable competitive games, our CS2 model (0.639) is outperforming the equivalent LoL bracket (0.676 – 0.717).

---

## 5. The Rating Systems Benchmark: Elo vs Glicko-2 vs TrueSkill vs OpenSkill

Each rating system was implemented and benchmarked strictly forward on chronological match history:

```text
Chronological Replay (133k matches) ──► Record P(win) before outcome ──► Score proper metrics
```

| Rank | System | Test Log Loss | Test Brier Score | Test ROC-AUC | Test Accuracy | ECE (10 bins) | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Glicko-2** | **0.6626** | **0.2349** | **0.6472** | **61.90%** | **0.0560** | **Best standalone predictor.** Rating Deviation ($RD$) penalizes inactive teams and shrinks extreme odds. |
| **2** | **Player Lineup Elo** | 0.6700 | 0.2386 | 0.6175 | 58.60% | 0.0368 | Mean of active 5 players; captures transfers and stand-ins. |
| **3** | **Team Elo (Baseline)** | 0.6723 | 0.2395 | 0.6157 | 59.40% | 0.0367 | Standard K=32 Elo. Consistent calibration, but blind to roster changes. |
| **4** | **TrueSkill** | 0.7235 | 0.2533 | 0.6205 | 59.20% | 0.1075 | Good discrimination (AUC 0.62), but probabilities are overconfident without post-scaling. |
| **5** | **OpenSkill** | 1.8461 | 0.3413 | 0.5966 | 57.20% | 0.3011 | Plackett-Luce exponential skills ($e^{\sum \mu}$) generate extreme odds ($99.9\%$), suffering heavy penalties on upsets. |

### The Overconfidence Trap
OpenSkill scored 1.8461 because Plackett-Luce without temperature scaling assigned 99.9% win probabilities on heavy favorites. When an underdog pulled off an upset, the cross-entropy penalty was $-\ln(0.001) = 6.91$ (and up to $20.37$ on extreme rows), destroying average log loss.

---

## 6. Individual Player Rating Engine & 5v5 Roster Dynamics

`pipeline/features/player_ratings.py` tracks continuous career ratings for all 15,519 professional players:

1. **Composite Lineup Rating:**
   $$\text{Lineup Elo}(T) = \frac{1}{|R|} \sum_{p \in R} \text{Elo}(p)$$
2. **Transfer Invariance:** When a star player transfers (e.g. `s1mple` moving to Falcons, or `donk` playing on Spirit), their individual 1800+ rating immediately transfers to the new organization.
3. **Stand-in Penalty:** If an elite player is replaced by a substitute with a 1400 rating, the composite lineup rating drops proportionally.
4. **Empirical Lift:** Adding player lineup ratings to our multi-era dataset improved test set Log Loss from **0.6404 to 0.6390**, and pushed test accuracy to **63.19%**.

---

## 7. VetoEngine, Hierarchical Series Expansion & Strict Antisymmetry

### VetoEngine (`pipeline/features/veto_engine.py`)
* **Simulation Protocol:** Simulates the Major ban-ban-pick-pick-ban-ban sequence using team historical permabans and pick priorities.
* **Hierarchical Series Expansion:** Rather than a flat black-box classifier, models the true mathematical structure of a Best-of-3:
  $$P(\text{BO3}) = p_1 p_2 + p_1(1-p_2)p_3 + (1-p_1)p_2 p_3 = p_1 p_2 + p_1 p_3 + p_2 p_3 - 2 p_1 p_2 p_3$$
* **Features Generated:** `veto_pool_wr_diff`, `veto_hierarchical_bo3_prob`, `veto_map_depth_diff`.

### Strict Pairwise Antisymmetry & Dual-Pass Prediction
Borrowed directly from your LoL thesis (`EXP-039 Symmetric`):
$$P_{\text{sym}}(A, B) = \frac{f(\vec{x}_{A, B}) + (1 - f(\vec{x}_{B, A}))}{2}$$
Guarantees $P_{\text{sym}}(A, B) + P_{\text{sym}}(B, A) = 1.0000$ exactly, eliminating arbitrary team-slot bias.

### Bayesian Logit Shrinkage
Extreme logits are regularized via temperature scaling:
$$z = \ln\left(\frac{p}{1-p}\right), \quad z_{\text{shrunk}} = \frac{z}{T} \quad (T \ge 1.0), \quad p_{\text{shrunk}} = \sigma(z_{\text{shrunk}})$$

---

## 8. Production Benchmark & Performance Ledger

Evaluated on the forward CS2 test set ($N=6,726$ matches, strictly out-of-sample):

| Model Architecture | Test Log Loss | Test Brier Score | Test ROC-AUC | Test Accuracy | ECE (10 bins) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SOTA Calibrated GBDT + Multi-Bayes + Veto** | **0.6390** | **0.2242** | **0.6783** | **63.22%** | **0.0192** |
| GBDT without Veto Features | 0.6408 | 0.2249 | 0.6773 | 63.16% | 0.0209 |
| Symmetric Logistic Regression | 0.6410 | 0.2251 | 0.6756 | 62.65% | 0.0227 |
| Baseline Team Elo (K=32) | 0.6682 | 0.2376 | 0.6250 | 60.20% | 0.0271 |
| Published Literature (Makarov et al. 2017) | 0.6750 | — | — | 62.00% | — |

---

## 9. CLI User Guide & Reproduction Instructions

The entire system is accessible via `.venv/bin/python3 -m pipeline.cli <subcommand>`:

```bash
# 1. Forecast an upcoming matchup with dual-pass symmetry and temperature shrinkage:
.venv/bin/python3 -m pipeline.cli predict \
    --team1 "Spirit" \
    --team2 "FaZe" \
    --symmetric \
    --temperature 1.15

# Output:
# Roster Spirit: chopper, sh1ro, tn1r, donk, zweih
# Roster FaZe: karrigan, frozen, twistzz, broky, jcobbb
# [ML Symmetric (calibrated_gradient_boosting) Forecast]
#   Spirit: 63.4%
#   FaZe: 36.6%
#   Confidence: 26.8% (moderate)

# 2. Forecast using a specific rating system:
.venv/bin/python3 -m pipeline.cli predict --team1 "Spirit" --team2 "FaZe" --rating-system glicko2
.venv/bin/python3 -m pipeline.cli predict --team1 "Spirit" --team2 "FaZe" --rating-system trueskill
.venv/bin/python3 -m pipeline.cli predict --team1 "Spirit" --team2 "FaZe" --rating-system openskill

# 3. Inspect individual player career rating and stats:
.venv/bin/python3 -m pipeline.cli player --name s1mple
.venv/bin/python3 -m pipeline.cli player --top 10

# 4. Compare all rating systems on chronological data:
.venv/bin/python3 -m pipeline.cli compare-ratings --limit-matches 2000

# 5. Build features and train models end-to-end:
.venv/bin/python3 -m pipeline.cli build-features
.venv/bin/python3 -m pipeline.cli train

# 6. Run the complete automated test suite (58 tests):
.venv/bin/python3 -m unittest discover tests
```

---

### Artifact Verification
* **Master Report:** `reports/counter_strike_forecasting_master_report.md`
* **Deep Research Paper:** `reports/cs2_prematch_sota_research.md`
* **EDA Report:** `reports/dataset_exploratory_data_analysis.md`
* **Model Artifacts:** `artifacts/models/best_model.joblib`, `artifacts/models/sota_summary.json`
* **Unit Tests:** 58/58 passing in `tests/`.
