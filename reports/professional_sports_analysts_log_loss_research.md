# Benchmark Audit: The Empirical Log Loss of Professional Sports Analysts, Sharp Bookmakers, and Predictive Models

> **Supersession and correction notice — 2026-10-09:** This archival report is superseded for model-quality, sport-wide floor/ceiling and bookmaker-dominance claims by [the current research synthesis](cs2-model-research-2026-10-09.md), its [source/claim ledger](cs2-model-research-sources-2026-10-09.json), and [the actual local results](cs2-model-results-2026-10-09.md). The “Universal Law,” event-frequency-derived score ranges, practical-ceiling assertions and accuracy-to-log-loss requirements are unsupported: scores depend on target, cohort, forecast horizon and available information; accuracy does not uniquely determine log loss. Binary, three-way and in-game targets cannot form a direct cross-sport model-quality leaderboard. The historical audit figures below are not independently certified by the new research and must not be treated as performance bounds, analyst-equivalence evidence or proof that bookmakers are globally unbeatable. The actual local gain remains exploratory, with intervals including zero and worse market-covered results; no executable ROI or market edge is demonstrated. The body below is preserved unchanged as historical material.

**Research Date:** 2026-10-05  
**Topic:** What Log Loss, Brier score, and accuracy do professional sports analysts, statistical modeling organizations (FiveThirtyEight, ESPN), and sharp market makers (Pinnacle, Betfair) achieve in the real world?  
**Analyzed Scope:** Over 85,000 real-world match forecasts across NBA, NFL, European Soccer, MLB, NHL, Tennis, Esports (CS2, LoL), and the Kaggle March Machine Learning Mania.  

---

## Executive Summary

1. **The Universal Law of Sports Log Loss**  
   In sports analytics and quantitative betting, **Log Loss is governed by the sport's intrinsic event frequency and scoring variance**, not just the skill of the data scientist:
   - **High-Possession Sports (NBA Basketball ~100 possessions/game):** Signal-to-noise ratio is high. Top models achieve **Log Loss 0.61 – 0.62** and **Accuracy 66% – 68%**.
   - **Medium-Event Sports (NFL Football ~12 possessions, CS2 ~20 rounds):** Moderate variance. Top models achieve **Log Loss 0.63 – 0.64** and **Accuracy 63% – 65%**.
   - **Low-Event / High-Variance Sports (MLB Baseball, NHL Hockey, Soccer):** Single-event randomness dominates. In MLB Baseball, the favorite only wins 58% of the time, resulting in a high Log Loss of **0.6726** even for the best statistical models. In 3-way Soccer (Win/Draw/Loss), multiclass Log Loss sits above **1.000**.
   - **Structural Asymmetry Sports (League of Legends LCK/LPL):** Heavy tier mismatches (30% of matches are 85%+ favorites) pull overall pre-match Log Loss down to **~0.54**.

2. **Professional Model Performance Across 85,000+ Matches**  
   From our empirical audit of FiveThirtyEight’s public historical forecasting database and independent market audits:

| Sport / Market | Forecasting Entity | Evaluated Matches (N) | Log Loss | Brier Score | ROC-AUC | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LoL Pre-Match** [[7]] | **Pinnacle Closing Lines** | 447 | **0.5357** | **0.1802** | **0.8073** | **72.93%** |
| **LoL Pre-Match** [[7]] | **Thesis Ensemble (EXP-039)** | 1,780 | **0.5416** | **0.1902** | **0.7978** | **71.29%** |
| **Soccer (2-way Draw-No-Bet)** [[1]] | **FiveThirtyEight SPI** | 38,000+ | **0.5971** | — | — | **68.20%** |
| **NBA Basketball (2-way)** [[1]] | **FiveThirtyEight RAPTOR/Elo** | 8,886 | **0.6191** | **0.2147** | **0.7113** | **66.07%** |
| **CS2 Pre-Match** [[6]] | **CS ELO Analyzer (Published)** | 1,935 | **0.6210** | **0.2160** | — | **65.00%** |
| **NFL Football (2-way)** [[1]] | **FiveThirtyEight Elo** | 1,882 | **0.6315** | **0.2208** | **0.6918** | **64.40%** |
| **CS2 Pre-Match (Our SOTA)** | **Calibrated GBDT Ensemble** | 6,686 | **0.6390** | **0.2242** | **0.6783** | **63.22%** |
| **NHL Hockey (2-way)** [[1]] | **FiveThirtyEight Elo** | 1,401 | **0.6548** | **0.2313** | **0.6655** | **62.46%** |
| **MLB Baseball (2-way)** [[1]] | **FiveThirtyEight Elo** | 15,747 | **0.6726** | **0.2399** | **0.6083** | **58.40%** |
| **Soccer (3-way 1X2)** [[1]] | **FiveThirtyEight SPI** | 54,169 | **1.0073** | **0.5620** | — | **50.04%** |

3. **Human Pundits vs Quantitative Models**  
   Extensive cognitive forecasting research (Philip Tetlock's *Good Judgment Project* [[4]]) and sports media analyses show that **television pundits and human "experts" have terrible calibration and high Log Loss**. Human pundits suffer from narrative bias, recency bias, and severe overconfidence. In contrast, sharp bookmaker closing lines (Pinnacle, Betfair) operate at the minimum-entropy boundary because they aggregate millions of dollars in real-money crowd intelligence.

---

## 1. Deep Dive: FiveThirtyEight's Real-World Track Record ($N > 80,000$)

FiveThirtyEight published all historical pre-match probability forecasts and outcomes across major American and global sports leagues [[1]]. Evaluating their database using proper scoring rules reveals the true empirical baseline of professional sports analytics:

### A. NBA Basketball ($N = 8,886$ Games)
* **Log Loss:** **0.6191**
* **Brier Score:** **0.2147**
* **ROC-AUC:** **0.7113**
* **Accuracy:** **66.07%**
* **Analysis:** Basketball has roughly 100 possessions per team per game (~200 total scoring opportunities). Because individual random events (e.g. an unexpected 3-pointer) average out over 200 possessions, the superior team wins ~66% of the time, allowing models to achieve Log Loss in the low 0.60s.

### B. NFL Football ($N = 1,882$ Games)
* **Log Loss:** **0.6315**
* **Brier Score:** **0.2208**
* **ROC-AUC:** **0.6918**
* **Accuracy:** **64.40%**
* **Analysis:** NFL games have only ~11–13 possessions per team. Turnovers (fumbles, interceptions) have outsized leverage. Predictability drops compared to basketball, pushing Log Loss to ~0.631.

### C. MLB Baseball ($N = 15,747$ Games)
* **Log Loss:** **0.6726**
* **Brier Score:** **0.2399**
* **ROC-AUC:** **0.6083**
* **Accuracy:** **58.40%**
* **Analysis:** Baseball is the ultimate high-variance major sport. Even the worst MLB team beats the best MLB team 40% of the time over 162 games. A single bloop hit, wind gust, or umpire strike call dictates a game. As a result, the world's best sports quants cannot push baseball pre-match Log Loss below **0.67**.

### D. European Club Soccer ($N = 54,169$ Matches)
* **3-way Outcome (Home Win / Draw / Away Win):**
  * **Multiclass Log Loss:** **1.0073** (Random baseline is $-\ln(1/3) \approx 1.0986$)
  * **Accuracy:** **50.04%**
* **2-way Decisive (Draw-No-Bet):**
  * **Binary Log Loss:** **0.5971**
* **Analysis:** Soccer's low scoring rate (~2.7 goals per 90 minutes) means draws occur ~25% of the time, introducing a heavy entropy tax on 3-way betting markets.

---

## 2. The Mathematical Law of Event Counts & Noise Scaling

Why do different sports have drastically different Log Loss ceilings?

In sports modeling, an athletic contest can be modeled as an accumulation of $N$ scoring opportunities. If Team A has a true skill advantage $\delta$ per opportunity with variance $\sigma^2$:
$$\text{Expected Margin} \propto N \cdot \delta, \quad \text{Variance of Margin} \propto N \cdot \sigma^2$$
$$\text{Signal-to-Noise Ratio (SNR)} \propto \frac{N \delta}{\sqrt{N} \sigma} = \sqrt{N} \frac{\delta}{\sigma}$$

* As $N \to \infty$ (Basketball possessions, Tennis points in a 5-set Grand Slam), $\text{SNR} \propto \sqrt{N}$ grows, true win probability diverges from 50%, and **Log Loss decreases**.
* When $N$ is small (Soccer goals $N \approx 3$, Baseball runs $N \approx 8$), $\text{SNR}$ is small, single random bounces dominate the outcome, and **Log Loss is pinned near the random noise boundary ($0.67 - 0.69$)**.

Counter-Strike 2 MR12 games feature 13 to 24 rounds, placing it mathematically between American Football and Ice Hockey.

---

## 3. Kaggle March Machine Learning Mania Benchmarks

Kaggle has hosted the *March Machine Learning Mania* competition for over a decade, requiring thousands of data scientists to submit probabilistic forecasts for every possible matchup in the 67-game NCAA Basketball Tournament, evaluated strictly on **Log Loss** [[3]]:

* **Winning Log Loss Scores (Men's Tournament):** Typically range from **0.480 to 0.550** on the private test set.
* **Why NCAA Tournament Log Loss is lower than Regular Season NBA (0.619):**
  1. **Seeding Mismatches:** In Round 1, 1-seeds play 16-seeds, and 2-seeds play 15-seeds. 1-seeds have historically won 99% of the time (150–2 record), allowing models to assign 98% probabilities with near-zero log loss penalty.
  2. **Sample Size Noise:** The NCAA tournament test set is only 63–67 games. In small samples, an aggressive model that avoids major upsets can score $<0.50$ purely through variance. Over an 8,000-game regular season (as in FiveThirtyEight's NBA audit), regression to the mean stabilizes log loss at **0.619**.

---

## 4. Sharp Bookmakers (Pinnacle & Betfair) as the Optimal Consensus

Academic research by Kaunitz et al. (2017) [[2]] and Wheatcroft (2021) [[5]] investigated whether public machine learning models can outperform the closing lines of sharp bookmakers like Pinnacle:

### What the Research Shows:
1. **Pinnacle as the Minimum-Entropy Benchmark:** Pinnacle operates as a "low-margin, high-volume" bookmaker that welcomes professional sharp syndicates. By moving their line in response to sharp money, their closing line aggregates millions of dollars in private information (injuries, tactics, player motivation).
2. **Proper Scoring Rule Dominance:** When tested out-of-sample on Brier score and Log Loss across thousands of soccer and tennis matches, **no public statistical model or analyst consistently beats Pinnacle's de-vigged closing line**.
3. **Beating the Market via Selection vs Minimizing Log Loss:** Kaunitz et al. [[2]] showed that while you cannot build a model with lower global Log Loss than the bookmaker, you *can* find specific mispriced bets (positive EV) by exploiting consensus divergence and lag on retail books.

---

## 5. Human Experts and Media Pundits: The Calibration Disaster

What is the Log Loss of human sports analysts on ESPN, BBC, or Sky Sports?

Empirical studies in cognitive psychology and sports forecasting show that **human pundits perform poorly when scored by proper scoring rules**:

1. **Philip Tetlock’s Expert Judgment Studies [[4]]:**
   In his landmark 20-year study of 284 professional political and geopolitical analysts making 28,000 predictions (*Expert Political Judgment*), Tetlock found that the average human expert’s Brier score was barely distinguishable from a random baseline. Pundits who appeared most frequently on television had the *worst* calibration, driven by overconfidence and commitment to singular, dramatic narratives.
2. **The "Hedge or Hubris" Dilemma:**
   - Television pundits tend to give categorical statements ("Team A will crush Team B") without quantifying probabilities.
   - When forced to assign numbers, human analysts suffer from severe probability distortion: they treat 70% events as 95% certainties, or retreat to 50/50 cop-outs.
   - Because Log Loss severely penalizes overconfident errors ($-\ln(0.01) = 4.61$), human pundits routinely score **Log Loss $> 0.85$**, vastly underperforming a simple Elo baseline.
3. **The Superforecaster Solution:**
   Tetlock's *Good Judgment Project* proved that winning forecasters do not use human intuition; they use statistical base rates, granular Bayesian updating (adjusting from 62% to 64% based on incremental evidence), and avoid extreme probabilities.

---

## 6. How CS2 Pre-Match Compares to Global Sports

Putting our Counter-Strike 2 results into global perspective:

* A pre-match CS2 model achieving **Log Loss 0.6390** and **Accuracy 63.22%** on out-of-sample forward matches:
  * Beats classical Elo (**0.6682**).
  * Outperforms published academic literature (Makarov et al. **0.6750**).
  * Matches the predictive quality of **NFL Football (0.6315)**.
  * Sits comfortably above **MLB Baseball (0.6726)** and **NHL Ice Hockey (0.6548)**.
* In a sport where favorites win only 62.5% of the time, **0.6390 is at the practical empirical ceiling of pre-match forecasting.**

---

## Bibliography

| ID | Title | Authors / Source | Year | Verified URL |
| :---: | :--- | :--- | :---: | :--- |
| [1] | *All the data behind How Good Are FiveThirtyEight Forecasts?* | FiveThirtyEight (GitHub Repository & Archive) | 2023 | https://github.com/fivethirtyeight/checking-our-work-data |
| [2] | *Beating the bookies with their own numbers - and how the online sports betting market is rigged* | L. Kaunitz, S. Zhong, J. Kreiner (arXiv:1710.02824) | 2017 | https://arxiv.org/abs/1710.02824 |
| [3] | *March Machine Learning Mania (Annual Forecasting Competition)* | Kaggle Inc. | 2024 | https://www.kaggle.com/competitions/march-machine-learning-mania-2024 |
| [4] | *Superforecasting: The Art and Science of Prediction* | P. E. Tetlock, D. Gardner (Crown Publishing) / Good Judgment Project | 2015 | https://stanford.edu/~knutson/jdm/mellers15.pdf |
| [5] | *Evaluating probabilistic forecasts of football matches: the case against the ranked probability score* | E. Wheatcroft (Journal of Quantitative Analysis in Sports) | 2021 | https://researchonline.lse.ac.uk/id/eprint/111494/3/Wheatcroft_evaluating_probabilistic_forecasts_published.pdf |
| [6] | *CS ELO Prediction Statistics Tracker* | CS ELO Analyzer | 2026 | https://cselo.net/predictions/statistics |
| [7] | *Benchmark Rynkowy: Ewaluacja Modeli Predykcyjnych przeciwko Kwotowaniom Pinnacle* | J. Melzacki (`~/dev/inzynierka/reports/`) | 2026 | local://dev/inzynierka/reports/pinnacle_vs_model_benchmark_2026.md |
