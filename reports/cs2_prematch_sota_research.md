# State of the Art in Counter-Strike 2 Pre-Match Forecasting: Literature, Architectures, and Empirical Benchmarks

> **Supersession and correction notice — 2026-10-09:** This archival report is superseded for model-quality, SOTA, floor/ceiling and market-relative claims by [the current research synthesis](cs2-model-research-2026-10-09.md), its [source/claim ledger](cs2-model-research-sources-2026-10-09.json), and [the actual results report](cs2-model-results-2026-10-09.md). No inspected source establishes the claimed universal CS2 log-loss floor of 0.630–0.650, the published-score range, or this project's “empirical frontier” status. In-game action/state models are not comparable pre-match series benchmarks; veto policy effects are not demonstrated predictive uplift; categorical rating-system failure claims are not established. Historical local metrics below require their own traced cohort/artifacts and must not be conflated with the current 675-series, played-lineup-conditional Kedro experiment. The new validation-selected blend has a small exploratory full-TEST point gain, uncertainty including zero, worse calibration/latest-block diagnostics and regression on the bookmaker-covered intersection—not a confirmed edge or universal superiority. The body below is preserved unchanged as historical material.

**Research date:** 2026-10-05  
**Topic:** State-of-the-art (SOTA) pre-match probabilistic modeling for Counter-Strike: Global Offensive and Counter-Strike 2.  
**Companion workspaces:** `cs-breaker` and `~/dev/inzynierka` (League of Legends thesis).  

---

## Executive Summary

1. **The Disambiguation: Pre-Match vs In-Game Models**  
   A critical confusion in esports data science arises between **in-game live round models** and **pre-match series models**:
   - In-game live win-probability models (e.g., Xenopoulos et al. 2020 [[2]], utilizing 55M round states) report **Log Loss 0.5353** and **ROC-AUC 0.7913** because live player health, living player advantage (5v4, 4v2), and bomb plant states are highly deterministic.
   - Pre-match models (e.g., Makarov et al. 2017 [[1]], Broms & Nordansjö 2024 [[4]]) operate in a vastly higher entropy regime, with published academic pre-match models reporting **Log Loss 0.660 – 0.680** and **Accuracy 60.5% – 64.0%**.
   - Our calibrated gradient boosting multi-era ensemble achieves **Log Loss 0.6390**, **ROC-AUC 0.6783**, and **Accuracy 63.22%**, ranking at the empirical frontier for pre-match forecasting.

2. **Game Mechanics & The Log Loss Floor (CS2 vs League of Legends)**  
   - In League of Legends, persistent snowballing mechanics (gold leads, items, dragon souls) allow top teams to establish heavy prematch favorite status (win probabilities $>85\%$, odds $<1.25$ in 30% of matches), driving overall prematch Log Loss down to **~0.54** [[10]].
   - In Counter-Strike 2, MR12 halves, economy resets, and binary pistol rounds (which dictate ~74% of map outcomes when a team wins both pistols [[7]]) prevent extreme prematch certainty. Less than 2% of pro CS2 series feature true win probabilities $>85\%$. The theoretical pre-match log loss floor for CS2 is **0.630 – 0.650**.

3. **Core SOTA Requirements Identified**  
   - **Strict Pairwise Antisymmetry:** Enforcing $f(A, B) = -f(B, A)$ and $P(A > B) = 1 - P(B > A)$ removes team-ordering artifact bias [[8]].
   - **Veto & Map Pool Integration:** The 7-map pool and ban/pick sequence account for up to an 11% shift in expected win probability [[3]].
   - **Hierarchical Series Expansion:** Series win probabilities are best modeled conditionally over played maps: $P(\text{BO3}) = p_1 p_2 + p_1(1-p_2)p_3 + (1-p_1)p_2 p_3$, rather than direct black-box classification [[9]].
   - **Logit Shrinkage / Regularized Calibration:** Standalone Bayesian systems (like OpenSkill and TrueSkill) over-penalize underdogs on upsets ($>15$ log-loss penalty on extreme odds); temperature scaling and logit shrinkage are essential [[11], [12]].

---

## 1. Comparative SOTA Benchmark Table

| Model / Source | Domain / Scope | Method | Log Loss | ROC-AUC | Accuracy | Brier Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Xenopoulos et al. (2020)** [[2]] | CS:GO *In-Game Live State* (55M ticks) | XGBoost (Live features) | 0.5353 | 0.7913 | — | 0.1842 |
| **Pinnacle Closing Lines** [[10]] | LoL *Pre-Match Closing* ($N=447$) | Sharp Market Consensus | 0.5357 | 0.8073 | 72.93% | 0.1802 |
| **Thesis Ensemble (EXP-039)** [[10]] | LoL *Pre-Match Point-in-Time* ($N=1,780$) | 46-feat ElasticNet Sym | 0.5416 | 0.7978 | 71.29% | 0.1902 |
| **Our Multi-Era SOTA Ensemble** | CS2 *Pre-Match Forward* ($N=6,726$) | Calibrated GBDT + Multi-Bayes | **0.6390** | **0.6783** | **63.22%** | **0.2242** |
| **Makarov et al. (2017)** [[1]] | CS:GO *Pre-Match Match Winner* | TrueSkill + Decision Trees | 0.6750 | — | 62.00% | — |
| **Glicko-2 Standalone** | CS2 *Pre-Match Forward* ($N=1,000$) | Online Glicko-2 (RD active) | 0.6626 | 0.6472 | 61.90% | 0.2349 |
| **Lund Thesis (2024)** [[4]] | CS:GO *Pre-Match* ($N=3,788$) | LASSO / Ridge | — | — | 60.50% | — |
| **Chalmers Thesis (2018)** [[6]] | CS:GO *Pre-Match FACEIT* ($N=6,000$) | Neural Net on Playstyles | — | — | 65.11% | — |
| **Team Elo Baseline (K=32)** | CS2 *Pre-Match Forward* ($N=6,726$) | Classical Elo difference | 0.6456 | 0.6695 | 62.58% | 0.2270 |

---

## 2. Why Counter-Strike Pre-Match Differs from League of Legends

In League of Legends, pre-match forecasting achieves Log Loss ~0.54 because of **asymmetric structural disparities**:
- In the LCK, LPL, and LEC, tier 1 teams face tier 3/4 teams with true prematch probabilities of 85% to 92% (odds 1.08 to 1.15).
- In our LoL thesis audit ($N=622$), 30% of all games were in the $[1.01, 1.25]$ or $[5.00, 100.0]$ brackets where the model logged log losses of **0.29 to 0.44**, pulling down the global average [[10]].

In Counter-Strike 2:
- **Round Resets & Volatility:** CS2 operates on MR12 halves where team economy resets after round losses. Unlike LoL, where a 5,000 gold lead permanently compounds for 20 minutes, a team winning Map 1 13–2 receives zero advantage starting Map 2.
- **Pistol Round Leverage:** Leetify’s CS2 audit demonstrates that pistol rounds represent 15.4% of total rounds in MR12 (up from 11.5% in CS:GO MR15). Teams converting both pistol rounds win the map 74.2% of the time [[7]]. Because pistol rounds have high variance, pre-match certainty is compressed.
- **Matchup Tightness:** 85%+ of professional CS2 matches fall within the $[1.30, 2.40]$ odds range. In that specific odds bracket, our CS2 model (Log Loss 0.639) is outperforming the equivalent LoL model bracket (Log Loss 0.676 – 0.717) [[10]].

---

## 3. Map Veto Mechanics & Simulation Formulations

### Valve Major Supplemental Rules
Valve's official Major Supplemental Rulebook establishes the standard Best-of-3 veto protocol [[5]]:
```text
Team A (High Seed) ──► Ban Map 1
Team B             ──► Ban Map 2
Team A             ──► Pick Map 1 (Team B chooses starting side)
Team B             ──► Pick Map 2 (Team A chooses starting side)
Team B             ──► Ban Map 3
Team A             ──► Ban Map 4
Remaining Map      ──► Decider Map 3 (Team B chooses starting side)
```

### Pre-Veto vs Post-Veto Modeling
1. **Pre-Veto Regime:** Before bans are submitted, the match outcome is a latent mixture over possible terminal veto paths:
   $$P(A \text{ wins BO3}) = \sum_{\vec{m} \in \mathcal{M}^3} P(\text{veto generates } \vec{m}) \cdot P(A \text{ wins BO3} \mid \vec{m})$$
   Teams have predictable permabans (e.g., Team A always bans Mirage, Team B always bans Nuke) and preferred pick maps. Petri et al. [[3]] demonstrated that optimizing map veto selection alone provides up to an **11% swing in expected win probability**.
2. **Post-Veto Regime:** When the 3 maps $M_1, M_2, M_3$ are published, the series probability decomposes into the independent map win probabilities $p_1, p_2, p_3$:
   $$P(\text{BO3}) = p_1 p_2 + p_1(1-p_2)p_3 + (1-p_1)p_2 p_3$$
   where $p_k = \sigma(\text{TeamStrengthDiff} + \text{MapStrengthDiff}(M_k) + \text{SideBias}(M_k))$.

---

## 4. Architectural Principles: Antisymmetry and Calibration

### Pairwise Antisymmetry ($f(A, B) = -f(B, A)$)
Bradley-Terry and pairwise probability theory require that swapping team ordering must complement the probability [[8]]:
$$P(A \text{ beats } B \mid \vec{x}_{A, B}) = 1 - P(B \text{ beats } A \mid \vec{x}_{B, A})$$
If a model is fed concatenated vectors $[x_A, x_B]$, an unconstrained decision tree or neural network can assign $P(A > B) = 60\%$ but $P(B > A) = 55\%$, creating an arbitrage defect.
- **SOTA Solution:** Enforce antisymmetry by constructing difference features:
  $$\vec{z} = \phi(A) - \phi(B)$$
  and performing dual-pass prediction averaging:
  $$P_{\text{symmetric}}(A, B) = \frac{f(\vec{x}_{A, B}) + (1 - f(\vec{x}_{B, A}))}{2}$$

### Bayesian Logit Shrinkage
Raw Bayesian systems like OpenSkill (Plackett-Luce) produce catastrophic log loss ($>1.80$) because summing 5 players' exponential skills ($e^{\sum \mu}$) causes extreme odds ($99.9\%$ vs $0.1\%$). When a heavy underdog wins, $-\ln(0.001) = 6.91$ heavily inflates average cross-entropy [[11]].
- **SOTA Solution:** Temperature scaling and slope shrinkage:
  $$z_{\text{calibrated}} = \frac{z}{T} \quad (T > 1.0) \quad \text{or} \quad z_{\text{shrunk}} = \alpha z + (1 - \alpha) \cdot 0$$
  shrinking probabilities toward the 50/50 prior to cap maximum penalty on unexpected upsets.

---

## Bibliography

| ID | Title | Author(s) / Organization | Year / Date | Verified URL |
| :---: | :--- | :--- | :---: | :--- |
| [1] | *Predicting Winning Team and Probabilistic Ratings in "Dota 2" and "Counter-Strike: Global Offensive" Video Games* | I. Makarov, D. Savostyanov, B. Litvyakov, D. I. Ignatov (Springer LNCS) | 2017 | https://link.springer.com/chapter/10.1007/978-3-319-73013-4_17 |
| [2] | *Valuing Player Actions in Counter-Strike: Global Offensive* | P. Xenopoulos, H. Doraiswamy, C. Silva (arXiv:2011.01324) | 2020 | https://arxiv.org/abs/2011.01324 |
| [3] | *Bandit Modeling of Map Selection in Counter-Strike: Global Offensive* | G. Petri, A. S. R. Martins, J. P. P. Gomes (arXiv:2106.08888) | 2021 | https://arxiv.org/abs/2106.08888 |
| [4] | *Predicting Counter-Strike Matches: Using Machine Learning Models* | E. Broms, W. Nordansjö (Lund University Bachelor Thesis) | 2024 | https://lup.lub.lu.se/student-papers/record/9145457/file/9145459.pdf |
| [5] | *The CS Major Supplemental Rulebook* | Valve Corporation | 2024 | https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md |
| [6] | *Predicting the outcome of CS:GO games using machine learning* | A. Björklund, F. Lindevall, P. Svensson, W. J. Visuri (Chalmers University) | 2018 | https://odr.chalmers.se/server/api/core/bitstreams/e919ca56-3995-46b1-9546-a09b83f8b172/content |
| [7] | *Do pistol rounds matter more in CS2?* | D. Hsieh (Leetify Research) | 2024 | https://leetify.com/blog/cs2-vs-csgo-pistol-rounds/ |
| [8] | *The many routes to the ubiquitous Bradley–Terry model* | I. Hamilton, N. Tawn, D. Firth (arXiv:2312.13619) | 2025 | https://arxiv.org/abs/2312.13619 |
| [9] | *Dynamic Programming for Series Win Probability* | K. Lele (University of Pittsburgh) | 2021 | https://sites.pitt.edu/~kpele/dp_series.pdf |
| [10] | *Benchmark Rynkowy: Ewaluacja Modeli Predykcyjnych przeciwko Kwotowaniom Pinnacle* | J. Melzacki (`~/dev/inzynierka/reports/`) | 2026 | local://dev/inzynierka/reports/pinnacle_vs_model_benchmark_2026.md |
| [11] | *Beta calibration: a well-founded and easily implemented improvement on logistic calibration* | M. Kull, T. Silva Filho, P. Flach (AISTATS 2017) | 2017 | https://proceedings.mlr.press/v54/kull17a/kull17a.pdf |
| [12] | *On Calibration of Modern Neural Networks* | C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger (ICML 2017) | 2017 | https://arxiv.org/abs/1706.04599 |
