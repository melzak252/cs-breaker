# Comprehensive Exploratory Data Analysis (EDA): Counter-Strike Multi-Era Dataset

> **Historical / superseded modeling claims (2026-10-09):** This EDA preserves the earlier multi-era analysis, not the current audited player-series cohort. Its counts and absolute “zero leakage” statements are not current model evidence. The earlier 7,038-row player benchmark was invalidated by roster and chronology defects. Current evidence: 675 lineup-conditional series, 38 predictors, 396/115/164 TRAIN/VALIDATION/TEST rows. See [observed results](cs2-model-results-2026-10-09.md) and [research synthesis](cs2-model-research-2026-10-09.md). The new recipe's small observed overall TEST gain is uncertain; ten-bin ECE, the final block and bookmaker-common scores worsen. No demonstrated statistical or betting edge.

**Analysis Date:** 2026-10-05  
**Data Sources:** Multi-era Counter-Strike Corpus (126,872 CS:GO games from `csgo_pro_games_data.csv` + 7,033 CS2 BO3 series from Kaggle + OpenCS2 telemetry).  
**Output Files:** `data/processed/train_features.csv` (127,015 rows), `data/processed/test_features.csv` (6,686 rows).  

---

## Executive Overview

| Metric | Measured Value | Insight |
| :--- | :---: | :--- |
| **Total Matches Analyzed** | **133,701** | Spans September 2012 to October 2025 (~13.1 years). |
| **Total Features Engineered** | **41 columns** | Zero null values across all processed features. |
| **Unique Professional Teams** | **6,267 teams** | Top 15 teams have $>1,500$ professional matches each. |
| **Unique Players Tracked** | **15,519 players** | Top veterans have $>2,000$ to $3,000$ games. |
| **Target Variable Balance** | **50.138%** | Perfectly balanced class distribution (no slot bias). |
| **Train / Test Split Date** | **2024-06-09** | Strict point-in-time boundary; zero future leakage. |

---

## 1. Temporal Evolution & Game Era Transition

Counter-Strike match volume grew rapidly from CS:GO's 2012 release, peaking during the open-circuit era (2017–2021) before transitioning to CS2 in late 2023:

```text
2012:    431 matches | CS:GO (MR15)
2013:  2,172 matches | CS:GO (MR15) ██
2014:  2,293 matches | CS:GO (MR15) ███
2015:  7,227 matches | CS:GO (MR15) █████████
2016: 11,243 matches | CS:GO (MR15) ██████████████
2017: 16,789 matches | CS:GO (MR15) ██████████████████████
2018: 17,925 matches | CS:GO (MR15) ███████████████████████  <-- Peak Volume
2019: 15,197 matches | CS:GO (MR15) ████████████████████
2020: 13,899 matches | CS:GO (MR15) ██████████████████      <-- Online Era
2021: 16,238 matches | CS:GO (MR15) █████████████████████
2022: 14,270 matches | CS:GO (MR15) ███████████████████
2023:  8,984 matches | CS:GO (MR15) ███████████             <-- CS2 Launch (Sept)
2024:  2,358 matches | CS2 (MR12)   ███
2025:  4,675 matches | CS2 (MR12)   ██████
```

---

## 2. Target Variable & Symmetry Audit

The target variable $y = 1$ if Team 1 wins, and $y = 0$ if Team 2 wins:
- **Full Corpus:** 67,035 wins for Team 1 out of 133,701 matches (**50.138%**).
- **Train Set ($N=127,015$):** **49.911%** win rate.
- **Test Set ($N=6,686$):** **54.457%** win rate.

Because the match list assigns Team 1 and Team 2 based on tournament bracket conventions rather than pre-match seeding, the training distribution is unbiased.

---

## 3. Team Dynamics & Most Active Organizations

Across 6,267 unique teams recorded, the top 15 organizations represent the historical backbone of competitive Counter-Strike:

| Rank | Organization | Matches Played | Era Coverage | Notable Core Lineups |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Ninjas in Pyjamas** | **2,285** | 2012–2025 | f0rest, GeT_RiGhT, Xizt, friberg, REZ |
| **2** | **fnatic** | **2,261** | 2012–2025 | JW, flusha, pronax, olofmeister, KRIMZ |
| **3** | **Virtus.pro** | **2,230** | 2012–2025 | TaZ, NEO, pashaBiceps, snax, byali, Jame |
| **4** | **MOUZ (mousesports)** | **2,126** | 2012–2025 | chrisJ, ropz, frozen, torzsi, siuhy |
| **5** | **Natus Vincere** | **2,066** | 2012–2025 | Zeus, Edward, GuardiaN, s1mple, b1t, Aleksib |
| **6** | **Tricked** | **1,924** | 2014–2023 | Danish talent incubator (HUNDEN, refrezh, acoR) |
| **7** | **Team Liquid** | **1,848** | 2015–2025 | EliGE, nitr0, Twistzz, NAF, Stewie2K |
| **8** | **ALTERNATE aTTaX** | **1,826** | 2012–2025 | kzy, stfN, syrsoN |
| **9** | **G2 Esports** | **1,788** | 2015–2025 | shox, ScreaM, kennyS, NiKo, m0NESY |
| **10** | **FORZE** | **1,779** | 2017–2025 | Jerry, FL1T, xsepower, zorte |
| **11** | **HellRaisers** | **1,775** | 2014–2021 | ANGE1, AdreN, Dosia, mou, woxic |
| **12** | **AGO** | **1,715** | 2017–2023 | Furlan, GruBy, snatchie, mhL |
| **13** | **LDLC** | **1,631** | 2013–2023 | NBK-, Happy, apEX, kioShiMa |
| **14** | **Sprout** | **1,608** | 2017–2024 | Spiidi, denis, faveN, Zyphon |
| **15** | **Astralis** | **1,598** | 2016–2025 | dev1ce, dupreeh, gla1ve, Magisk, Xyp9x |

---

## 4. Player Longevity & Historical Legends

Across **15,519 unique player nicknames**, Counter-Strike exhibits exceptional player career longevity:

| Rank | Player Handle | Matches Played | Major Titles / Era |
| :---: | :--- | :---: | :---: |
| **1** | **NiKo** (Nikola Kovač) | **3,076** | MOUZ, FaZe, G2 |
| **2** | **ALEX** (Alex McMeekin / Alex) | **2,632** | Vitality, LDLC, fnatic |
| **3** | **AdreN** (Dauren Kystaubayev) | **2,446** | HellRaisers, Gambit (Major Champion) |
| **4** | **TaZ** (Wiktor Wojtas) | **2,394** | Golden Five, Virtus.pro Major Champion |
| **5** | **bubble** (Kamen Kostadinov) | **2,359** | GPlay, E-Frag, Windigo, Fiend |
| **6** | **Snax** (Janusz Pogorzelski) | **2,253** | Virtus.pro Major Champion, G2 |
| **7** | **JW** (Jesper Wecksell) | **2,217** | 3x Major Champion (fnatic) |
| **8** | **Spiidi** (Timo Richter) | **2,200** | mousesports, Sprout |
| **9** | **NBK-** (Nathan Schmitt) | **2,194** | 2x Major Champion (LDLC, EnVyUs) |
| **10** | **apEX** (Dan Madesclaire) | **2,173** | 2x Major Champion (EnVyUs, Vitality IGL) |
| **11** | **flusha** (Robin Rönnquist) | **2,155** | 3x Major Champion, Major MVP |
| **12** | **Xyp9x** (Andreas Højsleth) | **2,147** | 4x Major Champion ("Clutch Minister") |
| **13** | **karrigan** (Finn Andersen) | **2,143** | Major Champion, TSM, Astralis, FaZe IGL |
| **14** | **tabseN** (Johannes Wodarz) | **2,125** | BIG Clan captain |
| **15** | **f0rest** (Patrik Lindberg) | **2,071** | NiP 87-0 era, Major Champion |
| **16** | **shox** (Richard Papillon) | **2,051** | LDLC, EnVyUs, G2, Vitality |

---

## 5. Map Pool Breakdown across Historical CS

Distribution of maps across the raw historical competitive matches ($N=127,695$):

| Map Code | Map Name | Matches Played | Share of Pool | Competitive Status |
| :--- | :--- | :---: | :---: | :--- |
| `mrg` | **de_mirage** | **23,936** | **18.7%** | Only map never removed from Active Duty. |
| `inf` | **de_inferno** | **22,262** | **17.4%** | Staple competitive map throughout CS:GO and CS2. |
| `ovp` | **de_overpass** | **15,009** | **11.8%** | Introduced in 2014; staple until CS2 2024 rotation. |
| `nuke` | **de_nuke** | **14,193** | **11.1%** | Heavily CT-sided in early CS:GO; rebalanced. |
| `d2` | **de_dust2** | **13,934** | **10.9%** | Classical aim map; rotated in and out of pool. |
| `trn` | **de_train** | **12,059** | **9.4%** | Rotated out in 2021; classical CT tactical map. |
| `cch` | **de_cache** | **8,947** | **7.0%** | Community map; Active Duty until 2019. |
| `vtg` | **de_vertigo** | **6,410** | **5.0%** | Added in 2019; contested A-ramp dynamic. |
| `cbl` | **de_cbble** | **5,799** | **4.5%** | Active Duty until 2018 (famous for Dragon Lore drops). |
| `anc` | **de_ancient** | **4,106** | **3.2%** | Modern pool addition (replaced Train in 2021). |
| `anb` | **de_anubis** | **984** | **0.8%** | Added late 2022; T-sided modern pool staple. |

---

## 6. Feature Correlation & Predictive Ranking

Linear Pearson correlation ($r$) and predictive ranking of the 41 engineered features against the target match outcome ($y$):

| Rank | Feature Name | Pearson $r$ | Feature Category | Signal Explanation |
| :---: | :--- | :---: | :--- | :--- |
| **1** | `elo_prob` | **+0.3046** | Rating Signal | Continuous logistic probability from global team Elo. |
| **2** | `elo_diff` | **+0.3029** | Rating Signal | Pure rating point gap between Team 1 and Team 2. |
| **3** | `veto_hierarchical_bo3_prob` | **-0.1705** | Veto Dynamics | Binomial series expansion across simulated map vetoes. |
| **4** | `veto_pool_wr_diff` | **-0.1679** | Veto Dynamics | Team 1 vs Team 2 win rate gap across the 7-map pool. |
| **5** | `round_diff` | **+0.1678** | Form / Momentum | Average round differential over past 10 matches. |
| **6** | `win_rate_diff_10` | **+0.1672** | Form / Momentum | Match win rate gap over previous 10 series. |
| **7** | `win_rate_diff_5` | **+0.1401** | Form / Momentum | Short-term form gap over previous 5 series. |
| **8** | `veto_map_depth_diff` | **+0.1261** | Veto Dynamics | Number of active competitive maps played with $>0$ matches. |
| **9** | `experience_diff` | **+0.1165** | Team Maturity | Total historical professional matches played gap. |
| **10** | `h2h_win_rate_team1` | **+0.0778** | Head-to-Head | Historical head-to-head win rate between the two teams. |
| **11** | `glicko2_rd_diff` | **-0.0614** | Uncertainty | Rating Deviation difference (penalizes uncertain rosters). |
| **12** | `trueskill_sigma_diff` | **-0.0542** | Uncertainty | Bayesian variance gap from Gaussian factor graph. |
| **13** | `openskill_ordinal_diff` | **+0.0535** | Bayesian Skill | Conservative skill metric ($\mu - 3\sigma$) differential. |
| **14** | `openskill_sigma_diff` | **-0.0524** | Uncertainty | Plackett-Luce variance differential. |
| **15** | `rest_days_diff` | **-0.0381** | Fatigue / Rest | Difference in days since previous match. |

---

## 7. Key EDA Takeaways for Model Engineering

1. **Elo and Veto are the Dominant Signals:** The combination of global Elo rating ($r \approx 0.30$) and map-pool veto advantage ($r \approx 0.17$) accounts for the majority of explainable pre-match variance.
2. **Uncertainty Features Add Value in Gradient Boosting:** While Glicko-2 $RD$ and TrueSkill $\sigma$ have modest linear correlation ($r \approx -0.06$), they act as non-linear gating variables: when uncertainty is high, tree models scale down predictions toward the 50/50 prior, preventing log loss blowups on upsets.
3. **No Target Leakage & Zero Nulls:** Every feature is computed online using strictly completed matches preceding the query timestamp. Zero nulls exist in the processed dataset.
