# Model Catalog — CS-Breaker

## 1. Candidate Architecture Matrix

| Model Identifier | Architecture | Feature Inputs | Test / Validation Log Loss | Status |
|---|---|---|---:|---|
| **`Baseline-Elo`** | Pure logistic Elo ($K=32$, 400-point scale) | Historical literal team names | **0.6660** (2,238 test matches) | Active Benchmark |
| **`Player-Elo-Nickname`** | Participant-level Elo ($K=48$) | StrandedPond player handles | **0.6962** (644 maps) | Research (trails prior 0.6864) |
| **`SteamID-Map-Elo`** | Lineup-level Elo ($K=24$) | OpenCS2 verified SteamIDs | **0.7010** (158 maps) | Research (trails coin flip 0.6931) |
| **`Blend-Player-Archive`** | 50/50 linear blend | Player Elo + Archive series Elo | **0.6401** (164 test series) | Preserved Candidate |
| **`Ridge-Ratings-Form`** | `ridge_C1_ratings_form_hnone` | Multi-system ratings + 20-series form | **0.6321** (419 matched matches) | Failed Bookmaker Challenge |
| **`Bayesian-Challenger`** | `ridge_C1_ratings_form_bayesian_hnone` | 16 dynamic Bayesian channels + ratings + form | **0.6248** (545 dev matches) | Published Research Challenger (`competitiveness_qualified: false`) |
| **`Hybrid-50`** | 50/50 logit-space Bayesian shrinkage | Model prediction + Shin de-vigged bookmaker quote | **0.6248** (shrinkage optimum) | Research Model |

---

## 2. Model Operational Contracts

### Bayesian Challenger (`artifacts/professional_model_20261010/selected_model.pkl`)
- **Schema**: Version 2 (`series_inference_state.pkl`).
- **Input Requirements**: `team1`, `team2`, `source_date` (strictly after history cutoff), format (`bo3` or `bo5`).
- **Output Schema**:
  - `prob_team1_win`, `prob_team2_win`
  - `model_selected`
  - `fit_through` (2026-04-30), `history_through` (2026-10-07)
  - `competitiveness_qualified`: strictly `false` (no unsupported claim of edge over sharp books)
  - `bayesian_channels_used`: list of 16 Gaussian dynamic channels

### Retrospective Elo Baseline (`cs_breaker.py` / `src/ratings/elo.py`)
- **Input Requirements**: Team names.
- **Output Schema**: Win probability via logistic Elo formula $P = 1 / (1 + 10^{(R_2 - R_1)/400})$.
- **Zero-Dependency**: Executes using Python standard library only.

### CS2 C0 Neural Attention Model (`src/models/c0_architecture.py`)
- **Architecture**: 16-map masked personal history attention + 5-player Team Set Transformer + Pairwise anti-symmetric win probability head ($19,617$ parameters).
- **Training**: Trained via `scripts/train_c0_model.py` with AdamW, Cosine Annealing, and symmetric BCE loss on 1,609 chronological matches (best val loss: 0.6556).
- **Checkpoint**: `data/06_models/c0_checkpoint_best.pt`.
- **Inference**: Programmatic and CLI serving via `CS2C0Predictor` in `src/models/c0_serving.py`.
- **Antisymmetry**: Guaranteed $P(A > B) + P(B > A) == 1.0$ to machine precision.
