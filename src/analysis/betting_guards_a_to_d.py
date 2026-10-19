"""Betting risk guards A through D and de-vigging comparison engine for CS2.

Implements:
- De-vigging methods: Multiplicative, Additive, Power, and Shin (with 2-way division-by-zero fix).
- Guard A: Conservative bound P_low (epistemic uncertainty).
- Guard B: Dynamic Logit Disagreement Guard (|logit(p_model) - logit(p_market)| <= delta).
  Scales sensitivity automatically: tight at tails (90%+ favorites / 10% underdogs),
  permissive around 50/50.
- Guard C: EV Haircut Multiplier (k * EV_raw - safety_margin) against Optimizer's Curse.
- Guard D: Closing Line Value (CLV) Proof against sharp closing odds.
- Guard W: Wasserstein Distributionally Robust Optimization (DRO) ambiguity ball (W_1 <= epsilon).
- Model Consensus Veto: Cross-architecture consensus (|p_challenger - p_baseline| <= threshold).
"""

from __future__ import annotations

import math
from typing import Tuple


# =============================================================================
# 1. DE-VIGGING METHODS
# =============================================================================

def devig_multiplicative(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Proportional normalization: p_i = (1/O_i) / (1/O_a + 1/O_b)."""
    if odds_a <= 1.0 or odds_b <= 1.0 or not math.isfinite(odds_a) or not math.isfinite(odds_b):
        raise ValueError(f"Odds must be > 1.0 and finite: {odds_a}, {odds_b}")
    qa = 1.0 / odds_a
    qb = 1.0 / odds_b
    tot = qa + qb
    return qa / tot, qb / tot


def devig_additive(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Additive devigging: subtracts half of overround equally from each outcome."""
    if odds_a <= 1.0 or odds_b <= 1.0 or not math.isfinite(odds_a) or not math.isfinite(odds_b):
        raise ValueError(f"Odds must be > 1.0 and finite: {odds_a}, {odds_b}")
    qa = 1.0 / odds_a
    qb = 1.0 / odds_b
    overround = (qa + qb) - 1.0
    half_ov = overround / 2.0
    pa = max(1e-6, min(1.0 - 1e-6, qa - half_ov))
    pb = max(1e-6, min(1.0 - 1e-6, qb - half_ov))
    tot = pa + pb
    return pa / tot, pb / tot


def devig_power(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Solve for exponent k such that sum((1/O_i)^k) = 1.0."""
    from scipy.optimize import brentq

    if odds_a <= 1.0 or odds_b <= 1.0 or not math.isfinite(odds_a) or not math.isfinite(odds_b):
        raise ValueError(f"Odds must be > 1.0 and finite: {odds_a}, {odds_b}")
    la = math.log(odds_a)
    lb = math.log(odds_b)

    def obj(k: float) -> float:
        return math.exp(-k * la) + math.exp(-k * lb) - 1.0

    k = brentq(obj, 0.0, math.log(2.0) / min(la, lb), xtol=1e-12, maxiter=200)
    pa = math.exp(-k * la)
    pb = math.exp(-k * lb)
    tot = pa + pb
    return pa / tot, pb / tot


def devig_shin(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Shin (1992) structural model with 2-way closed-form solution.

    For n = 2 outcomes, Shin's insider trading proportion z = S - 1, and:
    denom = 2 * (1 - z)
    p_i = (sqrt(z^2 + 4 * (1 - z) * (q_i^2 / S)) - z) / denom
    """
    if odds_a <= 1.0 or odds_b <= 1.0 or not math.isfinite(odds_a) or not math.isfinite(odds_b):
        raise ValueError(f"Odds must be > 1.0 and finite: {odds_a}, {odds_b}")
    qa = 1.0 / odds_a
    qb = 1.0 / odds_b
    s = qa + qb
    if s <= 1.0000000001:
        return qa / s, qb / s

    z = s - 1.0
    denom = 2.0 * (1.0 - z)
    if denom <= 1e-12:
        return qa / s, qb / s
    pa = (math.sqrt(z * z + 4.0 * (1.0 - z) * (qa * qa / s)) - z) / denom
    pb = (math.sqrt(z * z + 4.0 * (1.0 - z) * (qb * qb / s)) - z) / denom
    tot = pa + pb
    return pa / tot, pb / tot


# =============================================================================
# 2. GUARD B: DYNAMIC LOGIT DISAGREEMENT GUARD
# =============================================================================

def dynamic_logit_disagreement(prob_model: float, prob_market: float) -> float:
    """Calculate the absolute difference between model and market in logit space.

    Delta_logit = |logit(p_model) - logit(p_market)|.
    """
    if not (math.isfinite(prob_model) and math.isfinite(prob_market)):
        raise ValueError(f"Probabilities must be finite: model={prob_model}, market={prob_market}")
    if not (0.0 < prob_model < 1.0 and 0.0 < prob_market < 1.0):
        raise ValueError(f"Probabilities must be in (0, 1): model={prob_model}, market={prob_market}")

    eps = 1e-6
    pm_c = max(eps, min(1.0 - eps, prob_model))
    pk_c = max(eps, min(1.0 - eps, prob_market))

    lm = math.log(pm_c / (1.0 - pm_c))
    lk = math.log(pk_c / (1.0 - pk_c))
    return abs(lm - lk)


def apply_guard_b_dynamic_disagreement(
    prob_model: float,
    prob_market: float,
    max_logit_gap: float = 0.85,
) -> tuple[bool, float]:
    """Check if model and market agreement satisfies the dynamic logit bound.

    Returns (passes_guard, observed_logit_gap).
    """
    gap = dynamic_logit_disagreement(prob_model, prob_market)
    return gap <= max_logit_gap, gap


# =============================================================================
# 3. MODEL CONSENSUS VETO
# =============================================================================

def apply_model_consensus_veto(
    prob_challenger: float,
    prob_baseline: float,
    max_gap: float = 0.10,
) -> tuple[bool, float]:
    """Check agreement between primary model and secondary baseline.

    Suppresses false edges when complex models diverge heavily from simple form.
    Returns (passes_guard, absolute_gap).
    """
    gap = abs(prob_challenger - prob_baseline)
    return gap <= max_gap, gap


# =============================================================================
# 4. GUARD A: CONSERVATIVE BOUND (P_low)
# =============================================================================

def apply_guard_a_plow(
    prob_conservative: float,
    odds: float,
    tax_rate: float = 0.12,
    min_ev: float = 0.0,
) -> tuple[bool, float]:
    """Evaluate EV using conservative lower bound P_low.

    EV_safe = P_low * (odds * (1 - tax_rate)) - 1.0.
    Returns (passes_guard, ev_safe).
    """
    if prob_conservative is None or not math.isfinite(prob_conservative) or prob_conservative <= 0.0:
        return False, -1.0
    ev_safe = prob_conservative * (odds * (1.0 - tax_rate)) - 1.0
    return ev_safe > min_ev, ev_safe


# =============================================================================
# 5. GUARD C: EV HAIRCUT MULTIPLIER
# =============================================================================

def apply_guard_c_haircut(
    raw_ev: float,
    haircut_multiplier: float = 0.30,
    safety_margin: float = 0.02,
) -> tuple[bool, float]:
    """Apply empirical shrinkage factor to raw EV to protect against Optimizer's Curse.

    EV_discounted = haircut_multiplier * raw_ev - safety_margin.
    Returns (passes_guard, ev_discounted).
    """
    if not math.isfinite(raw_ev):
        return False, -1.0
    discounted_ev = haircut_multiplier * raw_ev - safety_margin
    return discounted_ev > 0.0, discounted_ev


# =============================================================================
# 6. GUARD D: CLOSING LINE VALUE (CLV) PROOF
# =============================================================================

def calculate_clv(
    taken_odds: float,
    closing_odds_novig: float,
    tax_rate: float = 0.12,
) -> float:
    """Calculate Closing Line Value (CLV) against the sharp closing line.

    CLV = (taken_odds * (1 - tax_rate)) / closing_odds_novig - 1.0.
    """
    if closing_odds_novig <= 0.0 or not math.isfinite(closing_odds_novig):
        return 0.0
    net_taken = taken_odds * (1.0 - tax_rate)
    return (net_taken / closing_odds_novig) - 1.0


def apply_guard_d_clv_drift(
    taken_market_prob: float,
    closing_market_prob: float,
    max_adverse_drift: float = 0.03,
) -> tuple[bool, float]:
    """Ensure the market did not move sharply against our selection before start.

    adverse_drift = taken_market_prob - closing_market_prob.
    Returns (passes_guard, drift).
    """
    drift = taken_market_prob - closing_market_prob
    return drift <= max_adverse_drift, drift


# =============================================================================
# 7. GUARD W: WASSERSTEIN DRO AMBIGUITY BALL (W_1 <= epsilon)
# =============================================================================

def wasserstein_dro_min_ev_hurdle(
    odds: float,
    tax_rate: float = 0.12,
    base_min_ev: float = 0.05,
    epsilon: float = 0.025,
) -> float:
    """Compute the odds-scaled minimum net EV hurdle induced by a 1-Wasserstein ball."""
    if odds <= 1.0 or not math.isfinite(odds):
        raise ValueError(f"odds must be > 1.0 and finite, got {odds}")
    if not (0.0 <= tax_rate < 1.0 and math.isfinite(tax_rate)):
        raise ValueError(f"tax_rate must be in [0, 1), got {tax_rate}")
    if not (0.0 <= epsilon < 1.0 and math.isfinite(epsilon)):
        raise ValueError(f"epsilon must be in [0, 1), got {epsilon}")
    if not (0.0 <= base_min_ev and math.isfinite(base_min_ev)):
        raise ValueError(f"base_min_ev must be >= 0 and finite, got {base_min_ev}")
    return max(base_min_ev, (1.0 - tax_rate) * odds * epsilon)


def apply_guard_w_wasserstein_dro(
    prob_model: float,
    odds: float,
    tax_rate: float = 0.12,
    base_min_ev: float = 0.05,
    epsilon: float = 0.025,
) -> tuple[bool, float, float]:
    """Evaluate worst-case net EV over a 1-Wasserstein Bernoulli ambiguity ball of radius epsilon.

    Returns (passes_guard, ev_dro, hurdle).
    """
    if not (0.0 < prob_model < 1.0 and math.isfinite(prob_model)):
        raise ValueError(f"prob_model must be in (0, 1), got {prob_model}")
    hurdle = wasserstein_dro_min_ev_hurdle(
        odds=odds,
        tax_rate=tax_rate,
        base_min_ev=base_min_ev,
        epsilon=epsilon,
    )
    return_factor = odds * (1.0 - tax_rate)
    ev_net = prob_model * return_factor - 1.0
    prob_dro = max(0.0, prob_model - epsilon)
    ev_dro = prob_dro * return_factor - 1.0
    passes = (ev_net >= base_min_ev) and (ev_dro > 0.0)
    return passes, ev_dro, hurdle
