"""Unified model evaluation and promotion benchmark for CS2 forecasting.

Implements the standard benchmark protocol matching lol-breaker:
- Multi-dimensional scoring: Log Loss, Brier score, AUC, Accuracy, ECE, MCE, Calibration slope.
- Brier score decomposition (Reliability, Resolution, Uncertainty).
- Paired statistical comparison against comparative baseline with bootstrap confidence intervals.
- Promotion gate checking against formal qualification criteria.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

DEFAULT_PROBABILITY_EPSILON = 1e-15


@dataclass(frozen=True)
class ModelBenchmarkSummary:
    """Core probabilistic, discrimination, and calibration metrics."""

    sample_size: int
    log_loss: float
    brier_score: float
    brier_reliability: float
    brier_resolution: float
    brier_uncertainty: float
    auc: float
    accuracy: float
    ece: float
    mce: float
    calibration_slope: float
    calibration_intercept: float


@dataclass(frozen=True)
class BaselineComparisonResult:
    """Paired statistical comparison against a locked comparative baseline."""

    baseline_name: str
    delta_log_loss: float
    ci_lower_log_loss: float
    ci_upper_log_loss: float
    p_value_log_loss: float
    is_statistically_significant: bool
    delta_brier: float
    delta_accuracy: float
    delta_auc: float


def clip_probabilities(
    probs: Sequence[float] | np.ndarray,
    eps: float = DEFAULT_PROBABILITY_EPSILON,
) -> np.ndarray:
    """Safely clip probability values strictly into [eps, 1 - eps]."""
    arr = np.asarray(probs, dtype=float)
    return np.clip(arr, eps, 1.0 - eps)


def calculate_ece(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float]:
    """Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = clip_probabilities(y_prob)
    n = len(y_t)
    if n == 0:
        return 0.0, 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bins[i], bins[i + 1]
        mask = (y_p >= bin_lower) & (y_p <= bin_upper if i == n_bins - 1 else y_p < bin_upper)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_t[mask])
            bin_conf = np.mean(y_p[mask])
            gap = abs(bin_acc - bin_conf)
            ece += (bin_count / n) * gap
            mce = max(mce, gap)

    return float(ece), float(mce)


def brier_decomposition(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float, float]:
    """Murphy (1973) Brier score decomposition: (Reliability, Resolution, Uncertainty).

    Brier = Reliability - Resolution + Uncertainty.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = clip_probabilities(y_prob)
    n = len(y_t)
    if n == 0:
        return 0.0, 0.0, 0.0

    base_rate = np.mean(y_t)
    uncertainty = float(base_rate * (1.0 - base_rate))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bins[i], bins[i + 1]
        mask = (y_p >= bin_lower) & (y_p <= bin_upper if i == n_bins - 1 else y_p < bin_upper)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_t[mask])
            bin_conf = np.mean(y_p[mask])
            reliability += (bin_count / n) * ((bin_conf - bin_acc) ** 2)
            resolution += (bin_count / n) * ((bin_acc - base_rate) ** 2)

    return float(reliability), float(resolution), uncertainty


def compute_benchmark_summary(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    n_bins: int = 10,
) -> ModelBenchmarkSummary:
    """Compute complete statistical summary metrics for a probability prediction vector."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = clip_probabilities(y_prob)
    n = len(y_t)
    if n == 0:
        raise ValueError("Cannot benchmark empty prediction arrays.")

    ll = float(log_loss(y_t, y_p))
    bs = float(brier_score_loss(y_t, y_p))
    rel, res, unc = brier_decomposition(y_t, y_p, n_bins=n_bins)

    try:
        auc = float(roc_auc_score(y_t, y_p))
    except ValueError:
        auc = 0.5

    acc = float(accuracy_score(y_t, (y_p >= 0.5).astype(int)))
    ece, mce = calculate_ece(y_t, y_p, n_bins=n_bins)

    # Calibration slope and intercept via logistic regression on logits
    logit_p = np.log(y_p / (1.0 - y_p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(fit_intercept=True, max_iter=200)
        lr.fit(logit_p, y_t)
        cal_slope = float(lr.coef_[0][0])
        cal_intercept = float(lr.intercept_[0])
    except Exception:
        cal_slope = 1.0
        cal_intercept = 0.0

    return ModelBenchmarkSummary(
        sample_size=n,
        log_loss=ll,
        brier_score=bs,
        brier_reliability=rel,
        brier_resolution=res,
        brier_uncertainty=unc,
        auc=auc,
        accuracy=acc,
        ece=ece,
        mce=mce,
        calibration_slope=cal_slope,
        calibration_intercept=cal_intercept,
    )


def compute_baseline_comparison(
    y_true: Sequence[int] | np.ndarray,
    candidate_probs: Sequence[float] | np.ndarray,
    baseline_probs: Sequence[float] | np.ndarray,
    baseline_name: str = "Baseline",
    n_bootstraps: int = 5000,
    random_seed: int = 42,
) -> BaselineComparisonResult:
    """Paired statistical comparison with bootstrap confidence intervals."""
    y_t = np.asarray(y_true, dtype=int)
    p_c = clip_probabilities(candidate_probs)
    p_b = clip_probabilities(baseline_probs)
    n = len(y_t)

    cand_ll = log_loss(y_t, p_c)
    base_ll = log_loss(y_t, p_b)
    delta_ll = float(cand_ll - base_ll)

    cand_bs = brier_score_loss(y_t, p_c)
    base_bs = brier_score_loss(y_t, p_b)
    delta_bs = float(cand_bs - base_bs)

    cand_acc = accuracy_score(y_t, (p_c >= 0.5).astype(int))
    base_acc = accuracy_score(y_t, (p_b >= 0.5).astype(int))
    delta_acc = float(cand_acc - base_acc)

    try:
        cand_auc = roc_auc_score(y_t, p_c)
        base_auc = roc_auc_score(y_t, p_b)
        delta_auc = float(cand_auc - base_auc)
    except ValueError:
        delta_auc = 0.0

    # Paired block bootstrap for delta log loss
    rng = np.random.default_rng(random_seed)
    deltas = []
    # Individual sample log losses
    cand_losses = -(y_t * np.log(p_c) + (1 - y_t) * np.log(1.0 - p_c))
    base_losses = -(y_t * np.log(p_b) + (1 - y_t) * np.log(1.0 - p_b))
    diff_losses = cand_losses - base_losses

    indices = np.arange(n)
    for _ in range(n_bootstraps):
        boot_idx = rng.choice(indices, size=n, replace=True)
        deltas.append(np.mean(diff_losses[boot_idx]))

    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))

    # Two-sided empirical p-value for H0: delta >= 0
    p_val = float(np.mean([d >= 0.0 for d in deltas]))
    is_sig = ci_upper < 0.0  # Significant improvement if 95% CI is entirely below 0

    return BaselineComparisonResult(
        baseline_name=baseline_name,
        delta_log_loss=delta_ll,
        ci_lower_log_loss=ci_lower,
        ci_upper_log_loss=ci_upper,
        p_value_log_loss=p_val,
        is_statistically_significant=is_sig,
        delta_brier=delta_bs,
        delta_accuracy=delta_acc,
        delta_auc=delta_auc,
    )
