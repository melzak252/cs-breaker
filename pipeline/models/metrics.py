"""Probability scoring and calibration metrics for binary forecasts."""

import math
from typing import Any, Dict, List, Tuple


def _validate_inputs(y_true: List[int], y_prob: List[float]) -> None:
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if any(label not in (0, 1) for label in y_true):
        raise ValueError("y_true values must be binary (0 or 1)")
    if any(not 0.0 <= probability <= 1.0 for probability in y_prob):
        raise ValueError("y_prob values must be between 0 and 1")


def brier_score(y_true: List[int], y_prob: List[float]) -> float:
    """Return the mean squared error between binary outcomes and probabilities."""
    _validate_inputs(y_true, y_prob)
    if not y_true:
        return 0.0
    return sum((probability - label) ** 2 for label, probability in zip(y_true, y_prob)) / len(y_true)


def log_loss_score(
    y_true: List[int], y_prob: List[float], eps: float = 1e-15
) -> float:
    """Return binary cross entropy, clipping probabilities away from 0 and 1."""
    _validate_inputs(y_true, y_prob)
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be between 0 and 0.5")
    if not y_true:
        return 0.0
    total = 0.0
    for label, probability in zip(y_true, y_prob):
        clipped = min(max(probability, eps), 1.0 - eps)
        total -= label * math.log(clipped) + (1 - label) * math.log1p(-clipped)
    return total / len(y_true)


def roc_auc_score_safe(y_true: List[int], y_prob: List[float]) -> float:
    """Return ROC AUC using the rank-sum formula, assigning average ranks to ties.

    An empty input or a target containing only one class has no defined ROC AUC;
    return 0.5 as a neutral, finite fallback.
    """
    _validate_inputs(y_true, y_prob)
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return 0.5

    ordered = sorted(range(len(y_prob)), key=y_prob.__getitem__)
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and y_prob[ordered[end]] == y_prob[ordered[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(y_true[ordered[index]] for index in range(start, end))
        start = end

    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def expected_calibration_error(
    y_true: List[int], y_prob: List[float], n_bins: int = 10
) -> Tuple[float, List[Dict[str, Any]]]:
    """Return equal-width expected calibration error and populated-bin summaries."""
    _validate_inputs(y_true, y_prob)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if not y_true:
        return 0.0, []

    counts = [0] * n_bins
    probability_sums = [0.0] * n_bins
    outcome_sums = [0] * n_bins
    for label, probability in zip(y_true, y_prob):
        # Put the closed upper endpoint (1.0) in the final bin.
        bin_index = min(int(probability * n_bins), n_bins - 1)
        counts[bin_index] += 1
        probability_sums[bin_index] += probability
        outcome_sums[bin_index] += label

    sample_count = len(y_true)
    ece = 0.0
    bins: List[Dict[str, Any]] = []
    for index, count in enumerate(counts):
        if count == 0:
            continue
        average_probability = probability_sums[index] / count
        win_rate = outcome_sums[index] / count
        ece += count / sample_count * abs(win_rate - average_probability)
        bins.append({
            "bin_lower": index / n_bins,
            "bin_upper": (index + 1) / n_bins,
            "count": count,
            "avg_prob": average_probability,
            "win_rate": win_rate,
        })
    return ece, bins


def evaluate_predictions(y_true: List[int], y_prob: List[float]) -> Dict[str, Any]:
    """Compute the standard forecast metric set and calibration-bin summaries."""
    _validate_inputs(y_true, y_prob)
    calibration_error, calibration_bins = expected_calibration_error(y_true, y_prob)
    accuracy = (
        sum((probability >= 0.5) == bool(label) for label, probability in zip(y_true, y_prob))
        / len(y_true)
        if y_true
        else 0.0
    )
    return {
        "brier": brier_score(y_true, y_prob),
        "log_loss": log_loss_score(y_true, y_prob),
        "roc_auc": roc_auc_score_safe(y_true, y_prob),
        "accuracy": accuracy,
        "ece": calibration_error,
        "calibration_bins": calibration_bins,
        "n_samples": len(y_true),
    }
