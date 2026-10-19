"""Non-inferiority decision rules for the frozen paired bookmaker challenge."""
from __future__ import annotations

from pipeline.models.bookmaker_comparison import BOOKS, _metrics, _paired_difference


def competitiveness_decision(consensus_rows, per_book, policy, *, test_start):
    """Apply declared support, proper-score and clustered-uncertainty thresholds.

    The input is one consensus observation per native match. Qualification is
    limited to these retrospective, time-aligned cases; it is not betting ROI,
    a claim of executable prices, or a guarantee about subsequent matches.
    """
    uncertainty = {
        label: {metric: _paired_difference(consensus_rows, metric, block_days=days, start_day=test_start)
                for metric in ("log_loss", "brier")}
        for label, days in (("source_day", 1), ("seven_day", 7))
    }
    matches = len(consensus_rows)
    source_days = uncertainty["source_day"]["log_loss"]["source_day_count"]
    checks = {
        "minimum_paired_matches": matches >= policy["minimum_paired_matches"],
        "minimum_source_days": source_days >= policy["minimum_source_days"],
    }
    for metric in ("log_loss", "brier"):
        point = uncertainty["source_day"][metric]["estimate"]
        checks[f"{metric}_point"] = (point is not None and point <= policy[metric]["maximum_point_model_minus_consensus"])
        for cluster in uncertainty:
            evidence = uncertainty[cluster][metric]
            upper = evidence["upper_confidence_bound_95"]
            checks[f"{metric}_{cluster}_upper"] = (
                evidence["block_count"] >= 2 and upper is not None
                and upper <= policy[metric]["maximum_one_sided_95_upper"])
    book_checks, sparse_books = {}, []
    for book, report in sorted(per_book.items()):
        if report["matches"] < policy["individual_book_minimum_matches_for_rule"]:
            sparse_books.append(book)
            continue
        point = report["paired_log_loss_difference"]["estimate"]
        book_checks[book] = point is not None and point <= policy["individual_book_maximum_point_log_loss_difference"]
        checks[f"individual_book_{book}_point"] = book_checks[book]
    return {
        "qualified": all(checks.values()), "matches": matches, "source_days": source_days,
        "checks": checks, "failed_checks": [name for name, passed in checks.items() if not passed],
        "uncertainty": uncertainty, "individual_book_checks": book_checks,
        "sparse_book_comparisons_inconclusive": sparse_books, "frozen_policy": policy,
        "scope": "previously unscored retrospective paired one-hour available-book consensus cohort",
        "execution_or_profitability_claim": False, "future_performance_guarantee": False,
    }


def history_lag_sensitivity(consensus_rows, paired_rows, probabilities, policy, *, test_start):
    """Re-score the identical quotes/population, including every individual book."""
    consensus = [{**row, "p_model": probabilities[row["hltv_match_id"]]} for row in consensus_rows]
    pairs = [{**row, "p_model": probabilities[row["hltv_match_id"]]} for row in paired_rows]
    per_book = {}
    for book in BOOKS:
        rows = [row for row in pairs if row["bookmaker"] == book]
        per_book[book] = {
            "matches": len(rows), "model": _metrics(rows, "p_model"),
            "bookmaker": _metrics(rows, "p_bookmaker"),
            "paired_log_loss_difference": _paired_difference(rows),
            "paired_brier_difference": _paired_difference(rows, "brier"),
        }
    result = competitiveness_decision(consensus, per_book, policy, test_start=test_start)
    result["per_book_paired_cohorts"] = per_book
    return result
