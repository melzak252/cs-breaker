#!/usr/bin/env python3
"""CLI runner for the standard walk-forward model benchmark for CS2.

Evaluates a candidate prediction column against ground truth and an optional
comparative baseline according to the standard promotion benchmark protocol.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.analysis.model_benchmark import (
    compute_baseline_comparison,
    compute_benchmark_summary,
)
from src.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unified model evaluation and promotion benchmark for CS2."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data/05_model_input/canonical_benchmark_paired.csv",
        help="Path to CSV or Parquet evaluation dataset.",
    )
    parser.add_argument(
        "--candidate-col",
        default="model_prob",
        help="Candidate model prediction column.",
    )
    parser.add_argument(
        "--baseline-col",
        default="book_prob",
        help="Comparative baseline prediction column.",
    )
    parser.add_argument(
        "--target-col",
        default="winner",
        help="Ground truth binary label column (default: winner).",
    )
    parser.add_argument(
        "--model-name",
        default="Candidate Model",
        help="Human-readable model name.",
    )
    parser.add_argument(
        "--baseline-name",
        default="Market Baseline",
        help="Comparative baseline name.",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=5000,
        help="Number of bootstrap resamples (default: 5000).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save JSON benchmark output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data.exists():
        alt = PROJECT_ROOT / "artifacts/bookmaker_comparison_matches.csv"
        if alt.exists():
            args.data = alt
        else:
            print(f"Error: dataset not found: {args.data}", file=sys.stderr)
            return 1

    df = pd.read_csv(args.data)
    if args.target_col not in df.columns:
        for c in ["target", "winner", "y_true", "y"]:
            if c in df.columns:
                args.target_col = c
                break

    if args.candidate_col not in df.columns:
        for c in ["p_model", "model_prob", "p_team1", "prob_team1", "selected_blend"]:
            if c in df.columns:
                args.candidate_col = c
                break

    if args.baseline_col not in df.columns:
        for c in ["p_bookmaker", "book_prob", "pinnacle_prob", "consensus_prob", "unibet_prob"]:
            if c in df.columns:
                args.baseline_col = c
                break

    if args.target_col not in df.columns:
        raise KeyError(f"Could not find target column in {df.columns.tolist()}")
    if args.candidate_col not in df.columns:
        raise KeyError(f"Could not find candidate column in {df.columns.tolist()}")
    y_true = df[args.target_col].astype(int).to_numpy()
    p_cand = df[args.candidate_col].astype(float).to_numpy()

    cand_summary = compute_benchmark_summary(y_true, p_cand)
    print(f"\n=== Model Benchmark: {args.model_name} (N = {cand_summary.sample_size}) ===")
    print(f"  Log Loss:              {cand_summary.log_loss:.6f}")
    print(f"  Brier Score:           {cand_summary.brier_score:.6f}")
    print(f"    - Reliability:       {cand_summary.brier_reliability:.6f}")
    print(f"    - Resolution:        {cand_summary.brier_resolution:.6f}")
    print(f"    - Uncertainty:       {cand_summary.brier_uncertainty:.6f}")
    print(f"  ROC AUC:               {cand_summary.auc:.4f}")
    print(f"  Accuracy:              {cand_summary.accuracy * 100:.2f}%")
    print(f"  ECE (10 Bins):         {cand_summary.ece:.4f}")
    print(f"  MCE:                   {cand_summary.mce:.4f}")
    print(f"  Calibration Slope:     {cand_summary.calibration_slope:.4f}")

    results = {
        "candidate": {
            "model_name": args.model_name,
            **cand_summary.__dict__,
        }
    }

    if args.baseline_col and args.baseline_col in df.columns:
        p_base = df[args.baseline_col].astype(float).to_numpy()
        base_summary = compute_benchmark_summary(y_true, p_base)
        comp = compute_baseline_comparison(
            y_true=y_true,
            candidate_probs=p_cand,
            baseline_probs=p_base,
            baseline_name=args.baseline_name,
            n_bootstraps=args.bootstraps,
        )
        print(f"\n=== Comparative Baseline: {args.baseline_name} ===")
        print(f"  Baseline Log Loss:     {base_summary.log_loss:.6f}")
        print(f"  Delta Log Loss:        {comp.delta_log_loss:+.6f}")
        print(f"  95% Bootstrap CI:      [{comp.ci_lower_log_loss:.6f}, {comp.ci_upper_log_loss:.6f}]")
        print(f"  Empirical p-value:     {comp.p_value_log_loss:.4f}")
        print(f"  Statistically Sig:     {'YES (beats baseline)' if comp.is_statistically_significant else 'NO (CI includes zero)'}")
        print(f"  Delta Brier:           {comp.delta_brier:+.6f}")
        print(f"  Delta Accuracy:        {comp.delta_accuracy * 100:+.2f}%")

        results["baseline"] = {
            "baseline_name": args.baseline_name,
            **base_summary.__dict__,
        }
        results["comparison"] = comp.__dict__

    if args.output_json:
        save_json(results, args.output_json)
        print(f"\nBenchmark results saved to: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
