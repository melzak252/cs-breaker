#!/usr/bin/env python3
"""Unified CLI Runner for CS-Breaker 5-Stage Simulation Pipeline.

Supports granular stage execution (--stage {stage_name}) and end-to-end runs (--stage all).
Enforces Kedro 8-layer architecture and integrates with DVC DAG workflows.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.simulation_pipeline import (
    run_stage1_verify_inputs,
    run_stage2_model_benchmark,
    run_stage3_betting_simulation,
    run_stage4_econometric_audit,
    run_stage5_generate_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Kedro 8-layer model & betting simulation pipeline for CS2."
    )
    parser.add_argument(
        "--stage",
        choices=[
            "verify_inputs",
            "model_benchmark",
            "betting_simulation",
            "econometric_audit",
            "generate_report",
            "all",
        ],
        default="all",
        help="Pipeline stage to execute (default: all).",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "conf/base/catalog.yml",
        help="Path to Kedro-style data catalog (default: conf/base/catalog.yml).",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=PROJECT_ROOT / "conf/base/parameters.yml",
        help="Path to simulation parameters (default: conf/base/parameters.yml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/08_reporting/simulation_pipeline",
        help="Output directory for generated stage artifacts.",
    )

    args = parser.parse_args()
    start_time = time.time()
    print(f"[pipeline] Executing simulation stage: '{args.stage}'...")

    try:
        if args.stage in ("verify_inputs", "all"):
            print("  -> Stage 1: Verifying input datasets against catalog specifications...")
            run_stage1_verify_inputs(args.catalog, args.params, args.output_dir)

        if args.stage in ("model_benchmark", "all"):
            print("  -> Stage 2: Running multi-model benchmark and Bayesian shrinkage grid...")
            run_stage2_model_benchmark(args.catalog, args.params, args.output_dir)

        if args.stage in ("betting_simulation", "all"):
            print("  -> Stage 3: Executing decision-time betting simulations under tax regimes...")
            run_stage3_betting_simulation(args.catalog, args.params, args.output_dir)

        if args.stage in ("econometric_audit", "all"):
            print("  -> Stage 4: Executing econometric audit and noise decay calculations...")
            run_stage4_econometric_audit(args.catalog, args.params, args.output_dir)

        if args.stage in ("generate_report", "all"):
            print("  -> Stage 5: Consolidating summary metrics and generating SIMULATION_REPORT.md...")
            run_stage5_generate_report(args.catalog, args.params, args.output_dir)

        elapsed = time.time() - start_time
        print(f"[pipeline] Stage '{args.stage}' completed successfully in {elapsed:.2f}s.")
        return 0

    except Exception as exc:
        print(f"[pipeline] ERROR in stage '{args.stage}': {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
