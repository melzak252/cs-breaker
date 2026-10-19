"""Unit tests for Kedro 8-layer simulation pipeline, model benchmark, risk guards, and Elo engine."""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.analysis.betting_guards_a_to_d import (
    apply_guard_a_plow,
    apply_guard_b_dynamic_disagreement,
    apply_guard_c_haircut,
    apply_guard_d_clv_drift,
    apply_guard_w_wasserstein_dro,
    apply_model_consensus_veto,
    calculate_clv,
    devig_additive,
    devig_multiplicative,
    devig_power,
    devig_shin,
    dynamic_logit_disagreement,
    wasserstein_dro_min_ev_hurdle,
)
from src.analysis.model_benchmark import (
    brier_decomposition,
    calculate_ece,
    clip_probabilities,
    compute_baseline_comparison,
    compute_benchmark_summary,
)
from src.analysis.simulation_pipeline import (
    run_stage1_verify_inputs,
    run_stage2_model_benchmark,
    run_stage3_betting_simulation,
    run_stage4_econometric_audit,
    run_stage5_generate_report,
)
from src.ratings.elo import EloRatingSystem, win_probability


class TestDeviggingMethods:
    def test_multiplicative_devig(self):
        pa, pb = devig_multiplicative(2.0, 2.0)
        assert pytest.approx(pa, 1e-6) == 0.5
        assert pytest.approx(pb, 1e-6) == 0.5

        # With overround: 1.90 and 1.90
        pa, pb = devig_multiplicative(1.90, 1.90)
        assert pytest.approx(pa + pb, 1e-6) == 1.0
        assert pytest.approx(pa, 1e-6) == 0.5

    def test_additive_devig(self):
        pa, pb = devig_additive(1.90, 1.90)
        assert pytest.approx(pa + pb, 1e-6) == 1.0
        assert pytest.approx(pa, 1e-6) == 0.5

    def test_power_devig(self):
        pa, pb = devig_power(1.80, 2.10)
        assert pytest.approx(pa + pb, 1e-6) == 1.0
        assert pa > pb

    def test_shin_devig_analytical_2way(self):
        pa, pb = devig_shin(1.90, 1.90)
        assert pytest.approx(pa + pb, 1e-6) == 1.0
        assert pytest.approx(pa, 1e-6) == 0.5

        pa2, pb2 = devig_shin(1.50, 2.70)
        assert pytest.approx(pa2 + pb2, 1e-6) == 1.0
        assert pa2 > pb2

    def test_invalid_odds_raise(self):
        with pytest.raises(ValueError):
            devig_multiplicative(0.9, 2.0)
        with pytest.raises(ValueError):
            devig_shin(2.0, -1.0)


class TestRiskGuards:
    def test_guard_b_dynamic_disagreement(self):
        # 0.50 vs 0.50 should have 0 disagreement
        assert dynamic_logit_disagreement(0.5, 0.5) == 0.0

        # Close agreement passes guard
        passes, gap = apply_guard_b_dynamic_disagreement(0.52, 0.50, max_logit_gap=0.85)
        assert passes is True
        assert gap < 0.85

        # Extreme divergence fails guard
        passes, gap = apply_guard_b_dynamic_disagreement(0.95, 0.20, max_logit_gap=0.85)
        assert passes is False
        assert gap > 0.85

    def test_model_consensus_veto(self):
        passes, gap = apply_model_consensus_veto(0.55, 0.52, max_gap=0.10)
        assert passes is True
        assert pytest.approx(gap, 1e-6) == 0.03

        passes, gap = apply_model_consensus_veto(0.70, 0.55, max_gap=0.10)
        assert passes is False
        assert pytest.approx(gap, 1e-6) == 0.15

    def test_guard_a_plow(self):
        # p_low = 0.60, odds = 2.0, tax = 0.12 -> net return = 2.0 * 0.88 = 1.76 -> EV = 0.60 * 1.76 - 1 = +0.056
        passes, ev = apply_guard_a_plow(0.60, 2.0, tax_rate=0.12, min_ev=0.0)
        assert passes is True
        assert ev > 0.0

    def test_guard_c_haircut(self):
        passes, ev_disc = apply_guard_c_haircut(raw_ev=0.15, haircut_multiplier=0.30, safety_margin=0.02)
        # 0.30 * 0.15 - 0.02 = 0.045 - 0.02 = 0.025 > 0
        assert passes is True
        assert pytest.approx(ev_disc, 1e-6) == 0.025

    def test_guard_d_clv(self):
        # taken odds 2.10, tax 12%, closing 1.80
        clv = calculate_clv(2.10, 1.80, tax_rate=0.12)
        # (2.10 * 0.88) / 1.80 - 1 = 1.848 / 1.80 - 1 = +0.02667
        assert clv > 0.0

        passes, drift = apply_guard_d_clv_drift(0.55, 0.54, max_adverse_drift=0.03)
        assert passes is True
        assert pytest.approx(drift, 1e-6) == 0.01

    def test_guard_w_wasserstein_dro(self):
        hurdle = wasserstein_dro_min_ev_hurdle(odds=2.0, tax_rate=0.12, base_min_ev=0.05, epsilon=0.025)
        assert hurdle >= 0.05

        passes, ev_dro, _ = apply_guard_w_wasserstein_dro(
            prob_model=0.65,
            odds=2.0,
            tax_rate=0.12,
            base_min_ev=0.05,
            epsilon=0.025,
        )
        assert passes is True
        assert ev_dro > 0.0


class TestModelBenchmarkMetrics:
    def test_calculate_ece_and_brier(self):
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.85, 0.15, 0.65, 0.75])

        ece, mce = calculate_ece(y_true, y_prob, n_bins=5)
        assert 0.0 <= ece <= 1.0
        assert 0.0 <= mce <= 1.0

        rel, res, unc = brier_decomposition(y_true, y_prob, n_bins=5)
        assert rel >= 0.0
        assert res >= 0.0
        assert unc >= 0.0

    def test_compute_benchmark_summary(self):
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.8, 0.2, 0.7, 0.9, 0.1, 0.3, 0.6, 0.4])

        summary = compute_benchmark_summary(y_true, y_prob)
        assert summary.sample_size == 8
        assert summary.log_loss > 0.0
        assert 0.0 <= summary.brier_score <= 1.0
        assert summary.auc >= 0.5
        assert summary.accuracy >= 0.5

    def test_compute_baseline_comparison(self):
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 10)
        p_cand = np.array([0.9, 0.1, 0.85, 0.95, 0.05, 0.15, 0.8, 0.2] * 10)
        p_base = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] * 10)

        comp = compute_baseline_comparison(y_true, p_cand, p_base, baseline_name="Constant_Prior", n_bootstraps=200)
        assert comp.delta_log_loss < 0.0  # Candidate model beats coin-flip baseline
        assert comp.delta_brier < 0.0


class TestEloRatingSystem:
    def test_win_probability_symmetry(self):
        p1 = win_probability(1600.0, 1500.0)
        p2 = win_probability(1500.0, 1600.0)
        assert pytest.approx(p1 + p2, 1e-6) == 1.0
        assert p1 > 0.5
        assert p2 < 0.5

    def test_elo_updates(self):
        elo = EloRatingSystem(k=32.0, default_rating=1500.0)
        r1, r2 = elo.update("TeamA", "TeamB", outcome_team1=1.0)
        assert r1 > 1500.0
        assert r2 < 1500.0
        assert pytest.approx(r1 - 1500.0, 1e-6) == pytest.approx(1500.0 - r2, 1e-6)


class TestSimulationPipelineStages:
    def test_end_to_end_stages(self, tmp_path):
        catalog = "conf/base/catalog.yml"
        params = "conf/base/parameters.yml"
        out_dir = tmp_path / "simulation_pipeline"

        # Stage 1: Verify inputs
        st1 = run_stage1_verify_inputs(catalog, params, out_dir)
        assert st1["stage"] == "verify_inputs"
        assert st1["datasets_verified"] > 0
        assert (out_dir / "stage1_input_manifest.json").exists()

        # Stage 2: Model benchmark
        st2 = run_stage2_model_benchmark(catalog, params, out_dir)
        assert st2["stage"] == "model_benchmark"
        assert len(st2["shrinkage_grid"]) > 0
        assert (out_dir / "stage2_model_benchmark.json").exists()
        assert (out_dir / "stage2_shrinkage_grid.csv").exists()

        # Stage 3: Betting simulation
        st3 = run_stage3_betting_simulation(catalog, params, out_dir)
        assert st3["stage"] == "betting_simulation"
        assert len(st3["strategies"]) == 4
        assert (out_dir / "stage3_betting_simulation.json").exists()
        assert (out_dir / "stage3_strategy_comparison.csv").exists()

        # Stage 4: Econometric audit
        st4 = run_stage4_econometric_audit(catalog, params, out_dir)
        assert st4["stage"] == "econometric_audit"
        assert len(st4["noise_decay"]) > 0
        assert (out_dir / "stage4_econometric_audit.json").exists()
        assert (out_dir / "stage4_noise_decay_table.csv").exists()

        # Stage 5: Generate report
        st5 = run_stage5_generate_report(catalog, params, out_dir)
        assert st5["pipeline_status"] == "completed"
        assert (out_dir / "summary_metrics.json").exists()
        assert (out_dir / "SIMULATION_REPORT.md").exists()
