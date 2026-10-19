"""Unit tests for SOTA components: VetoEngine, order antisymmetry, and temperature calibration."""

import unittest
from datetime import datetime, timezone

from pipeline.contracts import MatchFixture
from pipeline.features.builder import ChronologicalFeatureBuilder
from pipeline.features.veto_engine import VetoEngine
from pipeline.models.trainer import ModelTrainer, TemperatureCalibrator


class TestSotaPipeline(unittest.TestCase):
    def test_veto_engine_simulation_and_expansion(self):
        ve = VetoEngine()
        picked, decider = ve.simulate_veto("FaZe", "Spirit")
        self.assertEqual(len(picked), 2)
        self.assertIn(decider, ve.active_pool)
        self.assertNotIn(decider, picked)

        # Test exact BO3 probability expansion
        p_bo3 = ve.bo3_series_probability(0.60, 0.60, 0.60)
        self.assertAlmostEqual(p_bo3, 0.6480, places=4)
        p_equal = ve.bo3_series_probability(0.50, 0.50, 0.50)
        self.assertAlmostEqual(p_equal, 0.5000, places=4)

    def test_temperature_calibrator_shrinks_logits(self):
        cal = TemperatureCalibrator(temperature=1.5)
        # Moderate probability
        p_shrunk = cal.transform(0.85)
        self.assertLess(p_shrunk, 0.85)
        self.assertGreater(p_shrunk, 0.50)

        # Underdog probability
        p_underdog = cal.transform(0.15)
        self.assertGreater(p_underdog, 0.15)
        self.assertLess(p_underdog, 0.50)

        # Neutral 0.50 is invariant
        self.assertAlmostEqual(cal.transform(0.50), 0.50, places=6)

    def test_pairwise_antisymmetry_invariance(self):
        trainer = ModelTrainer()
        trainer.models = {"test": {"kind": "elo"}}
        trainer.best_model_name = "test"
        trainer.feature_columns = ["elo_diff", "elo_prob"]

        f_ab = {"elo_diff": 120.0, "elo_prob": 0.666}
        f_ba = {"elo_diff": -120.0, "elo_prob": 0.334}

        p_ab = trainer.predict_proba_symmetric("test", f_ab, inverted_feature_dict=f_ba)
        p_ba = trainer.predict_proba_symmetric("test", f_ba, inverted_feature_dict=f_ab)

        self.assertAlmostEqual(p_ab + p_ba, 1.0000, places=5)

    def test_feature_builder_includes_veto_features(self):
        builder = ChronologicalFeatureBuilder()
        fixture = MatchFixture(
            match_id="test_veto_m1",
            tournament_name="Major",
            timestamp=datetime.now(timezone.utc),
            team1="Spirit",
            team2="MOUZ",
            maps=["de_mirage", "de_nuke"],
        )
        features = builder.build_match_features(fixture)
        self.assertIn("veto_pool_wr_diff", features)
        self.assertIn("veto_hierarchical_bo3_prob", features)
        self.assertIn("veto_map_depth_diff", features)


if __name__ == "__main__":
    unittest.main()
