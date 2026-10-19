"""Unit tests for Glicko-2, TrueSkill, and OpenSkill rating systems and benchmark."""

import unittest
from datetime import datetime, timezone

from pipeline.contracts import MatchFixture
from pipeline.features.ratings_glicko2 import Glicko2RatingEngine
from pipeline.features.ratings_trueskill import TrueSkillRatingEngine
from pipeline.features.ratings_openskill import OpenSkillRatingEngine
from pipeline.models.rating_benchmark import RatingBenchmark


class TestAdvancedRatingEngines(unittest.TestCase):
    def setUp(self):
        self.t1_players = ["s1mple", "b1t", "electronic", "perfecto", "boombl4"]
        self.t2_players = ["device", "dupreeh", "gla1ve", "magisk", "xyp9x"]

    def test_glicko2_predict_and_update(self):
        engine = Glicko2RatingEngine()
        prob = engine.predict_win("NaVi", "Astralis", self.t1_players, self.t2_players)
        self.assertAlmostEqual(prob, 0.50, delta=0.01)

        features = engine.build_features("NaVi", "Astralis", self.t1_players, self.t2_players)
        self.assertIn("glicko2_rating_diff", features)
        self.assertIn("glicko2_rd_diff", features)
        self.assertIn("glicko2_prob", features)

        # Team 1 wins
        engine.update_match("NaVi", "Astralis", self.t1_players, self.t2_players, winner="team1")
        new_prob = engine.predict_win("NaVi", "Astralis", self.t1_players, self.t2_players)
        self.assertGreater(new_prob, 0.50)

    def test_trueskill_predict_and_update(self):
        engine = TrueSkillRatingEngine()
        prob = engine.predict_win("NaVi", "Astralis", self.t1_players, self.t2_players)
        self.assertAlmostEqual(prob, 0.50, delta=0.01)

        features = engine.build_features("NaVi", "Astralis", self.t1_players, self.t2_players)
        self.assertIn("trueskill_mu_diff", features)
        self.assertIn("trueskill_prob", features)
        self.assertIn("trueskill_sigma_diff", features)

        engine.update_match("NaVi", "Astralis", self.t1_players, self.t2_players, winner="team1")
        new_prob = engine.predict_win("NaVi", "Astralis", self.t1_players, self.t2_players)
        self.assertGreater(new_prob, 0.50)

    def test_openskill_predict_and_update(self):
        engine = OpenSkillRatingEngine()
        prob = engine.predict_win("NaVi", "Astralis", self.t1_players, self.t2_players)
        self.assertAlmostEqual(prob, 0.50, delta=0.05)

        features = engine.build_features("NaVi", "Astralis", self.t1_players, self.t2_players)
        self.assertIn("openskill_mu_diff", features)
        self.assertIn("openskill_prob", features)
        self.assertIn("openskill_ordinal_diff", features)

        engine.update_match("NaVi", "Astralis", self.t1_players, self.t2_players, winner="team1")
        new_prob = engine.predict_win("NaVi", "Astralis", self.t1_players, self.t2_players)
        self.assertGreater(new_prob, 0.50)


class TestRatingBenchmark(unittest.TestCase):
    def test_benchmark_runs_without_leakage(self):
        matches = [
            MatchFixture(
                match_id=f"m_{i}",
                tournament_name="Major",
                timestamp=datetime(2024, 1, 1 + i, tzinfo=timezone.utc),
                team1="TeamA",
                team2="TeamB",
                score1=16 if i % 2 == 0 else 10,
                score2=10 if i % 2 == 0 else 16,
                winner="team1" if i % 2 == 0 else "team2",
                team1_players=["p1", "p2", "p3", "p4", "p5"],
                team2_players=["q1", "q2", "q3", "q4", "q5"],
            )
            for i in range(20)
        ]
        bench = RatingBenchmark()
        results = bench.run_benchmark(matches, train_ratio=0.7)
        self.assertIn("leaderboard", results)
        self.assertEqual(len(results["leaderboard"]), 5)
        systems = [row["system"] for row in results["leaderboard"]]
        self.assertIn("glicko2", systems)
        self.assertIn("trueskill", systems)
        self.assertIn("openskill", systems)
        self.assertIn("team_elo", systems)
        self.assertIn("player_lineup_elo", systems)


if __name__ == "__main__":
    unittest.main()
