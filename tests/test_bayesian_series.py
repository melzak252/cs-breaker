"""Consumer-visible probability, observation and chronology invariants."""
import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import expit

from pipeline.features.bayesian_series import (
    BayesianSeriesEngine, SkillSpec, build_bayesian_features, score_distribution,
)


def game(identity, day, team1="A", team2="B", score1=2, score2=0, format="bo3"):
    return {"hltv_match_id": str(identity), "source_date": day, "team1": team1,
            "team2": team2, "score1": score1, "score2": score2, "format": format,
            "target": int(score1 > score2), "kind": "series" if format in ("bo3", "bo5") else "map"}


def spec(mode="score", half_life=None):
    return SkillSpec(name="example", update_mode=mode, prior_variance=1.0,
                     variance_per_day=0.0025 if half_life is None else 0,
                     mean_half_life_days=half_life)


class BayesianSeriesTests(unittest.TestCase):
    def test_stopped_score_probabilities_not_fixed_length_binomial(self):
        result = score_distribution(math.log(0.7 / 0.3), 0, "bo3")
        expected = {"2-0": 0.49, "2-1": 0.294, "0-2": 0.09, "1-2": 0.126}
        self.assertEqual(set(result), set(expected))
        for score, probability in expected.items():
            self.assertAlmostEqual(result[score], probability, places=12)

    def test_uncertainty_is_integrated_before_series_aggregation(self):
        mean, variance = 1.2, 2.5
        result = score_distribution(mean, variance, "bo5")
        expected = quad(lambda z: (expit(mean + math.sqrt(variance) * z) ** 3
                                  * (1 + 3 * expit(-mean - math.sqrt(variance) * z)
                                     + 6 * expit(-mean - math.sqrt(variance) * z) ** 2))
                        * math.exp(-z * z / 2) / math.sqrt(2 * math.pi), -10, 10,
                        epsabs=1e-12)[0]
        actual = sum(value for key, value in result.items() if key.startswith("3-"))
        self.assertAlmostEqual(actual, expected, places=7)
        plugin = sum(value for key, value in score_distribution(mean, 0, "bo5").items()
                     if key.startswith("3-"))
        self.assertGreater(plugin - actual, 0.05)

    def test_terminal_distribution_normalizes_and_reverses_sides(self):
        for format in ("bo3", "bo5"):
            for mean, variance in ((0, 2), (2, 4), (-4, 1), (15, 0.1)):
                with self.subTest(format=format, mean=mean, variance=variance):
                    result = score_distribution(mean, variance, format)
                    reverse = score_distribution(-mean, variance, format)
                    self.assertAlmostEqual(sum(result.values()), 1, places=12)
                    self.assertTrue(all(0 <= value <= 1 for value in result.values()))
                    for key, value in result.items():
                        self.assertAlmostEqual(value, reverse["-".join(reversed(key.split("-")))], places=12)

    def test_score_information_distinguishes_sweep_from_close_win(self):
        probabilities = {}
        for mode in ("winner", "score"):
            for loser_score in (0, 1):
                engine = BayesianSeriesEngine(specs=(spec(mode),))
                engine.update_day("2026-01-01", [game(1, "2026-01-01", score2=loser_score)])
                probabilities[mode, loser_score] = engine.features("A", "B", "2026-01-02")["example_prob"]
        self.assertAlmostEqual(probabilities["winner", 0], probabilities["winner", 1], places=12)
        self.assertGreater(probabilities["score", 0], probabilities["score", 1])
        self.assertGreater(probabilities["score", 1], 0.5)

    def test_map_round_margin_is_not_independent_match_evidence(self):
        forecasts = []
        for loser_score in (0, 11):
            engine = BayesianSeriesEngine(specs=(spec(),))
            engine.update_day("2026-01-01", [game(1, "2026-01-01", score1=13, score2=loser_score, format="mrg")])
            forecasts.append(engine.features("A", "B", "2026-01-02"))
            details = engine.diagnostics("A", "B", "2026-01-02", channel="example")
            self.assertEqual(details["team1_prior_maps"], 1)
            self.assertEqual(details["team1_prior_series"], 0)
        self.assertEqual(forecasts[0], forecasts[1])

    def test_same_day_batch_order_does_not_change_forecasts(self):
        games = [game(1, "2026-01-01"), game(2, "2026-01-01", "A", "C", 0, 2),
                 game(3, "2026-01-01", "B", "C", 2, 1)]
        first, second = BayesianSeriesEngine(), BayesianSeriesEngine()
        first.update_day("2026-01-01", games)
        second.update_day("2026-01-01", list(reversed(games)))
        self.assertEqual(first.features("A", "C", "2026-01-02"), second.features("A", "C", "2026-01-02"))
        for column, value in first.features("A", "B", "2026-01-02").items():
            self.assertAlmostEqual(value + first.features("B", "A", "2026-01-02")[column], 1, places=12)

    def test_invalid_day_batch_does_not_commit_partial_outcomes(self):
        engine = BayesianSeriesEngine(specs=(spec(),))
        before = engine.features("A", "B", "2026-01-02")
        invalid = game(2, "2026-01-01")
        invalid["target"] = 0
        with self.assertRaises(ValueError):
            engine.update_day("2026-01-01", [game(1, "2026-01-01"), invalid])
        self.assertEqual(engine.features("A", "B", "2026-01-02"), before)
        engine.update_day("2026-01-01", [game(1, "2026-01-01")])
        with self.assertRaises(ValueError):
            engine.update_day("2026-01-02", [game(1, "2026-01-02")])
        self.assertEqual(engine.history_through, "2026-01-01")

    def test_queries_do_not_advance_state_and_reject_historical_forecasts(self):
        engine = BayesianSeriesEngine(specs=(spec(),))
        engine.update_day("2026-01-01", [game(1, "2026-01-01")])
        before = engine.features("A", "B", "2026-01-02")
        engine.features("A", "unknown", "2027-01-01")
        engine.diagnostics("unknown", "B", "2027-01-01", channel="example")
        self.assertEqual(before, engine.features("A", "B", "2026-01-02"))
        with self.assertRaises(ValueError):
            engine.features("A", "B", "2026-01-01")
        with self.assertRaises(ValueError):
            engine.features("A", "A", "2026-01-02")

    def test_time_dynamics_increase_random_walk_uncertainty_and_revert_ou(self):
        walk = BayesianSeriesEngine(specs=(spec(),))
        reverting = BayesianSeriesEngine(specs=(spec(half_life=90),))
        for engine in (walk, reverting):
            engine.update_day("2026-01-01", [game(1, "2026-01-01")])
        near = walk.diagnostics("A", "B", "2026-01-02", channel="example")
        far = walk.diagnostics("A", "B", "2027-01-02", channel="example")
        self.assertGreater(far["variance_sum"], near["variance_sum"])
        self.assertAlmostEqual(far["mean_difference"], near["mean_difference"], places=12)
        near = reverting.diagnostics("A", "B", "2026-01-02", channel="example")
        far = reverting.diagnostics("A", "B", "2027-01-02", channel="example")
        self.assertLess(abs(far["mean_difference"]), abs(near["mean_difference"]))

    def test_future_results_do_not_rewrite_prior_forecasts(self):
        prefix = [game(1, "2026-01-01"), game(2, "2026-01-02", score1=0, score2=2),
                  game(3, "2026-01-02", "A", "C")]
        before, _ = build_bayesian_features(prefix)
        after, _ = build_bayesian_features(prefix + [game(4, "2026-01-03")])
        columns = [name for name in before if name.endswith("_prob")]
        np.testing.assert_array_equal(before[columns].to_numpy(), after.iloc[:len(before)][columns].to_numpy())
        same_day = before.loc[before.hltv_match_id.astype(str).isin(["2", "3"])]
        self.assertEqual(set(same_day.history_through), {"2026-01-01"})

    def test_two_day_lag_excludes_yesterday_but_returns_complete_state(self):
        games = [game(1, "2026-01-01"), game(2, "2026-01-02"), game(3, "2026-01-03")]
        frame, engine = build_bayesian_features(games, history_lag_days=2)
        row = frame.loc[frame.hltv_match_id.astype(str) == "2"].iloc[0]
        self.assertTrue(all(row[name] == 0.5 for name in frame if name.endswith("_prob")))
        self.assertEqual(frame.iloc[-1].history_through, "2026-01-01")
        self.assertEqual(engine.history_through, "2026-01-03")

    def test_specs_and_queries_reject_invalid_mathematical_inputs(self):
        for kwargs in (
            {"name": ""}, {"update_mode": "rounds"}, {"prior_variance": 0},
            {"prior_variance": float("nan")}, {"variance_per_day": 0},
            {"mean_half_life_days": 90}, {"mean_half_life_days": 0, "variance_per_day": 0},
        ):
            values = {"name": "valid", "update_mode": "score", "prior_variance": 1,
                      "variance_per_day": .0025, "mean_half_life_days": None, **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SkillSpec(**values)
        with self.assertRaises(ValueError):
            BayesianSeriesEngine(specs=(spec(), spec()))
        for mean, variance in ((0, -1), (float("inf"), 1), (0, float("nan"))):
            with self.subTest(mean=mean, variance=variance), self.assertRaises(ValueError):
                score_distribution(mean, variance, "bo3")
        engine = BayesianSeriesEngine(specs=(spec(),))
        for teams, day, format in (
            (("", "B"), "2026-01-01", "bo3"),
            (("A", "B"), "20260101", "bo3"),
            (("A", "B"), "2026-01-01", "mrg"),
        ):
            with self.subTest(teams=teams, day=day, format=format), self.assertRaises(ValueError):
                engine.features(*teams, day, format)
        with self.assertRaises(ValueError):
            engine.diagnostics("A", "B", "2026-01-01", channel="unknown")
        self.assertIsNone(engine.history_through)
        self.assertEqual(engine.teams, {})

    def test_score_posterior_matches_gaussian_conditioning_reference(self):
        # A sweep tilts the difference N(0, 2) by sigmoid(difference)**2.
        def integrand(x, power):
            difference = math.sqrt(2) * x
            return (difference ** power * expit(difference) ** 2
                    * math.exp(-x * x / 2) / math.sqrt(2 * math.pi))

        normalizer, first, second = [
            quad(lambda x: integrand(x, power), -10, 10, epsabs=1e-12)[0]
            for power in range(3)
        ]
        tilted_mean = first / normalizer
        tilted_variance = second / normalizer - tilted_mean ** 2
        engine = BayesianSeriesEngine(specs=(spec(),))
        engine.update_day("2026-01-01", [game(1, "2026-01-01")])
        detail = engine.diagnostics("A", "unseen", "2026-01-02", channel="example")
        self.assertAlmostEqual(detail["mean_difference"], .5 * tilted_mean, places=7)
        self.assertAlmostEqual(detail["variance_sum"],
                               .5 + .25 * tilted_variance + .0025 + 1, places=7)
        self.assertEqual(detail["team1_prior_series"], 1)
        self.assertEqual(detail["team1_prior_maps"], 2)

    def test_day_parallel_sites_add_natural_parameters_from_common_opening(self):
        observations = [game(1, "2026-01-01"), game(2, "2026-01-01", "A", "C", 1, 2)]
        single_moments = []
        for observation in observations:
            single = BayesianSeriesEngine(specs=(spec(),))
            single.update_day("2026-01-01", [observation])
            detail = single.diagnostics("A", "unseen", "2026-01-02", channel="example")
            single_moments.append((detail["mean_difference"], detail["variance_sum"] - 1 - .0025))
        expected_precision = sum(1 / variance for _, variance in single_moments) - 1
        expected_location = sum(mean / variance for mean, variance in single_moments)
        batch = BayesianSeriesEngine(specs=(spec(),))
        batch.update_day("2026-01-01", observations)
        detail = batch.diagnostics("A", "unseen", "2026-01-02", channel="example")
        self.assertAlmostEqual(detail["mean_difference"], expected_location / expected_precision, places=12)
        self.assertAlmostEqual(detail["variance_sum"], 1 / expected_precision + .0025 + 1, places=12)

    def test_reorienting_native_outcomes_leaves_team_posteriors_identical(self):
        original = game(1, "2026-01-01", "A", "B", 2, 1)
        reversed_sides = game(1, "2026-01-01", "B", "A", 1, 2)
        first, second = BayesianSeriesEngine(), BayesianSeriesEngine()
        first.update_day("2026-01-01", [original])
        second.update_day("2026-01-01", [reversed_sides])
        self.assertEqual(first.features("A", "B", "2026-01-02"),
                         second.features("A", "B", "2026-01-02"))


if __name__ == "__main__":
    unittest.main()
