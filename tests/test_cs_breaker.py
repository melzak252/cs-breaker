import csv
import math
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from cs_breaker import Elo, Match, evaluate, load_results, predict_matchup, score


def match(key, day, team1="A", team2="B", y=1):
    return Match(key, datetime(2024, 1, day, 12, tzinfo=timezone.utc), team1, team2, y)


class TemporalTests(unittest.TestCase):
    def test_same_day_results_do_not_change_any_prediction_that_day(self):
        model = Elo(32)
        rows = model.process_day([match("1", 1), match("2", 1, y=0)])
        self.assertEqual([r["p_elo"] for r in rows], [0.5, 0.5])
        self.assertEqual([r["prior_matches_team1"] for r in rows], [0, 0])
        self.assertEqual(model.ratings["A"], 1500)
        self.assertEqual(model.counts["A"], 2)

    def test_earlier_day_updates_strength_and_preserves_zero_sum(self):
        model = Elo(32)
        model.process_day([match("1", 1)])
        row = model.process_day([match("2", 2)])[0]
        self.assertAlmostEqual(row["p_elo"], 1 / (1 + 10 ** (-32 / 400)))
        self.assertEqual(row["prior_matches_team1"], 1)
        self.assertAlmostEqual(sum(model.ratings.values()), 3000)

    def test_batch_order_does_not_change_predictions_or_ratings(self):
        fixtures = [match("1", 1), match("2", 1, "A", "C", 0)]
        first, second = Elo(24), Elo(24)
        a = first.process_day(fixtures)
        b = second.process_day(list(reversed(fixtures)))
        self.assertEqual({r["match_id"]: r["p_elo"] for r in a},
                         {r["match_id"]: r["p_elo"] for r in b})
        self.assertEqual(first.ratings, second.ratings)

    def test_processing_the_same_or_earlier_day_is_rejected(self):
        model = Elo(32)
        model.process_day([match("1", 2)])
        for day in (1, 2):
            with self.assertRaises(ValueError):
                model.process_day([match("2", day)])

    def test_final_test_labels_do_not_change_selection_or_earlier_predictions(self):
        history = [match("1", 1), match("2", 2), match("3", 3),
                   match("4", 4, y=0), match("5", 5)]
        a, rows_a, _ = evaluate(history, date(2024, 1, 3), date(2024, 1, 5))
        changed = history[:-1] + [match("5", 5, y=0)]
        b, rows_b, _ = evaluate(changed, date(2024, 1, 3), date(2024, 1, 5))
        self.assertEqual(a["selected_k"], b["selected_k"])
        self.assertEqual(a["validation_candidates"], b["validation_candidates"])
        self.assertEqual([r["p_elo"] for r in rows_a], [r["p_elo"] for r in rows_b])
        self.assertEqual(rows_a[-1]["phase"], "test")

    def test_empty_cohort_is_not_silently_scored(self):
        with self.assertRaises(ValueError):
            evaluate([match("1", 1), match("2", 2)], date(2024, 1, 3), date(2024, 1, 5))


class InputTests(unittest.TestCase):
    def write_csv(self, path, rows):
        fields = ["hltv_match_id", "date", "team1_name", "team2_name", "winner",
                  "score_team1", "score_team2", "match_type"]
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, key="1", score1="2", score2="0", winner="team1"):
        return dict(hltv_match_id=key, date="2024-01-01T12:00:00Z",
                    team1_name="A", team2_name="B", winner=winner,
                    score_team1=score1, score_team2=score2, match_type="bo3")

    def test_inconsistent_format_rows_are_excluded_not_relabelled(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.csv"
            self.write_csv(path, [self.row(), self.row("2", "13", "6"),
                                  self.row("3", "3", "2")])
            matches, excluded = load_results(path)
        self.assertEqual([m.match_id for m in matches], ["1"])
        self.assertEqual([r["hltv_match_id"] for r in excluded], ["2", "3"])
        self.assertEqual(excluded[0]["score_team1"], "13")

    def test_duplicate_identity_and_conflicting_winner_are_errors(self):
        for rows in ([self.row(), self.row()], [self.row(winner="team2")]):
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "input.csv"
                self.write_csv(path, rows)
                with self.assertRaises(ValueError):
                    load_results(path)

    def test_timezone_offsets_are_converted_before_daily_batching(self):
        row = self.row()
        row["date"] = "2024-01-01T23:30:00-03:00"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.csv"
            self.write_csv(path, [row])
            matches, _ = load_results(path)
        self.assertEqual(matches[0].timestamp.date(), date(2024, 1, 2))


class MetricTests(unittest.TestCase):
    def test_log_loss_brier_auc_and_calibration_have_known_values(self):
        metrics = score([0, 1], [0.25, 0.75])
        self.assertAlmostEqual(metrics["log_loss"], -math.log(0.75))
        self.assertAlmostEqual(metrics["brier"], 0.0625)
        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["roc_auc"], 1)
        self.assertAlmostEqual(metrics["ece_10_bins"], 0.25)
        tied = score([0, 1], [0.5, 0.5])
        self.assertEqual(tied["roc_auc"], 0.5)
        self.assertEqual(tied["accuracy"], 0.5)

    def test_single_class_auc_is_undefined_and_inputs_must_align(self):
        self.assertIsNone(score([1], [0.9])["roc_auc"])
        for labels, probs in [([], []), ([0, 1], [0.5]), ([1], [float("nan")]),
                              ([0], [-0.1]), ([2], [0.5])]:
            with self.subTest(labels=labels, probs=probs), self.assertRaises(ValueError):
                score(labels, probs)


class PredictionTests(unittest.TestCase):
    def model(self):
        return dict(schema_version=1, selected_k=32, history_through="2024-01-01",
                    ratings={"A": 1516, "B": 1484, "MONEY CREW": 1500, "Money Crew": 1490},
                    counts={"A": 1, "B": 1, "MONEY CREW": 1, "Money Crew": 1},
                    last_played={"A": "2024-01-01", "B": "2024-01-01",
                                 "MONEY CREW": "2024-01-01", "Money Crew": "2024-01-01"})

    def test_orientation_complements_and_exact_names_take_precedence(self):
        model = self.model()
        ab = predict_matchup(model, "a", "B")
        ba = predict_matchup(model, "B", "A")
        self.assertAlmostEqual(ab["p_team1"] + ba["p_team1"], 1)
        self.assertEqual(ab["forecast_kind"], "historical_snapshot")
        self.assertEqual(predict_matchup(model, "MONEY CREW", "A")["team1"], "MONEY CREW")

    def test_unknown_ambiguous_and_self_matchups_are_rejected(self):
        model = self.model()
        for a, b in [("unknown", "A"), ("money crew", "A"), ("A", "a")]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                predict_matchup(model, a, b)


if __name__ == "__main__":
    unittest.main()
