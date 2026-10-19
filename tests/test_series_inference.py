"""Persisted organization forecasts cannot consume their own completed history."""
import unittest

import pandas as pd

from pipeline.features.series_history import SeriesHistoryEngine
from pipeline.features.valve_standings import StandingsHistory
from pipeline.models.series_forecast import fit_candidate, predict_series_scenario
from scripts.evaluate_bookmaker_challenge import forecast_origins


class SeriesInferenceTests(unittest.TestCase):
    def setUp(self):
        column = "series_elo_32_180_prob"
        train = pd.DataFrame({"source_date": ["2026-04-01", "2026-04-02"],
                              "target": [1, 0], column: [0.7, 0.3]})
        self.model = fit_candidate(
            {"name": "raw_example", "family": "rating", "feature_columns": [column],
             "parameters": {}, "recency_half_life_days": None}, train,
            {column: {"operation": "complement", "source": column}}, train_end="2026-04-30")
        engine = SeriesHistoryEngine()
        engine.update_day("2026-05-01", [{"hltv_match_id": "1", "source_date": "2026-05-01",
                                          "team1": "Alpha", "team2": "Beta", "target": 1,
                                          "format": "bo3", "score1": 2, "score2": 0, "kind": "series"}])
        self.state = {"schema_version": 2, "engine": engine, "standings": StandingsHistory([]),
                      "bayesian_engine": None,
                      "model_selected": "raw_example", "history_through": "2026-05-01",
                      "competitiveness_qualified": False, "availability_limits": []}

    def test_history_day_forecast_is_rejected_and_future_sides_complement(self):
        with self.assertRaises(ValueError):
            predict_series_scenario(self.model, self.state, "Alpha", "Beta", "2026-05-01")
        one = predict_series_scenario(self.model, self.state, "Alpha", "Beta", "2026-05-02")
        two = predict_series_scenario(self.model, self.state, "Beta", "Alpha", "2026-05-02")
        self.assertGreater(one["prob_team1_win"], 0.5)
        self.assertAlmostEqual(one["prob_team1_win"] + two["prob_team1_win"], 1)
        self.assertFalse(one["competitiveness_qualified"])
        self.assertEqual(self.state["engine"].history_through, "2026-05-01")

    def test_wrong_model_state_pair_is_rejected(self):
        self.state["model_selected"] = "different_model"
        with self.assertRaises(ValueError):
            predict_series_scenario(self.model, self.state, "Alpha", "Beta", "2026-05-02")

    def test_obsolete_state_cannot_masquerade_as_current_feature_state(self):
        self.state["schema_version"] = 1
        with self.assertRaises(ValueError):
            predict_series_scenario(self.model, self.state, "Alpha", "Beta", "2026-05-02")

    def test_bayesian_forecast_requires_matching_complete_history(self):
        from pipeline.features.bayesian_series import BayesianSeriesEngine

        column = "bayes_score_s1_rw0025_prob"
        train = pd.DataFrame({"source_date": ["2026-04-01", "2026-04-02"],
                              "target": [1, 0], column: [0.7, 0.3]})
        model = fit_candidate(
            {"name": "bayesian_example", "family": "rating", "feature_columns": [column],
             "parameters": {}, "recency_half_life_days": None}, train,
            {column: {"operation": "complement", "source": column}}, train_end="2026-04-30")
        self.state["model_selected"] = model.recipe["name"]
        with self.assertRaises(ValueError):
            predict_series_scenario(model, self.state, "Alpha", "Beta", "2026-05-02")
        engine = BayesianSeriesEngine()
        self.state["bayesian_engine"] = engine
        with self.assertRaises(ValueError):
            predict_series_scenario(model, self.state, "Alpha", "Beta", "2026-05-02")
        engine.update_day("2026-05-01", [{"hltv_match_id": "1", "source_date": "2026-05-01",
                                        "team1": "Alpha", "team2": "Beta", "target": 1,
                                        "format": "bo3", "score1": 2, "score2": 0, "kind": "series"}])
        one = predict_series_scenario(model, self.state, "Alpha", "Beta", "2026-05-02")
        two = predict_series_scenario(model, self.state, "Beta", "Alpha", "2026-05-02")
        self.assertGreater(one["prob_team1_win"], 0.5)
        self.assertAlmostEqual(one["prob_team1_win"] + two["prob_team1_win"], 1)
        self.assertFalse(one["competitiveness_qualified"])

    def test_unavailable_or_after_event_origins_are_not_fabricated(self):
        frame = pd.DataFrame({"hltv_match_id": ["1", "2", "3"], "source_date": ["2026-05-02"] * 3})
        snapshot = {"fixtures": [
            {"fixtureId": "early", "startTime": "2026-05-02T00:30:00Z"},
            {"fixtureId": "later", "startTime": "2026-05-02T18:00:00Z"}],
            "links": [
                {"hltv_match_id": "1", "fixtureId": "early", "mapping_status": "matched",
                 "label_status": "valid_explicit_series_result", "hltv_calendar_day": "2026-05-02"},
                {"hltv_match_id": "2", "fixtureId": "later", "mapping_status": "matched",
                 "label_status": "valid_explicit_series_result", "hltv_calendar_day": "2026-05-02"}]}
        self.assertEqual(forecast_origins(snapshot, "one_hour", frame),
                         [None, "2026-05-02T17:00:00Z", None])
