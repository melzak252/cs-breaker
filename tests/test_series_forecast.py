"""Forecast fitting boundaries, side invariance and persisted behavior."""
import copy
import pickle
import unittest

import numpy as np
import pandas as pd

from pipeline.models.series_forecast import fit_candidate, mirror_features, select_development_model


COLUMN = "series_elo_32_180_prob"
RULES = {COLUMN: {"operation": "complement", "source": COLUMN},
         "vrs_available": {"operation": "identity", "source": "vrs_available"}}


def frame(start="2026-04-01", count=40):
    rows = []
    for i in range(count):
        probability = (0.2, 0.35, 0.65, 0.8)[i % 4]
        rows.append({"hltv_match_id": str(i + 1),
                     "source_date": (pd.Timestamp(start) + pd.Timedelta(days=i // 4)).strftime("%Y-%m-%d"),
                     "target": int((i % 4 >= 2) != (i % 7 == 0)),
                     COLUMN: probability, "vrs_available": float(i % 3 != 0)})
    return pd.DataFrame(rows)


def recipe(family="ridge"):
    params = {"C": 0.1} if family == "ridge" else {}
    return {"name": family + "_example", "family": family,
            "feature_columns": [COLUMN] if family != "ridge" else [COLUMN, "vrs_available"],
            "parameters": params, "recency_half_life_days": None}


class SeriesForecastTests(unittest.TestCase):
    def test_coefficients_cannot_fit_a_row_after_training_boundary(self):
        data = frame()
        data.loc[0, "source_date"] = "2026-05-01"
        with self.assertRaises(ValueError):
            fit_candidate(recipe(), data, RULES, train_end="2026-04-30")

    def test_learned_and_calibrated_forecasts_complement_and_survive_serialization(self):
        data = frame()
        for family in ("ridge", "calibrated_rating"):
            with self.subTest(family=family):
                fitted = fit_candidate(recipe(family), data, RULES, train_end="2026-04-30")
                probability = fitted.predict(data)
                reversed_data = mirror_features(data, RULES)
                np.testing.assert_allclose(probability + fitted.predict(reversed_data), 1, atol=1e-12)
                restored = pickle.loads(pickle.dumps(fitted))
                np.testing.assert_array_equal(probability, restored.predict(data))
                changed = data.assign(target=1-data.target, result=999)
                np.testing.assert_array_equal(probability, fitted.predict(changed))

    def test_selection_rejects_reserved_dates_in_validation(self):
        train = frame()
        validation = frame("2026-07-01", count=8)
        with self.assertRaises(ValueError):
            select_development_model(
                train, validation, set(validation.hltv_match_id), [recipe()], RULES,
                train_end="2026-04-30", validation_end="2026-06-30")

    def test_outcomes_cannot_be_declared_predictors(self):
        bad = recipe()
        bad["feature_columns"] = ["target"]
        with self.assertRaises(ValueError):
            fit_candidate(bad, frame(), {"target": {"operation": "complement", "source": "target"}},
                          train_end="2026-04-30")

    def test_nonfinite_features_are_rejected_not_imputed_at_prediction(self):
        fitted = fit_candidate(recipe(), frame(), RULES, train_end="2026-04-30")
        corrupt = frame()
        corrupt.loc[0, COLUMN] = np.nan
        with self.assertRaises(ValueError):
            fitted.predict(corrupt)
