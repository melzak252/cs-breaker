"""Behavioral regression coverage for day-grouped player baseline evaluation."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline.models.player_baseline import predict_player_baseline, train_player_baseline


FEATURES = ["elo_diff", "elo_prob", "glicko_prob", "trueskill_prob", "openskill_prob"]


def cohort(days=10, series_per_day=6):
    rows = []
    for day in range(days):
        for series in range(series_per_day):
            target = series % 2
            difference = (2 * target - 1) * (1 + day / 10 + series / 100)
            probability = 1 / (1 + np.exp(-difference))
            rows.append({
                "hltv_match_id": str(day * series_per_day + series + 1),
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day, hours=series),
                "target": target,
                "team1": "Alpha",
                "team2": "Beta",
                "elo_diff": difference,
                "elo_prob": probability,
                "glicko_prob": probability,
                "trueskill_prob": probability,
                "openskill_prob": probability,
                "minimum_prior_player_series": day,
            })
    return pd.DataFrame(rows)

ARCHIVE_FEATURES = [
    "archive_elo_diff", "archive_elo_prob", "archive_decay_diff", "archive_decay_prob",
    "archive_team1_prior_matches", "archive_team2_prior_matches",
    "archive_minimum_prior_matches",
]
FORM_FEATURES = [
    "lineup_win_rate_diff", "lineup_last10_win_rate_diff", "player_kpr_diff",
    "player_dpr_diff", "player_adr_diff", "player_kast_diff",
]
FULL_FEATURES = FEATURES + ARCHIVE_FEATURES + FORM_FEATURES
NEW_RECIPES = (
    "pure_archive_elo", "pure_archive_decay", "calibrated_elo", "calibrated_glicko",
    "calibrated_trueskill", "calibrated_openskill", "ridge_rating_logits",
    "ridge_strength_form", "blend_player_archive_elo", "blend_player_archive_decay",
)


def archive_cohort(days=10):
    frame = cohort(days=days)
    for prefix, scale in (("archive_elo", 0.7), ("archive_decay", 0.45)):
        frame[f"{prefix}_diff"] = frame["elo_diff"] * 400 * scale
        frame[f"{prefix}_prob"] = 1 / (1 + np.exp(-frame["elo_diff"] * scale))
    frame["archive_team1_prior_matches"] = frame.index % 7 + 1
    frame["archive_team2_prior_matches"] = frame.index % 11 + 3
    frame["archive_minimum_prior_matches"] = frame[
        ["archive_team1_prior_matches", "archive_team2_prior_matches"]].min(axis=1)
    for column in FORM_FEATURES:
        frame[column] = frame["elo_diff"] * 0.1
    return frame


def probability_column(name):
    return f"{name.removeprefix('pure_')}_prob"


def reverse_sides(frame):
    result = frame.copy()
    for column in FULL_FEATURES:
        if column.endswith("_diff"):
            result[column] = -frame[column]
        elif column.endswith("_prob"):
            result[column] = 1 - frame[column]
    result["archive_team1_prior_matches"] = frame["archive_team2_prior_matches"]
    result["archive_team2_prior_matches"] = frame["archive_team1_prior_matches"]
    result["team1"], result["team2"] = frame["team2"], frame["team1"]
    return result


class PlayerBaselineTests(unittest.TestCase):
    def test_test_labels_cannot_change_any_candidate_probability_or_selected_recipe(self):
        frame = archive_cohort()
        metrics, predictions, bundle = train_player_baseline(frame, FULL_FEATURES)
        changed = frame.copy()
        mask = changed["date"].dt.day >= 9
        changed.loc[mask, "target"] = 1 - changed.loc[mask, "target"]
        other_metrics, other_predictions, other_bundle = train_player_baseline(changed, FULL_FEATURES)
        self.assertEqual(metrics["selection"], other_metrics["selection"])
        self.assertEqual(bundle["selected_model"], other_bundle["selected_model"])
        self.assertEqual(metrics["candidate_parameters"], other_metrics["candidate_parameters"])
        for name in metrics["models"]:
            np.testing.assert_array_equal(predictions[probability_column(name)],
                                          other_predictions[probability_column(name)])
        np.testing.assert_array_equal(predictions["prob_team1_win"],
                                      other_predictions["prob_team1_win"])
        np.testing.assert_array_equal(predict_player_baseline(bundle, predictions),
                                      predict_player_baseline(other_bundle, predictions))

    def test_validation_labels_choose_but_do_not_fit_candidate_forecasts(self):
        frame = archive_cohort()
        metrics, predictions, _ = train_player_baseline(frame, FULL_FEATURES)
        changed = frame.copy()
        mask = changed["date"].dt.day.between(7, 8)
        changed.loc[mask, "target"] = 1 - changed.loc[mask, "target"]
        other_metrics, other_predictions, _ = train_player_baseline(changed, FULL_FEATURES)
        self.assertEqual(metrics["candidate_parameters"], other_metrics["candidate_parameters"])
        self.assertEqual(metrics["development_walk_forward"],
                         other_metrics["development_walk_forward"])
        for name in metrics["models"]:
            np.testing.assert_array_equal(predictions[probability_column(name)],
                                          other_predictions[probability_column(name)])

    def test_calibration_is_bounded_train_log_loss_optimum_not_validation_fit(self):
        frame = cohort()
        sign = 2 * frame["target"] - 1
        frame["elo_diff"] = sign.astype(float)
        frame[FEATURES[1:]] = np.repeat(
            np.where(sign.to_numpy()[:, None] > 0, 0.8, 0.2), len(FEATURES) - 1, axis=1)
        # Each direction is correct in two of three series, on every day.
        mask = frame.index % 6 == 0
        mask |= frame.index % 6 == 5
        frame.loc[mask, "target"] = 1 - frame.loc[mask, "target"]
        metrics, predictions, bundle = train_player_baseline(frame, FEATURES)
        train = frame[frame["date"].dt.day <= 6]
        for channel in ("elo", "glicko", "trueskill", "openskill"):
            name = f"calibrated_{channel}"
            slope = metrics["candidate_parameters"][name]["slope"]
            self.assertGreater(slope, 0)
            self.assertLess(slope, 3)
            logits = np.log(train[f"{channel}_prob"] / (1 - train[f"{channel}_prob"]))
            fitted = 1 / (1 + np.exp(-slope * logits))
            derivative = np.mean(logits * (fitted - train["target"]))
            self.assertAlmostEqual(float(derivative), 0, delta=1e-6)
            np.testing.assert_allclose(predictions[f"{name}_prob"],
                                       np.where(predictions[f"{channel}_prob"] > 0.5, 2 / 3, 1 / 3),
                                       rtol=0, atol=1e-6)
        # The calibrated empirical probability is the unique log-loss optimum
        # here; regularized classifiers and raw .8/.2 forecasts are worse.
        self.assertTrue(bundle["selected_model"].startswith("calibrated_"))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calibrated.joblib"
            joblib.dump(bundle, path)
            restored = joblib.load(path)
        np.testing.assert_array_equal(predict_player_baseline(restored, predictions),
                                      predictions["prob_team1_win"])
        self.assertTrue(np.any(predictions["prob_team1_win"].to_numpy() !=
                               np.round(predictions["prob_team1_win"].to_numpy(), 4)))
        for invalid_slope in (-0.1, 3.1, float("nan"), "invalid"):
            corrupt = copy.deepcopy(restored)
            corrupt["slope"] = invalid_slope
            with self.subTest(slope=invalid_slope), self.assertRaises(ValueError):
                predict_player_baseline(corrupt, predictions)
        corrupt = copy.deepcopy(restored)
        corrupt["probability_column"] = "archive_elo_prob"
        with self.assertRaises(ValueError):
            predict_player_baseline(corrupt, predictions)

    def test_zero_training_logits_use_unit_slope_and_held_out_probabilities(self):
        frame = cohort()
        frame.loc[frame["date"].dt.day <= 6, FEATURES[1:]] = 0.5
        metrics, predictions, _ = train_player_baseline(frame, FEATURES)
        for channel in ("elo", "glicko", "trueskill", "openskill"):
            name = f"calibrated_{channel}"
            self.assertEqual(metrics["candidate_parameters"][name]["slope"], 1)
            np.testing.assert_allclose(predictions[f"{name}_prob"],
                                       predictions[f"{channel}_prob"], rtol=0, atol=1e-15)

    def test_calibration_saturates_at_zero_or_three_when_train_signal_demands_it(self):
        for reversed_labels, expected_slope in ((False, 3.0), (True, 0.0)):
            frame = cohort()
            if reversed_labels:
                mask = frame["date"].dt.day <= 6
                frame.loc[mask, "target"] = 1 - frame.loc[mask, "target"]
            metrics, predictions, _ = train_player_baseline(frame, FEATURES)
            for channel in ("elo", "glicko", "trueskill", "openskill"):
                name = f"calibrated_{channel}"
                with self.subTest(channel=channel, reversed_labels=reversed_labels):
                    slope = metrics["candidate_parameters"][name]["slope"]
                    self.assertAlmostEqual(slope, expected_slope, delta=1e-6)
                    logits = np.log(predictions[f"{channel}_prob"] /
                                    (1 - predictions[f"{channel}_prob"]))
                    np.testing.assert_allclose(predictions[f"{name}_prob"],
                                               1 / (1 + np.exp(-slope * logits)),
                                               rtol=0, atol=1e-14)

    def test_compact_forecasts_do_not_fit_validation_covariates_or_support(self):
        frame = archive_cohort()
        _, original, _ = train_player_baseline(frame, FULL_FEATURES)
        changed = frame.copy()
        validation = changed["date"].dt.day.between(7, 8)
        for column in FULL_FEATURES:
            if column.endswith("_diff"):
                changed.loc[validation, column] *= -100000
            elif column.endswith("_prob"):
                changed.loc[validation, column] = np.where(
                    changed.loc[validation, column] > 0.5, 1e-12, 1 - 1e-12)
        changed.loc[validation, "archive_team1_prior_matches"] = 100000
        changed.loc[validation, "archive_team2_prior_matches"] = 200000
        changed.loc[validation, "archive_minimum_prior_matches"] = 100000
        _, predictions, _ = train_player_baseline(changed, FULL_FEATURES)
        for name in ("ridge_rating_logits", "ridge_strength_form"):
            with self.subTest(recipe=name):
                np.testing.assert_array_equal(predictions[f"{name}_prob"],
                                              original[f"{name}_prob"])

    def test_fixed_archive_blends_use_equal_probability_weights(self):
        # Fixed blends average probabilities, not logits or fitted weights.
        _, predictions, _ = train_player_baseline(archive_cohort(), FULL_FEATURES)
        for channel in ("elo", "decay"):
            np.testing.assert_allclose(predictions[f"blend_player_archive_{channel}_prob"],
                                       0.5 * predictions["elo_prob"] +
                                       0.5 * predictions[f"archive_{channel}_prob"],
                                       rtol=0, atol=1e-15)

    def test_reduced_schema_omits_unavailable_archive_recipes_without_filling_them(self):
        metrics, predictions, _ = train_player_baseline(cohort(), FEATURES)
        for name in ("pure_archive_elo", "pure_archive_decay",
                     "blend_player_archive_elo", "blend_player_archive_decay"):
            self.assertNotIn(name, metrics["models"])
            self.assertNotIn(probability_column(name), predictions)
        for name in ("ridge_rating_logits", "ridge_strength_form"):
            self.assertIn(name, metrics["models"])
            self.assertTrue(np.isfinite(predictions[probability_column(name)]).all())
        frame = archive_cohort()
        frame["archive_unverified_diff"] = 0.0
        with self.assertRaisesRegex(ValueError, "mirror semantics"):
            train_player_baseline(frame, FULL_FEATURES + ["archive_unverified_diff"])
        columns = [column for column in FULL_FEATURES
                   if column != "archive_team2_prior_matches"]
        with self.assertRaises(ValueError):
            train_player_baseline(frame, columns)

    def test_all_new_recipes_complement_on_reversal_and_neutral_strengths_are_half(self):
        frame = archive_cohort()
        metrics, predictions, _ = train_player_baseline(frame, FULL_FEATURES)
        changed = frame.copy()
        mask = changed["date"].dt.day >= 9
        changed.loc[mask, :] = reverse_sides(frame.loc[mask])
        _, reversed_predictions, _ = train_player_baseline(changed, FULL_FEATURES)
        for name in NEW_RECIPES:
            with self.subTest(recipe=name):
                np.testing.assert_allclose(predictions[probability_column(name)] +
                                           reversed_predictions[probability_column(name)],
                                           1, rtol=0, atol=1e-14)
        for side, other in (("team1", "team2"), ("team2", "team1")):
            column = f"archive_{side}_prior_matches"
            self.assertEqual(metrics["mirror_rules"][column],
                             {"operation": "swap", "source": f"archive_{other}_prior_matches"})
            np.testing.assert_array_equal(reversed_predictions[column],
                                          predictions[f"archive_{other}_prior_matches"])
        for column in FULL_FEATURES:
            if column.endswith("_diff"):
                changed.loc[mask, column] = 0.0
            elif column.endswith("_prob"):
                changed.loc[mask, column] = 0.5
        changed.loc[mask, ["archive_team1_prior_matches", "archive_team2_prior_matches",
                           "archive_minimum_prior_matches"]] = 17
        _, neutral, _ = train_player_baseline(changed, FULL_FEATURES)
        for name in NEW_RECIPES:
            np.testing.assert_allclose(neutral[probability_column(name)], 0.5, rtol=0, atol=1e-15)

    def test_archive_signal_can_improve_selection_without_changing_raw_player_forecasts(self):
        frame = archive_cohort()
        frame["elo_diff"] = 0.0
        frame[FEATURES[1:]] = 0.5
        frame[FORM_FEATURES] = 0.0
        _, player_predictions, _ = train_player_baseline(frame, FEATURES)
        metrics, predictions, bundle = train_player_baseline(frame, FULL_FEATURES)
        for column in FEATURES[1:]:
            np.testing.assert_array_equal(predictions[column], player_predictions[column])
        self.assertLess(metrics["selection"]["selected_log_loss"], np.log(2) - 0.1)
        self.assertTrue(any(column.startswith("archive_") for column in bundle["feature_columns"]))
        self.assertGreater(float(np.ptp(predictions["prob_team1_win"])), 0.2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "archive.joblib"
            joblib.dump(bundle, path)
            restored = joblib.load(path)
        np.testing.assert_array_equal(predict_player_baseline(restored, predictions),
                                      predictions["prob_team1_win"])
        np.testing.assert_allclose(predict_player_baseline(restored, predictions) +
                                   predict_player_baseline(restored, reverse_sides(predictions)),
                                   1, rtol=0, atol=1e-14)
        for updates in ({"selected_model": "not_a_recipe"}, {"family": "not_a_family"},
                        {"version": 3}, {"mirror_rules": {}}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                predict_player_baseline(dict(restored, **updates), predictions)

    def test_development_folds_fit_only_strictly_earlier_train_days(self):
        frame = archive_cohort(days=20)
        metrics, _, _ = train_player_baseline(frame, FULL_FEATURES)
        folds = metrics["development_walk_forward"]["folds"]
        train_days = metrics["splits"]["train"]["dates"]
        self.assertEqual(len(folds), 3)
        remaining = train_days[len(train_days) // 2:]
        evaluated = []
        for fold in folds:
            fit_days, evaluation_days = fold["train_dates"], fold["validation_dates"]
            self.assertEqual(fit_days, [day for day in train_days if day < min(evaluation_days)])
            self.assertLess(max(fit_days), min(evaluation_days))
            evaluated.extend(evaluation_days)
            self.assertTrue(fold["models"])
        self.assertEqual(evaluated, remaining)
        # Changing the first day of the second inner block (both features and
        # labels) cannot affect the first fold, even at that exact boundary.
        boundary = folds[1]["validation_dates"][0]
        changed = frame.copy()
        mask = changed["date"].dt.strftime("%Y-%m-%d") >= boundary
        changed.loc[mask, "target"] = 1 - changed.loc[mask, "target"]
        for column in FULL_FEATURES:
            if column.endswith("_diff"):
                changed.loc[mask, column] *= -1000
            elif column.endswith("_prob"):
                changed.loc[mask, column] = 1 - changed.loc[mask, column]
        changed_metrics, _, _ = train_player_baseline(changed, FULL_FEATURES)
        self.assertEqual(folds[0], changed_metrics["development_walk_forward"]["folds"][0])

    def test_development_skips_single_class_prefix_without_using_future_labels(self):
        frame = archive_cohort(days=20)
        frame.loc[frame["date"].dt.day <= 6, "target"] = 1
        metrics, _, _ = train_player_baseline(frame, FULL_FEATURES)
        first = metrics["development_walk_forward"]["folds"][0]
        self.assertEqual(first["status"], "skipped_single_class")
        self.assertFalse(first["models"])

    def test_rating_forecaster_can_win_validation_instead_of_forcing_a_worse_classifier(self):
        frame = cohort()
        train_mask = frame["date"].dt.day <= 6
        frame.loc[train_mask, "target"] = 1 - frame.loc[train_mask, "target"]
        metrics, predictions, bundle = train_player_baseline(frame, FEATURES)
        self.assertEqual(metrics["selected_model"], "pure_elo")
        self.assertLess(metrics["selection"]["selected_log_loss"],
                        metrics["models"]["logistic_regression"]["validation"]["log_loss"])
        changed = frame.copy()
        changed.loc[changed["date"].dt.day >= 9, "target"] = 1 - changed.loc[changed["date"].dt.day >= 9, "target"]
        changed_metrics, _, _ = train_player_baseline(changed, FEATURES)
        self.assertEqual(changed_metrics["selection"], metrics["selection"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rating-model.joblib"
            joblib.dump(bundle, path)
            restored = joblib.load(path)
        forward = predict_player_baseline(restored, predictions)
        mirrored = predictions.copy()
        mirrored["elo_diff"] *= -1
        for column in FEATURES[1:]:
            mirrored[column] = 1 - mirrored[column]
        reverse = predict_player_baseline(restored, mirrored)
        np.testing.assert_allclose(forward + reverse, 1, atol=1e-15)
        self.assertGreater(float(np.ptp(forward)), 0.5)

    def test_shuffled_input_splits_whole_days_with_disjoint_chronological_boundaries(self):
        frame = cohort().sample(frac=1, random_state=7)
        metrics, predictions, _ = train_player_baseline(frame, FEATURES)
        splits = metrics["splits"]
        train = set(splits["train"]["dates"])
        validation = set(splits["validation"]["dates"])
        test = set(splits["test"]["dates"])
        self.assertEqual([len(train), len(validation), len(test)], [6, 2, 2])
        self.assertFalse(train & validation or train & test or validation & test)
        self.assertLess(max(train), min(validation))
        self.assertLess(max(validation), min(test))
        self.assertEqual([splits[s]["n_series"] for s in splits], [36, 12, 12])
        self.assertEqual(set(predictions["source_date"]), test)
        self.assertEqual(len(predictions), 12)
        self.assertEqual(len(set(predictions["hltv_match_id"])), 12)
        self.assertEqual(metrics["support"]["n_series"], 60)

    def test_reversed_and_shuffled_inputs_produce_identical_predictions_and_selection(self):
        frame = cohort()
        expected_metrics, expected, bundle = train_player_baseline(frame, FEATURES)
        for input_frame in (frame.iloc[::-1], frame.sample(frac=1, random_state=19)):
            actual_metrics, actual, actual_bundle = train_player_baseline(input_frame, FEATURES)
            pd.testing.assert_frame_equal(actual, expected)
            self.assertEqual(actual_bundle["selected_model"], bundle["selected_model"])
            self.assertEqual(actual_metrics, expected_metrics)

    def test_test_labels_and_features_do_not_choose_or_fit_the_model(self):
        frame = cohort()
        first_metrics, first_predictions, first = train_player_baseline(frame, FEATURES)
        changed = frame.copy()
        test_mask = changed["date"].dt.date >= pd.Timestamp("2024-01-09").date()
        changed.loc[test_mask, "target"] = 1 - changed.loc[test_mask, "target"]
        changed.loc[test_mask, "elo_diff"] = 1000
        for name in FEATURES[1:]:
            changed.loc[test_mask, name] = 0.99
        second_metrics, _, second = train_player_baseline(changed, FEATURES)
        self.assertEqual(first["selected_model"], second["selected_model"])
        self.assertEqual(first["training_cutoff"], second["training_cutoff"])
        self.assertEqual(first_metrics["selection"], second_metrics["selection"])
        for name in first_metrics["models"]:
            self.assertEqual(first_metrics["models"][name]["validation"],
                             second_metrics["models"][name]["validation"])
        np.testing.assert_array_equal(predict_player_baseline(first, first_predictions),
                                      predict_player_baseline(second, first_predictions))

    def test_single_class_test_has_null_auc_and_real_calibration_and_common_cohort(self):
        frame = cohort()
        frame.loc[frame["date"].dt.day >= 9, "target"] = 1
        metrics, predictions, _ = train_player_baseline(frame, FEATURES)
        for scores in metrics["models"].values():
            self.assertIsNone(scores["test"]["roc_auc"])
            self.assertEqual(scores["test"]["n_samples"], len(predictions))
            self.assertEqual(sum(b["count"] for b in scores["test"]["calibration_bins"]),
                             len(predictions))
            self.assertTrue(np.isfinite(scores["test"]["log_loss"]))
            self.assertGreaterEqual(scores["test"]["ece"], 0)
        json.dumps(metrics, allow_nan=False)

    def test_held_out_features_do_not_change_train_fitted_forecasts(self):
        frame = cohort()
        _, original_predictions, original_bundle = train_player_baseline(frame, FEATURES)
        frame.loc[frame["date"].dt.day >= 7, "elo_diff"] *= 10000
        metrics, predictions, bundle = train_player_baseline(frame, FEATURES)
        train = frame[frame["date"].dt.day <= 6]
        # Selection may change; each fitted candidate must still be TRAIN-only.
        changed_test = frame.copy()
        changed_test.loc[changed_test["date"].dt.day >= 9, FEATURES] = (
            cohort().loc[frame["date"].dt.day >= 9, FEATURES])
        _, restored_predictions, _ = train_player_baseline(changed_test, FEATURES)
        for name in metrics["models"]:
            column = f"{name.removeprefix('pure_')}_prob"
            np.testing.assert_array_equal(original_predictions[column], restored_predictions[column])
        np.testing.assert_array_equal(
            predict_player_baseline(original_bundle, original_predictions),
            predict_player_baseline(original_bundle, restored_predictions))
        self.assertEqual(bundle["training_cutoff"], "2024-01-06")
        np.testing.assert_array_equal(predictions["class_prior_prob"],
                                      np.full(len(predictions), train["target"].mean()))
        self.assertEqual(metrics["selection"]["fit_partition"], "train")

    def test_class_prior_does_not_use_validation_or_test_class_balance(self):
        frame = cohort()
        frame.loc[frame["date"].dt.day >= 7, "target"] = 1
        metrics, predictions, _ = train_player_baseline(frame, FEATURES)
        self.assertEqual(metrics["selection"]["class_prior"], 0.5)
        self.assertGreater(frame["target"].mean(), 0.5)
        np.testing.assert_array_equal(predictions["class_prior_prob"],
                                      np.full(len(predictions), 0.5))

    def test_five_day_cohort_keeps_nonempty_train_validation_and_test(self):
        metrics, predictions, _ = train_player_baseline(cohort(days=5), FEATURES)
        self.assertEqual([metrics["splits"][name]["n_days"]
                          for name in ("train", "validation", "test")], [3, 1, 1])
        self.assertEqual(len(predictions), 6)

    def test_serialized_estimator_reproduces_unrounded_saved_predictions(self):
        _, predictions, bundle = train_player_baseline(cohort(), FEATURES)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "baseline.joblib"
            joblib.dump(bundle, path)
            restored = joblib.load(path)
        probability = predict_player_baseline(restored, predictions)
        self.assertNotIn("p_team1", predictions.columns)
        self.assertNotIn("p_team2", predictions.columns)
        np.testing.assert_array_equal(predictions["prob_team1_win"], probability)
        np.testing.assert_array_equal(predictions["prob_team2_win"], 1 - probability)
        np.testing.assert_array_equal(probability, predict_player_baseline(
            restored, predictions[list(reversed(predictions.columns))]))
        self.assertTrue(np.any(probability != np.round(probability, 4)))
        for name in ("constant_0_5_prob", "class_prior_prob", "elo_prob", "glicko_prob",
                     "trueskill_prob", "openskill_prob", "logistic_regression_prob",
                     "gradient_boosting_prob"):
            self.assertIn(name, predictions.columns)
        self.assertEqual(set(predictions["predicted_winner"]), {"Alpha", "Beta"})

    def test_identical_strengths_predict_half_despite_unbalanced_training_classes(self):
        frame = cohort()
        frame.loc[frame.index % 6 == 4, "target"] = 1
        frame["elo_diff"] = 0.0
        frame[FEATURES[1:]] = 0.5
        metrics, predictions, bundle = train_player_baseline(frame, FEATURES)
        self.assertGreater(metrics["selection"]["class_prior"], 0.5)
        np.testing.assert_array_equal(predict_player_baseline(bundle, predictions),
                                      np.full(len(predictions), 0.5))
        for name in ("logistic_regression", "gradient_boosting"):
            np.testing.assert_array_equal(predictions[f"{name}_prob"],
                                          np.full(len(predictions), 0.5))
            self.assertAlmostEqual(metrics["models"][name]["validation"]["log_loss"], np.log(2))
            bins = metrics["models"][name]["validation"]["calibration_bins"]
            self.assertEqual(bins[0]["avg_prob"], 0.5)
        # The intentionally orientation-asymmetric prior is a control, not the forecast.
        np.testing.assert_array_equal(predictions["class_prior_prob"],
                                      np.full(len(predictions), metrics["selection"]["class_prior"]))

    def test_reversed_sides_complement_predictions_with_unbalanced_labels_and_support(self):
        frame = cohort()
        frame.loc[frame.index % 6 == 4, "target"] = 1
        differences = ["glicko_rd_diff", "trueskill_sigma_diff", "openskill_sigma_diff",
                       "player_kpr_diff", "player_last10_adr_diff"]
        invariants = ["glicko_joint_rd", "trueskill_joint_sigma", "openskill_joint_sigma",
                      "minimum_prior_matches", "performance_available", "minimum_prior_measured_maps"]
        for column in differences:
            frame[column] = frame["elo_diff"] * 0.25
        for column in invariants:
            frame[column] = frame.index % 5 + 1.0
        frame["team1_known_players"] = frame.index % 6
        frame["team2_known_players"] = 5 - frame["team1_known_players"]
        columns = FEATURES + differences + invariants + ["team1_known_players", "team2_known_players"]
        _, predictions, bundle = train_player_baseline(frame, columns)
        reversed_sides = predictions.copy()
        for column in ["elo_diff", *differences]:
            reversed_sides[column] = -predictions[column]
        for column in FEATURES[1:]:
            reversed_sides[column] = 1 - predictions[column]
        reversed_sides["team1_known_players"] = predictions["team2_known_players"]
        reversed_sides["team2_known_players"] = predictions["team1_known_players"]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "symmetric.joblib"
            joblib.dump(bundle, path)
            restored = joblib.load(path)
        np.testing.assert_allclose(
            predict_player_baseline(restored, predictions) +
            predict_player_baseline(restored, reversed_sides), 1.0, rtol=0, atol=1e-15)
        self.assertTrue(np.any(np.abs(predict_player_baseline(restored, predictions) - 0.5) > 1e-6))

    def test_unknown_feature_semantics_and_incomplete_side_pairs_are_rejected(self):
        frame = cohort()
        frame["unverified_diff"] = 0.0
        with self.assertRaisesRegex(ValueError, "mirror semantics"):
            train_player_baseline(frame, FEATURES + ["unverified_diff"])
        frame["team1_known_players"] = 5.0
        with self.assertRaisesRegex(ValueError, "both.*team1_known_players.*team2_known_players"):
            train_player_baseline(frame, FEATURES + ["team1_known_players"])
        _, predictions, bundle = train_player_baseline(frame, FEATURES)
        inconsistent = dict(bundle, mirror_rules={})
        with self.assertRaisesRegex(ValueError, "mirror rules"):
            predict_player_baseline(inconsistent, predictions)

    def test_duplicate_or_missing_ids_fail_before_training(self):
        frame = cohort()
        for invalid_id in (frame.loc[0, "hltv_match_id"], "", None):
            changed = frame.copy()
            changed.loc[1, "hltv_match_id"] = invalid_id
            with self.subTest(invalid_id=invalid_id), self.assertRaisesRegex(ValueError, "hltv_match_id"):
                train_player_baseline(changed, FEATURES)

    def test_invalid_features_targets_dates_or_insufficient_days_raise_meaningful_errors(self):
        for column, value, message in (("elo_diff", np.inf, "finite"),
                                       ("elo_prob", 1.1, "probabilities"),
                                       ("target", 2, "target"),
                                       ("date", None, "date")):
            frame = cohort()
            frame.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, message):
                train_player_baseline(frame, FEATURES)
        with self.assertRaisesRegex(ValueError, "distinct.*days"):
            train_player_baseline(cohort(days=4), FEATURES)
        one_class = cohort()
        one_class.loc[one_class["date"].dt.day <= 6, "target"] = 1
        with self.assertRaisesRegex(ValueError, "both.*classes"):
            train_player_baseline(one_class, FEATURES)
        with self.assertRaisesRegex(ValueError, "pre-result"):
            train_player_baseline(cohort(), FEATURES + ["target"])

    def test_inference_rejects_missing_nonfinite_or_inconsistent_feature_schema(self):
        _, predictions, bundle = train_player_baseline(cohort(), FEATURES)
        selected_column = bundle["feature_columns"][0]
        with self.assertRaisesRegex(ValueError, "Missing.*feature"):
            predict_player_baseline(bundle, predictions.drop(columns=[selected_column]))
        broken = predictions.copy()
        broken.loc[0, selected_column] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            predict_player_baseline(bundle, broken)

    def test_source_calendar_dates_are_not_converted_to_utc_days(self):
        frame = cohort()
        frame["source_date"] = frame["date"].dt.strftime("%Y-%m-%d")
        # Timestamps deliberately occupy a different UTC calendar day.
        frame["date"] = frame["source_date"] + "T23:30:00-06:00"
        metrics, predictions, _ = train_player_baseline(frame, FEATURES)
        self.assertEqual(metrics["splits"]["test"]["dates"], ["2024-01-09", "2024-01-10"])
        self.assertEqual(set(predictions["source_date"]), {"2024-01-09", "2024-01-10"})

    def test_timezone_bearing_dates_group_by_the_original_local_calendar(self):
        frame = cohort()
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d") + "T23:30:00-06:00"
        metrics, _, _ = train_player_baseline(frame, FEATURES)
        self.assertEqual(metrics["splits"]["train"]["start_date"], "2024-01-01")
        self.assertEqual(metrics["splits"]["test"]["end_date"], "2024-01-10")
