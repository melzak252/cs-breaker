"""Behavioral contracts for broad, source-day-frozen series history."""
import copy
import json
import math
import unittest

import pandas as pd

from pipeline.features.series_history import (
    FEATURE_COLUMNS, FEATURE_GROUPS, HISTORY_COLUMNS, MIRROR_RULES,
    SeriesHistoryEngine, build_series_features, load_history,
)
from pipeline.features.unified_player_ratings import glicko2_rating_period


def game(identity, day, left="Alpha", right="Beta", target=1):
    return {"hltv_match_id": str(identity), "source_date": day,
            "team1": left, "team2": right, "target": target,
            "format": "bo3", "kind": "series",
            "score1": 2 if target else 0, "score2": 0 if target else 2}


class SeriesHistoryTests(unittest.TestCase):
    def test_same_day_outcomes_do_not_change_any_opening_features(self):
        rows = [game(1, "2026-01-01"), game(2, "2026-01-02", "Alpha", "Gamma"),
                game(3, "2026-01-02", "Beta", "Gamma"),
                game(4, "2026-01-03", "Alpha", "Gamma")]
        changed = copy.deepcopy(rows)
        changed[1] = game(2, "2026-01-02", "Alpha", "Gamma", target=0)
        first, _ = build_series_features(rows)
        second, _ = build_series_features(changed)
        for identity in ("2", "3"):
            one = first.loc[first.hltv_match_id == identity, list(FEATURE_COLUMNS)].reset_index(drop=True)
            two = second.loc[second.hltv_match_id == identity, list(FEATURE_COLUMNS)].reset_index(drop=True)
            pd.testing.assert_frame_equal(one, two)
        self.assertNotEqual(
            first.loc[first.hltv_match_id == "4", "series_elo_32_180_prob"].iloc[0],
            second.loc[second.hltv_match_id == "4", "series_elo_32_180_prob"].iloc[0])

    def test_future_query_does_not_age_or_register_state_and_sides_complement(self):
        engine = SeriesHistoryEngine()
        engine.update_day("2026-01-01", [game(1, "2026-01-01")])
        before = engine.features("Alpha", "Gamma", "2026-01-02")
        engine.features("Unseen", "Another", "2030-01-01")
        self.assertEqual(before, engine.features("Alpha", "Gamma", "2026-01-02"))
        reverse = engine.features("Gamma", "Alpha", "2026-01-02")
        self.assertGreater(before["series_elo_32_180_prob"], 0.5)
        self.assertAlmostEqual(before["series_elo_32_180_prob"] + reverse["series_elo_32_180_prob"], 1.0)
        self.assertEqual(engine.history_through, "2026-01-01")

    def test_two_day_lag_cannot_see_yesterdays_result(self):
        rows = [game(1, "2026-01-01"), game(2, "2026-01-02", "Alpha", "Gamma")]
        one, _ = build_series_features(rows, history_lag_days=1)
        two, _ = build_series_features(rows, history_lag_days=2)
        self.assertGreater(one.loc[one.hltv_match_id == "2", "series_elo_32_180_prob"].iloc[0], 0.5)
        self.assertEqual(two.loc[two.hltv_match_id == "2", "series_elo_32_180_prob"].iloc[0], 0.5)

    def test_batch_reordering_and_whole_day_reordering_leave_replay_identical(self):
        rows = [game(8, "2026-01-01"), game(2, "2026-01-01", "Alpha", "Gamma", 0),
                game(3, "2026-01-02", "Alpha", "Gamma"),
                game(4, "2026-01-03", "Beta", "Gamma")]
        first, engine1 = build_series_features(rows)
        second, engine2 = build_series_features(list(reversed(rows)))
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(engine1.features("Alpha", "Beta", "2026-01-04"),
                         engine2.features("Alpha", "Beta", "2026-01-04"))

    def test_invalid_batch_never_partially_updates_or_reserves_ids(self):
        engine = SeriesHistoryEngine()
        engine.update_day("2026-01-01", [game(1, "2026-01-01")])
        opening = copy.deepcopy(engine.__dict__)
        invalid = [
            {**game(3, "2026-01-02"), "score1": 1},
            {**game(3, "2026-01-02"), "target": 0},
            {**game(3, "2026-01-02"), "source_date": "2026-01-03"},
            {**game(3, "2026-01-02"), "hltv_match_id": "01"},
            {**game(3, "2026-01-02"), "statistics": {
                "team1": totals(), "team2": {**totals(), "kills": float("nan")}}},
            game(1, "2026-01-02"),
        ]
        for bad in invalid:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                engine.update_day("2026-01-02", [game(2, "2026-01-02"), bad])
            self.assertEqual(engine.__dict__, opening)
        with self.assertRaises(ValueError):
            engine.update_day("2026-01-02", [game(2, "2026-01-02"), game(2, "2026-01-02")])
        self.assertEqual(engine.__dict__, opening)
        engine.update_day("2026-01-02", [game(2, "2026-01-02")])
        self.assertEqual(engine.history_through, "2026-01-02")
        with self.assertRaises(ValueError):
            engine.update_day("2026-01-01", [game(5, "2026-01-01")])
        with self.assertRaises(ValueError):
            engine.features("Alpha", "Beta", "2026-01-02")

    def test_queries_are_read_only_even_for_known_teams_after_inactivity(self):
        engine = SeriesHistoryEngine()
        engine.update_day("2026-01-01", [game(1, "2026-01-01")])
        state = copy.deepcopy(engine.__dict__)
        engine.features("Alpha", "Beta", "2030-01-01")
        engine.features("Unseen", "Other", "2030-01-01", format="bo5")
        self.assertEqual(engine.__dict__, state)

    def test_every_feature_has_explicit_side_mirror_semantics(self):
        engine = SeriesHistoryEngine()
        measured = {**game(1, "2026-01-01"), "statistics": {
            "team1": totals(kills=150, adr=100), "team2": totals(kills=50, adr=50)}}
        engine.update_day("2026-01-01", [measured])
        engine.update_day("2026-01-03", [game(2, "2026-01-03", "Alpha", "Gamma", 0)])
        forward = engine.features("Alpha", "Beta", "2026-01-04", format="bo5")
        reverse = engine.features("Beta", "Alpha", "2026-01-04", format="bo5")
        self.assertEqual(set(forward), set(FEATURE_COLUMNS))
        self.assertEqual(set(MIRROR_RULES), set(FEATURE_COLUMNS))
        self.assertEqual(set(FEATURE_GROUPS["all"]), set(FEATURE_COLUMNS))
        for name, rule in MIRROR_RULES.items():
            value = forward[rule["source"]]
            expected = -value if rule["operation"] == "negate" else (
                1 - value if rule["operation"] == "complement" else value)
            with self.subTest(feature=name):
                self.assertAlmostEqual(reverse[name], expected)

    def test_map_history_updates_only_map_channels_and_never_becomes_a_target(self):
        mapped = {**game(1, "2026-01-01"), "format": "mrg", "kind": "map",
                  "score1": 13, "score2": 7}
        rows, engine = build_series_features([mapped, game(2, "2026-01-02")])
        self.assertEqual(rows.hltv_match_id.tolist(), ["2"])
        self.assertEqual(rows.iloc[0]["series_elo_32_180_prob"], 0.5)
        self.assertEqual(rows.iloc[0]["glicko_200_prob"], 0.5)
        self.assertGreater(rows.iloc[0]["map_elo_32_180_prob"], 0.5)
        self.assertEqual(engine.history_through, "2026-01-02")

    def test_map_score_residual_has_no_outcome_dependent_multiplier(self):
        engine = SeriesHistoryEngine()
        engine.update_day("2026-01-01", [{**game(1, "2026-01-01"), "score2": 1}])
        feature = engine.features("Alpha", "Beta", "2026-01-02")
        self.assertAlmostEqual(feature["map_elo_32_180_diff"],
                               2 * 32 * (2 / 3 - 0.5) * 2 ** (-1 / 180))
        self.assertAlmostEqual(feature["series_elo_32_none_diff"], 32)

    def test_map_probability_uses_exact_bo3_and_bo5_majorities(self):
        engine = SeriesHistoryEngine()
        engine.update_day("2026-01-01", [game(1, "2026-01-01")])
        bo3 = engine.features("Alpha", "Beta", "2026-01-02", format="bo3")
        bo5 = engine.features("Alpha", "Beta", "2026-01-02", format="bo5")
        p = 1 / (1 + 10 ** (-bo3["map_elo_32_180_diff"] / 400))
        self.assertAlmostEqual(bo3["map_elo_32_180_prob"], 3 * p * p - 2 * p ** 3)
        self.assertAlmostEqual(bo5["map_elo_32_180_prob"],
                               10 * p ** 3 - 15 * p ** 4 + 6 * p ** 5)
        with self.assertRaises(ValueError):
            engine.features("Alpha", "Beta", "2026-01-02", format="mrg")

    def test_glicko_daily_period_matches_primitive_and_inactivity_is_lazy(self):
        engine = SeriesHistoryEngine()
        engine.update_day("2026-01-01", [game(1, "2026-01-01")])
        r1, rd1, sigma1 = glicko2_rating_period(1500, 200, .06, [(1500, 200, 1)], tau=.5)
        r2, rd2, sigma2 = glicko2_rating_period(1500, 200, .06, [(1500, 200, 0)], tau=.5)
        following = engine.features("Alpha", "Beta", "2026-01-02")
        self.assertAlmostEqual(following["glicko_200_diff"], r1 - r2)
        self.assertAlmostEqual(following["glicko_200_joint_rd"], math.hypot(rd1, rd2))
        inactive = engine.features("Alpha", "Beta", "2026-01-06")
        expected1 = min(350, math.sqrt(rd1 ** 2 + 4 * (173.7178 * sigma1) ** 2))
        expected2 = min(350, math.sqrt(rd2 ** 2 + 4 * (173.7178 * sigma2) ** 2))
        self.assertAlmostEqual(inactive["glicko_200_joint_rd"], math.hypot(expected1, expected2))
        far = engine.features("Alpha", "Beta", "2030-01-01")
        self.assertAlmostEqual(far["glicko_200_joint_rd"], math.hypot(350, 350))
        engine.update_day("2026-01-06", [game(2, "2026-01-06")])
        expected = glicko2_rating_period(r1, expected1, sigma1, [(r2, expected2, 1)], tau=.5)
        opposite = glicko2_rating_period(r2, expected2, sigma2, [(r1, expected1, 0)], tau=.5)
        next_day = engine.features("Alpha", "Beta", "2026-01-07")
        self.assertAlmostEqual(next_day["glicko_200_diff"], expected[0] - opposite[0])

    def test_same_day_statistics_and_missing_sides_are_not_measurements(self):
        measured = {**game(1, "2026-01-01"), "statistics": {
            "team1": totals(kills=150, adr=100), "team2": totals(kills=50, adr=50)}}
        rows, _ = build_series_features([measured, game(2, "2026-01-01"),
                                        game(3, "2026-01-02"),
                                        game(4, "2026-01-02", "Alpha", "Gamma")])
        features = rows.set_index("hltv_match_id")
        for identity in ("1", "2", "4"):
            self.assertEqual(features.loc[identity, "performance_available"], 0)
            self.assertEqual(features.loc[identity, "team_adr_diff"], 0)
        self.assertEqual(features.loc["3", "performance_available"], 1)
        self.assertEqual(features.loc["3", "team_adr_diff"], 50)
        self.assertEqual(features.loc["3", "minimum_prior_measured_maps"], 2)
        self.assertEqual(features.loc["3", "minimum_prior_measured_series"], 1)
        self.assertEqual(features.loc["3", "performance_age_days"], 1)
        self.assertAlmostEqual(features.loc["3", "team_kpr_diff"], .5)

    def test_map_only_days_do_not_refresh_series_elo_or_series_rest(self):
        engine = SeriesHistoryEngine()
        engine.update_day("2026-01-01", [game(1, "2026-01-01")])
        engine.update_day("2026-01-10", [{**game(2, "2026-01-10"), "kind": "map",
                                       "format": "nuke", "score1": 13, "score2": 5}])
        feature = engine.features("Alpha", "Gamma", "2026-01-11")
        self.assertAlmostEqual(feature["series_elo_32_180_diff"], 16 * 2 ** (-10 / 180))
        self.assertEqual(feature["series_rest_days_diff"], 10)
        self.assertEqual(feature["series_count_diff"], 1)

    def test_all_glicko_results_use_opening_opponents_not_within_day_winners(self):
        engine = SeriesHistoryEngine()
        engine.update_day("2026-01-01", [game(3, "2026-01-01", "Alpha", "Gamma"),
                                        game(1, "2026-01-01"),
                                        game(2, "2026-01-01", target=0)])
        expected = glicko2_rating_period(1500, 200, .06,
                                         [(1500, 200, 1), (1500, 200, 0), (1500, 200, 1)], tau=.5)
        feature = engine.features("Alpha", "Unseen", "2026-01-02")
        self.assertAlmostEqual(feature["glicko_200_diff"], expected[0] - 1500)
        self.assertAlmostEqual(feature["glicko_200_joint_rd"], math.hypot(expected[1], 200))
        self.assertEqual(feature["series_activity_30d_diff"], 3)

    def test_activity_counts_all_series_not_only_recent30_observation_storage(self):
        engine = SeriesHistoryEngine()
        engine.update_day("2026-01-01", [game(identity, "2026-01-01") for identity in range(1, 41)])
        feature = engine.features("Alpha", "Unseen", "2026-01-02")
        self.assertEqual(feature["series_activity_30d_diff"], 40)
        self.assertEqual(engine.features("Alpha", "Unseen", "2026-02-01")["series_activity_30d_diff"], 0)

    def test_lag_two_freezes_maps_and_measurements_and_returns_complete_future_state(self):
        measured = {**game(1, "2026-01-01"), "statistics": {
            "team1": totals(adr=100), "team2": totals(adr=50)}}
        mapped = {**game(2, "2026-01-02"), "kind": "map", "format": "inf",
                  "score1": 13, "score2": 4}
        rows = [measured, mapped, game(3, "2026-01-02"), game(4, "2026-01-03")]
        lag1, engine1 = build_series_features(rows, history_lag_days=1)
        lag2, engine2 = build_series_features(rows, history_lag_days=2)
        one, two = lag1.set_index("hltv_match_id"), lag2.set_index("hltv_match_id")
        self.assertEqual(two.loc["3", "performance_available"], 0)
        self.assertEqual(two.loc["4", "performance_available"], 1)
        self.assertNotEqual(one.loc["4", "map_elo_32_180_prob"], two.loc["4", "map_elo_32_180_prob"])
        self.assertEqual(engine1.features("Alpha", "Beta", "2026-01-04"),
                         engine2.features("Alpha", "Beta", "2026-01-04"))
        self.assertEqual(engine2.history_through, "2026-01-03")

    def test_measured_series_window_is_last_ten_measured_not_last_ten_results(self):
        engine = SeriesHistoryEngine()
        for identity in range(1, 12):
            day = f"2026-01-{identity:02}"
            measured = {**game(identity, day), "statistics": {
                "team1": totals(adr=1000 if identity == 1 else 100), "team2": totals(adr=50)}}
            engine.update_day(day, [measured])
        engine.update_day("2026-01-12", [game(12, "2026-01-12")])
        feature = engine.features("Alpha", "Beta", "2026-01-13")
        self.assertEqual(feature["team_last10_adr_diff"], 50)
        self.assertGreater(feature["team_adr_diff"], 50)
        self.assertEqual(feature["minimum_prior_measured_series"], 11)
        self.assertEqual(feature["minimum_prior_measured_maps"], 22)
        self.assertEqual(feature["performance_age_days"], 2)

    def test_performance_pools_exposure_rather_than_averaging_series_ratios(self):
        first = {**game(1, "2026-01-01"), "statistics": {
            "team1": totals(adr=100), "team2": totals(adr=50)}}
        second_totals = {**totals(adr=60), "rounds": 400, "adr_round_sum": 60 * 400,
                         "kast_round_sum": .75 * 400}
        second = {**game(2, "2026-01-02"), "statistics": {
            "team1": second_totals,
            "team2": {**second_totals, "adr_round_sum": 50 * 400}}}
        _, engine = build_series_features([first, second])
        feature = engine.features("Alpha", "Beta", "2026-01-03")
        self.assertAlmostEqual(feature["team_adr_diff"], (100 * 200 + 60 * 400) / 600 - 50)
        self.assertAlmostEqual(feature["team_last10_adr_diff"], feature["team_adr_diff"])

    def test_empty_and_invalid_lag_boundaries_do_not_create_state(self):
        rows, engine = build_series_features([])
        self.assertEqual(len(rows), 0)
        self.assertTrue(set(FEATURE_COLUMNS) <= set(rows.columns))
        self.assertEqual(engine.teams, {})
        self.assertIsNone(engine.history_through)
        for lag in (0, -1, True, 1.5):
            with self.subTest(lag=lag), self.assertRaises(ValueError):
                build_series_features([game(1, "2026-01-01")], history_lag_days=lag)
        with self.assertRaises(ValueError):
            engine.update_day("2026-01-01", [])
        self.assertIsNone(engine.history_through)
        self.assertFalse(set(("target", "score1", "score2", "format", "source_date")) & set(FEATURE_COLUMNS))
        self.assertTrue(set(FEATURE_GROUPS["probabilities"]) <= set(FEATURE_GROUPS["ratings"]))

    def test_map_measurements_update_career_but_not_measured_series_window(self):
        mapped = {**game(1, "2026-01-01"), "kind": "map", "format": "anc",
                  "score1": 13, "score2": 7,
                  "statistics": {
                      "team1": {**totals(adr=100), "maps": 1, "rounds": 100,
                                "adr_round_sum": 100 * 100, "kast_round_sum": .75 * 100},
                      "team2": {**totals(adr=50), "maps": 1, "rounds": 100,
                                "adr_round_sum": 50 * 100, "kast_round_sum": .75 * 100}}}
        engine = SeriesHistoryEngine()
        engine.update_day("2026-01-01", [mapped])
        feature = engine.features("Alpha", "Beta", "2026-01-02")
        self.assertEqual(feature["performance_available"], 1)
        self.assertEqual(feature["team_adr_diff"], 50)
        self.assertEqual(feature["minimum_prior_measured_maps"], 1)
        self.assertEqual(feature["minimum_prior_measured_series"], 0)
        self.assertEqual(feature["last10_performance_available"], 0)
        self.assertEqual(feature["team_last10_adr_diff"], 0)


def totals(kills=100, adr=80):
    return {"maps": 2, "rounds": 200, "kills": kills, "deaths": 75,
            "assists": 25, "adr_round_sum": adr * 200, "kast_round_sum": .75 * 200}


def native_row(identity=1, format="bo3", scores=(2, 0), day="2026-01-01"):
    row = {"hltv_match_id": str(identity), "date": day + " 12:00:00",
           "team1": "Alpha", "team2": "Beta", "format": format,
           "score1": str(scores[0]), "score2": str(scores[1]),
           "winner": "team1" if scores[0] > scores[1] else "team2",
           "match_url": f"https://www.hltv.org/matches/{identity}/alpha-vs-beta",
           "has_lineup": False, "player_match_statistics": ""}
    return row


def verified_row(identity=1):
    row = native_row(identity)
    row.update(has_lineup=True, lineup_source="strandedpond_historical_maps",
               lineup_status="complete", identity_namespace="strandedpond:nickname")
    measurements = {}
    for side in (1, 2):
        for slot in range(1, 6):
            key = f"strandedpond:nickname:{side}-{slot}"
            row[f"team{side}_player_{slot}_key"] = key
            measurements[key] = {"maps": 2, "rounds": 40,
                                 "kills": 30 if side == 1 else 10, "deaths": 15,
                                 "assists": 5, "adr_round_sum": 4000 if side == 1 else 2000,
                                 "kast_round_sum": 30 if side == 1 else 20}
    row["player_match_statistics"] = json.dumps(measurements)
    return row


class LoadHistoryTests(unittest.TestCase):
    def test_bounded_schema_does_not_request_scrape_time_snapshot_features(self):
        self.assertTrue(set(("hltv_match_id", "match_url", "player_match_statistics")) <= set(HISTORY_COLUMNS))
        self.assertFalse(any(name.startswith("kaggle_snapshot__") for name in HISTORY_COLUMNS))

    def test_complete_series_targets_and_modern_mr3_maps_are_distinct(self):
        rows = [native_row(1), native_row(2, "bo5", (3, 2))]
        for identity, scores in enumerate(((13, 0), (13, 11), (16, 12), (16, 14),
                                           (19, 15), (19, 17), (22, 18)), 3):
            rows.append(native_row(identity, "mrg", scores))
        games, audit = load_history(pd.DataFrame(rows))
        self.assertEqual(sum(g["kind"] == "series" for g in games), 2)
        self.assertEqual(sum(g["kind"] == "map" for g in games), 7)
        self.assertEqual(audit["accepted_series"], 2)
        self.assertEqual(audit["accepted_maps"], 7)
        targets, _ = build_series_features(games)
        self.assertEqual(targets.hltv_match_id.tolist(), ["1", "2"])

    def test_invalid_score_formats_are_excluded_without_series_label_invention(self):
        rows = [native_row(1, "def", (1, 0)), native_row(2, "bo2", (2, 0)),
                native_row(3, "bo1", (13, 8)), native_row(4, "bo3", (1, 0)),
                native_row(5, "bo5", (3, 3)), native_row(6, "mrg", (13, 12)),
                native_row(7, "mrg", (14, 12)), native_row(8, "mrg", (16, 11)),
                native_row(9, "mrg", (16, 15)), native_row(10, "mrg", (17, 15)),
                {**native_row(11), "score1": "2.5"},
                {**native_row(12), "winner": "team2"}]
        games, audit = load_history(pd.DataFrame(rows))
        self.assertEqual(games, [])
        self.assertEqual(sum(audit["exclusions"].values()), len(rows))

    def test_ids_urls_duplicates_and_source_calendar_are_validated(self):
        rows = [native_row(1), native_row(1),
                {**native_row(2), "match_url": "https://www.hltv.org/matches/3/wrong"},
                {**native_row(4), "hltv_match_id": "04"},
                native_row(5, day="2023-12-31"),
                {**native_row(6), "date": "2026-02-30"},
                {**native_row(7), "date": "2026-01-01T23:30:00-07:00",
                 "team1": " Alpha ", "team2": "alpha"}]
        games, audit = load_history(pd.DataFrame(rows))
        self.assertEqual([g["hltv_match_id"] for g in games], ["7"])
        self.assertEqual(games[0]["source_date"], "2026-01-01")
        self.assertEqual(games[0]["team1"], " Alpha ")
        self.assertEqual(audit["exclusions"]["duplicate_native_match_id"], 2)

    def test_verified_team_totals_pool_player_rounds_but_count_actual_maps_once(self):
        games, _ = load_history(pd.DataFrame([verified_row()]))
        left = games[0]["statistics"]["team1"]
        self.assertEqual(left["maps"], 2)
        self.assertEqual(left["rounds"], 200)
        self.assertEqual(left["kills"], 150)
        self.assertEqual(left["adr_round_sum"] / left["rounds"], 100)
        self.assertEqual(left["kast_round_sum"] / left["rounds"], .75)

    def test_missing_and_unverified_payloads_do_not_invent_performance(self):
        missing = verified_row()
        missing["player_match_statistics"] = ""
        unverified = native_row(2)
        unverified["player_match_statistics"] = "{broken claimed snapshot"
        games, audit = load_history(pd.DataFrame([missing, unverified]))
        self.assertEqual(len(games), 2)
        self.assertTrue(all("statistics" not in g for g in games))
        self.assertEqual(audit["accepted_measured_games"], 0)

    def test_malformed_verified_payloads_fail_without_conditioning_target_membership(self):
        original = verified_row()
        payload = json.loads(original["player_match_statistics"])
        first = next(iter(payload))
        malformed = []
        malformed.append({key: value for key, value in payload.items() if key != first})
        malformed.append({**payload, "strandedpond:nickname:outsider": payload[first]})
        for field, value in (("kills", -1), ("rounds", 0), ("adr_round_sum", float("inf")),
                             ("maps", 1), ("kast_round_sum", 100)):
            malformed.append({**payload, first: {**payload[first], field: value}})
        for measurement in malformed:
            with self.subTest(measurement=measurement):
                row = {**original, "player_match_statistics": json.dumps(measurement)}
                with self.assertRaisesRegex(ValueError, "invalid_verified_measurements"):
                    load_history(pd.DataFrame([row]))
        duplicate_roster = {**original, "team2_player_1_key": original["team1_player_1_key"]}
        with self.assertRaisesRegex(ValueError, "invalid_verified_measurements"):
            load_history(pd.DataFrame([duplicate_roster]))
        for payload_text in ("{not-json", "[]", "null", '{"duplicate": {}, "duplicate": {}}'):
            with self.subTest(payload=payload_text), self.assertRaisesRegex(ValueError, "invalid_verified_measurements"):
                load_history(pd.DataFrame([{**original, "player_match_statistics": payload_text}]))
