import unittest
import json
import csv
import io
import pickle

import pandas as pd

from pipeline.pipelines.data_consolidation.nodes import build_consolidated_dataset
from pipeline.pipelines.match_modeling.nodes import (
    generate_player_rankings_and_features,
    predict_player_match,
)


def match_row(identity="1", day="2026-01-01", winner="team1"):
    return {"match_id": identity, "date": day + " 12:00:00", "team1": "A", "team2": "B",
            "score1": "2" if winner == "team1" else "0", "score2": "0" if winner == "team1" else "2",
            "winner": winner, "format": "bo3", "tournament": "Event",
            "match_url": f"https://www.hltv.org/matches/{identity}/a-vs-b"}


def snapshot_row(identity="1"):
    row = {"hltv_match_id": identity, "team1_name": "A", "team2_name": "B"}
    for side in (1, 2):
        for slot in range(1, 6):
            row[f"team{side}_player_{slot}_name"] = f"snapshot{side}{slot}"
    return row


def captured_lineup(identity="1", source="hltv_live"):
    row = {"hltv_match_id": identity, "source": source, "lineup_status": "complete",
           "team1": "A", "team2": "B", "match_url": f"https://www.hltv.org/matches/{identity}/a-vs-b"}
    for side in (1, 2):
        for slot in range(1, 6):
            row[f"team{side}_p{slot}_name"] = f"actual{side}{slot}"
            row[f"team{side}_p{slot}_id"] = str(side * 10 + slot)
    return row


def consolidated(rows, lineups=None, snapshots=None):
    return build_consolidated_dataset(pd.DataFrame(rows), pd.DataFrame(snapshots or [snapshot_row()]),
                                      pd.DataFrame(lineups or [captured_lineup(source="kaggle")]))


class ConsolidationCorrectnessTests(unittest.TestCase):
    def test_snapshot_names_are_not_historical_participants(self):
        result = consolidated([match_row()])
        self.assertFalse(bool(result.iloc[0]["has_lineup"]))

    def test_captured_player_names_and_ids_override_unverified_snapshot_as_a_pair(self):
        result = consolidated([match_row()], [captured_lineup()])
        self.assertEqual(result.iloc[0]["team1_player_1_name"], "actual11")
        self.assertEqual(str(result.iloc[0]["team1_p1_id"]), "11")

    def test_one_missing_player_rejects_lineup(self):
        lineup = captured_lineup()
        lineup["team2_p5_name"] = ""
        result = consolidated([match_row()], [lineup])
        self.assertFalse(bool(result.iloc[0]["has_lineup"]))

    def test_conflicting_duplicate_native_match_ids_fail_instead_of_multiplying_rows(self):
        with self.assertRaises(ValueError):
            consolidated([match_row(), {**match_row(), "team2": "C"}])

    def test_capture_orientation_is_checked_before_player_assignment(self):
        lineup = captured_lineup()
        lineup["team1"], lineup["team2"] = "B", "A"
        result = consolidated([match_row()], [lineup])
        self.assertEqual(result.iloc[0]["team1_player_1_name"], "actual21")
        self.assertEqual(str(result.iloc[0]["team1_p1_id"]), "21")



class HistoricalSeriesEvidenceTests(unittest.TestCase):
    def fixture(self):
        maps, players = [], []
        for map_name, round_score in (("Nuke", (13, 7)), ("Mirage", (13, 9))):
            maps.append({"event": "Event", "date": "2026-01-01 12:00", "map": map_name,
                         "team_left": "A", "team_right": "B", "score_left": str(round_score[0]),
                         "score_right": str(round_score[1]), "total_rounds": str(sum(round_score)),
                         "winner": "A", "match_url": "https://www.hltv.org/matches/1/a-vs-b"})
            for team in ("A", "B"):
                for slot in range(5):
                    players.append({"event": "Event", "date": "2026-01-01 12:00", "map": map_name,
                                    "team": team, "player_name": team + str(slot), "kills": "10",
                                    "deaths": "8", "assists": "4", "adr": "70", "kast": "0.7"})
        return maps, players

    def consolidate(self, maps, players):
        return build_consolidated_dataset(pd.DataFrame([match_row()]), pd.DataFrame([snapshot_row()]),
                                          pd.DataFrame(), pd.DataFrame(maps), pd.DataFrame(players))

    def test_complete_historical_series_supplies_scoped_names_and_round_weighted_statistics(self):
        maps, players = self.fixture()
        row = self.consolidate(maps, players).iloc[0]
        self.assertTrue(bool(row["has_lineup"]))
        self.assertEqual(row["team1_player_1_key"], "strandedpond:nickname:A0")
        self.assertEqual(row["team1_p1_id"], "")
        measurements = json.loads(row["player_match_statistics"])["strandedpond:nickname:A0"]
        self.assertEqual(measurements["rounds"], 42)
        self.assertEqual(measurements["kills"], 20)
        self.assertEqual(measurements["adr_round_sum"] / measurements["rounds"], 70)

    def test_missing_or_conflicting_map_results_do_not_supply_a_series_lineup(self):
        maps, players = self.fixture()
        self.assertFalse(bool(self.consolidate(maps[:1], players).iloc[0]["has_lineup"]))
        changed = [maps[0], {**maps[1], "winner": "B"}]
        self.assertFalse(bool(self.consolidate(changed, players).iloc[0]["has_lineup"]))

    def test_one_substitute_or_cross_date_series_is_quarantined(self):
        maps, players = self.fixture()
        replaced = [{**p, "player_name": "substitute"} if p["map"] == "Mirage" and p["player_name"] == "A0"
                    else p for p in players]
        self.assertFalse(bool(self.consolidate(maps, replaced).iloc[0]["has_lineup"]))
        changed_maps = [maps[0], {**maps[1], "date": "2026-01-02 12:00"}]
        self.assertFalse(bool(self.consolidate(changed_maps, players).iloc[0]["has_lineup"]))

class ReplayCorrectnessTests(unittest.TestCase):
    def replay(self, rows):
        frame = consolidated(rows, [captured_lineup(r["match_id"]) for r in rows])
        return generate_player_rankings_and_features(frame)[0]

    def test_same_day_result_is_update_only_for_following_days(self):
        features = self.replay([match_row("1"), match_row("2"), match_row("3", "2026-01-02")])
        by_id = features.set_index("hltv_match_id")
        self.assertEqual(by_id.loc["2", "elo_prob"], 0.5)
        self.assertGreater(by_id.loc["3", "elo_prob"], 0.5)

    def test_flipping_result_does_not_change_other_same_day_predictions(self):
        first = self.replay([match_row("1"), match_row("2"), match_row("3", "2026-01-02")])
        second = self.replay([match_row("1", winner="team2"), match_row("2"), match_row("3", "2026-01-02")])
        a, b = first.set_index("hltv_match_id"), second.set_index("hltv_match_id")
        self.assertEqual(a.loc["2", "elo_prob"], b.loc["2", "elo_prob"])
        self.assertNotEqual(a.loc["3", "elo_prob"], b.loc["3", "elo_prob"])

    def test_forfeit_never_updates_participants(self):
        features = self.replay([{**match_row("1"), "format": "def"}, match_row("2", "2026-01-02")])
        self.assertEqual(features["hltv_match_id"].tolist(), ["2"])
        self.assertEqual(features.iloc[0]["elo_prob"], 0.5)

    def test_current_box_scores_are_not_used_until_a_following_day(self):
        rows = [match_row("1"), match_row("2"), match_row("3", "2026-01-02")]
        frame = consolidated(rows, [captured_lineup(r["match_id"]) for r in rows])
        statistics = {}
        for side in (1, 2):
            for slot in range(1, 6):
                key = f"hltv:player:{side * 10 + slot}"
                statistics[key] = {"maps": 2, "rounds": 40, "kills": 30 if side == 1 else 10,
                                   "deaths": 15, "assists": 5, "adr_round_sum": 4000 if side == 1 else 2000,
                                   "kast_round_sum": 30 if side == 1 else 20}
        frame.loc[frame["hltv_match_id"] == "1", "player_match_statistics"] = json.dumps(statistics)
        features = generate_player_rankings_and_features(frame)[0].set_index("hltv_match_id")
        self.assertEqual(features.loc["1", "performance_available"], 0)
        self.assertEqual(features.loc["2", "performance_available"], 0)
        self.assertEqual(features.loc["3", "performance_available"], 1)
        self.assertEqual(features.loc["3", "player_adr_diff"], 50)

    def test_native_player_ids_survive_csv_export_alongside_unmapped_nicknames(self):
        rows = [match_row("1"), match_row("2", "2026-01-02")]
        frame = consolidated(rows, [captured_lineup(r["match_id"]) for r in rows])
        frame.loc[frame["hltv_match_id"] == "2", "identity_namespace"] = "strandedpond:nickname"
        for side in (1, 2):
            for slot in range(1, 6):
                column = f"team{side}_player_{slot}_key"
                frame.loc[frame["hltv_match_id"] == "2", column] = f"strandedpond:nickname:actual{side}{slot}"
        rankings = generate_player_rankings_and_features(frame)[1]
        exported = list(csv.DictReader(io.StringIO(rankings.to_csv(index=False))))
        native = [r for r in exported if r["identity_namespace"] == "hltv:player"]
        self.assertEqual({r["player_id"] for r in native},
                         {str(side * 10 + slot) for side in (1, 2) for slot in range(1, 6)})
        self.assertTrue(all(r["player_id"] == "" for r in exported if r["identity_namespace"] == "strandedpond:nickname"))


class BroaderArchiveReplayTests(unittest.TestCase):
    def frame(self, rows, eligible_ids):
        return consolidated(rows, [captured_lineup(identity) for identity in eligible_ids])

    def replay(self, rows, eligible_ids):
        return generate_player_rankings_and_features(self.frame(rows, eligible_ids))

    def raw_archive_bundle(self, version=4):
        column = "archive_elo_prob"
        return {
            "version": version, "family": "rating_channel",
            "selected_model": "pure_archive_elo", "probability_column": column,
            "recipe": {"name": "pure_archive_elo", "family": "rating_channel",
                       "feature_columns": [column], "transform": "identity", "parameters": {}},
            "feature_columns": [column],
            "mirror_rules": {column: {"operation": "complement", "source": column}},
        }

    def rosters(self):
        return tuple(tuple(f"hltv:player:{side * 10 + slot}" for slot in range(1, 6))
                     for side in (1, 2))

    def test_no_lineup_history_changes_future_archive_forecasts_not_labels_or_player_channels(self):
        targets = [match_row("2", "2026-01-02"), match_row("3", "2026-01-03")]
        baseline, baseline_rankings, _, _ = self.replay(targets, ["2", "3"])
        features, rankings, audit, state = self.replay(
            [match_row("1"), *targets], ["2", "3"])
        self.assertEqual(features["hltv_match_id"].tolist(), ["2", "3"])
        self.assertEqual(features["target"].tolist(), baseline["target"].tolist())
        self.assertEqual(features["date"].tolist(), baseline["date"].tolist())
        self.assertEqual(features[["team1", "team2"]].to_dict("records"),
                         baseline[["team1", "team2"]].to_dict("records"))
        player_columns = [column for column in baseline if not column.startswith("archive_")]
        pd.testing.assert_frame_equal(features[player_columns], baseline[player_columns])
        pd.testing.assert_frame_equal(rankings, baseline_rankings)
        self.assertEqual(audit["eligible_series"], 2)
        self.assertGreater(features.iloc[0]["archive_elo_prob"], 0.5)
        self.assertEqual(features.iloc[0]["elo_prob"], 0.5)
        self.assertEqual(features.iloc[0]["archive_team1_prior_matches"], 1)
        self.assertEqual(state["schema_version"], 2)
        self.assertIn("team_history_engine", state)

    def test_archive_outcomes_are_frozen_for_all_same_day_target_features(self):
        rows = [match_row("1"), match_row("2"), match_row("3", "2026-01-02")]
        original = self.replay(rows, ["2", "3"])[0].set_index("hltv_match_id")
        changed = self.replay(
            [match_row("1", winner="team2"), *rows[1:]], ["2", "3"])[0].set_index("hltv_match_id")
        for channel in ("archive_elo", "archive_decay"):
            self.assertEqual(original.loc["2", channel + "_prob"], 0.5)
            self.assertEqual(original.loc["2", channel + "_prob"], changed.loc["2", channel + "_prob"])
            self.assertNotEqual(original.loc["3", channel + "_prob"], changed.loc["3", channel + "_prob"])
        self.assertEqual(original.loc["2", "archive_team1_prior_matches"], 0)
        self.assertEqual(original.loc["3", "archive_team1_prior_matches"], 2)

    def test_earlier_features_ignore_future_labels_and_input_row_order(self):
        rows = [match_row("1"), match_row("2", "2026-01-02"),
                match_row("3", "2026-01-03"), match_row("4", "2026-01-04")]
        original = self.replay(rows, ["2", "3", "4"])[0].set_index("hltv_match_id")
        changed_rows = [*rows[:3], match_row("4", "2026-01-04", winner="team2")]
        changed = self.replay(list(reversed(changed_rows)), ["2", "3", "4"])[0].set_index("hltv_match_id")
        pd.testing.assert_frame_equal(original.loc[["2", "3"]], changed.loc[["2", "3"]])

    def test_pre_modern_maps_forfeits_and_invalid_names_never_supply_archive_evidence(self):
        invalid_rows = [
            match_row("1", "2023-12-31"),
            {**match_row("2"), "format": "bo1", "score1": "13", "score2": "7"},
            {**match_row("3"), "format": "def"},
            {**match_row("4"), "score1": "1"},
            match_row("5"), match_row("6"), match_row("7"), match_row("8"),
        ]
        target = match_row("9", "2026-01-02")
        frame = self.frame([*invalid_rows, target], ["9"])
        # Change consolidated source fields so this regression isolates replay,
        # not the upstream capture/name matching rules.
        frame.loc[frame["hltv_match_id"] == "5", "team2"] = ""
        frame.loc[frame["hltv_match_id"] == "6", "team2"] = "A"
        frame.loc[frame["hltv_match_id"] == "7", "team1"] = "   "
        frame.loc[frame["hltv_match_id"] == "8", "date"] = "not-a-date"
        features, _, _, state = generate_player_rankings_and_features(frame)
        self.assertEqual(features["hltv_match_id"].tolist(), ["9"])
        row = features.iloc[0]
        self.assertEqual(row["archive_elo_prob"], 0.5)
        self.assertEqual(row["archive_decay_prob"], 0.5)
        self.assertEqual(row["archive_team1_prior_matches"], 0)
        self.assertEqual(row["archive_team2_prior_matches"], 0)
        self.assertEqual(state["history_through"], "2026-01-02")

    def test_source_names_remain_literal_in_node_replay(self):
        rows = [{**match_row("1"), "team1": "a", "team2": "b"},
                match_row("2", "2026-01-02")]
        features = self.replay(rows, ["2"])[0]
        self.assertEqual(features.iloc[0]["archive_elo_prob"], 0.5)
        self.assertEqual(features.iloc[0]["archive_minimum_prior_matches"], 0)

    def test_serialized_both_engines_use_maximum_history_day_for_explicit_inference(self):
        rows = [match_row("1"), match_row("2", "2026-01-03", winner="team2")]
        _, _, _, state = self.replay(rows, ["1"])
        self.assertEqual(state["schema_version"], 2)
        self.assertEqual(state["history_through"], "2026-01-03")
        restored = pickle.loads(pickle.dumps(state))
        bundle = self.raw_archive_bundle()
        roster1, roster2 = self.rosters()
        expected = state["team_history_engine"].compute_prematch_features(
            "A", "B", day="2026-01-04")["archive_elo_prob"]
        forward = predict_player_match(bundle, restored, "A", "B", roster1, roster2, "2026-01-04")
        reverse = predict_player_match(bundle, restored, "B", "A", roster2, roster1, "2026-01-04")
        self.assertEqual(forward["history_through"], "2026-01-03")
        self.assertAlmostEqual(forward["prob_team1_win"], expected)
        self.assertAlmostEqual(forward["prob_team1_win"] + reverse["prob_team1_win"], 1)
        player = restored["engine"].compute_prematch_features(
            "A", "B", roster1, roster2, day="2026-01-04")
        self.assertGreater(player["elo_prob"], 0.5)
        self.assertEqual(player["minimum_prior_matches"], 1)
        for day in ("2026-01-01", "2026-01-02", "2026-01-03"):
            with self.subTest(day=day), self.assertRaisesRegex(ValueError, "after|history"):
                predict_player_match(bundle, restored, "A", "B", roster1, roster2, day)

    def test_obsolete_state_and_artifact_are_rejected_before_scenario_prediction(self):
        _, _, _, state = self.replay([match_row("1")], ["1"])
        roster1, roster2 = self.rosters()
        obsolete = dict(state, schema_version=1)
        obsolete.pop("team_history_engine", None)
        with self.assertRaisesRegex(ValueError, "state|version|schema"):
            predict_player_match(self.raw_archive_bundle(), obsolete, "A", "B",
                                 roster1, roster2, "2026-01-02")
        with self.assertRaisesRegex(ValueError, "artifact|version"):
            predict_player_match(self.raw_archive_bundle(version=3), state, "A", "B",
                                 roster1, roster2, "2026-01-02")


if __name__ == "__main__":
    unittest.main()
