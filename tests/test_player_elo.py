import csv
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

from pipeline.ratings.player_elo import MapObservation, PlayerElo, evaluate, load_maps, predict_rosters


A = tuple(f"nick:a{i}" for i in range(5))
B = tuple(f"nick:b{i}" for i in range(5))


def observation(key, day, y=1, roster1=A, roster2=B, team1="A", team2="B"):
    return MapObservation(key, key.split(":")[0], datetime(2026, 1, day, 12),
                          "Nuke", team1, team2, roster1, roster2, y)


class PlayerRatingTests(unittest.TestCase):
    def test_roster_mean_drives_prediction_and_every_player_gets_map_residual(self):
        model = PlayerElo(32, 16)
        rows = model.process_day([observation("1:Nuke", 1)])
        self.assertEqual(rows[0]["p_player_elo"], 0.5)
        self.assertEqual({model.players[p] for p in A}, {1516})
        self.assertEqual({model.players[p] for p in B}, {1484})
        self.assertEqual(model.teams, {"A": 1508, "B": 1492})
        self.assertEqual(model.map_counts[A[0]], 1)
        self.assertEqual(model.series[A[0]], {"1"})

    def test_same_day_results_never_influence_same_day_predictions(self):
        model = PlayerElo(32, 32)
        rows = model.process_day([observation("1:Nuke", 1), observation("1:Mirage", 1)])
        self.assertEqual([r["p_player_elo"] for r in rows], [0.5, 0.5])
        self.assertEqual([r["minimum_prior_player_maps"] for r in rows], [0, 0])
        self.assertEqual(model.map_counts[A[0]], 2)
        self.assertEqual(len(model.series[A[0]]), 1)

    def test_player_skill_follows_identity_across_organizations(self):
        model = PlayerElo(32, 32)
        model.process_day([observation("1:Nuke", 1)])
        row = model.process_day([observation("2:Nuke", 2, team1="new org", team2="other org")])[0]
        self.assertGreater(row["p_player_elo"], 0.5)
        self.assertEqual(row["p_team_elo"], 0.5)
        self.assertEqual(row["minimum_prior_player_maps"], 1)

    def test_player_order_does_not_change_roster_probability(self):
        first, second = PlayerElo(32, 32), PlayerElo(32, 32)
        first.process_day([observation("1:Nuke", 1)])
        second.process_day([observation("1:Nuke", 1)])
        a = first.process_day([observation("2:Nuke", 2)])[0]
        b = second.process_day([observation("2:Nuke", 2, roster1=tuple(reversed(A)))])[0]
        self.assertEqual(a["p_player_elo"], b["p_player_elo"])
        self.assertEqual(first.players, second.players)

    def test_repeated_dates_duplicate_players_and_overlap_are_rejected(self):
        model = PlayerElo(32, 32)
        model.process_day([observation("1:Nuke", 2)])
        with self.assertRaises(ValueError):
            model.process_day([observation("2:Nuke", 2)])
        for bad in (observation("1:Nuke", 1, roster1=A[:-1]+(A[0],)),
                    observation("1:Nuke", 1, roster2=B[:-1]+(A[0],))):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                PlayerElo(32, 32).process_day([bad])

    def test_test_outcomes_do_not_select_k_or_change_their_own_predictions(self):
        history = [observation("1:Nuke", 1), observation("2:Nuke", 2),
                   observation("3:Nuke", 3), observation("4:Nuke", 4, 0),
                   observation("5:Nuke", 5)]
        a, rows_a, _ = evaluate(history, date(2026, 1, 3), date(2026, 1, 5))
        b, rows_b, _ = evaluate(history[:-1]+[replace(history[-1], y=0)],
                               date(2026, 1, 3), date(2026, 1, 5))
        self.assertEqual(a["selected_k"], b["selected_k"])
        self.assertEqual(a["validation_candidates"], b["validation_candidates"])
        self.assertEqual([r["p_player_elo"] for r in rows_a], [r["p_player_elo"] for r in rows_b])


class JoinTests(unittest.TestCase):
    def fixture(self, key="1", map_name="Nuke", day="2026-01-01 12:00"):
        row = dict(event="event", date=day, map=map_name, team_left="A", team_right="B",
                   score_left="13", score_right="7", total_rounds="20", winner="A",
                   match_url=f"https://www.hltv.org/matches/{key}/a-vs-b")
        players = [dict(event="event", date=day, map=map_name, team=team,
                        player_name=f"{team}{i}") for team in ("A", "B") for i in range(5)]
        return row, players

    def load(self, maps, players):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp)/"maps.csv", Path(temp)/"players.csv"]
            for path, rows in zip(paths, (maps, players)):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            return load_maps(*paths)

    def test_exact_duplicates_do_not_inflate_maps_or_rosters(self):
        row, players = self.fixture()
        maps, audit, rejected = self.load([row, row], players+players)
        self.assertEqual(len(maps), 1)
        self.assertEqual(len(maps[0].roster1), 5)
        self.assertEqual(audit["exact_duplicate_map_rows"], 1)
        self.assertEqual(audit["exact_duplicate_player_rows"], 10)
        self.assertEqual(rejected, [])

    def test_date_conflicting_series_is_quarantined_not_time_corrected(self):
        a, pa = self.fixture()
        b, pb = self.fixture(map_name="Mirage", day="2026-01-02 03:00")
        maps, _, rejected = self.load([a, b], pa+pb)
        self.assertEqual(maps, [])
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all("series_calendar_date_conflict" in r["reasons"] for r in rejected))

    def test_ambiguous_join_does_not_assign_one_roster_to_two_matches(self):
        a, players = self.fixture()
        b, _ = self.fixture("2")
        maps, _, rejected = self.load([a, b], players)
        self.assertEqual(maps, [])
        self.assertTrue(all("ambiguous_roster_join" in r["reasons"] for r in rejected))

    def test_sixth_player_and_cross_team_overlap_are_not_silently_dropped(self):
        row, players = self.fixture()
        extra = dict(players[0], player_name="sixth")
        self.assertEqual(self.load([row], players+[extra])[0], [])
        overlap = [dict(p, player_name="A0") if p["player_name"]=="B0" else p for p in players]
        self.assertEqual(self.load([row], overlap)[0], [])


class RosterScenarioTests(unittest.TestCase):
    def test_saved_snapshot_uses_explicit_rosters_and_reversal_complements(self):
        model = dict(schema_version=1, identity_namespace="strandedpond:nickname:",
                     history_through="2026-01-01", selected_k={"player":32,"team":32},
                     players={p:1516 if p in A else 1484 for p in A+B},
                     map_counts={p:1 for p in A+B})
        a = predict_rosters(model, A, B)
        b = predict_rosters(model, B, A)
        self.assertAlmostEqual(a["p_roster1"]+b["p_roster1"], 1)
        self.assertEqual(a["target"], "map_winner")
        self.assertEqual(a["forecast_kind"], "historical_roster_scenario")
        with self.assertRaises(ValueError):
            predict_rosters(model, A[:-1]+("unknown",), B)


if __name__ == "__main__":
    unittest.main()
