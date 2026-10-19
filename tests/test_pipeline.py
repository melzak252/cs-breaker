"""Contract, ingestion, parsing, feature, model, and CLI pipeline tests."""

import hashlib
import math
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from pipeline.contracts import MatchFixture, ParsedMapResult, PlayerMapStats, Tournament
from pipeline.ingestion.downloader import ChecksumVerificationError, DemoDownloader
from pipeline.ingestion.liquipedia import LiquipediaClient
from pipeline.parser.aggregator import EventAggregator, OpenCS2Adapter
from pipeline.parser.demoparser import CS2DemoParser
from pipeline.features.builder import ChronologicalFeatureBuilder
from pipeline.features.dataset import DatasetPipeline
from pipeline.models.metrics import (
    brier_score,
    evaluate_predictions,
    expected_calibration_error,
    log_loss_score,
)
from pipeline.models.trainer import ModelTrainer


class TestContracts(unittest.TestCase):
    def test_tournament_serializes_all_metadata(self):
        item = Tournament("id", "Major", "S", "2024-01-01", "2024-01-05", "Paris", "https://example.test")
        self.assertEqual(item.to_dict()["name"], "Major")
        self.assertEqual(item.to_dict()["liquipedia_url"], "https://example.test")

    def test_match_fixture_serializes_timestamp_and_maps(self):
        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        item = MatchFixture("m1", "Major", timestamp, "A", "B", "bo3", 2, 1, "team1", maps=["Mirage"])
        data = item.to_dict()
        self.assertEqual(data["timestamp"], timestamp.isoformat())
        self.assertEqual(data["maps"], ["Mirage"])
        self.assertEqual(data["winner"], "team1")

    def test_player_stats_provide_safe_derived_rates(self):
        player = PlayerMapStats("p", "Player", "A", 2, kills=6, deaths=3, headshot_kills=3, damage_dealt=240, rounds_played=12)
        self.assertEqual(player.adr, 20)
        self.assertEqual(player.kd_ratio, 2)
        self.assertEqual(player.hs_percent, 50)
        self.assertEqual(player.to_dict()["adr"], 20)
        zero_deaths = PlayerMapStats("q", "Zero", "A", 2, kills=4)
        self.assertEqual(zero_deaths.kd_ratio, 4)

    def test_parsed_map_result_serializes_nested_player_contracts(self):
        player = PlayerMapStats("p", "Player", "A", 2)
        result = ParsedMapResult("m", "de_mirage", "A", "B", 13, 8, 21, "A", [player])
        data = result.to_dict()
        self.assertEqual(data["winner_team"], "A")
        self.assertEqual(data["player_stats"][0]["player_id"], "p")


class TestIngestion(unittest.TestCase):
    def test_liquipedia_uses_offline_fallback_on_request_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            client = LiquipediaClient(cache_dir=directory)
            with mock.patch.object(client, "_cargo_query", side_effect=OSError("offline")):
                tournaments = client.get_tournaments(2025)
                matches = client.get_tournament_matches("Example Cup")
        self.assertTrue(tournaments)
        self.assertEqual(tournaments[0].tournament_id, "offline-2025-major")
        self.assertEqual(matches[0].team1, "Team Alpha")

    def test_liquipedia_pacing_sleeps_only_for_remaining_interval(self):
        client = LiquipediaClient()
        client._last_request_at = 10.0
        with mock.patch("pipeline.ingestion.liquipedia.time.monotonic", side_effect=[11.25, 11.25]), mock.patch(
            "pipeline.ingestion.liquipedia.time.sleep"
        ) as sleep:
            client._pace_request()
        sleep.assert_called_once_with(0.75)
        self.assertEqual(client._last_request_at, 11.25)

    def test_downloader_creates_synthetic_fixture_and_checks_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(DemoDownloader.create_dummy_test_demo(str(Path(directory) / "nested" / "sample.dem"), "de_nuke"))
            content = path.read_bytes()
            self.assertTrue(content.startswith(b"PBDEMS2\x00"))
            self.assertIn(b"de_nuke", content)
            downloader = DemoDownloader()
            self.assertEqual(downloader._hash_file(path), hashlib.sha256(content).hexdigest())

    def test_downloader_accepts_matching_and_rejects_wrong_checksum(self):
        payload = b"demo payload"

        class Response:
            status = 200
            headers = {"Content-Length": str(len(payload))}

            def __init__(self):
                self.remaining = payload

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size=-1):
                result, self.remaining = self.remaining[:size], self.remaining[size:]
                return result

            def getcode(self):
                return 200

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "pipeline.ingestion.downloader.urllib.request.urlopen",
            side_effect=[Response(), Response()],
        ):
            downloader = DemoDownloader()
            destination = str(Path(directory) / "demo.dem")
            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(downloader.download_file("https://example.test/demo", destination, expected), destination)
            with self.assertRaises(ChecksumVerificationError):
                downloader.download_file("https://example.test/demo", destination + ".bad", "0" * 64)

class TestCSGOHistoricalAdapter(unittest.TestCase):
    def test_clean_round_score(self):
        from pipeline.ingestion.csgo_archive import CSGOHistoricalAdapter
        self.assertEqual(CSGOHistoricalAdapter._clean_round_score(" (16)"), 16)
        self.assertEqual(CSGOHistoricalAdapter._clean_round_score("(4)"), 4)
        self.assertIsNone(CSGOHistoricalAdapter._clean_round_score(""))

    def test_load_historical_fixtures_from_csv(self):
        from pipeline.ingestion.csgo_archive import CSGOHistoricalAdapter
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("date_ymd,date_unix_iso,date_unix,team1,team2,team1_rounds,team2_rounds,map_name_short,event_name,game_link\n")
            f.write("2023-07-13,2023-07-13 22:15:00,1689282900000,Take Flyte,ex-Snakes Den , (4),(16),inf,Thunderpick NA,/stats/matches/mapstatsid/159588/tf-vs-sd\n")
            f.write("2023-07-13,2023-07-13 20:30:00,1689276600000,Boca Juniors ,AJF ,(16),(13),mrg,CCT SA,/stats/matches/mapstatsid/159589/bj-vs-ajf\n")
            temp_path = f.name
        try:
            fixtures = CSGOHistoricalAdapter.load_historical_fixtures(temp_path)
            self.assertEqual(len(fixtures), 2)
            self.assertEqual(fixtures[0].team1, "Boca Juniors")
            self.assertEqual(fixtures[0].score1, 16)
            self.assertEqual(fixtures[0].score2, 13)
            self.assertEqual(fixtures[0].winner, "team1")
            self.assertEqual(fixtures[0].maps, ["de_mirage"])
            self.assertEqual(fixtures[1].team2, "ex-Snakes Den")
            self.assertEqual(fixtures[1].winner, "team2")
            self.assertEqual(fixtures[1].maps, ["de_inferno"])
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestDemoParser(unittest.TestCase):
    def test_synthetic_demo_returns_normalized_empty_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = DemoDownloader.create_dummy_test_demo(str(Path(directory) / "test.dem"))
            with mock.patch("pipeline.parser.demoparser.DemoParser", None):
                result = CS2DemoParser(path).parse_all_events()
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["deaths"], [])
        self.assertIn("map_name", result["header"])

    def test_parser_extracts_normalized_mock_events(self):
        class Backend:
            def __init__(self, path): pass
            def parse_header(self): return {"mapname": "de_mirage", "tickrate": 64}
            def parse_event(self, name, **kwargs):
                return {"round_end": [{"round_num": 1, "winner": "TERRORIST"}],
                        "player_death": [{"attacker_name": "A", "user_name": "B"}],
                        "player_hurt": [{"attacker_name": "A", "user_name": "B", "dmg_health": 40}]}.get(name, [])
        with tempfile.TemporaryDirectory() as directory:
            path = DemoDownloader.create_dummy_test_demo(str(Path(directory) / "test.dem"))
            with mock.patch("pipeline.parser.demoparser.DemoParser", Backend):
                events = CS2DemoParser(path).parse_all_events()
        self.assertEqual(events["header"]["map_name"], "de_mirage")
        self.assertEqual(events["rounds"][0]["winner"], "T")
        self.assertEqual(events["damages"][0]["attacker"], "A")

    def test_aggregator_calculates_adr_kd_and_opening_duels(self):
        events = {
            "deaths": [
                {"round_num": 1, "attacker_steamid": "a", "attacker_name": "Alpha", "attacker_team_num": 2, "user_steamid": "b", "user_name": "Bravo", "user_team_num": 3, "headshot": True},
                {"round_num": 2, "attacker_steamid": "a", "attacker_name": "Alpha", "attacker_team_num": 2, "user_steamid": "c", "user_name": "Charlie", "user_team_num": 3},
            ],
            "damages": [{"round_num": 1, "attacker_steamid": "a", "attacker_name": "Alpha", "attacker_team_num": 2, "dmg_health": 100},
                        {"round_num": 2, "attacker_steamid": "a", "attacker_name": "Alpha", "attacker_team_num": 2, "dmg_health": 80}],
            "rounds": [{"round_num": 1, "winner": "team1"}, {"round_num": 2, "winner": "team1"}],
        }
        result = EventAggregator.aggregate_from_events("match", "de_mirage", events, "A", "B")
        players = {p.player_name: p for p in result.player_stats}
        alpha, bravo = players["Alpha"], players["Bravo"]
        self.assertEqual((alpha.kills, alpha.deaths, alpha.kd_ratio), (2, 0, 2))
        self.assertEqual(alpha.adr, 90)
        self.assertEqual((alpha.first_kills, bravo.first_deaths), (2, 1))
        self.assertEqual(result.winner_team, "A")

    def test_opencs2_adapter_reads_local_parquet_shard(self):
        directory = Path("data/raw/opencs2")
        shards = sorted(directory.glob("*.parquet"))
        self.assertTrue(shards, "expected checked-in OpenCS2 parquet shards")
        try:
            results = OpenCS2Adapter.parse_opencs2_parquet(str(shards[0]))
        except ImportError as error:
            self.skipTest(str(error))
        self.assertTrue(results)
        self.assertIsInstance(results[0], ParsedMapResult)
        self.assertTrue(results[0].map_name)


class TestFeatureEngineering(unittest.TestCase):
    @staticmethod
    def fixture(match_id, day, team1="A", team2="B", winner="team1"):
        return MatchFixture(match_id, "Cup", datetime(2024, 1, day, tzinfo=timezone.utc), team1, team2, "bo3", 2, 0, winner)

    def test_first_match_features_exclude_its_own_outcome(self):
        matches = [self.fixture("1", 1), self.fixture("2", 2)]
        rows = ChronologicalFeatureBuilder().process_matches(matches)
        self.assertEqual(rows[0]["experience_diff"], 0)
        self.assertEqual(rows[0]["h2h_matches"], 0)
        self.assertEqual(rows[1]["experience_diff"], 0)
        self.assertEqual(rows[1]["h2h_matches"], 1)
        self.assertEqual(rows[1]["h2h_win_rate_team1"], 1)

    def test_rolling_form_and_chronological_split_are_consistent(self):
        matches = [self.fixture("1", 1, "A", "B", "team1"), self.fixture("2", 2, "A", "C", "team2"), self.fixture("3", 3, "A", "D", "team1")]
        rows = ChronologicalFeatureBuilder().process_matches(matches)
        self.assertEqual(rows[2]["win_rate_diff_5"], 0.5)
        pipeline = DatasetPipeline(matches=matches)
        train, test, manifest = pipeline.build_chronological_dataset(2 / 3)
        self.assertEqual(manifest["match_count"], 3)
        self.assertTrue(DatasetPipeline.verify_zero_leakage(train, test))

    def test_leakage_audit_rejects_label_feature_and_wrong_history(self):
        good = {"date": "2024-01-01T00:00:00+00:00", "team1": "A", "team2": "B", "prior_matches_team1": 0, "prior_matches_team2": 0, "features": {"elo_diff": 0}}
        bad = dict(good, features={"winner": "A"})
        with self.assertRaises(AssertionError):
            DatasetPipeline.verify_zero_leakage([bad], [])
        next_row = dict(good, date="2024-01-02T00:00:00+00:00", prior_matches_team1=0, prior_matches_team2=1)
        with self.assertRaises(AssertionError):
            DatasetPipeline.verify_zero_leakage([good, next_row], [])


class TestModels(unittest.TestCase):
    def test_binary_metrics_have_expected_values_and_validation(self):
        labels, probabilities = [0, 1], [0.25, 0.75]
        self.assertAlmostEqual(brier_score(labels, probabilities), 0.0625)
        self.assertAlmostEqual(log_loss_score(labels, probabilities), -math.log(0.75))
        ece, bins = expected_calibration_error(labels, probabilities, n_bins=2)
        self.assertAlmostEqual(ece, 0.25)
        self.assertEqual([entry["count"] for entry in bins], [1, 1])
        report = evaluate_predictions(labels, probabilities)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["n_samples"], 2)
        with self.assertRaises(ValueError):
            brier_score([0], [1.1])

    def test_trainer_fit_evaluate_roundtrip_and_predicts_probabilities(self):
        train = [{"elo_diff": float(index % 7 - 3), "form": float(index % 3), "y": index % 2} for index in range(20)]
        test = [{"elo_diff": -1.0, "form": 0.0, "y": 0}, {"elo_diff": 1.0, "form": 2.0, "y": 1}]
        trainer = ModelTrainer()
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                summary = trainer.train_and_evaluate(train, test)
                name = summary["best_model"]
                saved = trainer.save_model(name, str(Path(directory) / "model.joblib"))
                bundle = ModelTrainer.load_model(saved)
                self.assertEqual(bundle["model_name"], name)
                probability = trainer.predict_proba(name, {"elo_diff": 1.0, "form": 2.0})
            finally:
                os.chdir(previous)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        self.assertIn("test", summary["models"][name])


class TestPlayerRatingEngine(unittest.TestCase):
    def test_player_rating_initialization_and_update(self):
        from pipeline.features.player_ratings import IndividualPlayerRatingEngine
        engine = IndividualPlayerRatingEngine()
        p = engine.get_player("s1mple")
        self.assertEqual(p.elo, 1500.0)
        self.assertEqual(p.matches_played, 0)

        engine.update_match(
            team1="NaVi",
            team2="FaZe",
            team1_players=["s1mple", "b1t", "electronic", "perfecto", "boombl4"],
            team2_players=["karrigan", "rain", "broky", "twistzz", "ropz"],
            winner="team1",
            date_iso="2022-05-22",
        )
        self.assertGreater(engine.get_player("s1mple").elo, 1500.0)
        self.assertLess(engine.get_player("karrigan").elo, 1500.0)
        self.assertEqual(engine.get_player("s1mple").wins, 1)

        # Test lineup feature generation
        features = engine.build_lineup_features(
            team1="NaVi",
            team2="FaZe",
            team1_players=["s1mple", "b1t", "electronic", "perfecto", "boombl4"],
            team2_players=["karrigan", "rain", "broky", "twistzz", "ropz"],
        )
        self.assertIn("player_lineup_elo_diff", features)
        self.assertGreater(features["player_lineup_elo_diff"], 0)

        # Test top players ranking
        top = engine.top_players(limit=5, min_matches=1)
        self.assertEqual(top[0]["player_name"], "s1mple")

class TestCLI(unittest.TestCase):
    def test_cli_parses_and_dispatches_run_and_build_features(self):
        from pipeline import cli

        with mock.patch.object(cli, "_run", return_value=0) as run:
            self.assertEqual(cli.main(["run", "--train-ratio", "0.6", "--output-dir", "out"]), 0)
        run.assert_called_once()
        run_args = run.call_args.args[0]
        self.assertEqual(run_args.train_ratio, 0.6)
        self.assertEqual(run_args.output_dir, "out")

        with mock.patch.object(cli, "_build_features", return_value=0) as build:
            self.assertEqual(cli.main(["build-features", "--results", "matches.csv", "--train-ratio", "0.7"]), 0)
        build.assert_called_once()
        build_args = build.call_args.args[0]
        self.assertEqual(build_args.results, "matches.csv")
        self.assertEqual(build_args.train_ratio, 0.7)


if __name__ == "__main__":
    unittest.main()
