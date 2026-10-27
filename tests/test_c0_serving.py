"""Unit tests for CS2 C0 serving module and inference predictor."""

import json
from pathlib import Path
import tempfile

import pytest
import torch

from src.features.player_sequence_builder import PlayerHistoryTracker
from src.models import CS2C0Model, CS2C0Predictor
from src.models.c0_architecture import CS2TokenConfig
from src.models.c0_serving import build_cli_parser, main
from src.models.c0_trainer import C0Trainer


@pytest.fixture
def sample_tracker() -> PlayerHistoryTracker:
    tracker = PlayerHistoryTracker(max_history=16)
    tracker.add_player_map_record("p1", "2026-01-01", "mirage", "CT", {"kills": 22}, 1)
    tracker.add_player_map_record("p1", "2026-01-02", "nuke", "T", {"kills": 18}, 0)
    tracker.add_player_map_record("q1", "2026-01-03", "inferno", "CT", {"kills": 15}, 1)
    return tracker


@pytest.fixture
def sample_model() -> CS2C0Model:
    model = CS2C0Model(token_config=CS2TokenConfig(hidden_dim=32))
    model.eval()
    return model


class TestCS2C0Predictor:
    def test_imports(self):
        from src.models.c0_serving import CS2C0Predictor as ServingPredictor
        from src.models import CS2C0Predictor as ModelsPredictor

        assert ServingPredictor is ModelsPredictor

    def test_predict_matchup_output_format(self, sample_tracker, sample_model):
        predictor = CS2C0Predictor(
            model=sample_model,
            tracker=sample_tracker,
        )

        team1 = ["p1", "p2", "p3", "p4", "p5"]
        team2 = ["q1", "q2", "q3", "q4", "q5"]

        res = predictor.predict_matchup(team1, team2, cutoff_date="2026-01-05")

        assert isinstance(res, dict)
        assert "prob_team1" in res
        assert "prob_team2" in res
        assert "team1_history_counts" in res
        assert "team2_history_counts" in res
        assert "cutoff_date" in res
        assert "antisymmetry_verified" in res
        assert "model_version" in res

        assert isinstance(res["prob_team1"], float)
        assert isinstance(res["prob_team2"], float)
        assert 0.0 <= res["prob_team1"] <= 1.0
        assert 0.0 <= res["prob_team2"] <= 1.0

        # Exact anti-symmetry verification
        assert res["antisymmetry_verified"] is True
        assert abs((res["prob_team1"] + res["prob_team2"]) - 1.0) <= 1e-5

        assert res["cutoff_date"] == "2026-01-05"
        assert res["team1_history_counts"] == [2, 0, 0, 0, 0]
        assert res["team2_history_counts"] == [1, 0, 0, 0, 0]

    def test_roster_validation_failures(self, sample_tracker, sample_model):
        predictor = CS2C0Predictor(model=sample_model, tracker=sample_tracker)

        # Team 1 has only 4 players
        with pytest.raises(ValueError, match="Team 1 must have exactly 5 players"):
            predictor.predict_matchup(["p1", "p2", "p3", "p4"], ["q1", "q2", "q3", "q4", "q5"])

        # Team 2 has 6 players
        with pytest.raises(ValueError, match="Team 2 must have exactly 5 players"):
            predictor.predict_matchup(["p1", "p2", "p3", "p4", "p5"], ["q1", "q2", "q3", "q4", "q5", "q6"])

        # Team 1 has duplicate players
        with pytest.raises(ValueError, match="Team 1 must contain 5 distinct players"):
            predictor.predict_matchup(["p1", "p1", "p3", "p4", "p5"], ["q1", "q2", "q3", "q4", "q5"])

        # Teams have overlapping players
        with pytest.raises(ValueError, match="rosters must be disjoint"):
            predictor.predict_matchup(["p1", "p2", "p3", "p4", "p5"], ["p1", "q2", "q3", "q4", "q5"])

    def test_checkpoint_and_bank_loading(self, sample_tracker, sample_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ckpt_path = tmp_path / "model.pt"
            bank_path = tmp_path / "bank.pt"

            trainer = C0Trainer(model=sample_model)
            trainer.save_checkpoint(ckpt_path, model_version="c0_test_v1")
            sample_tracker.save(bank_path)

            predictor = CS2C0Predictor(
                checkpoint_path=ckpt_path,
                sequence_bank_path=bank_path,
            )

            assert predictor.model_version == "c0_test_v1"
            assert predictor.tracker is not None
            assert predictor.model is not None

            team1 = ["p1", "p2", "p3", "p4", "p5"]
            team2 = ["q1", "q2", "q3", "q4", "q5"]
            res = predictor.predict_matchup(team1, team2)

            assert res["antisymmetry_verified"] is True
            assert res["model_version"] == "c0_test_v1"

    def test_missing_files_error_or_fallback(self):
        # Strict mode raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            CS2C0Predictor(
                checkpoint_path="nonexistent_checkpoint.pt",
                sequence_bank_path="nonexistent_bank.pt",
                allow_missing=False,
            )

        # Allow missing mode falls back gracefully
        predictor = CS2C0Predictor(
            checkpoint_path="nonexistent_checkpoint.pt",
            sequence_bank_path="nonexistent_bank.pt",
            allow_missing=True,
        )
        assert predictor.model is not None
        assert predictor.tracker is not None

        res = predictor.predict_matchup(
            ["p1", "p2", "p3", "p4", "p5"],
            ["q1", "q2", "q3", "q4", "q5"],
        )
        assert res["antisymmetry_verified"] is True


class TestServingCLI:
    def test_cli_parser(self):
        parser = build_cli_parser()
        args = parser.parse_args([
            "--team1", "a", "b", "c", "d", "e",
            "--team2", "f", "g", "h", "i", "j",
            "--cutoff-date", "2026-06-01",
        ])
        assert args.team1 == ["a", "b", "c", "d", "e"]
        assert args.team2 == ["f", "g", "h", "i", "j"]
        assert args.cutoff_date == "2026-06-01"

    def test_cli_main_execution(self, capsys):
        code = main([
            "--team1", "a", "b", "c", "d", "e",
            "--team2", "f", "g", "h", "i", "j",
        ])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "prob_team1" in data
        assert "prob_team2" in data
        assert data["antisymmetry_verified"] is True
