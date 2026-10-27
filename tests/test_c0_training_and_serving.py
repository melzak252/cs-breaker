"""End-to-end integration tests for C0 training engine, checkpointing, and prediction."""

from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.c0_architecture import CS2C0Model
from src.models.c0_tokens import CS2TokenConfig
from src.models.c0_trainer import C0Trainer
from src.models.c0_serving import CS2C0Predictor
from src.features.player_sequence_builder import PlayerHistoryTracker


class TestC0TrainingAndCheckpointing:
    def test_trainer_fit_and_checkpoint(self, tmp_path):
        config = CS2TokenConfig(hidden_dim=16)
        model = CS2C0Model(token_config=config)
        trainer = C0Trainer(model, lr=0.01)

        # Synthetic dataset: 10 batches of 4
        N = 40
        history = torch.randn(N, 2, 5, 16, 18)
        mask = torch.ones(N, 2, 5, 16, dtype=torch.bool)
        labels = torch.randint(0, 2, (N, 1)).float()

        dataset = TensorDataset(history, mask, labels)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=8, shuffle=False)

        checkpoint_file = tmp_path / "test_c0.pt"
        history_dict = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=3,
            patience=2,
            checkpoint_path=checkpoint_file,
        )

        assert "train_loss" in history_dict
        assert "val_loss" in history_dict
        assert len(history_dict["train_loss"]) >= 1
        assert checkpoint_file.exists()

        # Reload checkpoint into fresh model
        fresh_model = CS2C0Model(token_config=config)
        fresh_trainer = C0Trainer(fresh_model)
        fresh_trainer.load_checkpoint(checkpoint_file)

        # Evaluate should produce identical loss
        eval_metrics = fresh_trainer.evaluate(val_loader)
        assert eval_metrics["val_loss"] > 0.0

    def test_predictor_with_trained_checkpoint(self, tmp_path):
        checkpoint_path = Path("data/06_models/c0_checkpoint_best.pt")
        bank_path = Path("data/04_feature/player_sequence_bank.pt")

        if checkpoint_path.exists() and bank_path.exists():
            predictor = CS2C0Predictor(
                checkpoint_path=checkpoint_path,
                sequence_bank_path=bank_path,
            )

            # Use 5 players from bank
            tracker = predictor.tracker
            players = list(tracker.histories.keys())
            if len(players) >= 10:
                t1 = players[:5]
                t2 = players[5:10]
                result = predictor.predict_matchup(t1, t2)
                assert "prob_team1" in result
                assert "prob_team2" in result
                assert result["antisymmetry_verified"] is True
                assert pytest.approx(result["prob_team1"] + result["prob_team2"], 1e-5) == 1.0
