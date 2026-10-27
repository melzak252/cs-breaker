"""CS2 C0 matchup win probability predictor serving and inference module.

Provides:
- CS2C0Predictor: Serving interface that loads a trained CS2C0Model checkpoint
  and PlayerHistoryTracker sequence bank to forecast 5v5 CS2 matchups.
  Mathematically verifies anti-symmetry P(A > B) + P(B > A) == 1.0.
- CLI support:
    python -m src.models.c0_serving --team1 p1 p2 p3 p4 p5 --team2 q1 q2 q3 q4 q5
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import logging
from pathlib import Path
import sys
from typing import Any

import torch

from src.models.c0_architecture import CS2C0Model
from src.models.c0_tokens import CS2TokenConfig

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = "data/06_models/c0_checkpoint_best.pt"
DEFAULT_SEQUENCE_BANK_PATH = "data/04_feature/player_sequence_bank.pt"

__all__ = [
    "CS2C0Predictor",
    "build_cli_parser",
    "main",
]


class CS2C0Predictor:
    """CS2 C0 matchup win probability predictor and serving interface.

    Loads trained model weights into CS2C0Model, sets evaluation mode, and
    queries PlayerHistoryTracker for temporal player sequence features.
    Guarantees pairwise anti-symmetry: P(A > B) + P(B > A) == 1.0.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        sequence_bank_path: str | Path = DEFAULT_SEQUENCE_BANK_PATH,
        device: str | torch.device | None = None,
        model: CS2C0Model | None = None,
        tracker: PlayerHistoryTracker | None = None,
        allow_missing: bool = False,
    ) -> None:
        """Initialize predictor.

        Args:
            checkpoint_path: Path to serialized model checkpoint (.pt).
            sequence_bank_path: Path to serialized PlayerHistoryTracker bank (.pt).
            device: Optional PyTorch device or device string ('cpu', 'cuda').
            model: Optional pre-instantiated CS2C0Model instance.
            tracker: Optional pre-instantiated PlayerHistoryTracker instance.
            allow_missing: If True, falls back to an untrained model and empty tracker
                if files do not exist on disk instead of raising FileNotFoundError.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.checkpoint_path = Path(checkpoint_path)
        self.sequence_bank_path = Path(sequence_bank_path)
        self.model_version = "c0_v1"

        from src.features.player_sequence_builder import PlayerHistoryTracker
        # 1. Initialize or load PlayerHistoryTracker
        if tracker is not None:
            self.tracker = tracker
        elif self.sequence_bank_path.is_file():
            logger.info("Loading player sequence bank from %s...", self.sequence_bank_path)
            self.tracker = PlayerHistoryTracker.load(self.sequence_bank_path)
        elif allow_missing:
            logger.warning(
                "Sequence bank file not found at %s. Initializing empty PlayerHistoryTracker.",
                self.sequence_bank_path,
            )
            self.tracker = PlayerHistoryTracker(max_history=16)
        else:
            raise FileNotFoundError(f"Sequence bank file not found: {self.sequence_bank_path}")

        # 2. Initialize or load CS2C0Model
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
            if hasattr(model, "model_version"):
                self.model_version = str(model.model_version)
        elif self.checkpoint_path.is_file():
            logger.info("Loading C0 model checkpoint from %s...", self.checkpoint_path)
            self._load_checkpoint(self.checkpoint_path)
        elif allow_missing:
            logger.warning(
                "Checkpoint file not found at %s. Initializing default CS2C0Model.",
                self.checkpoint_path,
            )
            self.model = CS2C0Model().to(self.device)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

    def _load_checkpoint(self, path: Path) -> None:
        """Load model state and metadata from checkpoint path."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        state_dict: dict[str, Any]
        metadata: dict[str, Any] = {}

        if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
            state_dict = dict(checkpoint["model_state_dict"])
            metadata = dict(checkpoint.get("metadata", {}))
            if "model_version" in checkpoint:
                self.model_version = str(checkpoint["model_version"])
            elif "model_version" in metadata:
                self.model_version = str(metadata["model_version"])
            elif "version" in metadata:
                self.model_version = str(metadata["version"])
        elif isinstance(checkpoint, Mapping):
            state_dict = dict(checkpoint)
            if "model_version" in state_dict:
                self.model_version = str(state_dict.pop("model_version"))
        elif isinstance(checkpoint, CS2C0Model):
            self.model = checkpoint.to(self.device)
            self.model.eval()
            if hasattr(checkpoint, "model_version"):
                self.model_version = str(checkpoint.model_version)
            return
        else:
            raise ValueError(f"Unrecognized checkpoint object type: {type(checkpoint)}")

        # Infer model hidden dimensions from saved tensors or metadata
        hidden_dim = 32
        head_hidden_dim = 64

        if "history_attention.attn_score.0.weight" in state_dict:
            hidden_dim = state_dict["history_attention.attn_score.0.weight"].shape[0]
        elif "config" in metadata and isinstance(metadata["config"], Mapping):
            hidden_dim = int(metadata["config"].get("hidden_dim", 32))

        if "head.0.weight" in state_dict:
            head_hidden_dim = state_dict["head.0.weight"].shape[0]

        token_config = CS2TokenConfig(hidden_dim=hidden_dim)
        model = CS2C0Model(token_config=token_config, head_hidden_dim=head_hidden_dim)
        model.load_state_dict(state_dict)
        self.model = model.to(self.device)
        self.model.eval()

    @staticmethod
    def _validate_rosters(
        team1_players: Sequence[str],
        team2_players: Sequence[str],
    ) -> tuple[list[str], list[str]]:
        """Validate that both teams have exactly 5 distinct players and disjoint rosters."""
        if not isinstance(team1_players, (list, tuple, Sequence)) or isinstance(team1_players, (str, bytes)):
            raise ValueError(f"team1_players must be a sequence of player IDs, got {type(team1_players).__name__}")
        if not isinstance(team2_players, (list, tuple, Sequence)) or isinstance(team2_players, (str, bytes)):
            raise ValueError(f"team2_players must be a sequence of player IDs, got {type(team2_players).__name__}")

        t1 = [str(p).strip() for p in team1_players]
        t2 = [str(p).strip() for p in team2_players]

        if any(not p for p in t1):
            raise ValueError("team1_players cannot contain empty player IDs")
        if any(not p for p in t2):
            raise ValueError("team2_players cannot contain empty player IDs")

        if len(t1) != 5:
            raise ValueError(f"Team 1 must have exactly 5 players, got {len(t1)}")
        if len(t2) != 5:
            raise ValueError(f"Team 2 must have exactly 5 players, got {len(t2)}")

        if len(set(t1)) != 5:
            raise ValueError(f"Team 1 must contain 5 distinct players, got duplicates: {t1}")
        if len(set(t2)) != 5:
            raise ValueError(f"Team 2 must contain 5 distinct players, got duplicates: {t2}")

        overlap = set(t1) & set(t2)
        if overlap:
            raise ValueError(f"Team 1 and Team 2 rosters must be disjoint; overlapping players: {sorted(overlap)}")

        return t1, t2

    def predict_matchup(
        self,
        team1_players: Sequence[str],
        team2_players: Sequence[str],
        cutoff_date: Any = None,
    ) -> dict[str, Any]:
        """Predict win probability for team1 vs team2 with anti-symmetry verification.

        Args:
            team1_players: Sequence of 5 distinct player IDs for Team 1.
            team2_players: Sequence of 5 distinct player IDs for Team 2.
            cutoff_date: Optional temporal cutoff timestamp (date string or datetime).
                Strictly records with date < cutoff_date are extracted from player history.

        Returns:
            Dictionary containing:
                - "prob_team1": float, win probability of team 1 over team 2
                - "prob_team2": float, win probability of team 2 over team 1
                - "team1_history_counts": list[int], prior maps per player for team 1
                - "team2_history_counts": list[int], prior maps per player for team 2
                - "cutoff_date": str or None, formatted cutoff date
                - "antisymmetry_verified": bool, True if prob_team1 + prob_team2 == 1.0 within 1e-5
                - "model_version": str, model identifier
        """
        from src.features.player_sequence_builder import build_match_input_tensors
        t1, t2 = self._validate_rosters(team1_players, team2_players)

        # 1. Forward matchup: Team 1 vs Team 2
        fwd_tensors = build_match_input_tensors(
            tracker=self.tracker,
            team1_players=t1,
            team2_players=t2,
            cutoff_date=cutoff_date,
            device=self.device,
        )

        with torch.no_grad():
            prob_t1_tensor = self.model(
                history=fwd_tensors["history"],
                mask=fwd_tensors["mask"],
            )
            prob_team1 = float(prob_t1_tensor.squeeze().item())

        # 2. Reversed matchup: Team 2 vs Team 1 to mathematically verify anti-symmetry
        rev_tensors = build_match_input_tensors(
            tracker=self.tracker,
            team1_players=t2,
            team2_players=t1,
            cutoff_date=cutoff_date,
            device=self.device,
        )

        with torch.no_grad():
            prob_t2_tensor = self.model(
                history=rev_tensors["history"],
                mask=rev_tensors["mask"],
            )
            prob_team2 = float(prob_t2_tensor.squeeze().item())

        # 3. Mathematical anti-symmetry verification: P(A > B) + P(B > A) == 1.0 within 1e-5
        prob_sum = prob_team1 + prob_team2
        antisymmetry_verified = bool(abs(prob_sum - 1.0) <= 1e-5)

        # 4. Extract prior map counts per player
        team1_history_counts = [int(c) for c in fwd_tensors["player_counts"][0, 0].cpu().tolist()]
        team2_history_counts = [int(c) for c in fwd_tensors["player_counts"][0, 1].cpu().tolist()]

        # 5. Format cutoff date
        cutoff_date_str: str | None
        if cutoff_date is None:
            cutoff_date_str = None
        elif hasattr(cutoff_date, "isoformat"):
            cutoff_date_str = cutoff_date.isoformat()
        else:
            cutoff_date_str = str(cutoff_date)

        return {
            "prob_team1": prob_team1,
            "prob_team2": prob_team2,
            "team1_history_counts": team1_history_counts,
            "team2_history_counts": team2_history_counts,
            "cutoff_date": cutoff_date_str,
            "antisymmetry_verified": antisymmetry_verified,
            "model_version": str(self.model_version),
        }

    def predict_proba(
        self,
        team1_players: Sequence[str],
        team2_players: Sequence[str],
        cutoff_date: Any = None,
    ) -> float:
        """Predict win probability for team 1 against team 2."""
        return self.predict_matchup(team1_players, team2_players, cutoff_date=cutoff_date)["prob_team1"]

    def __repr__(self) -> str:
        return (
            f"CS2C0Predictor(model_version={self.model_version!r}, "
            f"device={self.device!r})"
        )


def build_cli_parser() -> argparse.ArgumentParser:
    """Build argument parser for C0 model serving CLI."""
    parser = argparse.ArgumentParser(
        description="Serve CS2 C0 anti-symmetric matchup win predictions.",
    )
    parser.add_argument(
        "--team1",
        nargs="+",
        required=True,
        help="Five player IDs for Team 1 (space-separated).",
    )
    parser.add_argument(
        "--team2",
        nargs="+",
        required=True,
        help="Five player IDs for Team 2 (space-separated).",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=None,
        help="Optional cutoff date/timestamp (YYYY-MM-DD or ISO timestamp).",
    )
    parser.add_argument(
        "--checkpoint",
        "--checkpoint-path",
        dest="checkpoint_path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to trained C0 model checkpoint (.pt).",
    )
    parser.add_argument(
        "--sequence-bank",
        "--sequence-bank-path",
        dest="sequence_bank_path",
        type=str,
        default=DEFAULT_SEQUENCE_BANK_PATH,
        help="Path to serialized PlayerHistoryTracker sequence bank (.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device ('cpu', 'cuda', etc.). Defaults to auto-detect.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for CS2 C0 matchup predictions."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    try:
        ckpt_path = Path(args.checkpoint_path)
        seq_path = Path(args.sequence_bank_path)
        fallback = not (ckpt_path.exists() and seq_path.exists())

        predictor = CS2C0Predictor(
            checkpoint_path=ckpt_path,
            sequence_bank_path=seq_path,
            device=args.device,
            allow_missing=fallback,
        )

        forecast = predictor.predict_matchup(
            team1_players=args.team1,
            team2_players=args.team2,
            cutoff_date=args.cutoff_date,
        )
        print(json.dumps(forecast, indent=2))
        return 0
    except Exception as exc:
        logger.error("Forecast failed: %s", exc)
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
