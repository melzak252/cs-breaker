#!/usr/bin/env python3
"""Counter-Strike 2 C0 neural model training CLI.

Orchestrates:
1. Loading the precomputed PlayerHistoryTracker sequence bank.
2. Ingesting historical match and player performance records.
3. Chronological train/validation/test splitting preserving strict temporal causality.
4. PyTorch DataLoader instantiation with anti-symmetric batch collation.
5. CS2C0Model instantiation with CS2TokenConfig player embeddings.
6. C0Trainer fit with AdamW, Cosine Annealing, early stopping, and anti-symmetric loss.
7. Out-of-sample evaluation on held-out test matches (Log Loss, Brier Score, Accuracy).
8. Best model checkpoint and reporting metrics persistence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.features.c0_dataset import (
    CS2MatchDataset,
    collate_c0_batch,
    create_chronological_splits,
)
from src.features.player_sequence_builder import PlayerHistoryTracker
from src.models.c0_architecture import CS2C0Model
from src.models.c0_tokens import CS2TokenConfig
from src.models.c0_trainer import C0Trainer

logger = logging.getLogger("train_c0_model")


def set_seed(seed: int) -> None:
    """Set global random seeds for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CS2 C0 anti-symmetric neural match prediction model."
    )
    parser.add_argument(
        "--sequence-bank",
        type=str,
        default="data/04_feature/player_sequence_bank.pt",
        help="Path to serialized player sequence bank (.pt).",
    )
    parser.add_argument(
        "--matches-csv",
        type=str,
        default="data/raw/strandedpond/matches.csv",
        help="Path to matches CSV dataset.",
    )
    parser.add_argument(
        "--player-stats-csv",
        type=str,
        default="data/raw/strandedpond/player_stats.csv",
        help="Path to player stats CSV dataset.",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=str,
        default="data/06_models/c0_checkpoint_best.pt",
        help="Destination path for best model checkpoint (.pt).",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="data/08_reporting/c0_training_metrics.json",
        help="Destination path for training and evaluation metrics JSON.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Maximum training epochs (default: 15).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size for DataLoader (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate for AdamW optimizer (default: 0.001).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="Hidden representation dimension for CS2TokenConfig and model (default: 32).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience in epochs on validation loss (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cuda', 'cpu', or None for auto-detect).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of chronological matches allocated to train (default: 0.70).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of chronological matches allocated to validation (default: 0.15).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker subprocesses (default: 0).",
    )

    return parser.parse_args(argv)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Execute complete C0 model training and evaluation workflow."""
    # 1. Deterministic seeding
    set_seed(args.seed)
    logger.info("Deterministic seed set to %d", args.seed)

    # 2. Sequence bank loading
    seq_bank_path = Path(args.sequence_bank)
    if not seq_bank_path.exists():
        raise FileNotFoundError(
            f"Required player sequence bank not found at: {seq_bank_path}. "
            "Please run 'scripts/build_player_sequence_bank.py' first."
        )

    logger.info("Loading player sequence bank from %s...", seq_bank_path)
    tracker = PlayerHistoryTracker.load(seq_bank_path)
    logger.info("Loaded PlayerHistoryTracker with %d distinct player histories.", len(tracker))

    # 3. Chronological dataset splits
    matches_path = Path(args.matches_csv)
    if not matches_path.exists():
        raise FileNotFoundError(f"Matches CSV file not found at: {matches_path}")

    stats_path: Path | None = Path(args.player_stats_csv)
    if stats_path is not None and not stats_path.exists():
        logger.warning(
            "Player stats CSV not found at %s; will attempt single-table schema match parsing.",
            stats_path,
        )
        stats_path = None

    logger.info(
        "Partitioning chronological matches from %s (stats: %s, train_ratio=%.2f, val_ratio=%.2f)...",
        matches_path,
        stats_path,
        args.train_ratio,
        args.val_ratio,
    )
    train_ds, val_ds, test_ds = create_chronological_splits(
        tracker=tracker,
        matches_csv_path=matches_path,
        player_stats_csv_path=stats_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    logger.info(
        "Chronological splits ready: %d train, %d val, %d test matches (total %d).",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        len(train_ds) + len(val_ds) + len(test_ds),
    )

    if len(train_ds) == 0:
        raise ValueError(
            f"Extracted 0 valid 5v5 training matches from {matches_path}. Cannot proceed with training."
        )

    # 4. DataLoader creation
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_c0_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_c0_batch,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_c0_batch,
        num_workers=args.num_workers,
    )

    # 5. Model instantiation
    token_config = CS2TokenConfig(hidden_dim=args.hidden_dim)
    model = CS2C0Model(token_config)
    logger.info(
        "Initialized CS2C0Model with token hidden_dim=%d, total parameters=%d.",
        args.hidden_dim,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # 6. Trainer instantiation
    checkpoint_out_path = Path(args.checkpoint_out)
    checkpoint_out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_out_path = Path(args.metrics_out)
    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)

    trainer = C0Trainer(
        model=model,
        lr=args.lr,
        device=args.device,
        t_max=args.epochs,
    )
    logger.info("Initialized C0Trainer on device: %s", trainer.device)

    # 7. Fit model with validation tracking and early stopping
    logger.info(
        "Starting training fit for up to %d epochs (patience=%d, lr=%.6f)...",
        args.epochs,
        args.patience,
        args.lr,
    )
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader if len(val_ds) > 0 else None,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=checkpoint_out_path,
        verbose=True,
    )

    # Ensure checkpoint file exists on disk even if 0 epochs or edge cases occurred
    if not checkpoint_out_path.exists():
        trainer.save_checkpoint(
            checkpoint_out_path,
            epoch=len(history.get("train_loss", [])),
            val_loss=min(history["val_loss"]) if history.get("val_loss") else None,
        )

    # 8. Evaluate best checkpoint on test set
    if checkpoint_out_path.exists():
        logger.info("Loading best saved checkpoint from %s for test evaluation...", checkpoint_out_path)
        trainer.load_checkpoint(checkpoint_out_path)

    test_metrics: dict[str, float] = {}
    if len(test_ds) > 0:
        logger.info("Evaluating best model on held-out test split (%d matches)...", len(test_ds))
        eval_result = trainer.evaluate(test_loader)
        test_metrics = {
            "test_log_loss": float(eval_result["loss"]),
            "test_brier_score": float(eval_result["brier"]),
            "test_accuracy": float(eval_result["accuracy"]),
        }
    else:
        logger.warning("Test dataset is empty; test metrics set to 0.0.")
        test_metrics = {
            "test_log_loss": 0.0,
            "test_brier_score": 0.0,
            "test_accuracy": 0.0,
        }

    # 9. Save metrics and configuration metadata
    best_val_loss = min(history["val_loss"]) if history.get("val_loss") else None
    metrics_payload: dict[str, Any] = {
        "test_log_loss": test_metrics["test_log_loss"],
        "test_brier_score": test_metrics["test_brier_score"],
        "test_accuracy": test_metrics["test_accuracy"],
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history.get("train_loss", [])),
        "test_metrics": test_metrics,
        "training_history": history,
        "dataset_splits": {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "total_samples": len(train_ds) + len(val_ds) + len(test_ds),
        },
        "config": {
            "sequence_bank": str(args.sequence_bank),
            "matches_csv": str(args.matches_csv),
            "player_stats_csv": str(args.player_stats_csv),
            "checkpoint_out": str(args.checkpoint_out),
            "metrics_out": str(args.metrics_out),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "patience": args.patience,
            "seed": args.seed,
            "device": str(trainer.device),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("Saved training and test evaluation metrics to %s", metrics_out_path)

    # 10. Execution Summary Output
    val_loss_str = f"{best_val_loss:.4f}" if best_val_loss is not None else "N/A"
    summary_lines = [
        "=" * 60,
        "                CS2 C0 MODEL TRAINING SUMMARY",
        "=" * 60,
        f"  Device              : {trainer.device}",
        f"  Random Seed         : {args.seed}",
        f"  Model Hidden Dim    : {args.hidden_dim}",
        f"  Dataset Splits      : {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test",
        f"  Epochs Completed    : {len(history.get('train_loss', []))} / {args.epochs}",
        f"  Best Validation Loss: {val_loss_str}",
        "-" * 60,
        "  Out-of-Sample Test Evaluation:",
        f"    Log Loss (BCE)    : {test_metrics['test_log_loss']:.4f}",
        f"    Brier Score       : {test_metrics['test_brier_score']:.4f}",
        f"    Accuracy          : {test_metrics['test_accuracy'] * 100:.2f}%",
        "-" * 60,
        f"  Best Checkpoint     : {checkpoint_out_path}",
        f"  Metrics Report      : {metrics_out_path}",
        "=" * 60,
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text + "\n")

    return metrics_payload


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
