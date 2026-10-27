"""PyTorch Dataset, batch collator, and chronological data split generator for CS2 C0 training.

Implements:
1. CS2MatchDataset:
   - Stores list of 5v5 match dictionaries with team rosters, cutoff dates, and outcome labels.
   - Holds reference to PlayerHistoryTracker.
   - Extracts anti-symmetric feature tensors on demand using strict cutoff dates to preserve temporal causality.
2. collate_c0_batch:
   - Batch collator for PyTorch DataLoader.
   - Stacks history -> (B, 2, 5, 16, 18), mask -> (B, 2, 5, 16), label -> (B, 1).
3. create_chronological_splits:
   - Ingests and joins historical matches and player lineups.
   - Filters to complete 5v5 matches where both teams have 5 named players and a definitive winner.
   - Sorts strictly chronologically ascending by match timestamp.
   - Partitions into train, validation, and test CS2MatchDataset instances.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.player_sequence_builder import (
    PlayerHistoryTracker,
    build_match_input_tensors,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CS2MatchDataset",
    "collate_c0_batch",
    "create_chronological_splits",
    "extract_5v5_matches",
]


def _clean_str(val: Any) -> str:
    """Return stripped string or empty string if None/NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none", "<na>", "null") else s


def _normalize_match_date(date_val: Any) -> pd.Timestamp:
    """Parse date value to a timezone-naive pd.Timestamp."""
    if isinstance(date_val, pd.Timestamp):
        return date_val.tz_localize(None) if date_val.tzinfo is not None else date_val
    try:
        ts = pd.to_datetime(date_val)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts
    except Exception:
        return pd.Timestamp("1970-01-01")


def _parse_player_sequence(val: Any) -> list[str]:
    """Parse player sequence from list, JSON string, or delimiter-separated text."""
    if isinstance(val, (list, tuple)):
        return [_clean_str(p) for p in val if _clean_str(p)]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [_clean_str(p) for p in parsed if _clean_str(p)]
            except Exception:
                pass
        delimiter = ";" if ";" in s else ","
        return [_clean_str(p) for p in s.split(delimiter) if _clean_str(p)]
    return []


def _parse_label(val: Any) -> float | None:
    """Parse match outcome into binary float (1.0 for team1 win, 0.0 for team2 win)."""
    if isinstance(val, (bool, np.bool_)):
        return 1.0 if val else 0.0
    if isinstance(val, (int, float, np.integer, np.floating)):
        if np.isnan(val):
            return None
        return 1.0 if float(val) > 0.5 else 0.0
    if isinstance(val, torch.Tensor):
        return 1.0 if float(val.item()) > 0.5 else 0.0
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("1", "1.0", "team1", "left", "true", "t1", "team_1", "team_left"):
            return 1.0
        if s in ("0", "0.0", "team2", "right", "false", "t2", "team_2", "team_right"):
            return 0.0
    return None


def _determine_winner(row: Mapping[str, Any] | pd.Series, team1_name: Any, team2_name: Any) -> float | None:
    """Determine definitive winner label (1.0 for team1, 0.0 for team2, None if ambiguous/tie)."""
    # 1. Winner string/ID matching
    w = row.get("winner") if hasattr(row, "get") else getattr(row, "winner", None)
    if pd.notna(w):
        w_str = str(w).strip()
        t1_str = str(team1_name).strip()
        t2_str = str(team2_name).strip()
        if t1_str and w_str == t1_str:
            return 1.0
        if t2_str and w_str == t2_str:
            return 0.0
        parsed = _parse_label(w_str)
        if parsed is not None:
            return parsed

    # 2. Label or y column
    if hasattr(row, "__contains__"):
        if "y" in row and pd.notna(row["y"]):
            parsed = _parse_label(row["y"])
            if parsed is not None:
                return parsed
        if "label" in row and pd.notna(row["label"]):
            parsed = _parse_label(row["label"])
            if parsed is not None:
                return parsed

    # 3. Score columns comparison
    get_val = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
    s1 = get_val("score_left") or get_val("score1") or get_val("team1_score")
    s2 = get_val("score_right") or get_val("score2") or get_val("team2_score")
    try:
        if s1 is not None and s2 is not None and pd.notna(s1) and pd.notna(s2):
            f1, f2 = float(s1), float(s2)
            if f1 > f2:
                return 1.0
            if f2 > f1:
                return 0.0
    except (ValueError, TypeError):
        pass

    return None


def extract_5v5_matches(
    matches_input: str | Path | pd.DataFrame,
    player_stats_input: str | Path | pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Extract valid 5v5 matches where both teams have 5 named players and a definitive winner.

    Supports:
    1. Direct lineup columns (e.g. team1_p1_name... or team1_players).
    2. Two-table join between matches CSV (map/match metadata) and player_stats CSV (player rosters).

    Args:
        matches_input: Path to matches CSV or DataFrame.
        player_stats_input: Optional path to player stats CSV or DataFrame.

    Returns:
        List of match dicts containing team1_players, team2_players, date, and y.
    """
    if isinstance(matches_input, pd.DataFrame):
        matches_df = matches_input.copy()
    else:
        p = Path(matches_input)
        if not p.exists():
            raise FileNotFoundError(f"Matches CSV file not found: {matches_input}")
        matches_df = pd.read_csv(p)

    players_df: pd.DataFrame | None = None
    if player_stats_input is not None:
        if isinstance(player_stats_input, pd.DataFrame):
            players_df = player_stats_input.copy()
        else:
            pp = Path(player_stats_input)
            if pp.exists():
                players_df = pd.read_csv(pp)
            else:
                raise FileNotFoundError(f"Player stats CSV file not found: {player_stats_input}")

    extracted: list[dict[str, Any]] = []

    # Schema A: direct player list columns (e.g. team1_players, team2_players)
    has_player_list_cols = (
        ("team1_players" in matches_df.columns or "team1" in matches_df.columns)
        and ("team2_players" in matches_df.columns or "team2" in matches_df.columns)
    )
    if has_player_list_cols:
        t1_col = "team1_players" if "team1_players" in matches_df.columns else "team1"
        t2_col = "team2_players" if "team2_players" in matches_df.columns else "team2"
        sample_s = matches_df[t1_col].dropna()
        if not sample_s.empty:
            sample_list = _parse_player_sequence(sample_s.iloc[0])
            if len(sample_list) == 5:
                for _, row in matches_df.iterrows():
                    t1 = _parse_player_sequence(row.get(t1_col))
                    t2 = _parse_player_sequence(row.get(t2_col))
                    if len(t1) != 5 or len(t2) != 5:
                        continue
                    y = _determine_winner(row, "team1", "team2")
                    if y is None:
                        continue
                    dt = _normalize_match_date(row.get("date"))
                    extracted.append(
                        {
                            "team1_players": t1,
                            "team2_players": t2,
                            "date": dt,
                            "y": y,
                        }
                    )
                if extracted:
                    return extracted

    # Schema B: individual player slot columns (team1_p1_name..team1_p5_name, team1_player_1..team1_player_5)
    for prefix in [("team1_p", "team2_p"), ("team1_player_", "team2_player_")]:
        t1_cols = [f"{prefix[0]}{i}_name" for i in range(1, 6)]
        t2_cols = [f"{prefix[1]}{i}_name" for i in range(1, 6)]
        if not all(c in matches_df.columns for c in t1_cols + t2_cols):
            t1_cols = [f"{prefix[0]}{i}" for i in range(1, 6)]
            t2_cols = [f"{prefix[1]}{i}" for i in range(1, 6)]
        if all(c in matches_df.columns for c in t1_cols + t2_cols):
            for _, row in matches_df.iterrows():
                t1 = [_clean_str(row.get(c)) for c in t1_cols]
                t2 = [_clean_str(row.get(c)) for c in t2_cols]
                t1 = [p for p in t1 if p]
                t2 = [p for p in t2 if p]
                if len(t1) != 5 or len(t2) != 5:
                    continue
                y = _determine_winner(row, row.get("team1", "team1"), row.get("team2", "team2"))
                if y is None:
                    continue
                dt = _normalize_match_date(row.get("date"))
                extracted.append(
                    {
                        "team1_players": t1,
                        "team2_players": t2,
                        "date": dt,
                        "y": y,
                        "team1": row.get("team1", ""),
                        "team2": row.get("team2", ""),
                    }
                )
            if extracted:
                return extracted

    # Schema C: two-table join with player_stats
    if players_df is not None:
        pcol = (
            "player_name"
            if "player_name" in players_df.columns
            else "player_id"
            if "player_id" in players_df.columns
            else "player"
        )
        tcol = "team" if "team" in players_df.columns else "team_name"

        players_df[pcol] = players_df[pcol].astype(str).str.strip()
        valid_p = players_df[
            ~players_df[pcol].isin(["", "nan", "None", "<NA>", "null"])
        ].copy()

        # Check potential join keys between matches and player_stats
        candidate_keys = [
            ["event", "date", "map"],
            ["match_id"],
            ["match_url", "map"],
            ["date", "map"],
            ["event", "date"],
            ["date"],
        ]
        join_keys: list[str] = []
        for ck in candidate_keys:
            if all(k in matches_df.columns and k in valid_p.columns for k in ck):
                join_keys = ck
                break

        group_cols = join_keys + [tcol]
        rosters: dict[Any, list[str]] = {}
        for g_val, sub_df in valid_p.groupby(group_cols):
            uniq = [_clean_str(x) for x in sub_df[pcol].unique() if _clean_str(x)]
            rosters[g_val] = uniq

        for _, row in matches_df.iterrows():
            t1 = (
                row.get("team_left")
                if "team_left" in matches_df.columns
                else row.get("team1")
            )
            t2 = (
                row.get("team_right")
                if "team_right" in matches_df.columns
                else row.get("team2")
            )
            if not t1 or not t2:
                continue

            if join_keys:
                prefix = tuple(row[k] for k in join_keys)
                k1 = prefix + (t1,)
                k2 = prefix + (t2,)
            else:
                k1 = (t1,)
                k2 = (t2,)

            r1 = rosters.get(k1)
            r2 = rosters.get(k2)
            if r1 is None or len(r1) != 5 or r2 is None or len(r2) != 5:
                continue

            y = _determine_winner(row, t1, t2)
            if y is None:
                continue

            dt = _normalize_match_date(row.get("date"))
            extracted.append(
                {
                    "team1_players": list(r1),
                    "team2_players": list(r2),
                    "date": dt,
                    "y": y,
                    "team1": str(t1),
                    "team2": str(t2),
                }
            )

    return extracted


class CS2MatchDataset(Dataset):
    """PyTorch Dataset for CS2 5v5 matches.

    Each item yields anti-symmetric input tensors constructed using historical
    player sequences strictly prior to the match timestamp.
    """

    def __init__(
        self,
        matches: Sequence[Mapping[str, Any]],
        tracker: PlayerHistoryTracker,
        max_len: int = 16,
    ) -> None:
        """Initialize dataset.

        Args:
            matches: Sequence of match dicts with 'team1_players', 'team2_players',
                'date', and 'y' (0 or 1).
            tracker: PlayerHistoryTracker instance containing historical records.
            max_len: Number of sequence steps per player (default 16).
        """
        self.matches = list(matches)
        self.tracker = tracker
        self.max_len = max_len

    def __len__(self) -> int:
        """Return the number of matches in dataset."""
        return len(self.matches)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Fetch item at index.

        Calls build_match_input_tensors with strict cutoff_date to prevent
        lookahead leakage, squeezes batch dimension, and returns cleanly shaped tensors.

        Returns:
            dict with:
                - "history": float32 tensor of shape (2, 5, 16, feature_dim)
                - "mask": bool tensor of shape (2, 5, 16)
                - "label": float32 tensor of shape (1,)
        """
        match = self.matches[idx]
        t1 = match.get("team1_players", ())
        t2 = match.get("team2_players", ())
        cutoff_date = match.get("date", match.get("cutoff_date", None))

        tensors = build_match_input_tensors(
            self.tracker,
            t1,
            t2,
            cutoff_date=cutoff_date,
            max_len=self.max_len,
        )

        history = tensors["history"].squeeze(0)  # (2, 5, max_len, feature_dim)
        mask = tensors["mask"].squeeze(0)        # (2, 5, max_len)

        # Defense-in-depth against NaNs
        torch.nan_to_num_(history, nan=0.0, posinf=0.0, neginf=0.0)

        raw_label = match.get("y", match.get("label", 0.0))
        label_val = _parse_label(raw_label)
        if label_val is None:
            label_val = 0.0
        label = torch.tensor([label_val], dtype=torch.float32)

        return {
            "history": history,
            "mask": mask,
            "label": label,
        }


def collate_c0_batch(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate individual match dictionary samples into a batched dictionary.

    Args:
        batch: Sequence of dicts from CS2MatchDataset.__getitem__.

    Returns:
        dict with:
            - "history": shape (B, 2, 5, 16, 18) float32 tensor
            - "mask": shape (B, 2, 5, 16) bool tensor
            - "label": shape (B, 1) float32 tensor
    """
    if not batch:
        return {
            "history": torch.empty((0, 2, 5, 16, 18), dtype=torch.float32),
            "mask": torch.empty((0, 2, 5, 16), dtype=torch.bool),
            "label": torch.empty((0, 1), dtype=torch.float32),
        }

    history = torch.stack([item["history"] for item in batch], dim=0)
    mask = torch.stack([item["mask"] for item in batch], dim=0)

    # Ensure each label is shaped (1,) before stacking to (B, 1)
    labels = [
        item["label"].view(1) if item["label"].dim() == 0 or item["label"].shape != (1,)
        else item["label"]
        for item in batch
    ]
    label = torch.stack(labels, dim=0)

    out: dict[str, torch.Tensor] = {
        "history": history,
        "mask": mask,
        "label": label,
    }

    if "player_counts" in batch[0]:
        out["player_counts"] = torch.stack([item["player_counts"] for item in batch], dim=0)

    return out


def create_chronological_splits(
    tracker: PlayerHistoryTracker,
    matches_csv_path: str | Path | pd.DataFrame,
    player_stats_csv_path: str | Path | pd.DataFrame | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    max_len: int = 16,
    populate_tracker: bool = False,
) -> tuple[CS2MatchDataset, CS2MatchDataset, CS2MatchDataset]:
    """Create strictly chronological train, validation, and test datasets.

    Extracts 5v5 matches where both teams have 5 named players and a definitive winner,
    sorts strictly chronologically ascending by match date, and splits into train,
    validation, and test CS2MatchDataset instances.

    Args:
        tracker: PlayerHistoryTracker instance.
        matches_csv_path: Path to matches CSV or in-memory DataFrame.
        player_stats_csv_path: Optional path to player stats CSV or in-memory DataFrame.
        train_ratio: Proportion of matches allocated to train (default 0.70).
        val_ratio: Proportion of matches allocated to validation (default 0.15).
        max_len: Number of sequence steps per player (default 16).
        populate_tracker: If True and tracker is empty, ingests player records from
            player_stats_csv_path.

    Returns:
        tuple of (train_dataset, val_dataset, test_dataset)
    """
    if train_ratio < 0.0 or val_ratio < 0.0 or (train_ratio + val_ratio) > 1.0:
        raise ValueError(
            f"train_ratio ({train_ratio}) and val_ratio ({val_ratio}) must be non-negative and sum to <= 1.0"
        )

    # Optional tracker population if tracker is empty
    if populate_tracker and len(tracker) == 0 and player_stats_csv_path is not None:
        if isinstance(player_stats_csv_path, pd.DataFrame):
            stats_df = player_stats_csv_path
        else:
            p_path = Path(player_stats_csv_path)
            stats_df = pd.read_csv(p_path) if p_path.exists() else None
        if stats_df is not None and not stats_df.empty:
            tracker.ingest_records_from_dataframe(stats_df)

    # Extract 5v5 matches with definitive winners
    matches = extract_5v5_matches(matches_csv_path, player_stats_csv_path)

    # Strictly sort chronologically ascending by date
    matches.sort(key=lambda m: _normalize_match_date(m["date"]))

    total_matches = len(matches)
    if total_matches == 0:
        logger.warning("No valid 5v5 matches extracted from %s", matches_csv_path)
        return (
            CS2MatchDataset([], tracker=tracker, max_len=max_len),
            CS2MatchDataset([], tracker=tracker, max_len=max_len),
            CS2MatchDataset([], tracker=tracker, max_len=max_len),
        )

    n_train = max(0, min(total_matches, int(round(total_matches * train_ratio))))
    n_val = max(0, min(total_matches - n_train, int(round(total_matches * val_ratio))))

    train_matches = matches[:n_train]
    val_matches = matches[n_train : n_train + n_val]
    test_matches = matches[n_train + n_val :]

    logger.info(
        "Created chronological splits: %d train, %d val, %d test (total %d)",
        len(train_matches),
        len(val_matches),
        len(test_matches),
        total_matches,
    )

    return (
        CS2MatchDataset(train_matches, tracker=tracker, max_len=max_len),
        CS2MatchDataset(val_matches, tracker=tracker, max_len=max_len),
        CS2MatchDataset(test_matches, tracker=tracker, max_len=max_len),
    )
