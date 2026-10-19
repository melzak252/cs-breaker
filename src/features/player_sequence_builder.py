"""Multi-year player sequence tensor and history extractor for CS2 C0 models.

Provides:
- PlayerHistoryTracker: rolling per-player history deque of past map records
  strictly enforcing temporal causality (record.date < cutoff_date).
- build_match_input_tensors: builds (1, 2, 5, 16, feature_dim) PyTorch tensors
  for match inputs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.models.c0_tokens import (
    CS2TokenConfig,
    encode_player_map_token,
    get_map_id,
    get_side_id,
)

__all__ = [
    "PlayerMapRecord",
    "PlayerHistoryTracker",
    "build_match_input_tensors",
    "build_batch_match_input_tensors",
]


def _normalize_timestamp(date_val: Any) -> pd.Timestamp:
    """Normalize arbitrary date inputs to a timezone-naive pd.Timestamp."""
    if date_val is None:
        raise ValueError("date cannot be None")
    if isinstance(date_val, pd.Timestamp):
        ts = date_val
    else:
        ts = pd.to_datetime(date_val)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def _normalize_player_id(player_id: Any) -> str:
    """Normalize player ID to a clean string identifier."""
    if player_id is None:
        return ""
    return str(player_id).strip()


def _parse_won_flag(won: Any) -> float:
    """Parse won value into 1.0 (win) or 0.0 (loss)."""
    if won is None:
        return 0.0
    if isinstance(won, (bool, np.bool_)):
        return 1.0 if won else 0.0
    if isinstance(won, (int, float, np.integer, np.floating)):
        if math.isnan(won):
            return 0.0
        return 1.0 if won > 0 else 0.0
    if isinstance(won, str):
        cleaned = won.strip().lower()
        if cleaned in ("1", "true", "t", "won", "win", "w", "yes"):
            return 1.0
        return 0.0
    return 0.0


@dataclass(slots=True)
class PlayerMapRecord:
    """Single competitive map performance record for a player."""

    player_id: str
    date: pd.Timestamp
    map_name: str
    side: str
    won: float
    stats_dict: dict[str, Any] = field(default_factory=dict)
    features: np.ndarray = field(default_factory=lambda: np.zeros(18, dtype=np.float32))


class _HistoryStore(dict[str, deque[PlayerMapRecord]]):
    """Internal dictionary supporting string normalization and automatic deque creation."""

    def __init__(self, max_history: int = 16) -> None:
        super().__init__()
        self.max_history = max_history

    def __getitem__(self, key: Any) -> deque[PlayerMapRecord]:
        norm_key = _normalize_player_id(key)
        if norm_key not in self:
            d: deque[PlayerMapRecord] = deque(maxlen=self.max_history)
            super().__setitem__(norm_key, d)
            return d
        return super().__getitem__(norm_key)

    def __contains__(self, key: Any) -> bool:
        return super().__contains__(_normalize_player_id(key))

    def get(self, key: Any, default: Any = None) -> Any:  # type: ignore[override]
        norm_key = _normalize_player_id(key)
        if norm_key in self:
            return super().__getitem__(norm_key)
        return default


class PlayerHistoryTracker:
    """Maintains rolling per-player history deque of past map records.

    Ingests chronological match events and extracts feature sequences with strict
    temporal filtering (record.date < cutoff_date) to prevent lookahead leakage.
    """

    def __init__(
        self,
        max_history: int = 16,
        feature_dim: int = 18,
        include_map_side: bool = True,
        padding: str = "right",
    ) -> None:
        """Initialize the history tracker.

        Args:
            max_history: Maximum number of map records maintained in each player's rolling deque.
            feature_dim: Dimension of the encoded feature vector per map token (default 18).
            include_map_side: If True, features prepend [map_id, side_id] (dim 18).
                If False, features only contain 16 continuous stats.
            padding: Padding alignment; "right" pads at the end (left-aligned data),
                "left" pads at the beginning (right-aligned data).
        """
        self.max_history = max_history
        self.feature_dim = feature_dim
        self.include_map_side = include_map_side
        self.padding = padding
        self.histories: dict[str, deque[PlayerMapRecord]] = _HistoryStore(max_history=max_history)

    def __len__(self) -> int:
        """Return the number of distinct tracked players."""
        return len(self.histories)
    def add_player_map_record(
        self,
        player_id: Any,
        date: Any,
        map_name: str,
        side: str,
        stats_dict: Mapping[str, Any] | Sequence[float] | np.ndarray | None = None,
        won: Any = None,
    ) -> PlayerMapRecord:
        """Ingest a chronological match event for a player.

        Args:
            player_id: Unique player identifier.
            date: Timestamp or parseable date string of the match.
            map_name: Map name (e.g. "Mirage", "de_dust2").
            side: Side starting on or played ("CT", "T", "Neutral").
            stats_dict: Dict of player stats or pre-encoded numeric vector.
            won: Whether the player won the map (bool, int, or str).

        Returns:
            The created PlayerMapRecord appended to the player's rolling deque.
        """
        norm_pid = _normalize_player_id(player_id)
        ts = _normalize_timestamp(date)

        # Parse won flag
        won_val = _parse_won_flag(won)
        if won is None and isinstance(stats_dict, Mapping) and "won" in stats_dict:
            won_val = _parse_won_flag(stats_dict["won"])

        # Encode feature vector
        feat_vec: np.ndarray
        if isinstance(stats_dict, np.ndarray):
            if stats_dict.ndim == 1 and stats_dict.size == self.feature_dim:
                feat_vec = stats_dict.astype(np.float32, copy=False)
            elif stats_dict.ndim == 1 and stats_dict.size == 16 and self.feature_dim == 18:
                feat_vec = np.empty(18, dtype=np.float32)
                feat_vec[0] = float(get_map_id(map_name))
                feat_vec[1] = float(get_side_id(side))
                feat_vec[2:] = stats_dict.astype(np.float32, copy=False)
            else:
                raw = stats_dict.astype(np.float32, copy=False)
                feat_vec = np.zeros(self.feature_dim, dtype=np.float32)
                n = min(len(raw), self.feature_dim)
                feat_vec[:n] = raw[:n]
        elif isinstance(stats_dict, Sequence) and not isinstance(stats_dict, (str, bytes)):
            raw = np.array(stats_dict, dtype=np.float32)
            if raw.size == self.feature_dim:
                feat_vec = raw
            elif raw.size == 16 and self.feature_dim == 18:
                feat_vec = np.empty(18, dtype=np.float32)
                feat_vec[0] = float(get_map_id(map_name))
                feat_vec[1] = float(get_side_id(side))
                feat_vec[2:] = raw
            else:
                feat_vec = np.zeros(self.feature_dim, dtype=np.float32)
                n = min(len(raw), self.feature_dim)
                feat_vec[:n] = raw[:n]
        else:
            # Mapping or None -> use encode_player_map_token
            stats_map: Mapping[str, Any] | None = stats_dict if isinstance(stats_dict, Mapping) else None
            encoded = encode_player_map_token(
                record=stats_map,
                map_name=map_name,
                side_name=side,
                won=won_val,
                include_map_side=self.include_map_side,
            )
            if encoded.shape[0] == self.feature_dim:
                feat_vec = encoded
            else:
                feat_vec = np.zeros(self.feature_dim, dtype=np.float32)
                n = min(len(encoded), self.feature_dim)
                feat_vec[:n] = encoded[:n]

        record = PlayerMapRecord(
            player_id=norm_pid,
            date=ts,
            map_name=str(map_name) if map_name is not None else "",
            side=str(side) if side is not None else "",
            won=won_val,
            stats_dict=dict(stats_dict) if isinstance(stats_dict, Mapping) else {},
            features=feat_vec,
        )

        self.histories[norm_pid].append(record)
        return record

    def get_player_sequence(
        self,
        player_id: Any,
        cutoff_date: Any = None,
        max_len: int = 16,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract chronological player sequence matrix and observation mask.

        Strictly enforces `record.date < cutoff_date` to prevent lookahead leakage.
        Handles empty or unknown players gracefully with an all-False mask and zero tensor.

        Args:
            player_id: Player identifier.
            cutoff_date: Timestamp or parseable date string. Only records strictly
                prior to this date (record.date < cutoff_date) are returned.
                If None, all available records are considered.
            max_len: Maximum sequence length (number of past maps, default 16).

        Returns:
            tuple of:
                - sequence_matrix: np.ndarray of shape (max_len, feature_dim), float32.
                - mask: np.ndarray of shape (max_len,), bool (True for observed, False for empty/pad).
        """
        sequence_matrix = np.zeros((max_len, self.feature_dim), dtype=np.float32)
        mask = np.zeros(max_len, dtype=bool)

        norm_pid = _normalize_player_id(player_id)
        if not norm_pid or norm_pid not in self.histories:
            return sequence_matrix, mask

        # Temporal filter
        cutoff_ts: pd.Timestamp | None = (
            _normalize_timestamp(cutoff_date) if cutoff_date is not None else None
        )

        records = self.histories[norm_pid]
        valid_records: list[PlayerMapRecord] = [
            r for r in records if cutoff_ts is None or r.date < cutoff_ts
        ]

        if not valid_records:
            return sequence_matrix, mask

        # Order chronologically (oldest to newest) and select up to max_len most recent
        valid_records.sort(key=lambda r: r.date)
        selected = valid_records[-max_len:]
        num_valid = len(selected)

        if self.padding == "left":
            offset = max_len - num_valid
            for idx, rec in enumerate(selected):
                sequence_matrix[offset + idx] = rec.features
                mask[offset + idx] = True
        else:  # "right" padding (standard left-aligned data)
            for idx, rec in enumerate(selected):
                sequence_matrix[idx] = rec.features
                mask[idx] = True

        return sequence_matrix, mask

    def get_player_record_count(self, player_id: Any, cutoff_date: Any = None) -> int:
        """Count the number of past records for a player prior to cutoff_date."""
        norm_pid = _normalize_player_id(player_id)
        if not norm_pid or norm_pid not in self.histories:
            return 0
        cutoff_ts: pd.Timestamp | None = (
            _normalize_timestamp(cutoff_date) if cutoff_date is not None else None
        )
        return sum(
            1 for r in self.histories[norm_pid] if cutoff_ts is None or r.date < cutoff_ts
        )

    def num_players(self) -> int:
        """Return the number of players currently tracked."""
        return len(self.histories)

    def clear(self) -> None:
        """Clear all player history records."""
        self.histories.clear()

    def ingest_records_from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        player_col: str = "player_name",
        date_col: str = "date",
        map_col: str = "map",
        side_col: str = "side",
        won_col: str = "won",
    ) -> int:
        """Ingest a DataFrame of player performance records chronologically.

        Sorts by date prior to ingestion to maintain chronological deque order.
        """
        if df.empty:
            return 0

        work_df = df.copy()
        if date_col in work_df.columns:
            work_df["_sort_dt"] = pd.to_datetime(work_df[date_col])
            work_df = work_df.sort_values("_sort_dt")

        count = 0
        for _, row in work_df.iterrows():
            pid = row.get(player_col) or row.get("player_id")
            dt = row.get(date_col)
            m_name = row.get(map_col) or row.get("map_name") or ""
            s_name = row.get(side_col) or row.get("side_name") or row.get("left_team_started") or ""
            w_val = row.get(won_col)

            # Build stats dictionary from row items
            row_dict = row.to_dict()
            self.add_player_map_record(
                player_id=pid,
                date=dt,
                map_name=m_name,
                side=s_name,
                stats_dict=row_dict,
                won=w_val,
            )
            count += 1
        return count

    def __getitem__(self, key: str) -> Any:
        """Dictionary-like access for compatibility."""
        if key == "tracker":
            return self
        if key in ("summary", "summary_statistics", "metadata", "stats"):
            return getattr(self, "summary", {})
        if key == "histories":
            return self.histories
        raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        return key in ("tracker", "summary", "summary_statistics", "metadata", "stats", "histories")

    def get(self, key: str, default: Any = None) -> Any:
        if key == "tracker":
            return self
        if key in ("summary", "summary_statistics", "metadata", "stats"):
            return getattr(self, "summary", default)
        if key == "histories":
            return self.histories
        return default

    def to_state_dict(self) -> dict[str, Any]:
        """Serialize tracker configuration and history state into a dict."""
        histories_dict: dict[str, list[dict[str, Any]]] = {}
        for pid, recs in self.histories.items():
            histories_dict[pid] = [
                {
                    "player_id": r.player_id,
                    "date": r.date,
                    "map_name": r.map_name,
                    "side": r.side,
                    "won": r.won,
                    "stats_dict": r.stats_dict,
                    "features": r.features,
                }
                for r in recs
            ]
        return {
            "max_history": self.max_history,
            "feature_dim": self.feature_dim,
            "include_map_side": self.include_map_side,
            "padding": self.padding,
            "histories": histories_dict,
            "summary": getattr(self, "summary", {}),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> PlayerHistoryTracker:
        """Instantiate tracker from serialized state dictionary."""
        tracker = cls(
            max_history=state.get("max_history", 16),
            feature_dim=state.get("feature_dim", 18),
            include_map_side=state.get("include_map_side", True),
            padding=state.get("padding", "right"),
        )
        for pid, recs in state.get("histories", {}).items():
            for r in recs:
                if isinstance(r, PlayerMapRecord):
                    tracker.histories[pid].append(r)
                elif isinstance(r, dict):
                    tracker.histories[pid].append(
                        PlayerMapRecord(
                            player_id=r["player_id"],
                            date=r["date"],
                            map_name=r.get("map_name", ""),
                            side=r.get("side", ""),
                            won=r.get("won", 0.0),
                            stats_dict=r.get("stats_dict", {}),
                            features=r.get("features", np.zeros(tracker.feature_dim, dtype=np.float32)),
                        )
                    )
        if "summary" in state:
            tracker.summary = state["summary"]
        return tracker

    def save(self, path: str | Path, summary: Mapping[str, Any] | None = None) -> None:
        """Save serialized tracker instance to a .pt file."""
        if summary is not None:
            self.summary = dict(summary)
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, save_path)

    @classmethod
    def load(cls, path: str | Path) -> PlayerHistoryTracker:
        """Load tracker from a .pt file."""
        obj = torch.load(path, weights_only=False)
        if isinstance(obj, PlayerHistoryTracker):
            return obj
        if isinstance(obj, dict):
            if "tracker" in obj and isinstance(obj["tracker"], PlayerHistoryTracker):
                return obj["tracker"]
            if "histories" in obj:
                return cls.from_state_dict(obj)
        raise ValueError(f"Unrecognized tracker object format in {path}: {type(obj)}")


def build_match_input_tensors(
    tracker: PlayerHistoryTracker,
    team1_players: Sequence[Any],
    team2_players: Sequence[Any],
    cutoff_date: Any,
    max_len: int = 16,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Build input tensors for a single CS2 match between two 5-player teams.

    Args:
        tracker: PlayerHistoryTracker instance containing historical records.
        team1_players: Sequence of up to 5 player IDs for team 1.
        team2_players: Sequence of up to 5 player IDs for team 2.
        cutoff_date: Match cutoff timestamp. Strictly records with date < cutoff_date
            are included to prevent lookahead leakage.
        max_len: Number of sequence steps (maps) per player (default 16).
        device: Optional PyTorch device for returned tensors.

    Returns:
        dict with:
            - "history": shape (1, 2, 5, max_len, feature_dim) float32 tensor
            - "mask": shape (1, 2, 5, max_len) bool tensor (True for observed, False for pad)
            - "player_counts": shape (1, 2, 5) int64 tensor of prior games observed.
    """
    feature_dim = tracker.feature_dim

    history_t = torch.zeros((1, 2, 5, max_len, feature_dim), dtype=torch.float32)
    mask_t = torch.zeros((1, 2, 5, max_len), dtype=torch.bool)
    counts_t = torch.zeros((1, 2, 5), dtype=torch.long)

    t1_list = list(team1_players)[:5] if team1_players is not None else []
    t2_list = list(team2_players)[:5] if team2_players is not None else []

    # Team 1 (index 0)
    for p_idx in range(5):
        if p_idx < len(t1_list) and t1_list[p_idx] is not None:
            pid = t1_list[p_idx]
            seq_mat, m = tracker.get_player_sequence(pid, cutoff_date=cutoff_date, max_len=max_len)
            history_t[0, 0, p_idx] = torch.from_numpy(seq_mat)
            mask_t[0, 0, p_idx] = torch.from_numpy(m)
            counts_t[0, 0, p_idx] = int(m.sum())

    # Team 2 (index 1)
    for p_idx in range(5):
        if p_idx < len(t2_list) and t2_list[p_idx] is not None:
            pid = t2_list[p_idx]
            seq_mat, m = tracker.get_player_sequence(pid, cutoff_date=cutoff_date, max_len=max_len)
            history_t[0, 1, p_idx] = torch.from_numpy(seq_mat)
            mask_t[0, 1, p_idx] = torch.from_numpy(m)
            counts_t[0, 1, p_idx] = int(m.sum())

    if device is not None:
        history_t = history_t.to(device)
        mask_t = mask_t.to(device)
        counts_t = counts_t.to(device)

    return {
        "history": history_t,
        "mask": mask_t,
        "player_counts": counts_t,
    }


def build_batch_match_input_tensors(
    tracker: PlayerHistoryTracker,
    matches: Sequence[Mapping[str, Any] | tuple[Sequence[Any], Sequence[Any], Any]],
    max_len: int = 16,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Build batched input tensors for multiple matches.

    Args:
        tracker: PlayerHistoryTracker instance.
        matches: Sequence of match dicts (with 'team1_players', 'team2_players', 'cutoff_date')
            or tuples of (team1_players, team2_players, cutoff_date).
        max_len: Number of sequence steps per player.
        device: Optional PyTorch device.

    Returns:
        dict with:
            - "history": shape (batch_size, 2, 5, max_len, feature_dim) float32 tensor
            - "mask": shape (batch_size, 2, 5, max_len) bool tensor
            - "player_counts": shape (batch_size, 2, 5) int64 tensor
    """
    batch_size = len(matches)
    feature_dim = tracker.feature_dim

    history_t = torch.zeros((batch_size, 2, 5, max_len, feature_dim), dtype=torch.float32)
    mask_t = torch.zeros((batch_size, 2, 5, max_len), dtype=torch.bool)
    counts_t = torch.zeros((batch_size, 2, 5), dtype=torch.long)

    for b_idx, match in enumerate(matches):
        if isinstance(match, Mapping):
            t1 = match.get("team1_players", ())
            t2 = match.get("team2_players", ())
            cutoff = match.get("cutoff_date")
        else:
            t1, t2, cutoff = match

        single = build_match_input_tensors(
            tracker=tracker,
            team1_players=t1,
            team2_players=t2,
            cutoff_date=cutoff,
            max_len=max_len,
            device=None,
        )
        history_t[b_idx] = single["history"][0]
        mask_t[b_idx] = single["mask"][0]
        counts_t[b_idx] = single["player_counts"][0]

    if device is not None:
        history_t = history_t.to(device)
        mask_t = mask_t.to(device)
        counts_t = counts_t.to(device)

    return {
        "history": history_t,
        "mask": mask_t,
        "player_counts": counts_t,
    }
