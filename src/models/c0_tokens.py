"""CS2 tokenization, map/side vocabulary, and player map token encoding specs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Map Vocabulary Constants and Mappings
# ---------------------------------------------------------------------------

MAP_PAD: int = 0
MAP_UNKNOWN: int = 0
MAP_MIRAGE: int = 1
MAP_INFERNO: int = 2
MAP_NUKE: int = 3
MAP_DUST2: int = 4
MAP_ANCIENT: int = 5
MAP_ANUBIS: int = 6
MAP_VERTIGO: int = 7
MAP_OVERPASS: int = 8
MAP_TRAIN: int = 9
MAP_CACHE: int = 10

CANONICAL_MAPS: tuple[str, ...] = (
    "mirage",
    "inferno",
    "nuke",
    "dust2",
    "ancient",
    "anubis",
    "vertigo",
    "overpass",
    "train",
    "cache",
)

MAP_TO_ID: dict[str, int] = {
    "pad": MAP_PAD,
    "unknown": MAP_UNKNOWN,
    "mirage": MAP_MIRAGE,
    "inferno": MAP_INFERNO,
    "nuke": MAP_NUKE,
    "dust2": MAP_DUST2,
    "ancient": MAP_ANCIENT,
    "anubis": MAP_ANUBIS,
    "vertigo": MAP_VERTIGO,
    "overpass": MAP_OVERPASS,
    "train": MAP_TRAIN,
    "cache": MAP_CACHE,
}

ID_TO_MAP: dict[int, str] = {
    MAP_PAD: "pad",
    MAP_MIRAGE: "mirage",
    MAP_INFERNO: "inferno",
    MAP_NUKE: "nuke",
    MAP_DUST2: "dust2",
    MAP_ANCIENT: "ancient",
    MAP_ANUBIS: "anubis",
    MAP_VERTIGO: "vertigo",
    MAP_OVERPASS: "overpass",
    MAP_TRAIN: "train",
    MAP_CACHE: "cache",
}

# ---------------------------------------------------------------------------
# Side Vocabulary Constants and Mappings
# ---------------------------------------------------------------------------

SIDE_NEUTRAL: int = 0
SIDE_NONE: int = 0
SIDE_CT: int = 1
SIDE_T: int = 2

SIDE_TO_ID: dict[str, int] = {
    "neutral": SIDE_NEUTRAL,
    "none": SIDE_NONE,
    "pad": SIDE_NEUTRAL,
    "unknown": SIDE_NEUTRAL,
    "": SIDE_NEUTRAL,
    "ct": SIDE_CT,
    "counter-terrorist": SIDE_CT,
    "counter_terrorist": SIDE_CT,
    "counterterrorist": SIDE_CT,
    "counter terrorists": SIDE_CT,
    "counter-terrorists": SIDE_CT,
    "t": SIDE_T,
    "terrorist": SIDE_T,
    "terrorists": SIDE_T,
}

ID_TO_SIDE: dict[int, str] = {
    SIDE_NEUTRAL: "neutral",
    SIDE_CT: "ct",
    SIDE_T: "t",
}

# ---------------------------------------------------------------------------
# Canonical Player Stat Features (16 dimensions)
# ---------------------------------------------------------------------------

STAT_FEATURE_NAMES: tuple[str, ...] = (
    "kills",
    "deaths",
    "assists",
    "adr",
    "kast",
    "rating",
    "hs_pct",
    "entry_diff",
    "rounds_won",
    "rounds_lost",
    "won",
    "multi_kills",
    "clutches",
    "kd_ratio",
    "round_swing",
    "total_rounds",
)

TOKEN_FEATURE_NAMES: tuple[str, ...] = ("map_id", "side_id") + STAT_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CS2TokenConfig:
    """Configuration for CS2 player map token representations and embeddings."""

    max_history_maps: int = 16
    num_maps: int = 16
    map_embed_dim: int = 16
    side_embed_dim: int = 8
    player_stat_dim: int = 16
    hidden_dim: int = 32
    num_sides: int = 4
    stat_features: tuple[str, ...] = field(default_factory=lambda: STAT_FEATURE_NAMES)

    @property
    def total_token_dim(self) -> int:
        """Total dimension when discrete IDs (map_id, side_id) are prepended."""
        return self.player_stat_dim + 2


# ---------------------------------------------------------------------------
# Helper Parsing Functions
# ---------------------------------------------------------------------------

def _clean_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip().lower()
    return str(val).strip().lower()


def get_map_id(map_name: str | None) -> int:
    """Return integer ID for a CS map (case-insensitive, alias-tolerant).

    Returns 0 for unknown/pad maps.
    """
    cleaned = _clean_str(map_name)
    if not cleaned:
        return MAP_PAD

    # Strip prefixes like "de_" or "cs_"
    if cleaned.startswith("de_") or cleaned.startswith("cs_"):
        cleaned = cleaned[3:]

    # Remove delimiters and spaces
    compact = cleaned.replace(" ", "").replace("_", "").replace("-", "")

    # Handle common variants
    if compact in ("dust2", "dustii", "dust"):
        return MAP_DUST2

    return MAP_TO_ID.get(compact, MAP_UNKNOWN)


def get_map_name(map_id: int) -> str:
    """Return canonical lowercase map name from integer ID."""
    return ID_TO_MAP.get(int(map_id), "unknown")


def get_side_id(side_name: str | None) -> int:
    """Return integer ID for a side (case-insensitive).

    0 = Neutral / None / Unknown
    1 = CT
    2 = T
    """
    cleaned = _clean_str(side_name)
    if not cleaned:
        return SIDE_NEUTRAL

    norm = cleaned.replace("-", "_").replace(" ", "_")
    return SIDE_TO_ID.get(norm, SIDE_NEUTRAL)


def get_side_name(side_id: int) -> str:
    """Return side name from integer ID."""
    return ID_TO_SIDE.get(int(side_id), "neutral")


def _to_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float, np.integer, np.floating)):
        if math.isnan(val):
            return default
        return float(val)
    if isinstance(val, str):
        cleaned = val.strip()
        if not cleaned:
            return default
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


def _parse_percentage(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float, np.integer, np.floating)):
        if math.isnan(val):
            return default
        return float(val)
    if isinstance(val, str):
        cleaned = val.replace("%", "").strip()
        if not cleaned:
            return default
        try:
            f = float(cleaned)
            return f / 100.0 if "%" in val else f
        except ValueError:
            return default
    return default


def _parse_kast(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float, np.integer, np.floating)):
        if math.isnan(val):
            return default
        f = float(val)
        return f / 100.0 if f > 1.0 else f
    if isinstance(val, str):
        cleaned = val.replace("%", "").strip()
        if not cleaned:
            return default
        try:
            f = float(cleaned)
            return f / 100.0 if (f > 1.0 or "%" in val) else f
        except ValueError:
            return default
    return default


def _parse_entry_diff(entry_diff_val: Any, opening_kd_val: Any, default: float = 0.0) -> float:
    if entry_diff_val is not None:
        val = _to_float(entry_diff_val, default=float("nan"))
        if not math.isnan(val):
            return val
    if opening_kd_val is not None:
        s = str(opening_kd_val).strip()
        if ":" in s:
            parts = s.split(":")
            try:
                k = float(parts[0].strip())
                d = float(parts[1].strip())
                return k - d
            except (ValueError, IndexError):
                pass
        elif "-" in s and not s.startswith("-"):
            parts = s.split("-")
            try:
                k = float(parts[0].strip())
                d = float(parts[1].strip())
                return k - d
            except (ValueError, IndexError):
                pass
    return default


def _parse_hs_pct(hs_pct_val: Any, headshots: float, kills: float, default: float = 0.0) -> float:
    if hs_pct_val is not None:
        val = _parse_percentage(hs_pct_val, default=float("nan"))
        if not math.isnan(val):
            return val / 100.0 if val > 1.0 else val
    if kills > 0:
        return float(headshots) / float(kills)
    return default


def _parse_kd_ratio(kd_ratio_val: Any, kills: float, deaths: float, default: float = 0.0) -> float:
    if kd_ratio_val is not None:
        val = _to_float(kd_ratio_val, default=float("nan"))
        if not math.isnan(val):
            return val
    return kills / max(1.0, deaths)


def _parse_won(val: Any) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (bool, np.bool_)):
        return 1.0 if val else 0.0
    if isinstance(val, (int, float, np.integer, np.floating)):
        if math.isnan(val):
            return 0.0
        return 1.0 if val > 0 else 0.0
    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned in ("1", "true", "t", "won", "win", "w", "yes"):
            return 1.0
        return 0.0
    return 0.0


def _parse_round_swing(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float, np.integer, np.floating)):
        if math.isnan(val):
            return default
        return float(val)
    if isinstance(val, str):
        cleaned = val.replace("%", "").strip()
        if not cleaned:
            return default
        try:
            f = float(cleaned)
            return f / 100.0 if "%" in val else f
        except ValueError:
            return default
    return default


# ---------------------------------------------------------------------------
# Token Encoding Functions
# ---------------------------------------------------------------------------

def encode_player_stats(
    record: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Encode player performance record into a 16-dimensional float32 vector.

    Order of features:
    0: kills
    1: deaths
    2: assists
    3: adr
    4: kast (in [0, 1])
    5: rating
    6: hs_pct (in [0, 1])
    7: entry_diff
    8: rounds_won
    9: rounds_lost
    10: won (1.0 or 0.0)
    11: multi_kills
    12: clutches
    13: kd_ratio
    14: round_swing
    15: total_rounds
    """
    data: dict[str, Any] = {}
    if record is not None:
        if hasattr(record, "to_dict"):
            data.update(record.to_dict())  # e.g. pd.Series
        elif isinstance(record, Mapping):
            data.update(record)
    data.update(kwargs)

    kills = _to_float(data.get("kills"))
    deaths = _to_float(data.get("deaths"))
    assists = _to_float(data.get("assists"))
    adr = _to_float(data.get("adr"))
    kast = _parse_kast(data.get("kast"))
    rating = _to_float(data.get("rating"))

    headshots = _to_float(data.get("headshots"))
    hs_pct = _parse_hs_pct(data.get("hs_pct"), headshots=headshots, kills=kills)

    entry_diff = _parse_entry_diff(data.get("entry_diff"), data.get("opening_kd"))

    rounds_won = _to_float(data.get("rounds_won"))
    rounds_lost = _to_float(data.get("rounds_lost"))
    won = _parse_won(data.get("won"))

    multi_kills = _to_float(data.get("multi_kills"))
    clutches = _to_float(data.get("clutches"))

    kd_ratio = _parse_kd_ratio(data.get("kd_ratio"), kills=kills, deaths=deaths)
    round_swing = _parse_round_swing(data.get("round_swing"))

    total_rounds_val = data.get("total_rounds")
    if total_rounds_val is not None:
        total_rounds = _to_float(total_rounds_val)
    else:
        total_rounds = rounds_won + rounds_lost

    vec = np.array(
        [
            kills,
            deaths,
            assists,
            adr,
            kast,
            rating,
            hs_pct,
            entry_diff,
            rounds_won,
            rounds_lost,
            won,
            multi_kills,
            clutches,
            kd_ratio,
            round_swing,
            total_rounds,
        ],
        dtype=np.float32,
    )
    return vec


def encode_player_map_token(
    record: Mapping[str, Any] | None = None,
    *,
    map_name: str | None = None,
    side_name: str | None = None,
    won: Any = None,
    include_map_side: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    """Convert a single map performance record into a numeric vector.

    Args:
        record: Mapping or pd.Series containing player stats.
        map_name: CS map name (e.g. "Mirage", "de_nuke").
        side_name: Side name (e.g. "CT", "T", "Neutral").
        won: Optional win indicator (1.0/0.0 or bool).
        include_map_side: If True, returns an 18-dim vector prepending
            `[map_id, side_id]`. If False, returns the 16-dim player stat vector.
        **kwargs: Additional or overriding stat fields.

    Returns:
        np.ndarray of shape (16,) if include_map_side=False, or (18,) if True.
    """
    merged_kwargs = dict(kwargs)
    if won is not None:
        merged_kwargs["won"] = won

    stat_vec = encode_player_stats(record=record, **merged_kwargs)

    if not include_map_side:
        return stat_vec

    data: dict[str, Any] = {}
    if record is not None:
        if hasattr(record, "to_dict"):
            data.update(record.to_dict())
        elif isinstance(record, Mapping):
            data.update(record)
    data.update(kwargs)

    resolved_map = map_name if map_name is not None else (data.get("map") or data.get("map_name"))
    resolved_side = (
        side_name
        if side_name is not None
        else (data.get("side") or data.get("side_name") or data.get("left_team_started"))
    )

    map_id = float(get_map_id(resolved_map))
    side_id = float(get_side_id(resolved_side))

    full_vec = np.empty(18, dtype=np.float32)
    full_vec[0] = map_id
    full_vec[1] = side_id
    full_vec[2:] = stat_vec
    return full_vec


# ---------------------------------------------------------------------------
# PyTorch Token Embedding Module
# ---------------------------------------------------------------------------

class CS2TokenEmbedding(nn.Module):
    """PyTorch embedding layer for CS2 player map tokens.

    Maps discrete map_id and side_id via learned embeddings and projects
    continuous statistics into a unified hidden space of dimension `hidden_dim`.
    """

    def __init__(self, config: CS2TokenConfig | None = None) -> None:
        super().__init__()
        self.config = config or CS2TokenConfig()
        self.map_embedding = nn.Embedding(
            num_embeddings=self.config.num_maps,
            embedding_dim=self.config.map_embed_dim,
            padding_idx=MAP_PAD,
        )
        self.side_embedding = nn.Embedding(
            num_embeddings=self.config.num_sides,
            embedding_dim=self.config.side_embed_dim,
            padding_idx=SIDE_NEUTRAL,
        )
        in_dim = (
            self.config.map_embed_dim
            + self.config.side_embed_dim
            + self.config.player_stat_dim
        )
        self.projection = nn.Sequential(
            nn.Linear(in_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
        )

    def forward(
        self,
        map_ids: torch.Tensor | None = None,
        side_ids: torch.Tensor | None = None,
        stats: torch.Tensor | None = None,
        tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Accepts either:
          - `tokens`: Tensor of shape (..., 18) where index 0 is map_id,
            index 1 is side_id, and indices 2..18 are stats.
          - Or separate `map_ids`, `side_ids`, `stats` tensors.

        Returns:
            Tensor of shape (..., hidden_dim).
        """
        if tokens is None and map_ids is not None and side_ids is None and stats is None:
            tokens = map_ids

        if tokens is not None:
            m_ids = tokens[..., 0].long()
            s_ids = tokens[..., 1].long()
            st = tokens[..., 2:].float()
        elif map_ids is not None and side_ids is not None and stats is not None:
            m_ids = map_ids.long()
            s_ids = side_ids.long()
            st = stats.float()
        else:
            raise ValueError(
                "Either `tokens` of shape (..., 18) or all of (map_ids, side_ids, stats) must be provided."
            )

        map_emb = self.map_embedding(m_ids)
        side_emb = self.side_embedding(s_ids)
        combined = torch.cat([map_emb, side_emb, st], dim=-1)
        return self.projection(combined)
