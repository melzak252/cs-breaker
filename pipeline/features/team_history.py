"""Literal-team modern-series history with frozen, atomic source-day updates.

Names are source identities, not organization IDs or inferred aliases. Both Elo
channels use a 1500 prior, scale 400 and K=32; the second lazily reverts to the
prior with a fixed 180-day half-life. Queries never register teams or age state.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
import math
import re


ARCHIVE_FEATURES = [
    "archive_elo_diff", "archive_elo_prob", "archive_decay_diff", "archive_decay_prob",
    "archive_team1_prior_matches", "archive_team2_prior_matches", "archive_minimum_prior_matches",
]


def _rating_day(value: str) -> date:
    if not isinstance(value, str):
        raise ValueError("day must be a canonical YYYY-MM-DD string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("day must be a canonical YYYY-MM-DD string") from exc
    if parsed.isoformat() != value:
        raise ValueError("day must be a canonical YYYY-MM-DD string")
    return parsed


def _teams(team1: str, team2: str) -> None:
    if any(not isinstance(team, str) or not team.strip() for team in (team1, team2)):
        raise ValueError("literal team names must be nonempty strings")
    if team1 == team2:
        raise ValueError("opposing literal teams must be distinct")


def _probability(difference: float) -> float:
    exponent = math.log(10.0) * difference / 400.0
    tail = math.exp(-abs(exponent))
    return 1.0 / (1.0 + tail) if exponent >= 0 else tail / (1.0 + tail)


@dataclass(frozen=True)
class TeamHistoryProfile:
    elo: float = 1500.0
    decay: float = 1500.0
    matches_played: int = 0
    last_active: date | None = None


class TeamHistoryEngine:
    """Two fixed archive rating channels keyed by unmodified source team names."""

    def __init__(self) -> None:
        self.teams: dict[str, TeamHistoryProfile] = {}
        self._last_day: date | None = None
        self._match_ids: set[str] = set()

    def _future_day(self, day: str) -> date:
        current = _rating_day(day)
        if self._last_day is not None and current <= self._last_day:
            raise ValueError("day must be later than the last completed history update day")
        return current

    @staticmethod
    def _opening(profile: TeamHistoryProfile, current: date) -> tuple[float, float]:
        elapsed = (current - profile.last_active).days if profile.last_active is not None else 0
        decay = 1500.0 + (profile.decay - 1500.0) * 2.0 ** (-elapsed / 180.0)
        return profile.elo, decay

    def compute_prematch_features(self, team1: str, team2: str, *, day: str) -> dict[str, float | int]:
        current = self._future_day(day)
        _teams(team1, team2)
        left = self.teams.get(team1, TeamHistoryProfile())
        right = self.teams.get(team2, TeamHistoryProfile())
        left_ratings = self._opening(left, current)
        right_ratings = self._opening(right, current)
        elo_diff = left_ratings[0] - right_ratings[0]
        decay_diff = left_ratings[1] - right_ratings[1]
        return {
            "archive_elo_diff": elo_diff,
            "archive_elo_prob": _probability(elo_diff),
            "archive_decay_diff": decay_diff,
            "archive_decay_prob": _probability(decay_diff),
            "archive_team1_prior_matches": left.matches_played,
            "archive_team2_prior_matches": right.matches_played,
            "archive_minimum_prior_matches": min(left.matches_played, right.matches_played),
        }

    def update_day(self, day: str, games: list[dict]) -> None:
        """Validate the entire batch before computing or committing any change.

        Every expectation uses the same opening ratings. Deterministic native-ID
        order plus fsum makes input permutations equivalent, even for repeated
        team appearances. No empty or invalid batch advances the saved cutoff.
        """
        current = self._future_day(day)
        if not isinstance(games, list) or not games:
            raise ValueError("games must be a nonempty list of series records")
        ordered = []
        match_ids = set()
        for game in games:
            if not isinstance(game, dict):
                raise ValueError("each series must be a dictionary")
            identity = game.get("hltv_match_id")
            if not isinstance(identity, str) or re.fullmatch(r"[1-9][0-9]*", identity) is None:
                raise ValueError("hltv_match_id requires a positive native HLTV integer ID")
            if identity in match_ids or identity in self._match_ids:
                raise ValueError("duplicate series match ID")
            match_ids.add(identity)
            if "day" in game and game["day"] != day:
                raise ValueError("series day must match its update day")
            team1, team2 = game.get("team1"), game.get("team2")
            _teams(team1, team2)
            y = game.get("y")
            if isinstance(y, bool) or not isinstance(y, (int, float)) or not math.isfinite(y) or y not in (0, 1):
                raise ValueError("series outcome y must be finite zero or one")
            ordered.append((identity, team1, team2, int(y)))
        ordered.sort(key=lambda game: int(game[0]))

        profiles = {}
        openings = {}
        deltas = defaultdict(lambda: ([], []))
        counts = defaultdict(int)
        for _, team1, team2, y in ordered:
            for team in (team1, team2):
                if team not in profiles:
                    profiles[team] = self.teams.get(team, TeamHistoryProfile())
                    openings[team] = self._opening(profiles[team], current)
                counts[team] += 1
            for channel in (0, 1):
                expected = _probability(openings[team1][channel] - openings[team2][channel])
                delta = 32.0 * (y - expected)
                deltas[team1][channel].append(delta)
                deltas[team2][channel].append(-delta)
        updated = {
            team: TeamHistoryProfile(
                elo=openings[team][0] + math.fsum(deltas[team][0]),
                decay=openings[team][1] + math.fsum(deltas[team][1]),
                matches_played=profile.matches_played + counts[team],
                last_active=current,
            )
            for team, profile in profiles.items()
        }
        self.teams.update(updated)
        self._match_ids.update(match_ids)
        self._last_day = current
