"""Leak-free online feature generation for chronological CS2 match data."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

from pipeline.features.player_ratings import IndividualPlayerRatingEngine
from pipeline.features.ratings_glicko2 import Glicko2RatingEngine
from pipeline.features.ratings_openskill import OpenSkillRatingEngine
from pipeline.features.ratings_trueskill import TrueSkillRatingEngine
from pipeline.features.veto_engine import VetoEngine

COMPETITIVE_MAPS = (
    "de_mirage",
    "de_inferno",
    "de_nuke",
    "de_dust2",
    "de_ancient",
    "de_anubis",
    "de_vertigo",
)


def _canonical_map(name: str) -> str:
    normalized = name.strip().lower()
    return normalized if normalized.startswith("de_") else f"de_{normalized}"


def _same_team(left: str, right: str) -> bool:
    return left.strip().casefold() == right.strip().casefold()


@dataclass
class _TeamState:
    elo: float = 1500.0
    matches: int = 0
    outcomes: Deque[int] = field(default_factory=lambda: deque(maxlen=10))
    round_differentials: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    last_played: Optional[datetime] = None
    map_games: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    map_wins: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    demo_count: int = 0
    adr_total: float = 0.0
    entry_rate_total: float = 0.0
    utility_efficiency_total: float = 0.0

    @property
    def win_rate_5(self) -> float:
        recent = list(self.outcomes)[-5:]
        return sum(recent) / len(recent) if recent else 0.0

    @property
    def win_rate_10(self) -> float:
        return sum(self.outcomes) / len(self.outcomes) if self.outcomes else 0.0

    @property
    def round_differential(self) -> float:
        return sum(self.round_differentials) / len(self.round_differentials) if self.round_differentials else 0.0

    @property
    def adr(self) -> float:
        return self.adr_total / self.demo_count if self.demo_count else 0.0

    @property
    def entry_frag_rate(self) -> float:
        return self.entry_rate_total / self.demo_count if self.demo_count else 0.0

    @property
    def utility_efficiency(self) -> float:
        return self.utility_efficiency_total / self.demo_count if self.demo_count else 0.0


class ChronologicalFeatureBuilder:
    """Build numeric match features using only state from earlier matches."""

    K_FACTOR = 32.0

    def __init__(self, enable_advanced_ratings: bool = True) -> None:
        self._teams: Dict[str, _TeamState] = defaultdict(_TeamState)
        self._head_to_head: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.player_engine = IndividualPlayerRatingEngine()
        self.enable_advanced_ratings = enable_advanced_ratings
        self.glicko2_engine = Glicko2RatingEngine() if enable_advanced_ratings else None
        self.trueskill_engine = TrueSkillRatingEngine() if enable_advanced_ratings else None
        self.openskill_engine = OpenSkillRatingEngine() if enable_advanced_ratings else None
        self.veto_engine = VetoEngine()

    def _team(self, name: str) -> _TeamState:
        return self._teams[name.strip().casefold()]


    @staticmethod
    def _h2h_key(team1: str, team2: str) -> Tuple[str, str]:
        a, b = team1.casefold(), team2.casefold()
        return (a, b) if a <= b else (b, a)

    def _h2h_team1_wins(self, team1: str, team2: str) -> int:
        history = self._head_to_head[self._h2h_key(team1, team2)]
        # Stored from the lexicographically first team's perspective.
        return sum(history) if team1.casefold() <= team2.casefold() else len(history) - sum(history)

    @staticmethod
    def _rest_days(team: _TeamState, timestamp: datetime) -> float:
        if team.last_played is None:
            return 0.0
        return max((timestamp - team.last_played).total_seconds() / 86400.0, 0.0)

    @staticmethod
    def _map_win_rate(team: _TeamState, map_name: str) -> float:
        key = _canonical_map(map_name)
        games = team.map_games.get(key, 0)
        return team.map_wins.get(key, 0) / games if games else 0.0

    def build_match_features(
        self, match: MatchFixture, map_name: Optional[str] = None
    ) -> Dict[str, float]:
        """Emit a numeric feature row from current (strictly prior) state."""
        team1 = self._team(match.team1)
        team2 = self._team(match.team2)
        elo_diff = team1.elo - team2.elo
        count = len(self._head_to_head[self._h2h_key(match.team1, match.team2)])
        team1_h2h_wins = self._h2h_team1_wins(match.team1, match.team2)
        features = {
            "elo_diff": float(elo_diff),
            "elo_prob": float(1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))),
            "win_rate_diff_5": team1.win_rate_5 - team2.win_rate_5,
            "win_rate_diff_10": team1.win_rate_10 - team2.win_rate_10,
            "round_diff": team1.round_differential - team2.round_differential,
            "experience_diff": float(team1.matches - team2.matches),
            "rest_days_diff": self._rest_days(team1, match.timestamp)
            - self._rest_days(team2, match.timestamp),
            "h2h_matches": float(count),
            "h2h_win_rate_team1": team1_h2h_wins / max(count, 1),
            "adr_diff": team1.adr - team2.adr,
            "entry_frag_rate_diff": team1.entry_frag_rate - team2.entry_frag_rate,
            "utility_efficiency_diff": team1.utility_efficiency - team2.utility_efficiency,
        }
        # Integrate individual player lineup rating features
        lineup_feats = self.player_engine.build_lineup_features(
            team1=match.team1,
            team2=match.team2,
            team1_players=match.team1_players,
            team2_players=match.team2_players,
        )
        veto_features = self.veto_engine.build_veto_features(
            match.team1, match.team2, match.maps
        )
        features.update(veto_features)
        features.update(lineup_feats)
        if self.enable_advanced_ratings:
            features.update(
                self.glicko2_engine.build_features(
                    match.team1, match.team2, match.team1_players, match.team2_players
                )
            )
            features.update(
                self.trueskill_engine.build_features(
                    match.team1, match.team2, match.team1_players, match.team2_players
                )
            )
            features.update(
                self.openskill_engine.build_features(
                    match.team1, match.team2, match.team1_players, match.team2_players
                )
            )
        if map_name is not None:
            features["map_win_rate_diff"] = self._map_win_rate(team1, map_name) - self._map_win_rate(team2, map_name)
        return features

    def _add_demo_stats(self, team_name: str, players: list) -> None:
        if not players:
            return
        team = self._team(team_name)
        # Store one team-level observation per map, rather than weighting maps
        # with larger rosters more heavily.
        team.demo_count += 1
        team.adr_total += sum(player.adr for player in players) / len(players)
        rounds = sum(player.rounds_played for player in players)
        first_kills = sum(player.first_kills for player in players)
        utility_damage = sum(player.utility_damage for player in players)
        team.entry_rate_total += first_kills / max(rounds, 1)
        team.utility_efficiency_total += utility_damage / max(rounds, 1)

    def update_state(
        self,
        match: MatchFixture,
        map_results: Optional[List[ParsedMapResult]] = None,
    ) -> None:
        """Apply this completed match's outcome after its feature row is emitted."""
        self.veto_engine.update_match(match, map_results)
        if match.winner not in ("team1", "team2"):
            return

        team1 = self._team(match.team1)
        team2 = self._team(match.team2)
        team1_won = match.winner == "team1"
        expected1 = 1.0 / (1.0 + 10.0 ** ((team2.elo - team1.elo) / 400.0))
        actual1 = 1.0 if team1_won else 0.0
        change = self.K_FACTOR * (actual1 - expected1)
        team1.elo += change
        team2.elo -= change

        team1.matches += 1
        team2.matches += 1
        team1.outcomes.append(1 if team1_won else 0)
        team2.outcomes.append(0 if team1_won else 1)

        # Update individual player ratings
        self.player_engine.update_match(
            team1=match.team1,
            team2=match.team2,
            team1_players=match.team1_players,
            team2_players=match.team2_players,
            winner=match.winner,
            date_iso=match.timestamp.isoformat(),
        )
        if self.enable_advanced_ratings:
            self.glicko2_engine.update_match(
                match.team1,
                match.team2,
                match.team1_players,
                match.team2_players,
                match.winner,
                match.timestamp,
            )
            self.trueskill_engine.update_match(
                match.team1,
                match.team2,
                match.team1_players,
                match.team2_players,
                match.winner,
                match.timestamp,
            )
            self.openskill_engine.update_match(
                match.team1,
                match.team2,
                match.team1_players,
                match.team2_players,
                match.winner,
                match.timestamp,
            )
        if match.score1 is not None and match.score2 is not None:
            difference = float(match.score1 - match.score2)
            team1.round_differentials.append(difference)
            team2.round_differentials.append(-difference)
        team1.last_played = match.timestamp
        team2.last_played = match.timestamp

        history = self._head_to_head[self._h2h_key(match.team1, match.team2)]
        history.append(
            int(team1_won) if match.team1.casefold() <= match.team2.casefold()
            else int(not team1_won)
        )

        for result in map_results or []:
            map_key = _canonical_map(result.map_name)
            if map_key not in COMPETITIVE_MAPS:
                continue
            result_team1_is_match_team1 = _same_team(result.team1_name, match.team1)
            result_team2_is_match_team1 = _same_team(result.team2_name, match.team1)
            if result_team1_is_match_team1:
                match_team1_name, match_team2_name = result.team1_name, result.team2_name
                match_team1_won = _same_team(result.winner_team, result.team1_name)
            elif result_team2_is_match_team1:
                match_team1_name, match_team2_name = result.team2_name, result.team1_name
                match_team1_won = _same_team(result.winner_team, result.team2_name)
            else:
                # Do not assign map observations to the wrong fixture teams.
                continue
            team1_state = self._team(match.team1)
            team2_state = self._team(match.team2)
            team1_state.map_games[map_key] += 1
            team2_state.map_games[map_key] += 1
            if match_team1_won:
                team1_state.map_wins[map_key] += 1
            else:
                team2_state.map_wins[map_key] += 1
            team1_players = [
                player for player in result.player_stats
                if _same_team(player.team_name, match_team1_name)
            ]
            team2_players = [
                player for player in result.player_stats
                if _same_team(player.team_name, match_team2_name)
            ]
            self._add_demo_stats(match.team1, team1_players)
            self._add_demo_stats(match.team2, team2_players)

    def process_matches(
        self,
        matches: List[MatchFixture],
        demo_map_lookup: Optional[Dict[str, List[ParsedMapResult]]] = None,
    ) -> List[Dict[str, Any]]:
        """Emit rows in chronology, deferring tied-time updates to avoid leakage."""
        ordered = sorted(matches, key=lambda match: (match.timestamp, match.match_id))
        rows: List[Dict[str, Any]] = []
        index = 0
        while index < len(ordered):
            end = index + 1
            timestamp = ordered[index].timestamp
            while end < len(ordered) and ordered[end].timestamp == timestamp:
                end += 1
            # Every match at an identical timestamp sees the same pre-time state.
            for match in ordered[index:end]:
                row: Dict[str, Any] = self.build_match_features(match)
                row["y"] = 1 if match.winner == "team1" else 0
                rows.append(row)
            for match in ordered[index:end]:
                self.update_state(match, (demo_map_lookup or {}).get(match.match_id))
            index = end
        return rows
