"""Glicko-2 ratings for Counter-Strike teams and players."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import glicko2


class Glicko2RatingEngine:
    """Track individual and team Glicko-2 ratings across matches."""

    DEFAULT_RATING = 1500.0
    DEFAULT_RD = 350.0
    DEFAULT_VOLATILITY = 0.06

    def __init__(
        self,
        default_rating: float = DEFAULT_RATING,
        default_rd: float = DEFAULT_RD,
        default_volatility: float = DEFAULT_VOLATILITY,
    ) -> None:
        self.default_rating = default_rating
        self.default_rd = default_rd
        self.default_volatility = default_volatility
        self.players: Dict[str, glicko2.Player] = {}
        self.teams: Dict[str, glicko2.Player] = {}
        self._player_names: Dict[str, str] = {}
        self._matches_played: Dict[str, int] = {}
        self._wins: Dict[str, int] = {}

    def _new_player(self, rating: Optional[float] = None, rd: Optional[float] = None) -> glicko2.Player:
        return glicko2.Player(
            rating=self.default_rating if rating is None else rating,
            rd=self.default_rd if rd is None else rd,
            vol=self.default_volatility,
        )

    @staticmethod
    def _key(name: str) -> str:
        return name.strip().casefold()

    def _get_team(self, name: str) -> glicko2.Player:
        key = self._key(name)
        if key not in self.teams:
            self.teams[key] = self._new_player()
        return self.teams[key]

    def _get_player(self, name: str, team: Optional[str] = None) -> glicko2.Player:
        key = self._key(name)
        if key not in self.players:
            fallback = self._get_team(team) if team else None
            self.players[key] = self._new_player(
                fallback.rating if fallback else None,
                fallback.rd if fallback else None,
            )
            self._player_names[key] = name.strip()
            self._matches_played[key] = 0
            self._wins[key] = 0
        return self.players[key]

    def _aggregate(self, team: str, player_names: Sequence[str]) -> Tuple[float, float]:
        """Return mean roster rating and uncertainty of that mean."""
        roster = [name.strip() for name in player_names if name and name.strip()]
        if not roster:
            rating = self._get_team(team)
            return rating.rating, rating.rd

        ratings: List[float] = []
        deviations: List[float] = []
        team_key = self._key(team)
        for name in roster:
            player = self.players.get(self._key(name))
            if player is not None:
                ratings.append(player.rating)
                deviations.append(player.rd)
            else:
                fallback = self.teams.get(team_key)
                ratings.append(fallback.rating if fallback else self.default_rating)
                deviations.append(fallback.rd if fallback else self.default_rd)
        count = len(roster)
        return sum(ratings) / count, math.sqrt(sum(rd * rd for rd in deviations)) / count

    @staticmethod
    def _probability(rating1: float, rd1: float, rating2: float, rd2: float) -> float:
        q = math.log(10.0) / 400.0
        pooled_rd = math.sqrt(rd1 * rd1 + rd2 * rd2)
        g = 1.0 / math.sqrt(1.0 + 3.0 * q * q * pooled_rd * pooled_rd / (math.pi * math.pi))
        exponent = -g * (rating1 - rating2) / 400.0
        # Stable logistic evaluation, including extreme rating differences.
        if exponent >= 0:
            exp_value = math.exp(-exponent)
            return exp_value / (1.0 + exp_value)
        return 1.0 / (1.0 + math.exp(exponent))

    def predict_win(
        self,
        team1: str,
        team2: str,
        team1_players: Sequence[str],
        team2_players: Sequence[str],
    ) -> float:
        """Return the pre-match probability that team1 wins."""
        rating1, rd1 = self._aggregate(team1, team1_players)
        rating2, rd2 = self._aggregate(team2, team2_players)
        return self._probability(rating1, rd1, rating2, rd2)

    def build_features(
        self,
        team1: str,
        team2: str,
        team1_players: Sequence[str],
        team2_players: Sequence[str],
    ) -> Dict[str, float]:
        """Build pre-match rating, probability, and uncertainty features."""
        rating1, rd1 = self._aggregate(team1, team1_players)
        rating2, rd2 = self._aggregate(team2, team2_players)
        return {
            "glicko2_rating_diff": rating1 - rating2,
            "glicko2_prob": self._probability(rating1, rd1, rating2, rd2),
            "glicko2_rd_diff": rd1 - rd2,
            "glicko2_team1_rd": rd1,
            "glicko2_team2_rd": rd2,
            "glicko2_team1_conservative": rating1 - 2.0 * rd1,
        }

    @staticmethod
    def _update(rating: glicko2.Player, opponent_rating: float, opponent_rd: float, score: float) -> None:
        rating.update_player([opponent_rating], [opponent_rd], [score])

    def update_match(
        self,
        team1: str,
        team2: str,
        team1_players: Sequence[str],
        team2_players: Sequence[str],
        winner: str,
        timestamp: Optional[Any] = None,
    ) -> None:
        """Update player and team states after the match result is known."""
        winner_key = winner.strip().casefold()
        if winner_key not in {"team1", "team2"}:
            raise ValueError("winner must be 'team1' or 'team2'")

        rating1, rd1 = self._aggregate(team1, team1_players)
        rating2, rd2 = self._aggregate(team2, team2_players)
        score1 = 1.0 if winner_key == "team1" else 0.0
        score2 = 1.0 - score1

        team1_roster = list(dict.fromkeys(name.strip() for name in team1_players if name and name.strip()))
        team2_roster = list(dict.fromkeys(name.strip() for name in team2_players if name and name.strip()))
        for name in team1_roster:
            key = self._key(name)
            player = self._get_player(name, team1)
            self._update(player, rating2, rd2, score1)
            self._matches_played[key] += 1
            self._wins[key] += int(score1)
        for name in team2_roster:
            key = self._key(name)
            player = self._get_player(name, team2)
            self._update(player, rating1, rd1, score2)
            self._matches_played[key] += 1
            self._wins[key] += int(score2)

        team1_state = self._get_team(team1)
        team2_state = self._get_team(team2)
        self._update(team1_state, rating2, rd2, score1)
        self._update(team2_state, rating1, rd1, score2)

    def top_players(self, limit: int = 25, min_matches: int = 20) -> List[Dict[str, Any]]:
        """Return players ranked by conservative rating (rating minus twice RD)."""
        qualified = [
            key for key, matches in self._matches_played.items()
            if matches >= min_matches
        ]
        qualified.sort(key=lambda key: -(self.players[key].rating - 2.0 * self.players[key].rd))
        return [
            {
                "player_name": self._player_names[key],
                "rating": self.players[key].rating,
                "rd": self.players[key].rd,
                "volatility": self.players[key].vol,
                "conservative_rating": self.players[key].rating - 2.0 * self.players[key].rd,
                "matches_played": self._matches_played[key],
                "wins": self._wins[key],
            }
            for key in qualified[:limit]
        ]
