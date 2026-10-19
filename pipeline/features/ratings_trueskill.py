"""TrueSkill player ratings and pre-match lineup features."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import trueskill


class TrueSkillRatingEngine:
    """Track player TrueSkill posteriors across changing Counter-Strike rosters."""

    def __init__(self) -> None:
        self.env = trueskill.TrueSkill(draw_probability=0.0)
        self.default_rating = self.env.create_rating(mu=25.0, sigma=25.0 / 3.0)
        self.players: Dict[str, trueskill.Rating] = {}
        self.matches_played: Dict[str, int] = {}
        self.wins: Dict[str, int] = {}
        self.player_names: Dict[str, str] = {}
        self.team_ratings: Dict[str, trueskill.Rating] = {}

    @staticmethod
    def _clean_players(player_names: Sequence[str]) -> List[Tuple[str, str]]:
        return [(name.strip().casefold(), name.strip()) for name in player_names if name and name.strip()]

    def _lineup_ratings(self, team: str, player_names: Sequence[str]) -> List[trueskill.Rating]:
        """Resolve known players and cold-start players without mutating state."""
        team_key = team.strip().casefold()
        fallback = self.team_ratings.get(team_key, self.default_rating)
        lineup = [
            self.players.get(key, fallback)
            for key, _ in self._clean_players(player_names)
        ]
        if not lineup:
            # An empty roster still has a defined contest rating from team history/default.
            lineup.append(fallback)
        return lineup

    @staticmethod
    def _team_stats(ratings: Sequence[trueskill.Rating]) -> Tuple[float, float]:
        mu = sum(rating.mu for rating in ratings)
        sigma = math.sqrt(sum(rating.sigma * rating.sigma for rating in ratings))
        return mu, sigma

    def _win_probability(
        self, ratings1: Sequence[trueskill.Rating], ratings2: Sequence[trueskill.Rating]
    ) -> float:
        mu1, _ = self._team_stats(ratings1)
        mu2, _ = self._team_stats(ratings2)
        variance1 = sum(rating.sigma * rating.sigma for rating in ratings1)
        variance2 = sum(rating.sigma * rating.sigma for rating in ratings2)
        denominator = math.sqrt(
            (len(ratings1) + len(ratings2)) * self.env.beta**2 + variance1 + variance2
        )
        return self.env.cdf((mu1 - mu2) / denominator)

    def predict_win(
        self,
        team1: str,
        team2: str,
        team1_players: Sequence[str],
        team2_players: Sequence[str],
    ) -> float:
        """Return the exact Gaussian probability that team 1 wins."""
        ratings1 = self._lineup_ratings(team1, team1_players)
        ratings2 = self._lineup_ratings(team2, team2_players)
        return self._win_probability(ratings1, ratings2)

    def build_features(
        self,
        team1: str,
        team2: str,
        team1_players: Sequence[str],
        team2_players: Sequence[str],
    ) -> Dict[str, float]:
        ratings1 = self._lineup_ratings(team1, team1_players)
        ratings2 = self._lineup_ratings(team2, team2_players)
        mu1, sigma1 = self._team_stats(ratings1)
        mu2, sigma2 = self._team_stats(ratings2)
        return {
            "trueskill_mu_diff": mu1 - mu2,
            "trueskill_prob": self._win_probability(ratings1, ratings2),
            "trueskill_sigma_diff": sigma1 - sigma2,
            "trueskill_team1_conservative": mu1 - 3.0 * sigma1,
            "trueskill_team2_conservative": mu2 - 3.0 * sigma2,
        }

    def update_match(
        self,
        team1: str,
        team2: str,
        team1_players: Sequence[str],
        team2_players: Sequence[str],
        winner: str,
        timestamp: Optional[Any] = None,
    ) -> None:
        """Update participating players after the match result is known."""
        clean1 = self._clean_players(team1_players)
        clean2 = self._clean_players(team2_players)
        ratings1 = self._lineup_ratings(team1, team1_players)
        ratings2 = self._lineup_ratings(team2, team2_players)
        rated1, rated2 = self.env.rate(
            [ratings1, ratings2], ranks=[0, 1] if winner == "team1" else [1, 0]
        )

        for (key, display_name), rating in zip(clean1, rated1):
            self.players[key] = rating
            self.player_names.setdefault(key, display_name)
            self.matches_played[key] = self.matches_played.get(key, 0) + 1
            if winner == "team1":
                self.wins[key] = self.wins.get(key, 0) + 1
            else:
                self.wins.setdefault(key, 0)
        for (key, display_name), rating in zip(clean2, rated2):
            self.players[key] = rating
            self.player_names.setdefault(key, display_name)
            self.matches_played[key] = self.matches_played.get(key, 0) + 1
            if winner != "team1":
                self.wins[key] = self.wins.get(key, 0) + 1
            else:
                self.wins.setdefault(key, 0)

        if clean1:
            self.team_ratings[team1.strip().casefold()] = self._average_rating(
                [self.players[key] for key, _ in clean1]
            )
        if clean2:
            self.team_ratings[team2.strip().casefold()] = self._average_rating(
                [self.players[key] for key, _ in clean2]
            )

    def _average_rating(self, ratings: Sequence[trueskill.Rating]) -> trueskill.Rating:
        return self.env.create_rating(
            mu=sum(rating.mu for rating in ratings) / len(ratings),
            sigma=math.sqrt(sum(rating.sigma**2 for rating in ratings) / len(ratings)),
        )

    def top_players(self, limit: int = 25, min_matches: int = 20) -> List[Dict[str, Any]]:
        """Return players meeting the match threshold, ranked by conservative skill."""
        qualified = [
            key for key in self.players
            if self.matches_played.get(key, 0) >= min_matches
        ]
        qualified.sort(key=lambda key: -(self.players[key].mu - 3.0 * self.players[key].sigma))
        return [
            {
                "player_name": self.player_names[key],
                "mu": self.players[key].mu,
                "sigma": self.players[key].sigma,
                "conservative_skill": self.players[key].mu - 3.0 * self.players[key].sigma,
                "matches_played": self.matches_played[key],
                "wins": self.wins.get(key, 0),
            }
            for key in qualified[:limit]
        ]
