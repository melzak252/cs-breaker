"""OpenSkill Plackett-Luce ratings for Counter-Strike player lineups."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from openskill.models import PlackettLuce


class OpenSkillRatingEngine:
    """Track individual OpenSkill ratings across matches and changing rosters."""

    def __init__(self) -> None:
        self.model = PlackettLuce()
        self.players: Dict[str, Any] = {}
        self.matches_played: Dict[str, int] = {}
        self.team_ratings: Dict[str, tuple[float, float]] = {}

    @staticmethod
    def _clean_players(player_names: Sequence[str]) -> List[str]:
        return [name.strip() for name in player_names if name and name.strip()]

    def _team_rating(self, team: str) -> Any:
        """Return a team's cached fallback rating, or the default rating."""
        values = self.team_ratings.get(team.strip().casefold())
        if values is None:
            return self.model.rating()
        return self.model.rating(mu=values[0], sigma=values[1])

    def _lineup_ratings(self, team: str, player_names: Sequence[str]) -> List[Any]:
        fallback = self._team_rating(team)
        names = self._clean_players(player_names)
        if not names:
            return [fallback]
        return [self.players.get(name.casefold(), fallback) for name in names]

    @staticmethod
    def _lineup_stats(ratings: Sequence[Any]) -> tuple[float, float, float]:
        if not ratings:
            return 25.0, 25.0 / 3.0, 25.0 - 3.0 * (25.0 / 3.0)
        mean_mu = sum(rating.mu for rating in ratings) / len(ratings)
        mean_sigma = sum(rating.sigma for rating in ratings) / len(ratings)
        return mean_mu, mean_sigma, mean_mu - 3.0 * mean_sigma

    def predict_win(
        self,
        team1: str,
        team2: str,
        team1_players: Sequence[str],
        team2_players: Sequence[str],
    ) -> float:
        """Estimate the probability that team1 wins using Plackett-Luce."""
        t1_ratings = self._lineup_ratings(team1, team1_players)
        t2_ratings = self._lineup_ratings(team2, team2_players)
        return float(self.model.predict_win([t1_ratings, t2_ratings])[0])

    def build_features(
        self,
        team1: str,
        team2: str,
        team1_players: Sequence[str],
        team2_players: Sequence[str],
    ) -> Dict[str, float]:
        """Build rating, win-probability, and uncertainty differentials."""
        t1_ratings = self._lineup_ratings(team1, team1_players)
        t2_ratings = self._lineup_ratings(team2, team2_players)
        t1_mu, t1_sigma, t1_ordinal = self._lineup_stats(t1_ratings)
        t2_mu, t2_sigma, t2_ordinal = self._lineup_stats(t2_ratings)
        return {
            "openskill_mu_diff": t1_mu - t2_mu,
            "openskill_prob": float(self.model.predict_win([t1_ratings, t2_ratings])[0]),
            "openskill_sigma_diff": t1_sigma - t2_sigma,
            "openskill_ordinal_diff": t1_ordinal - t2_ordinal,
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
        """Update ratings after a match; winner is ``team1`` or ``team2``."""
        t1_names = self._clean_players(team1_players)
        t2_names = self._clean_players(team2_players)
        t1_ratings = self._lineup_ratings(team1, t1_names)
        t2_ratings = self._lineup_ratings(team2, t2_names)
        ranks = [1, 2] if winner == "team1" else [2, 1]
        updated = self.model.rate([t1_ratings, t2_ratings], ranks=ranks)

        for names, ratings in ((t1_names, updated[0]), (t2_names, updated[1])):
            for name, rating in zip(names, ratings):
                key = name.casefold()
                self.players[key] = rating
                self.matches_played[key] = self.matches_played.get(key, 0) + 1

        for team, ratings in ((team1, updated[0]), (team2, updated[1])):
            if ratings:
                team_key = team.strip().casefold()
                self.team_ratings[team_key] = (
                    sum(rating.mu for rating in ratings) / len(ratings),
                    sum(rating.sigma for rating in ratings) / len(ratings),
                )

    def top_players(self, limit: int = 25, min_matches: int = 20) -> List[Dict[str, Any]]:
        """Return qualified players ranked by ordinal (mu - 3*sigma)."""
        ranked = [
            {
                "player": name,
                "mu": rating.mu,
                "sigma": rating.sigma,
                "ordinal": rating.mu - 3.0 * rating.sigma,
                "matches_played": self.matches_played.get(name, 0),
            }
            for name, rating in self.players.items()
            if self.matches_played.get(name, 0) >= min_matches
        ]
        ranked.sort(key=lambda player: -player["ordinal"])
        return ranked[:limit]
