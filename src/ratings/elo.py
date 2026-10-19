"""Elo rating system implementation for CS2 match outcome forecasting."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple


def win_probability(rating1: float, rating2: float) -> float:
    """Calculate Elo logistic win probability for team 1 against team 2 with a 400-point base-10 scale."""
    diff = rating1 - rating2
    z = diff * math.log(10) / 400.0
    z = max(-700.0, min(700.0, z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


class EloRatingSystem:
    """Retrospective Elo rating engine with atomic date-batch updates."""

    def __init__(self, k: float = 32.0, default_rating: float = 1500.0) -> None:
        self.k = float(k)
        self.default_rating = float(default_rating)
        self.ratings: dict[str, float] = {}
        self.match_counts: dict[str, int] = {}

    def get_rating(self, team: str) -> float:
        """Get the current rating of a team, defaulting to default_rating."""
        return self.ratings.get(team, self.default_rating)

    def predict(self, team1: str, team2: str) -> float:
        """Predict win probability for team1 facing team2."""
        r1 = self.get_rating(team1)
        r2 = self.get_rating(team2)
        return win_probability(r1, r2)

    def update(self, team1: str, team2: str, outcome_team1: float) -> tuple[float, float]:
        """Update ratings for team1 and team2 after observing outcome_team1 in [0.0, 1.0]."""
        p1 = self.predict(team1, team2)
        r1 = self.get_rating(team1)
        r2 = self.get_rating(team2)

        delta1 = self.k * (outcome_team1 - p1)
        delta2 = -delta1

        self.ratings[team1] = r1 + delta1
        self.ratings[team2] = r2 + delta2

        self.match_counts[team1] = self.match_counts.get(team1, 0) + 1
        self.match_counts[team2] = self.match_counts.get(team2, 0) + 1

        return self.ratings[team1], self.ratings[team2]
