"""Individual player rating engine tracking career Elo and dynamic lineup ratings."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class PlayerCareerProfile:
    """Historical career profile and rating for an individual player."""
    player_name: str
    elo: float = 1500.0
    matches_played: int = 0
    wins: int = 0
    rounds_played: int = 0
    teams: List[str] = field(default_factory=list)
    last_active: Optional[str] = None

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.matches_played, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_name": self.player_name,
            "elo": round(self.elo, 1),
            "matches_played": self.matches_played,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 3),
            "teams": self.teams[-5:],
            "last_active": self.last_active,
        }


class IndividualPlayerRatingEngine:
    """Tracks continuous individual player ratings across matches and rosters."""

    def __init__(self, default_elo: float = 1500.0, k_base: float = 24.0) -> None:
        self.default_elo = default_elo
        self.k_base = k_base
        self.players: Dict[str, PlayerCareerProfile] = {}
        # Team default rating cache for teams with unknown players
        self.team_ratings: Dict[str, float] = {}

    def get_player(self, name: str) -> PlayerCareerProfile:
        """Get or initialize player profile."""
        key = name.strip().casefold()
        if key not in self.players:
            self.players[key] = PlayerCareerProfile(player_name=name.strip(), elo=self.default_elo)
        return self.players[key]

    def compute_lineup_rating(self, player_names: Sequence[str], team_fallback: Optional[str] = None) -> Tuple[float, float, float]:
        """Compute (mean_elo, mean_matches_played, mean_win_rate) for a 5-man roster."""
        clean_names = [p.strip() for p in player_names if p and p.strip()]
        if not clean_names:
            # Fallback to team rating or default
            fallback_elo = self.team_ratings.get((team_fallback or "").strip().casefold(), self.default_elo)
            return fallback_elo, 0.0, 0.50

        elos: List[float] = []
        matches: List[int] = []
        win_rates: List[float] = []

        for p_name in clean_names:
            key = p_name.casefold()
            if key in self.players:
                profile = self.players[key]
                elos.append(profile.elo)
                matches.append(profile.matches_played)
                win_rates.append(profile.win_rate)
            else:
                # Cold start player uses team rating if available, else default
                team_key = (team_fallback or "").strip().casefold()
                p_elo = self.team_ratings.get(team_key, self.default_elo)
                elos.append(p_elo)
                matches.append(0)
                win_rates.append(0.50)

        mean_elo = sum(elos) / len(elos)
        mean_matches = sum(matches) / len(matches)
        mean_wr = sum(win_rates) / len(win_rates)
        return mean_elo, mean_matches, mean_wr

    def build_lineup_features(
        self,
        team1: str,
        team2: str,
        team1_players: Sequence[str],
        team2_players: Sequence[str],
    ) -> Dict[str, float]:
        """Generate pre-match differential features derived from individual player ratings."""
        t1_elo, t1_exp, t1_wr = self.compute_lineup_rating(team1_players, team_fallback=team1)
        t2_elo, t2_exp, t2_wr = self.compute_lineup_rating(team2_players, team_fallback=team2)

        lineup_elo_diff = t1_elo - t2_elo
        exp_z = math.exp(-lineup_elo_diff * math.log(10) / 400.0)
        lineup_prob = 1.0 / (1.0 + exp_z) if math.isfinite(exp_z) else (0.0 if lineup_elo_diff < 0 else 1.0)

        return {
            "player_lineup_elo_diff": round(lineup_elo_diff, 2),
            "player_lineup_prob": round(lineup_prob, 4),
            "player_lineup_exp_diff": round(t1_exp - t2_exp, 1),
            "player_lineup_wr_diff": round(t1_wr - t2_wr, 4),
        }

    def update_match(
        self,
        team1: str,
        team2: str,
        team1_players: Sequence[str],
        team2_players: Sequence[str],
        winner: str,  # 'team1' or 'team2'
        date_iso: Optional[str] = None,
    ) -> None:
        """Update individual player ratings and records strictly after pre-match features were emitted."""
        t1_elo, _, _ = self.compute_lineup_rating(team1_players, team_fallback=team1)
        t2_elo, _, _ = self.compute_lineup_rating(team2_players, team_fallback=team2)

        # Expected outcome for team 1
        z = (t1_elo - t2_elo) * math.log(10) / 400.0
        p1 = 1.0 / (1.0 + math.exp(-z)) if abs(z) < 30 else (1.0 if z > 0 else 0.0)

        actual_1 = 1.0 if winner == "team1" else 0.0
        delta = self.k_base * (actual_1 - p1)

        # Update Team 1 players
        t1_clean = [p.strip() for p in team1_players if p and p.strip()]
        for p_name in t1_clean:
            prof = self.get_player(p_name)
            # Adaptive K: higher K for younger players, lower for established players
            k_mult = 1.5 if prof.matches_played < 20 else (1.0 if prof.matches_played < 100 else 0.75)
            prof.elo += delta * k_mult
            prof.matches_played += 1
            if actual_1 == 1.0:
                prof.wins += 1
            if team1 not in prof.teams:
                prof.teams.append(team1)
            if date_iso:
                prof.last_active = date_iso

        # Update Team 2 players
        t2_clean = [p.strip() for p in team2_players if p and p.strip()]
        for p_name in t2_clean:
            prof = self.get_player(p_name)
            k_mult = 1.5 if prof.matches_played < 20 else (1.0 if prof.matches_played < 100 else 0.75)
            prof.elo -= delta * k_mult
            prof.matches_played += 1
            if actual_1 == 0.0:
                prof.wins += 1
            if team2 not in prof.teams:
                prof.teams.append(team2)
            if date_iso:
                prof.last_active = date_iso

        # Update team fallback rating
        if t1_clean:
            self.team_ratings[team1.strip().casefold()] = sum(self.get_player(p).elo for p in t1_clean) / len(t1_clean)
        if t2_clean:
            self.team_ratings[team2.strip().casefold()] = sum(self.get_player(p).elo for p in t2_clean) / len(t2_clean)

    def top_players(self, limit: int = 25, min_matches: int = 50) -> List[Dict[str, Any]]:
        """Return highest-rated players with sufficient sample size."""
        qualified = [p for p in self.players.values() if p.matches_played >= min_matches]
        qualified.sort(key=lambda p: -p.elo)
        return [p.to_dict() for p in qualified[:limit]]
