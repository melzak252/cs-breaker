"""Map-pool history and pre-match veto features for CS2."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Tuple


ACTIVE_MAP_POOL = (
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


def _team_key(name: str) -> str:
    return name.strip().casefold()


class VetoEngine:
    """Track team map records, simulate the Major BO3 veto, and emit features."""
    active_pool: Tuple[str, ...] = ACTIVE_MAP_POOL

    # A small symmetric Beta prior keeps unseen/sparse map rates near 50%.
    MAP_PRIOR_WINS = 2.0
    MAP_PRIOR_LOSSES = 2.0
    def __init__(self) -> None:
        self.map_matches: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.map_wins: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.map_bans: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def update_state(
        self,
        team1: str,
        team2: str,
        played_maps: List[str],
        map_winners: Optional[Mapping[str, str]] = None,
        banned_maps: Optional[Mapping[str, List[str]]] = None,
    ) -> None:
        """Record completed maps and bans; map_winners maps each map to a team.

        Winners may be team names or the relative labels ``team1``/``team2``.
        Bans are supplied as ``{team_name: [map, ...]}``.
        """
        key1, key2 = _team_key(team1), _team_key(team2)
        winner_by_map = {
            _canonical_map(map_name): winner
            for map_name, winner in (map_winners or {}).items()
        }
        for map_name in played_maps:
            map_key = _canonical_map(map_name)
            self.map_matches[key1][map_key] += 1
            self.map_matches[key2][map_key] += 1
            winner = winner_by_map.get(map_key)
            if winner is None:
                continue
            winner_key = _team_key(winner)
            if winner_key == "team1":
                winner_key = key1
            elif winner_key == "team2":
                winner_key = key2
            if winner_key == key1:
                self.map_wins[key1][map_key] += 1
            elif winner_key == key2:
                self.map_wins[key2][map_key] += 1

        if banned_maps:
            for team_name, maps in banned_maps.items():
                team_key = _team_key(team_name)
                if team_key == "team1":
                    team_key = key1
                elif team_key == "team2":
                    team_key = key2
                for map_name in maps:
                    self.map_bans[team_key][_canonical_map(map_name)] += 1

    def update_match(
        self,
        match: object,
        map_results: Optional[List[object]] = None,
        banned_maps: Optional[Mapping[str, List[str]]] = None,
    ) -> None:
        """Consume a fixture, optionally using per-map results and veto bans.

        A fixture's series winner is not enough to infer individual map
        winners, so wins are recorded only when per-map outcomes are supplied.
        """
        winners: Dict[str, str] = {}
        if map_results:
            for result in map_results:
                winners[result.map_name] = result.winner_team
        self.update_state(
            match.team1,
            match.team2,
            list(match.maps),
            winners,
            banned_maps if banned_maps is not None else getattr(match, "banned_maps", None),
        )

    def _win_rate(self, team: str, map_name: str) -> float:
        team_key, map_key = _team_key(team), _canonical_map(map_name)
        matches = self.map_matches[team_key][map_key]
        wins = self.map_wins[team_key][map_key]
        return (wins + self.MAP_PRIOR_WINS) / (
            matches + self.MAP_PRIOR_WINS + self.MAP_PRIOR_LOSSES
        )

    def _ban_priority(self, team: str, map_name: str) -> Tuple[int, float, int]:
        team_key = _team_key(team)
        matches = self.map_matches[team_key][map_name]
        # Least-played maps are the default permaban; among equally played
        # maps, prefer the map with the lower rate, with pool order as tie-break.
        return (matches, self._win_rate(team, map_name), ACTIVE_MAP_POOL.index(map_name))

    def _pick_priority(self, team: str, map_name: str) -> Tuple[float, int, int]:
        team_key = _team_key(team)
        return (
            -self._win_rate(team, map_name),
            -self.map_matches[team_key][map_name],
            ACTIVE_MAP_POOL.index(map_name),
        )

    def simulate_veto(self, team1: str, team2: str) -> Tuple[List[str], str]:
        """Simulate the seven-map Major BO3 sequence and return both picks + decider."""
        available = list(ACTIVE_MAP_POOL)

        def remove_best_ban(team: str) -> None:
            chosen = min(available, key=lambda map_name: self._ban_priority(team, map_name))
            available.remove(chosen)

        def choose_pick(team: str) -> str:
            chosen = min(available, key=lambda map_name: self._pick_priority(team, map_name))
            available.remove(chosen)
            return chosen

        # Team 1 ban, team 2 ban, team 1 pick, team 2 pick,
        # team 2 ban, team 1 ban, then the sole remaining decider.
        remove_best_ban(team1)
        remove_best_ban(team2)
        picked_maps = [choose_pick(team1), choose_pick(team2)]
        remove_best_ban(team2)
        remove_best_ban(team1)
        return picked_maps, available[0]

    @staticmethod
    def bo3_series_probability(p1: float, p2: float, p3: float) -> float:
        """Exact probability of winning two of three maps with map-specific odds."""
        return p1 * p2 + p1 * (1.0 - p2) * p3 + (1.0 - p1) * p2 * p3

    def _map_probability(self, team1: str, team2: str, map_name: str) -> float:
        rate1 = self._win_rate(team1, map_name)
        rate2 = self._win_rate(team2, map_name)
        # Comparing posterior logits makes swapping teams exactly antisymmetric.
        logit_diff = math.log(rate1 / (1.0 - rate1)) - math.log(rate2 / (1.0 - rate2))
        return 1.0 / (1.0 + math.exp(-logit_diff))

    def build_veto_features(
        self,
        team1: str,
        team2: str,
        played_maps: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Build antisymmetric map-pool and hierarchical series features."""
        pool_diffs = [
            self._win_rate(team1, map_name) - self._win_rate(team2, map_name)
            for map_name in ACTIVE_MAP_POOL
        ]
        depth1 = sum(self.map_matches[_team_key(team1)][map_name] > 0 for map_name in ACTIVE_MAP_POOL)
        depth2 = sum(self.map_matches[_team_key(team2)][map_name] > 0 for map_name in ACTIVE_MAP_POOL)

        # Keep map selection independent of query orientation so the matchup
        # probability remains exactly complementary when team order is swapped.
        ordered_teams = sorted((team1, team2), key=_team_key)
        if played_maps is None:
            picks, decider = self.simulate_veto(*ordered_teams)
            series_maps = [*picks, decider]
        else:
            series_maps = [_canonical_map(map_name) for map_name in played_maps[:3]]
            if len(series_maps) != 3:
                picks, decider = self.simulate_veto(*ordered_teams)
                series_maps = [*picks, decider]

        probabilities = [self._map_probability(team1, team2, map_name) for map_name in series_maps]
        return {
            "veto_pool_wr_diff": float(sum(pool_diffs) / len(pool_diffs)),
            "veto_hierarchical_bo3_prob": float(
                self.bo3_series_probability(*probabilities)
            ),
            "veto_map_depth_diff": float(depth1 - depth2),
        }
