"""Core contracts and dataclasses for the cs-breaker pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Tournament:
    """Tournament metadata extracted from Liquipedia or manifests."""
    tournament_id: str
    name: str
    tier: str  # S-Tier, A-Tier, Qualifier, etc.
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    location: Optional[str] = None
    liquipedia_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MatchFixture:
    """Scheduled or completed match fixture."""
    match_id: str
    tournament_name: str
    timestamp: datetime
    team1: str
    team2: str
    format: str = "bo3"  # bo1, bo3, bo5
    score1: Optional[int] = None
    score2: Optional[int] = None
    winner: Optional[str] = None  # "team1" or "team2"
    demo_url: Optional[str] = None
    maps: List[str] = field(default_factory=list)
    team1_players: List[str] = field(default_factory=list)
    team2_players: List[str] = field(default_factory=list)
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass(frozen=True)
class DemoFileMeta:
    """Metadata describing a downloaded or local .dem file."""
    demo_id: str
    match_id: str
    file_path: str
    map_name: str
    file_size_bytes: int
    sha256: str
    download_url: Optional[str] = None


@dataclass(frozen=True)
class PlayerMapStats:
    """Parsed per-player statistics for a single map from demoparser2."""
    player_id: str  # SteamID64 or handle
    player_name: str
    team_name: str
    team_num: int  # 2 = T, 3 = CT
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    headshot_kills: int = 0
    damage_dealt: float = 0.0
    rounds_played: int = 0
    first_kills: int = 0
    first_deaths: int = 0
    utility_damage: float = 0.0
    flash_assists: int = 0
    clutches_won: int = 0

    @property
    def adr(self) -> float:
        return self.damage_dealt / max(self.rounds_played, 1)

    @property
    def kd_ratio(self) -> float:
        return self.kills / max(self.deaths, 1)

    @property
    def hs_percent(self) -> float:
        return (self.headshot_kills / max(self.kills, 1)) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["adr"] = round(self.adr, 2)
        d["kd_ratio"] = round(self.kd_ratio, 2)
        d["hs_percent"] = round(self.hs_percent, 1)
        return d


@dataclass(frozen=True)
class ParsedMapResult:
    """Parsed full map result derived from demoparser2 events."""
    match_id: str
    map_name: str
    team1_name: str
    team2_name: str
    team1_score: int
    team2_score: int
    total_rounds: int
    winner_team: str
    player_stats: List[PlayerMapStats] = field(default_factory=list)
    rounds_data: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "map_name": self.map_name,
            "team1_name": self.team1_name,
            "team2_name": self.team2_name,
            "team1_score": self.team1_score,
            "team2_score": self.team2_score,
            "total_rounds": self.total_rounds,
            "winner_team": self.winner_team,
            "player_stats": [p.to_dict() for p in self.player_stats],
            "rounds_data": self.rounds_data,
        }
