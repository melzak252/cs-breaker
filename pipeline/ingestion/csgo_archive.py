"""Adapter to load and normalize historical CS:GO pro matches (2012-2023)."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline.contracts import MatchFixture

# Map short codes to standardized Counter-Strike map names
MAP_NAME_MAPPING: Dict[str, str] = {
    "inf": "de_inferno",
    "mrg": "de_mirage",
    "mirage": "de_mirage",
    "inferno": "de_inferno",
    "d2": "de_dust2",
    "dust2": "de_dust2",
    "nuke": "de_nuke",
    "ovp": "de_overpass",
    "overpass": "de_overpass",
    "trn": "de_train",
    "train": "de_train",
    "cbl": "de_cbble",
    "cbble": "de_cbble",
    "cobblestone": "de_cbble",
    "cache": "de_cache",
    "vert": "de_vertigo",
    "vertigo": "de_vertigo",
    "vtg": "de_vertigo",
    "anc": "de_ancient",
    "ancient": "de_ancient",
    "anb": "de_anubis",
    "anubis": "de_anubis",
    "tuscan": "de_tuscan",
    "season": "de_season",
}


class CSGOHistoricalAdapter:
    """Ingest and normalize 127k+ historical CS:GO matches from 2012 through 2023."""

    @staticmethod
    def _clean_round_score(value: str) -> Optional[int]:
        """Extract clean integer score from string like ' (16)' or '(4)'."""
        if not value:
            return None
        digits = re.sub(r"[^\d]", "", value)
        return int(digits) if digits else None

    @staticmethod
    def _parse_timestamp(row: Dict[str, str]) -> datetime:
        """Parse timestamp reliably from datetime, date_unix (ms), date_unix_iso, or date_ymd."""
        dt_str = row.get("datetime", "").strip()
        if dt_str:
            try:
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                if dt.utcoffset() is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except (ValueError, TypeError):
                pass

        unix_ms_str = row.get("date_unix", "").strip()
        if unix_ms_str and unix_ms_str.isdigit():
            epoch_seconds = int(unix_ms_str) / 1000.0
            return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        
        iso_str = row.get("date_unix_iso", "").strip()
        if iso_str:
            try:
                dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
                if dt.utcoffset() is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except (ValueError, TypeError):
                pass

        ymd = row.get("date_ymd", "2012-01-01").strip()
        return datetime.fromisoformat(f"{ymd}T00:00:00+00:00")

    @classmethod
    def load_historical_fixtures(
        cls,
        csv_path: Optional[str] = None,
        exclude_ties: bool = True,
        limit: Optional[int] = None,
    ) -> List[MatchFixture]:
        """Load historical games as normalized MatchFixture objects sorted chronologically."""
        if csv_path is None:
            default_full = Path("data/raw/csgo_historical/csgo_pro_games_data.csv")
            default_list = Path("data/raw/csgo_historical/historic_games_list.csv")
            csv_path = str(default_full) if default_full.exists() else str(default_list)

        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CS:GO historical CSV not found at {csv_path}")

        fixtures: List[MatchFixture] = []
        seen_ids = set()

        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            for i, row in enumerate(reader):
                if limit is not None and len(fixtures) >= limit:
                    break

                team1 = row.get("team1", "").strip()
                team2 = row.get("team2", "").strip()
                if not team1 or not team2 or team1.casefold() == team2.casefold():
                    continue

                s1 = cls._clean_round_score(row.get("team1_rounds", ""))
                s2 = cls._clean_round_score(row.get("team2_rounds", ""))
                if s1 is None or s2 is None or s1 < 0 or s2 < 0:
                    continue

                if s1 == s2 and exclude_ties:
                    continue

                winner = "team1" if s1 > s2 else "team2"
                timestamp = cls._parse_timestamp(row)

                # Extract mapstats id from game_link
                link = row.get("game_link", "").strip()
                match_id_match = re.search(r"/mapstatsid/(\d+)/", link)
                if match_id_match:
                    match_id = f"csgo_m{match_id_match.group(1)}"
                else:
                    match_id = f"csgo_row_{i}"

                # Ensure uniqueness if duplicated
                base_id = match_id
                suffix = 1
                while match_id in seen_ids:
                    match_id = f"{base_id}_{suffix}"
                    suffix += 1
                seen_ids.add(match_id)

                raw_map = row.get("map_name_short", "").strip().lower()
                clean_map = MAP_NAME_MAPPING.get(raw_map, f"de_{raw_map}" if raw_map else "unknown")

                t1_players = [
                    row.get(f"team1_p{k}_name", "").strip()
                    for k in range(1, 6)
                    if row.get(f"team1_p{k}_name", "").strip()
                ]
                t2_players = [
                    row.get(f"team2_p{k}_name", "").strip()
                    for k in range(1, 6)
                    if row.get(f"team2_p{k}_name", "").strip()
                ]

                fixture = MatchFixture(
                    match_id=match_id,
                    tournament_name=row.get("event_name", "Unknown Tournament").strip(),
                    timestamp=timestamp,
                    team1=team1,
                    team2=team2,
                    format="bo1",
                    score1=s1,
                    score2=s2,
                    winner=winner,
                    demo_url=None,
                    maps=[clean_map],
                    team1_players=t1_players,
                    team2_players=t2_players,
                )
                fixtures.append(fixture)

        # Sort strictly chronologically
        fixtures.sort(key=lambda m: (m.timestamp, m.match_id))
        return fixtures
