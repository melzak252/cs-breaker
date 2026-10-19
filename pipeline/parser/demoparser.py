"""CS2 demo event parsing backed by :mod:`demoparser2` when available."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from demoparser2 import DemoParser
except Exception:  # Import errors (including broken optional native installs).
    DemoParser = None  # type: ignore[assignment,misc]


class CS2DemoParser:
    """Extract useful match events from a CS2 ``.dem`` file.

    Parsing failures are intentionally non-fatal: callers can still consume a
    well-formed result with empty event lists for synthetic fixtures or damaged
    demos.
    """

    def __init__(self, demo_path: str):
        self.demo_path = str(demo_path)
        self._path = Path(self.demo_path)
        if not self._path.is_file():
            raise FileNotFoundError(f"Demo file does not exist: {self.demo_path}")
        self._parser: Optional[Any] = None
        self._error: Optional[str] = None
        if DemoParser is None:
            self._error = "demoparser2 is unavailable"
        else:
            try:
                self._parser = DemoParser(self.demo_path)
            except Exception as exc:
                self._error = str(exc)

    @staticmethod
    def _records(value: Any) -> List[Dict[str, Any]]:
        """Convert demoparser's pandas/list result to plain record dictionaries."""
        if value is None:
            return []
        if hasattr(value, "to_dict"):
            try:
                records = value.to_dict("records")
            except TypeError:
                records = value.to_dict()
                if isinstance(records, dict):
                    return [records]
            return [dict(row) for row in records]
        if isinstance(value, dict):
            return [dict(value)]
        try:
            return [dict(row) for row in value]
        except (TypeError, ValueError):
            return []

    @staticmethod
    def _value(row: Dict[str, Any], *keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in row and row[key] is not None:
                return row[key]
        return default

    def _event(self, name: str, *, player_extra_info: bool = False) -> List[Dict[str, Any]]:
        if self._parser is None:
            return []
        try:
            kwargs = {"player_extra_info": True} if player_extra_info else {}
            return self._records(self._parser.parse_event(name, **kwargs))
        except Exception as exc:
            self._error = str(exc)
            return []

    def parse_header(self) -> Dict[str, Any]:
        """Return map, tick-rate, and server metadata when present."""
        header: Dict[str, Any] = {}
        if self._parser is not None:
            try:
                raw = self._parser.parse_header()
                if isinstance(raw, dict):
                    header = dict(raw)
                elif hasattr(raw, "to_dict"):
                    header = dict(raw.to_dict())
            except Exception as exc:
                self._error = str(exc)
        # Preserve the backend's original header keys while presenting stable
        # convenient names for consumers.
        header.setdefault("map_name", self._value(header, "map_name", "mapname", "map"))
        header.setdefault("tick_rate", self._value(header, "tick_rate", "tickrate", "tick_rate_num"))
        header.setdefault("server_name", self._value(header, "server_name", "servername", "server"))
        header.setdefault("server_info", self._value(header, "server_info", "server_name", "servername"))
        return header

    def parse_rounds(self) -> List[Dict[str, Any]]:
        """Parse round-end events with winner, reason, and score information."""
        events = []
        for name in ("round_end", "round_officially_ended"):
            events.extend(
                {**event, "event_name": name}
                for event in self._event(name)
            )
        rounds: List[Dict[str, Any]] = []
        for event in events:
            winner = self._value(event, "winner", "winning_team", "team")
            if isinstance(winner, (int, float)):
                winner = {2: "T", 3: "CT"}.get(int(winner), winner)
            elif isinstance(winner, str):
                normalized = winner.upper()
                if normalized in {"TERRORIST", "TERRORISTS"}:
                    winner = "T"
                elif normalized in {"COUNTERTERRORIST", "COUNTERTERRORISTS", "COUNTER-TERRORIST"}:
                    winner = "CT"
                else:
                    winner = normalized
            rounds.append({
                "event_name": self._value(event, "event_name", default="round_end"),
                "round": self._value(event, "total_rounds_played", "round", "round_num"),
                "tick": self._value(event, "tick"),
                "winner": winner,
                "reason": self._value(event, "reason", "round_end_reason"),
                "team1_score": self._value(event, "team1_score", "ct_score", "score_ct"),
                "team2_score": self._value(event, "team2_score", "t_score", "score_t"),
                "ct_score": self._value(event, "ct_score", "score_ct"),
                "t_score": self._value(event, "t_score", "score_t"),
            })
        return rounds

    def parse_deaths(self) -> List[Dict[str, Any]]:
        """Parse player deaths, requesting demoparser's extra player fields."""
        fields = ("attacker_steamid", "user_steamid", "assister_steamid", "weapon", "headshot", "penetrated")
        return [
            {**event, **{field: event.get(field) for field in fields}}
            for event in self._event("player_death", player_extra_info=True)
        ]

    def parse_damages(self) -> List[Dict[str, Any]]:
        """Parse damage events with attacker, victim, damage, and hitgroup."""
        aliases = {
            "attacker": ("attacker_name", "attacker"),
            "victim": ("user_name", "victim_name", "victim"),
            "dmg_health": ("dmg_health", "damage", "health_damage"),
            "hitgroup": ("hitgroup",),
        }
        result = []
        for event in self._event("player_hurt"):
            record = dict(event)
            for field, keys in aliases.items():
                record[field] = self._value(event, *keys)
            result.append(record)
        return result

    def parse_utility(self) -> List[Dict[str, Any]]:
        """Parse grenade detonation events."""
        result = []
        for name in ("smokegrenade_detonate", "flashbang_detonate", "hegrenade_detonate"):
            for event in self._event(name):
                record = dict(event)
                record.setdefault("event_name", name)
                result.append(record)
        return result

    def parse_all_events(self) -> Dict[str, Any]:
        """Return normalized header and event collections for a demo."""
        result = {
            "demo_path": self.demo_path,
            "header": self.parse_header(),
            "rounds": self.parse_rounds(),
            "deaths": self.parse_deaths(),
            "damages": self.parse_damages(),
            "utility": self.parse_utility(),
            "synthetic": self._parser is None or self._error is not None,
        }
        if self._error:
            result["parse_error"] = self._error
        return result
